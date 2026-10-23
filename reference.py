"""MOSS-Image forward with a clean reference latent in the sequence.

The sequence is ``[image | ref | text]``. Ref tokens go through ``video_embedder``
and get no noise. With ``ref_timestep`` (the LoRA's metadata; v2 trains at 0) they get
their own block modulation from that timestep, image and text keep the step's; without
it (v1) they share the step's. On RoPE they sit at ``r = 1`` and
the image's ``f`` (= text length), with their ``(h, w)`` stretched onto the target
grid, ``(i + 0.5)·H/h_ref − 0.5``, so a half-size ref lines up with the target
position for position. The head reads the image rows only. With ``ref_isolate`` (the
LoRA's metadata; colorize v3) the ref queries attend only to the ref keys, image and
text still read the ref, and a block the ``patches_replace`` dispatch skips (the
uncond block-22 skip) still runs the ref rows, so every branch reads the same ref.

Isolated, with its own timestep, the ref's hidden states depend on nothing but the
page: not the step, the prompt or the CFG branch. ``RefKV`` runs the ref through
the blocks once and keeps each block's ref keys (before RoPE) and values; every
later forward runs the image and text rows alone, their queries reading the cached
ref keys rotated for that call's text length (``_main_block``).

``forward_with_ref`` is the ``moss-image`` node's ``MOSSImageTransformer.forward``
(OpenMOSS, Apache-2.0) with the ref rows spliced in; its ``patches_replace``
dispatch is kept, so the uncond block-22 skip (SLG) still applies. It mirrors
anima_lora's ``library/moss/model.py`` (``ref_span`` / ``ref_isolate`` /
``_ref_only``), which the colorize LoRAs were trained against.
"""

import logging
import sys

import torch

logger = logging.getLogger(__name__)


def ref_rope(dit, height, width, ref_hw, text_length, device):
    """RoPE rows ``(1, h·w, head_dim)`` for a ``ref_hw = (h, w)`` grid over an
    ``H × W`` image, built as ``_RoPE.forward`` builds the image rows."""
    bands = dit.rope._default_bands(device)
    rh, rw = ref_hw
    h, w = torch.meshgrid(
        (torch.arange(rh, device=device) + 0.5) * height / rh - 0.5,
        (torch.arange(rw, device=device) + 0.5) * width / rw - 0.5,
        indexing="ij",
    )
    coords = torch.stack(
        (torch.ones_like(h), torch.full_like(h, text_length), h, w), -1
    ).flatten(0, 1)
    coords[:, 2] -= (height - 1) / 2
    coords[:, 3] -= (width - 1) / 2
    phase = coords.unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


def _time_mods(dit, times):
    """``(block (B,6,D), head (B,2,D))`` modulations at ``times`` (B,), as training."""
    # Training computes the sinusoid in float64 and feeds float32 into its MLP.
    half = dit.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    time = dit.time_embedding(embedding.to(dit.time_embedding[0].weight.dtype)).float()
    mod = dit.shared_block_modulation(time.to(dit.shared_block_modulation.weight.dtype))
    head = dit.video_head_modulation(time.to(dit.video_head_modulation.weight.dtype))
    return mod.view(times.shape[0], 6, -1), head.view(times.shape[0], 2, -1)


def _spans(y, span, main, ref):
    """``y`` with ``main`` applied outside ``span = (start, end)`` and ``ref`` inside."""
    s, e = span
    return torch.cat((main(y[:, :s]), ref(y[:, s:e]), main(y[:, e:])), 1)


def _isolated_attention(rotate, attend, q, k, v, rope, heads, span):
    """``moss-image``'s ``_attention`` with the queries in ``span`` attending only to
    its keys: anima_lora's ``_attention`` with ``ref_isolate``."""
    s, e = span
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    main = attend(torch.cat((q[:, :, :s], q[:, :, e:]), 2), k, v, heads, skip_reshape=True)
    ref = attend(q[:, :, s:e], k[:, :, s:e], v[:, :, s:e], heads, skip_reshape=True)
    return torch.cat((main[:, :s], ref, main[:, s:]), 1)


def _ref_block(layer, ops, x, rope, mod, ref_mod, span, isolate):
    """``moss-image``'s ``_Block.forward`` with the rows in ``span`` modulated by
    ``ref_mod`` and, with ``isolate``, attending only to themselves: op for op
    anima_lora's ``_Block`` with ``ref_span`` / ``ref_isolate``. ``ops`` is the
    ``moss-image`` model module's ``(_attention, _situ, _rotate, optimized_attention)``."""
    attention, situ, rotate, attend = ops
    a = mod.unbind(1)
    r = ref_mod.unbind(1)

    def affine(y, i):
        return _spans(
            y,
            span,
            lambda part: part * (1 + a[i + 1][:, None]) + a[i][:, None],
            lambda part: part * (1 + r[i + 1][:, None]) + r[i][:, None],
        )

    def gated(y, i):
        return _spans(y, span, lambda part: part * a[i][:, None], lambda part: part * r[i][:, None])

    y = affine(layer.norm1(x), 0)
    dtype = layer.proj["q"].weight.dtype
    y = y.to(dtype)
    q = layer.q_norm(layer.proj["q"](y).unflatten(-1, (layer.heads, -1))).flatten(2)
    k = layer.k_norm(layer.proj["k"](y).unflatten(-1, (layer.heads, -1))).flatten(2)
    if isolate:
        attn = _isolated_attention(rotate, attend, q, k, layer.proj["v"](y), rope, layer.heads, span)
    else:
        attn = attention(q, k, layer.proj["v"](y), rope, layer.heads)
    x = x + gated(layer.proj["o"](attn), 2)
    y = affine(layer.norm2(x), 3)
    hidden, gate = layer.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
    return x + gated(layer.proj["ffn_down"](situ(hidden, gate)), 5)


def _ref_pass(layer, ops, x, rope, mod):
    """``moss-image``'s ``_Block.forward`` on the ref rows alone (what they get under
    ``ref_isolate``), plus their keys (``k_norm``'d, before RoPE) and values."""
    attention, situ, _, _ = ops
    a = mod.unbind(1)
    y = layer.norm1(x) * (1 + a[1][:, None]) + a[0][:, None]
    dtype = layer.proj["q"].weight.dtype
    y = y.to(dtype)
    q = layer.q_norm(layer.proj["q"](y).unflatten(-1, (layer.heads, -1))).flatten(2)
    k = layer.k_norm(layer.proj["k"](y).unflatten(-1, (layer.heads, -1))).flatten(2)
    v = layer.proj["v"](y)
    x = x + layer.proj["o"](attention(q, k, v, rope, layer.heads)) * a[2][:, None]
    y = layer.norm2(x) * (1 + a[4][:, None]) + a[3][:, None]
    hidden, gate = layer.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
    return x + layer.proj["ffn_down"](situ(hidden, gate)) * a[5][:, None], k, v


def _main_block(layer, ops, x, rope, mod, k_ref, v_ref, ref_rope, n_img):
    """``moss-image``'s ``_Block.forward`` on ``x = [image | text]``, the keys and
    values of an isolated ref (``_ref_pass``'s, ``ref_rope`` for this call's text
    length) spliced in after the image's: the image and text rows of the whole
    block, without running the ref rows."""
    _, situ, rotate, attend = ops
    a = mod.unbind(1)
    y = layer.norm1(x) * (1 + a[1][:, None]) + a[0][:, None]
    dtype = layer.proj["q"].weight.dtype
    y = y.to(dtype)
    heads = layer.heads
    shape = (*x.shape[:2], heads, -1)
    ref_shape = (*k_ref.shape[:2], heads, -1)
    q = layer.q_norm(layer.proj["q"](y).unflatten(-1, (heads, -1)))
    k = layer.k_norm(layer.proj["k"](y).unflatten(-1, (heads, -1)))
    q = rotate(q, rope).transpose(1, 2)
    k = rotate(k, rope).transpose(1, 2)
    v = layer.proj["v"](y).reshape(shape).transpose(1, 2)
    k_ref = rotate(k_ref.reshape(ref_shape), ref_rope).transpose(1, 2)
    v_ref = v_ref.reshape(ref_shape).transpose(1, 2)
    k = torch.cat((k[:, :, :n_img], k_ref, k[:, :, n_img:]), 2)
    v = torch.cat((v[:, :, :n_img], v_ref, v[:, :, n_img:]), 2)
    attn = attend(q, k, v, heads, skip_reshape=True)
    x = x + layer.proj["o"](attn) * a[2][:, None]
    y = layer.norm2(x) * (1 + a[4][:, None]) + a[3][:, None]
    hidden, gate = layer.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
    return x + layer.proj["ffn_down"](situ(hidden, gate)) * a[5][:, None]


_compiled = {}


def _on(fn, layer, *args):
    """``fn(block, ops, *args)`` on ``layer``. A compiled block (an
    ``OptimizedModule``, e.g. the Spectrum KSampler's compile) runs ``fn`` compiled
    too: one inductor graph with dynamic shapes, shared by all 42 blocks, since
    their parameters are graph inputs rather than guards."""
    base = getattr(layer, "_orig_mod", layer)
    ops = sys.modules[type(base).__module__]
    if base is not layer:
        if fn not in _compiled:
            _compiled[fn] = torch.compile(fn, backend="inductor", dynamic=True)
        fn = _compiled[fn]
    fns = (ops._attention, ops._situ, ops._rotate, ops.optimized_attention)
    return fn(base, fns, *args)


def _hooked(layer, args, out):
    """``out`` through ``layer``'s forward hooks (DAVE's), which a block function
    called in place of ``layer(...)`` would otherwise skip."""
    for hook in layer._forward_hooks.values():
        result = hook(layer, args, out)
        if result is not None:
            out = result
    return out


def ref_block(layer, x, rope, mod, ref_mod, span, isolate=False):
    """``_ref_block`` on ``layer``, its forward hooks included."""
    out = _on(_ref_block, layer, x, rope, mod, ref_mod, span, isolate)
    return _hooked(layer, (x, rope, mod), out)


def main_block(layer, x, rope, mod, k_ref, v_ref, ref_rope, n_img):
    """``_main_block`` on ``layer``, its forward hooks included."""
    out = _on(_main_block, layer, x, rope, mod, k_ref, v_ref, ref_rope, n_img)
    return _hooked(layer, (x, rope, mod), out)


class RefKV:
    """The isolated ref's per-block keys (before RoPE) and values, one list per ref
    row, built on first use and rebuilt when ``key`` (the model's weight patches,
    device, target size) changes. ``store = "cpu"`` keeps them in pinned host
    memory and copies each block's in per forward."""

    def __init__(self, store="gpu"):
        self.store = store
        self.key = None
        self.rows = None

    def get(self, key, build):
        if key != self.key:
            self.key, self.rows = None, None
            self.rows = build()
            self.key = key
        return self.rows

    def build(self, dit, ref, ref_mod, height, width, text_length, device):
        rows, size = [], 0
        for r in range(ref.shape[0]):
            x = _embed_latent(dit, ref[r][None])
            rope = ref_rope(dit, height, width, tuple(ref.shape[-2:]), text_length, device)
            rope = rope.to(x.dtype)
            layers = []
            for layer in dit.layers:
                x, k, v = _on(_ref_pass, layer, x, rope, ref_mod)
                if self.store == "cpu":
                    k, v = k.cpu().pin_memory(), v.cpu().pin_memory()
                layers.append((k, v))
                size += k.nbytes + v.nbytes
            rows.append(layers)
        logger.info("MOSS i2i: ref K/V cached, %.2f GB on %s", size / 2**30, self.store)
        return rows


def _ref_only(layer, x, out, rope, ref_mod, span):
    """``out`` with the ref rows of the block input ``x`` through ``layer`` on their
    own: what the whole block gives them under ``ref_isolate``. Used where the block
    was skipped, so the skipped branch reads the same ref as the others."""
    s, e = span
    return torch.cat((out[:, :s], layer(x[:, s:e], rope[:, s:e], ref_mod), out[:, e:]), 1)


def _embed_latent(dit, latent):
    tokens = latent.flatten(2).transpose(1, 2)
    return dit.video_embedder(tokens.to(dit.video_embedder.weight.dtype).contiguous())


def forward_with_ref(
    dit,
    ref,
    x,
    timestep,
    context,
    attention_mask=None,
    transformer_options={},
    ref_timestep=None,
    ref_isolate=False,
    ref_kv=None,
    model_key=None,
    **kwargs,
):
    """``ref (1 or B, 128, h, w)``: clean reference latents, in every CFG branch.
    ``ref_timestep`` (DiT units, 0–1000): the ref rows' own modulation timestep;
    None shares the step's. ``ref_isolate``: the ref rows attend only to themselves
    and still run through skipped blocks. ``ref_kv`` (a ``RefKV``; isolated ref with
    its own timestep only): run the ref once, keyed on ``model_key``."""
    transformer_options = transformer_options.copy()
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    hidden_states, encoder_hidden_states = x, context
    if hidden_states.ndim != 4 or hidden_states.shape[1] != dit.latent_channels:
        raise ValueError("hidden_states must be [B, latent_channels, H, W].")
    batch, _, height, width = hidden_states.shape
    if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
        raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
    mask = attention_mask
    if mask is None:
        mask = torch.ones(
            encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
        )
    if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
    ref = ref.to(hidden_states.device)
    ref_hw = tuple(ref.shape[-2:])
    n_img = height * width
    times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
    times = times.expand(batch)
    mod, head = _time_mods(dit, times)
    ref_mod = None
    if ref_timestep is not None:
        ref_mod, _ = _time_mods(dit, torch.full_like(times, float(ref_timestep)))
        ref_mod = ref_mod.to(mod.dtype)
    cached = None
    if ref_kv is not None and ref_isolate and ref_mod is not None:
        length = int(mask[0].sum())
        key = (model_key, hidden_states.device, height, width, float(ref_timestep))
        cached = ref_kv.get(
            key,
            lambda: ref_kv.build(
                dit, ref, ref_mod[:1], height, width, length, hidden_states.device
            ),
        )
    outputs = []
    for index in range(batch):
        if cached is not None:
            outputs.append(
                _forward_cached(
                    dit,
                    cached[min(index, ref.shape[0] - 1)],
                    ref_hw,
                    hidden_states[index : index + 1],
                    encoder_hidden_states[index : index + 1, mask[index].bool()],
                    mod[index : index + 1],
                    head[index : index + 1],
                    blocks_replace,
                    transformer_options,
                )
            )
            continue
        context = encoder_hidden_states[index : index + 1, mask[index].bool()]
        if context.shape[1] == 0:
            raise ValueError("Each sample needs at least one valid text token.")
        video = _embed_latent(dit, hidden_states[index : index + 1])
        ref_tokens = _embed_latent(dit, ref[min(index, ref.shape[0] - 1)][None])
        # Match the reference activation dtype under fp32 weights + autocast.
        rope = dit.rope(height, width, context.shape[1], hidden_states.device)
        rope = torch.cat(
            (
                rope[:, :n_img],
                ref_rope(dit, height, width, ref_hw, context.shape[1], hidden_states.device),
                rope[:, n_img:],
            ),
            1,
        ).to(video.dtype)
        n_lead = n_img + ref_tokens.shape[1]
        span = (n_img, n_lead)
        text = dit.text_embedder(context, rope[:, n_lead:])
        x = torch.cat((video, ref_tokens, text), 1)

        def ref_vec(vec, index=index):
            return vec if ref_mod is None else ref_mod[index : index + 1]

        def run(layer, x, rope, vec, span=span):
            if ref_mod is None and not ref_isolate:
                return layer(x, rope, vec)
            return ref_block(layer, x, rope, vec, ref_vec(vec), span, ref_isolate)

        transformer_options["total_blocks"] = len(dit.layers)
        transformer_options["block_type"] = "single"
        for layer_index, layer in enumerate(dit.layers):
            transformer_options["block_index"] = layer_index
            if ("single_block", layer_index) in blocks_replace:
                ran = []

                def block_wrap(args, layer=layer, ran=ran):
                    ran.append(True)
                    return {"img": run(layer, args["img"], args["pe"], args["vec"])}

                out = blocks_replace[("single_block", layer_index)](
                    {
                        "img": x,
                        "pe": rope,
                        "vec": mod[index : index + 1],
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )
                if ref_isolate and not ran:
                    vec = mod[index : index + 1]
                    out["img"] = _ref_only(layer, x, out["img"], rope, ref_vec(vec), span)
                x = out["img"]
            else:
                x = run(layer, x, rope, mod[index : index + 1])
        x = dit.final_norm_video(x[:, :n_img].to(dit.final_norm_video.weight.dtype))
        x = x * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
        x = dit.final_linear_video(x.to(dit.final_linear_video.weight.dtype))
        outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
    return torch.cat(outputs)


def _forward_cached(dit, kv, ref_hw, latent, context, vec, head, blocks_replace, transformer_options):
    """One sample of ``forward_with_ref`` on the image and text rows alone, with the
    ref's cached keys / values ``kv`` (``RefKV``) in every block. A skipped block
    leaves the image and text as they are; the cache already holds the ref's pass
    through it."""
    if context.shape[1] == 0:
        raise ValueError("Each sample needs at least one valid text token.")
    _, _, height, width = latent.shape
    n_img = height * width
    device = latent.device
    video = _embed_latent(dit, latent)
    # Match the reference activation dtype under fp32 weights + autocast.
    rope = dit.rope(height, width, context.shape[1], device).to(video.dtype)
    rrope = ref_rope(dit, height, width, ref_hw, context.shape[1], device).to(video.dtype)
    text = dit.text_embedder(context, rope[:, n_img:])
    x = torch.cat((video, text), 1)

    def run(layer, x, rope, vec, layer_index):
        k, v = kv[layer_index]
        k, v = k.to(device, non_blocking=True), v.to(device, non_blocking=True)
        return main_block(layer, x, rope, vec, k, v, rrope, n_img)

    transformer_options["total_blocks"] = len(dit.layers)
    transformer_options["block_type"] = "single"
    for layer_index, layer in enumerate(dit.layers):
        transformer_options["block_index"] = layer_index
        if ("single_block", layer_index) in blocks_replace:

            def block_wrap(args, layer=layer, layer_index=layer_index):
                return {"img": run(layer, args["img"], args["pe"], args["vec"], layer_index)}

            out = blocks_replace[("single_block", layer_index)](
                {"img": x, "pe": rope, "vec": vec, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = run(layer, x, rope, vec, layer_index)
    x = dit.final_norm_video(x[:, :n_img].to(dit.final_norm_video.weight.dtype))
    x = x * (1 + head[:, 1, None]) + head[:, 0, None]
    x = dit.final_linear_video(x.to(dit.final_linear_video.weight.dtype))
    return x.transpose(1, 2).reshape(1, -1, height, width)


def make_apply_model_wrapper(ref, ref_timestep=None, ref_isolate=False, ref_cache="gpu"):
    """``APPLY_MODEL`` wrapper: for the length of one ``_apply_model`` call, the live
    ``diffusion_model`` runs ``forward_with_ref``. Spectrum's cached steps never
    reach ``apply_model``, so they cost no ref rows. ``ref_cache`` (``gpu`` / ``cpu``
    / ``off``): where an isolated ref's ``RefKV`` lives; it outlasts the run, so a
    reseed of the same page reuses it.

    The wrapper holds the ref tensor and its cache only (no patcher, no module), and
    resolves the module from ``executor.class_obj`` each call, so a reloaded model
    is followed; the cache is keyed on the model's weight patches."""
    cache = {}
    ref_kv = None if ref_cache == "off" else RefKV(ref_cache)

    def wrapper(executor, *args, **kwargs):
        dit = executor.class_obj.diffusion_model
        model_key = (id(dit), getattr(executor.class_obj, "current_weight_patches_uuid", None))
        x = args[0]
        key = x.device
        if key not in cache:
            cache.clear()
            cache[key] = ref.to(x.device)
        device_ref = cache[key]
        previous = dit.__dict__.get("forward")

        def forward(*f_args, **f_kwargs):
            return forward_with_ref(
                dit,
                device_ref,
                *f_args,
                ref_timestep=ref_timestep,
                ref_isolate=ref_isolate,
                ref_kv=ref_kv,
                model_key=model_key,
                **f_kwargs,
            )

        dit.forward = forward
        try:
            return executor(*args, **kwargs)
        finally:
            if previous is None:
                del dit.forward
            else:
                dit.forward = previous

    return wrapper

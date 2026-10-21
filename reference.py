"""MOSS-Image forward with a clean reference latent in the sequence.

The sequence is ``[image | ref | text]``. Ref tokens go through ``video_embedder``
and get no noise. With ``ref_timestep`` (the LoRA's metadata; v2 trains at 0) they get
their own block modulation from that timestep, image and text keep the step's; without
it (v1) they share the step's. On RoPE they sit at ``r = 1`` and
the image's ``f`` (= text length), with their ``(h, w)`` stretched onto the target
grid, ``(i + 0.5)·H/h_ref − 0.5``, so a half-size ref lines up with the target
position for position. The head reads the image rows only.

``forward_with_ref`` is the ``moss-image`` node's ``MOSSImageTransformer.forward``
(OpenMOSS, Apache-2.0) with the ref rows spliced in; its ``patches_replace``
dispatch is kept, so the uncond block-22 skip (SLG) still applies. It mirrors
anima_lora's ``library/moss/model.py`` on the ``i2i_probe`` branch, which the
colorize LoRAs were trained against.
"""

import sys

import torch


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


def ref_block(layer, x, rope, mod, ref_mod, span):
    """``moss-image``'s ``_Block.forward`` with the rows in ``span`` modulated by
    ``ref_mod``: op for op anima_lora's ``_Block`` with ``ref_span``."""
    # A compiled block (Spectrum KSampler's compile) is an OptimizedModule wrapper.
    layer = getattr(layer, "_orig_mod", layer)
    ops = sys.modules[type(layer).__module__]
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
    attn = ops._attention(q, k, layer.proj["v"](y), rope, layer.heads)
    x = x + gated(layer.proj["o"](attn), 2)
    y = affine(layer.norm2(x), 3)
    hidden, gate = layer.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
    return x + gated(layer.proj["ffn_down"](ops._situ(hidden, gate)), 5)


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
    **kwargs,
):
    """``ref (1 or B, 128, h, w)``: clean reference latents, in every CFG branch.
    ``ref_timestep`` (DiT units, 0–1000): the ref rows' own modulation timestep;
    None shares the step's."""
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
    outputs = []
    for index in range(batch):
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

        def run(layer, x, rope, vec, index=index, span=span):
            if ref_mod is None:
                return layer(x, rope, vec)
            return ref_block(layer, x, rope, vec, ref_mod[index : index + 1], span)

        transformer_options["total_blocks"] = len(dit.layers)
        transformer_options["block_type"] = "single"
        for layer_index, layer in enumerate(dit.layers):
            transformer_options["block_index"] = layer_index
            if ("single_block", layer_index) in blocks_replace:

                def block_wrap(args, layer=layer):
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
                x = out["img"]
            else:
                x = run(layer, x, rope, mod[index : index + 1])
        x = dit.final_norm_video(x[:, :n_img].to(dit.final_norm_video.weight.dtype))
        x = x * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
        x = dit.final_linear_video(x.to(dit.final_linear_video.weight.dtype))
        outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
    return torch.cat(outputs)


def make_apply_model_wrapper(ref, ref_timestep=None):
    """``APPLY_MODEL`` wrapper: for the length of one ``_apply_model`` call, the live
    ``diffusion_model`` runs ``forward_with_ref``. Spectrum's cached steps never
    reach ``apply_model``, so they cost no ref rows.

    The wrapper holds the ref tensor only (no patcher, no module), and resolves the
    module from ``executor.class_obj`` each call, so a reloaded model is followed."""
    cache = {}

    def wrapper(executor, *args, **kwargs):
        dit = executor.class_obj.diffusion_model
        x = args[0]
        key = x.device
        if key not in cache:
            cache.clear()
            cache[key] = ref.to(x.device)
        device_ref = cache[key]
        previous = dit.__dict__.get("forward")

        def forward(*f_args, **f_kwargs):
            return forward_with_ref(
                dit, device_ref, *f_args, ref_timestep=ref_timestep, **f_kwargs
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

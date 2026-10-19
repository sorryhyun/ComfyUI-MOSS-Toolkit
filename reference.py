"""MOSS-Image forward with a clean reference latent in the sequence.

The sequence is ``[image | ref | text]``. Ref tokens go through ``video_embedder``,
get no noise, and share the step's modulation. On RoPE they sit at ``r = 1`` and
the image's ``f`` (= text length), with their ``(h, w)`` stretched onto the target
grid, ``(i + 0.5)·H/h_ref − 0.5``, so a half-size ref lines up with the target
position for position. The head reads the image rows only.

``forward_with_ref`` is the ``moss-image`` node's ``MOSSImageTransformer.forward``
(OpenMOSS, Apache-2.0) with the ref rows spliced in; its ``patches_replace``
dispatch is kept, so the uncond block-22 skip (SLG) still applies. It mirrors
anima_lora's ``library/moss/model.py`` on the ``i2i_probe`` branch, which the
colorize LoRAs were trained against.
"""

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


def _embed_latent(dit, latent):
    tokens = latent.flatten(2).transpose(1, 2)
    return dit.video_embedder(tokens.to(dit.video_embedder.weight.dtype).contiguous())


def forward_with_ref(
    dit, ref, x, timestep, context, attention_mask=None, transformer_options={}, **kwargs
):
    """``ref (1 or B, 128, h, w)``: clean reference latents, in every CFG branch."""
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
    # Training computes the sinusoid in float64 and feeds float32 into its MLP.
    half = dit.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    time = dit.time_embedding(embedding.to(dit.time_embedding[0].weight.dtype)).float()
    mod = dit.shared_block_modulation(time.to(dit.shared_block_modulation.weight.dtype)).view(
        batch, 6, -1
    )
    head = dit.video_head_modulation(time.to(dit.video_head_modulation.weight.dtype)).view(
        batch, 2, -1
    )
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
        text = dit.text_embedder(context, rope[:, n_lead:])
        x = torch.cat((video, ref_tokens, text), 1)
        transformer_options["total_blocks"] = len(dit.layers)
        transformer_options["block_type"] = "single"
        for layer_index, layer in enumerate(dit.layers):
            transformer_options["block_index"] = layer_index
            if ("single_block", layer_index) in blocks_replace:

                def block_wrap(args, layer=layer):
                    return {"img": layer(args["img"], args["pe"], args["vec"])}

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
                x = layer(x, rope, mod[index : index + 1])
        x = dit.final_norm_video(x[:, :n_img].to(dit.final_norm_video.weight.dtype))
        x = x * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
        x = dit.final_linear_video(x.to(dit.final_linear_video.weight.dtype))
        outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
    return torch.cat(outputs)


def make_apply_model_wrapper(ref):
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
            return forward_with_ref(dit, device_ref, *f_args, **f_kwargs)

        dit.forward = forward
        try:
            return executor(*args, **kwargs)
        finally:
            if previous is None:
                del dit.forward
            else:
                dit.forward = previous

    return wrapper

"""MOSS-Image LoRA files onto whichever MOSS-Image MODEL is loaded.

Two DiT layouts are in use: the vendor's separate ``layers.N.proj.{q,k,v}`` (the
``moss-image`` node, and ComfyUI's ``MOSSImage`` on the vendor shards) and the
Comfy-Org repack's fused ``layers.N.proj.qkv`` (ComfyUI's ``MOSSImage``, ai-toolkit).
ComfyUI maps a split LoRA onto a fused model by row slices, but not the reverse, so a
fused LoRA (ai-toolkit's LoKr, ai-toolkit saves before 2026-10-17, any LoRA trained on
the repack) patches nothing on a split model. ``to_model_layout`` splits those groups
by rows: exact for LoRA / LoHa (the down side is shared), and for LoKr when ``w1``'s
rows divide by 3 (else the group becomes three full ``diff`` patches).

``transformer.`` and bare module paths get the ``diffusion_model.`` prefix, which both
models' key maps carry; the ``moss-image`` model's map has no bare paths.
"""

import logging

import comfy.lora
import comfy.lora_convert
import torch

logger = logging.getLogger(__name__)

ROOTS = (
    "layers.",
    "text_embedder.",
    "time_embedding.",
    "video_embedder",
    "shared_block_modulation",
    "video_head_modulation",
    "final_linear_video",
)
# split by rows across q / k / v; the rest of a group is shared
ROW_SPLIT = ("lora_up.weight", "lora_B.weight", "hada_w1_a", "hada_w2_a", "diff")
SHARED = (
    "lora_down.weight",
    "lora_A.weight",
    "lora_mid.weight",
    "alpha",
    "hada_w1_b",
    "hada_w2_b",
)
LOKR = (
    "lokr_w1",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_t2",
)
SUFFIXES = (
    ROW_SPLIT
    + SHARED
    + LOKR
    + ("dora_scale", "diff_b", "set_weight", "w_norm", "b_norm")
)


def split_key(key):
    """``(module, suffix)`` of an adapter key, or None."""
    for suffix in SUFFIXES:
        if key.endswith("." + suffix):
            return key[: -len(suffix) - 1], suffix
    return None


def _module_path(module):
    if module.startswith("transformer."):
        return "diffusion_model." + module[len("transformer.") :]
    if module.startswith(ROOTS):
        return "diffusion_model." + module
    return module


def _fused_parts(module):
    """The q / k / v module names of a fused ``proj.qkv`` module, or None."""
    if module.endswith((".proj.qkv", "_proj_qkv")):
        return [module[: -len("qkv")] + p for p in "qkv"]
    return None


def _split_lokr(group):
    """A fused LoKr group as three groups, scale folded into ``w1`` (ComfyUI scales
    ``alpha / rank`` of the last decomposed factor; ``alpha`` is dropped)."""
    w1 = group.get("lokr_w1")
    dim = None
    if w1 is None:
        w1 = group["lokr_w1_a"].float() @ group["lokr_w1_b"].float()
        dim = group["lokr_w1_b"].shape[0]
    if "lokr_w2_b" in group:
        dim = group["lokr_w2_b"].shape[0]
    alpha = group.get("alpha")
    scale = alpha.item() / dim if alpha is not None and dim is not None else 1.0
    w1 = w1.float() * scale
    w2_keys = [
        k for k in ("lokr_w2", "lokr_w2_a", "lokr_w2_b", "lokr_t2") if k in group
    ]
    if w1.shape[0] % 3 == 0:
        # kron rows are (w1 row, w2 row) pairs, w1 major: a third of w1's rows is a third of the weight
        return [
            {
                "lokr_w1": part.to(group[w2_keys[0]].dtype).contiguous(),
                **{k: group[k] for k in w2_keys},
            }
            for part in w1.chunk(3, 0)
        ]
    w2 = group.get("lokr_w2")
    if w2 is None:
        w2 = group["lokr_w2_a"].float() @ group["lokr_w2_b"].float()
    diff = torch.kron(w1, w2.float()).to(group[w2_keys[0]].dtype)
    return [{"diff": part.contiguous()} for part in diff.chunk(3, 0)]


def _split_group(group):
    """A fused ``proj.qkv`` adapter group as three, or None when it cannot split."""
    if "dora_scale" in group:
        return None
    if any(k in group for k in LOKR):
        return _split_lokr(group)
    if not all(k in ROW_SPLIT or k in SHARED for k in group):
        return None
    parts = [{}, {}, {}]
    for suffix, value in group.items():
        chunks = value.chunk(3, 0) if suffix in ROW_SPLIT else (value,) * 3
        for part, chunk in zip(parts, chunks):
            part[suffix] = chunk.contiguous()
    return parts


def model_is_fused(model):
    """Whether ``model``'s DiT blocks fuse q / k / v (the Comfy-Org repack)."""
    return "qkv" in model.get_model_object("diffusion_model").layers[0].proj


def to_model_layout(lora, fused):
    """``lora`` with its module paths prefixed and, on a split model (``fused`` False),
    its fused ``proj.qkv`` groups split into q / k / v."""
    groups, out = {}, {}
    for key, value in lora.items():
        parsed = split_key(key)
        if parsed is None:
            out[key] = value
            continue
        module, suffix = parsed
        groups.setdefault(_module_path(module), {})[suffix] = value
    split = kept = 0
    for module, group in groups.items():
        names = None if fused else _fused_parts(module)
        parts = _split_group(group) if names else None
        if parts is None:
            kept += names is not None
            out.update({f"{module}.{s}": v for s, v in group.items()})
            continue
        split += 1
        for name, part in zip(names, parts):
            out.update({f"{name}.{s}": v for s, v in part.items()})
    if split:
        logger.info(
            "MOSS LoRA: %d fused q/k/v groups split onto the separate projections",
            split,
        )
    if kept:
        logger.warning(
            "MOSS LoRA: %d fused q/k/v groups could not be split (DoRA / unknown keys)",
            kept,
        )
    return out


def load(model, lora, strength, name):
    """``model`` patched with ``lora`` at ``strength``. Logs every non-block module it
    patches (``shared_block_modulation``, …) and the adapter modules this model has no
    weight for; raises when nothing patches."""
    lora = to_model_layout(lora, model_is_fused(model))
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    lora = comfy.lora_convert.convert_lora(lora)
    patches = comfy.lora.load_lora(lora, key_map, log_missing=False)
    modules = {split_key(k)[0] for k in lora if split_key(k)}
    missing = sorted(m for m in modules if m not in key_map)
    if not patches:
        raise ValueError(
            f"MOSS LoRA {name}: none of its {len(modules)} modules matches this MOSS-Image "
            f"model (e.g. {missing[:3]})"
        )
    out = model.clone()
    out.add_patches(patches, strength)
    targets = {k[0] if isinstance(k, tuple) else k for k in patches}
    extra = sorted(
        {
            t[len("diffusion_model.") : -len(".weight")]
            for t in targets
            if ".layers." not in t
        }
    )
    logger.info(
        "MOSS LoRA %s: %d weights patched%s",
        name,
        len(targets),
        f", incl. {', '.join(extra)}" if extra else "",
    )
    if missing:
        logger.warning(
            "MOSS LoRA %s: %d modules not loaded (no such weight in this model): %s",
            name,
            len(missing),
            ", ".join(missing[:8]) + (" …" if len(missing) > 8 else ""),
        )
    return out

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

``load(..., bypass=True)`` adds the adapters' output in the forward and leaves the base
weights alone. A weight patch on an INT8 DiT requantizes each patched linear (fresh row
scales, stochastic rounding), which loses a GPTQ file's weights; bypass keeps them.
ComfyUI's own ``load_bypass_lora_for_models`` drops adapters on row slices of a fused
``proj.qkv``, binds its hooks to the modules of the patcher it was given (a non-dynamic
clone for compile rebuilds the DiT) and, chained, overwrites the previous LoRA's
injection; ``Deltas`` and ``BypassInjection`` do neither.

On a ConvRot INT8 linear, ``DownRows`` runs the ``down`` matrices as one small INT8 GEMM
in the weight's layout and only ``up`` in float. ``BypassForwardHook`` would set
``comfy_force_forward``, which drops ``ffn_down``'s situ fold; ``linear_input_act`` below
keeps it. Any other linear (bf16, LoKr, LoHa) takes the hook.
"""

import functools
import inspect
import logging

import comfy.lora
import comfy.lora_convert
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
import comfy.utils
import comfy.weight_adapter
import torch
from comfy.weight_adapter.bypass import BypassForwardHook

logger = logging.getLogger(__name__)

BYPASS_KEY = "moss_toolkit_lora_bypass"

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


class Deltas:
    """Every bypass adapter on one linear, in ``BypassForwardHook``'s adapter role:
    ``(rows, adapter)`` pairs, ``rows`` a slice of the output axis or None for all."""

    def __init__(self, parts=()):
        self.parts = list(parts)
        self.device = None

    def _to(self, device):
        for _, adapter in self.parts:
            adapter.weights = tuple(
                w.to(device) if isinstance(w, torch.Tensor) else w
                for w in adapter.weights
            )
        self.device = device

    def h(self, x, base_out):
        if self.device != x.device:
            self._to(x.device)
        out = torch.zeros_like(base_out)
        for rows, adapter in self.parts:
            if rows is None:
                out += adapter.h(x, base_out)
            else:
                out[..., rows] += adapter.h(x, base_out[..., rows])
        return out

    def g(self, y):
        return y


class DownRows:
    """Every LoRA on one linear as an INT8 GEMM of their ``down`` rows, with the strength
    folded into ``up``. Built on the first forward after inject, dropped at eject."""

    def __init__(self, module, parts):
        self.module = module
        self.parts = parts
        self.original_forward = None
        self.device = None
        self.down = None
        self.ups = []
        self.in_act = False

    @staticmethod
    def fits(module, parts):
        weight = module.weight
        return (
            isinstance(weight, comfy.quant_ops.QuantizedTensor)
            and getattr(weight.params, "convrot", False)
            and not weight.params.transposed
            and all(
                isinstance(adapter, comfy.weight_adapter.LoRAAdapter)
                and adapter.weights[3] is None
                and adapter.weights[5] is None
                for _, adapter in parts
            )
        )

    def _build(self, device):
        dtype = self.module.weight.params.orig_dtype
        downs, self.ups = [], []
        at = 0
        for rows, adapter in self.parts:
            up, down, alpha = adapter.weights[:3]
            rank = down.shape[0]
            scale = adapter.multiplier * (
                float(alpha) / rank if alpha is not None else 1.0
            )
            downs.append(down.to(device).float())
            self.ups.append((rows, at, at + rank, up.to(device, dtype) * scale))
            at += rank
        self.down = self.module.weight.requantize_from_float(
            torch.cat(downs), scale="recalculate", stochastic_rounding=0
        )
        self.device = device

    def add(self, x, out, input_act=None, act_weight=None, act_eps=0.0):
        if self.device != x.device:
            self._build(x.device)
        if input_act is None:
            latent = torch.nn.functional.linear(x, self.down)
        else:
            latent = self.down.linear_input_act(
                x,
                None,
                input_act=input_act,
                input_act_weight=act_weight,
                input_act_eps=act_eps,
            )
        latent = latent.to(out.dtype)
        for rows, start, stop, up in self.ups:
            delta = torch.nn.functional.linear(latent[..., start:stop], up)
            if rows is None:
                out += delta
            else:
                out[..., rows] += delta
        return out

    def forward(self, x, *args, **kwargs):
        out = self.original_forward(x, *args, **kwargs)
        return out if self.in_act else self.add(x, out)

    def inject(self):
        self.original_forward = self.module.forward
        self.module.forward = self.forward
        self.module.moss_lora_rows = self

    def eject(self):
        self.module.forward = self.original_forward
        self.original_forward = None
        del self.module.moss_lora_rows
        self.device = self.down = None
        self.ups = []


_linear_input_act = inspect.unwrap(comfy.ops.linear_input_act)


@functools.wraps(_linear_input_act)
def linear_input_act(
    linear,
    x,
    input_act,
    act_weight=None,
    act_eps=0.0,
    residual=None,
    residual_scale=None,
):
    rows = getattr(linear, "moss_lora_rows", None)
    if rows is None:
        return _linear_input_act(
            linear, x, input_act, act_weight, act_eps, residual, residual_scale
        )
    rows.in_act = True
    try:
        out = _linear_input_act(
            linear, x, input_act, act_weight, act_eps, residual, residual_scale
        )
    finally:
        rows.in_act = False
    return rows.add(x, out, input_act, act_weight, act_eps)


comfy.ops.linear_input_act = linear_input_act


class BypassInjection(comfy.patcher_extension.PatcherInjection):
    """``deltas`` (``{module path: Deltas}``) on the patcher's own model, resolved at
    inject."""

    def __init__(self, deltas):
        super().__init__(self.apply, self.remove)
        self.deltas = deltas

    def apply(self, patcher):
        model = patcher.model
        hooks = []
        for path, delta in self.deltas.items():
            module = comfy.utils.get_attr(model, path)
            fits = DownRows.fits(module, delta.parts)
            hooks.append(
                DownRows(module, delta.parts)
                if fits
                else BypassForwardHook(module, delta)
            )
        for hook in hooks:
            hook.inject()
        model.moss_lora_hooks = hooks

    def remove(self, patcher):
        for hook in patcher.model.__dict__.pop("moss_lora_hooks", ()):
            hook.eject()


def _bypassable(key, patch):
    """Whether ``patch`` runs as a delta on the linear's output. A key with its own
    function or a slice off the output axis, and a DoRA LoRA, need the weight."""
    if not isinstance(patch, comfy.weight_adapter.WeightAdapterBase):
        return False
    if isinstance(key, tuple) and (len(key) > 2 or key[1][0] != 0):
        return False
    return not (
        isinstance(patch, comfy.weight_adapter.LoRAAdapter)
        and patch.weights[4] is not None
    )


def _add_bypass(model, patches, strength):
    """``patches`` on ``model`` in bypass, on top of an earlier bypass LoRA; returns the
    patches that went on as weight patches."""
    injection = model.get_injections(BYPASS_KEY)
    deltas = (
        {path: Deltas(delta.parts) for path, delta in injection[0].deltas.items()}
        if injection
        else {}
    )
    regular = {}
    for key, patch in patches.items():
        if not _bypassable(key, patch):
            regular[key] = patch
            continue
        rows = None
        target = key
        if isinstance(key, tuple):
            target, (_, offset, size) = key
            rows = slice(offset, offset + size)
        patch.multiplier = strength
        path = target[: -len(".weight")]
        deltas.setdefault(path, Deltas()).parts.append((rows, patch))
    if regular:
        model.add_patches(regular, strength)
    model.set_injections(BYPASS_KEY, [BypassInjection(deltas)])
    return regular


def load(model, lora, strength, name, bypass=False):
    """``model`` with ``lora`` at ``strength``, on its weights or, with ``bypass``, in
    its forward. Logs every non-block module it patches (``shared_block_modulation``,
    …) and the adapter modules this model has no weight for; raises when nothing
    patches."""
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
    if bypass:
        regular = _add_bypass(out, patches, strength)
        if regular:
            logger.warning(
                "MOSS LoRA %s: %d patches stay weight patches in bypass",
                name,
                len(regular),
            )
    else:
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
        "MOSS LoRA %s: %d weights %s%s",
        name,
        len(targets),
        "in bypass" if bypass else "patched",
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

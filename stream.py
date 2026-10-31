"""fp32 residual stream for MOSS-Image, as anima_lora's ``GenerateRequest.stream``
(``library/moss/loader.py::stream_fp32_``): the stream, modulation, norms, embedders and
head run in fp32 around the block GEMMs, which stay in the model's own format (bf16, or
INT8 / fp8 on a quantized file).

The two MOSS-Image models get there differently:

* ``moss-image`` node: each op casts its input to its weight's dtype, so the weights
  have to change. ``stream_fp32`` casts every DiT parameter but the block linears' and
  ``q_norm`` / ``k_norm`` weights, which needs the plain ``ModelPatcher`` (the dynamic
  one's weights stay mmapped in the file's dtype), so it runs at load time. The module's
  ``_rotate`` is patched to return its input's dtype, so the fp32 RoPE rows do not lift
  bf16 queries and keys past the bf16 values.
* ComfyUI's ``MOSSImage``: activations follow ``x.dtype`` and non-block ops cast their
  weight to it, so no weight changes and any loaded MODEL can be patched, dynamic VRAM
  and INT8 included. ``stream_fp32_core`` adds an ``APPLY_MODEL`` wrapper that hands the
  forward an fp32 latent and casts the block GEMMs' stream input back to the compute
  dtype; their outputs and the attention (``q_norm`` / ``k_norm``, RoPE) stay there.

On the core model an isolated ref's keys and values (``ref_kv``, cached or not) come from
a stream in the compute dtype, since they join the blocks' attention there. A joint ref,
in the sequence, runs in fp32.
"""

import logging
import re
import sys

import comfy.model_management
import comfy.patcher_extension
import torch

logger = logging.getLogger(__name__)

WRAPPER_KEY = "moss_toolkit_stream_fp32"
# Kept in the weight dtype under the fp32 stream: the block GEMMs, and q/k norms whose
# input is a bf16 GEMM output.
BLOCK_KEEP = re.compile(r"layers\.\d+\.(proj\.\w+|[qk]_norm)")
# The core model's block linears that read the stream; ``o`` and ``ffn_down`` read GEMM
# outputs.
STREAM_READERS = ("qkv", "q", "k", "v", "ffn_up_gate")
# ``ref_latents_method``s whose refs ride in the sequence; any other builds ``ref_kv``,
# whose keys and values join the blocks' attention in the compute dtype.
SEQUENCE_REFS = ("index", "index_timestep_zero")


def _rotate_in_input_dtype(module):
    """``module._rotate`` with its result cast back to ``x``'s dtype; for a bf16 stream
    (bf16 RoPE rows) that is the identity."""
    rotate = module._rotate
    if getattr(rotate, "keeps_dtype", False):
        return

    def _rotate(x, rope):
        return rotate(x, rope).to(x.dtype)

    _rotate.keeps_dtype = True
    module._rotate = _rotate


def stream_fp32(patcher):
    """Every plain floating DiT parameter outside ``BLOCK_KEEP`` to fp32."""
    dit = patcher.model.diffusion_model
    keep = {id(m.weight) for n, m in dit.named_modules() if BLOCK_KEEP.fullmatch(n)}
    for p in dit.parameters():
        if id(p) not in keep and type(p.data) is torch.Tensor and p.is_floating_point():
            p.data = p.data.float()
    _rotate_in_input_dtype(sys.modules[type(dit).__module__])


def apply_model_wrapper(executor, x, t, c_concat=None, *args, **kwargs):
    """One ``_apply_model`` with ``x`` kept fp32 (it would truncate it to the compute
    dtype) and pre-hooks casting the blocks' stream input to that dtype. Both last the
    one call, so clones of the model without the patch stay untouched."""
    base = executor.class_obj
    dit = base.diffusion_model
    compute = base.get_dtype_inference()
    if compute == torch.float32:
        return executor(x, t, c_concat, *args, **kwargs)

    xc = base.model_sampling.calculate_input(t, x)
    if c_concat is not None:
        xc = torch.cat((xc, comfy.model_management.cast_to_device(c_concat, xc.device, xc.dtype)), dim=1)
    xc = xc.float()

    def to_compute(_module, inputs):
        return (inputs[0].to(compute), *inputs[1:])

    call = dit.forward
    previous = dit.__dict__.get("forward")

    def forward(_xc, timestep, context=None, **f_kwargs):
        if f_kwargs.get("ref_latents") and f_kwargs.get("ref_latents_method") in SEQUENCE_REFS:
            f_kwargs["ref_latents"] = [ref.float() for ref in f_kwargs["ref_latents"]]
        if f_kwargs.get("context_attention_bias") is not None:
            f_kwargs["context_attention_bias"] = f_kwargs["context_attention_bias"].float()
        return call(xc, timestep, context.float(), **f_kwargs)

    hooks = [
        linear.register_forward_pre_hook(to_compute)
        for layer in dit.layers
        for name, linear in layer.proj.items()
        if name in STREAM_READERS
    ]
    dit.forward = forward
    try:
        return executor(x, t, c_concat, *args, **kwargs)
    finally:
        for hook in hooks:
            hook.remove()
        if previous is None:
            del dit.forward
        else:
            dit.forward = previous


def stream_fp32_core(model):
    """A clone of a ComfyUI ``MOSSImage`` MODEL with the fp32 stream. Dynamic VRAM casts
    every module already; on a plain patcher a fully loaded module would not, and the
    flag costs nothing when a module's input matches its weight."""
    for module in model.get_model_object("diffusion_model").modules():
        if hasattr(module, "comfy_cast_weights"):
            module.comfy_cast_weights = True
    out = model.clone()
    out.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, WRAPPER_KEY)
    out.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, WRAPPER_KEY, apply_model_wrapper)
    return out


class MOSSStreamFP32:
    """The fp32 residual stream as a MODEL patch, for a MODEL from any ComfyUI loader."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "MOSS-Image"
    DESCRIPTION = (
        "Runs ComfyUI's MOSS-Image DiT with an fp32 residual stream, modulation, norms, "
        "embedders and head around the block GEMMs (bf16 or INT8). No weight is reloaded, "
        "so it works with dynamic VRAM, LoRAs and the INT8 files. The moss-image node's "
        "DiT needs the stream knob on MOSS-Image Loader instead."
    )

    def patch(self, model):
        dit = model.get_model_object("diffusion_model")
        if not hasattr(dit, "shared_block_modulation"):
            raise ValueError("MOSS-Image fp32 Stream needs a MOSS-Image MODEL.")
        if hasattr(dit, "preprocess_refs"):
            return (stream_fp32_core(model),)
        if dit.video_embedder.weight.dtype == torch.float32:
            return (model,)
        raise ValueError(
            "MOSS-Image fp32 Stream: the moss-image node's DiT casts its weights for the "
            "stream, which has to happen at load time. Set stream = fp32 on MOSS-Image "
            "Loader, or load the file with ComfyUI's Load Diffusion Model."
        )


NODE_CLASS_MAPPINGS = {"MOSSStreamFP32": MOSSStreamFP32}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSStreamFP32": "MOSS-Image fp32 Stream"}

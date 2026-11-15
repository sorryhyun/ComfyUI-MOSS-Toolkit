"""fp32 residual stream for MOSS-Image, as anima_lora's ``GenerateRequest.stream``
(``library/moss/loader.py::stream_fp32_``): the stream, modulation, norms, embedders and
head run in fp32 around the block GEMMs, which stay in the model's own format (bf16, or
INT8 / fp8 on a quantized file).

Activations on ComfyUI's ``MOSSImage`` follow ``x.dtype`` and non-block ops cast their
weight to it, so no weight changes and any loaded MODEL can be patched, dynamic VRAM and
INT8 included. ``stream_fp32_core`` adds an ``APPLY_MODEL`` wrapper that hands the
forward an fp32 latent and casts the block GEMMs' stream input back to the compute
dtype; their outputs and the attention (``q_norm`` / ``k_norm``, RoPE) stay there. The
two text blocks run in fp32 whole, attention included, on SDPA: FlashAttention refuses
fp32.

An isolated ref's keys and values (``ref_kv``, cached or not) come from
a stream in the compute dtype, since they join the blocks' attention there. A joint ref,
in the sequence, runs in fp32.

The text embedder reads the prompt, the text rows of the RoPE table and the text mask,
not the latent or the timestep, so ``TextCache`` runs it once per distinct prompt in a
queued run instead of once per forward: in fp32 its two blocks are the larger part of
the per-step cost outside the 42 blocks.
"""

import logging

import comfy.model_management
import comfy.patcher_extension
import torch
from comfy.ldm.modules.attention import attention_pytorch

from .reference import is_core_moss

logger = logging.getLogger(__name__)

WRAPPER_KEY = "moss_toolkit_stream_fp32"
# The core model's block linears that read the stream; ``o`` and ``ffn_down`` read GEMM
# outputs.
STREAM_READERS = ("qkv", "q", "k", "v", "ffn_up_gate")
# ``ref_latents_method``s whose refs ride in the sequence; any other builds ``ref_kv``,
# whose keys and values join the blocks' attention in the compute dtype.
SEQUENCE_REFS = ("index", "index_timestep_zero")
# Text embedder outputs kept per model: cond, uncond, and a few more for area or
# multiple conds.
TEXT_CACHE_SIZE = 8

try:
    from comfy_execution.utils import get_executing_context
except ImportError:
    def get_executing_context():
        return None


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.device == b.device and a.dtype == b.dtype and torch.equal(a, b)


class TextCache:
    """``text_embedder`` outputs ``(B, S, 3072)`` in fp32 for one model and one queued
    run. An entry matches on the context (after ``.float()``), the text RoPE rows (which
    carry ``f_offset``) and the text mask, compared by value: ComfyUI builds the context
    anew each step. A changed ``model_key`` (``id(dit)``, weight patches, ``prompt_id``)
    drops every entry; a bypass LoRA's strength changes only with a new ``prompt_id``.
    ``_forward`` copies the output into the sequence with ``torch.cat``, so an entry is
    returned as is."""

    def __init__(self, size=TEXT_CACHE_SIZE):
        self.size = size
        self.model_key = None
        self.entries = []
        self.builds = 0
        self.hits = 0

    def begin(self, model_key):
        if model_key != self.model_key:
            self.model_key = model_key
            self.entries.clear()

    def lookup(self, context, pe, mask, build):
        for c, p, m, tokens in self.entries:
            if _same(c, context) and _same(p, pe) and _same(m, mask):
                self.hits += 1
                return tokens
        tokens = build()
        self.entries.append((context, pe.clone(), None if mask is None else mask.clone(), tokens))
        del self.entries[: -self.size]
        self.builds += 1
        return tokens


def make_apply_model_wrapper(text_cache=None):
    """The fp32 stream wrapper, holding ``text_cache`` (a ``TextCache``, ``None`` for
    none). Clones of the model share it; its key keeps them apart."""

    def wrapper(executor, x, t, c_concat=None, *args, **kwargs):
        return apply_model_wrapper(executor, x, t, c_concat, *args, text_cache=text_cache, **kwargs)

    return wrapper


def apply_model_wrapper(executor, x, t, c_concat=None, *args, text_cache=None, **kwargs):
    """One ``_apply_model`` with ``x`` kept fp32 (it would truncate it to the compute
    dtype), pre-hooks casting the blocks' stream input to that dtype and every op
    casting its weight, which dynamic VRAM does anyway and a fully loaded plain
    patcher does not. With ``text_cache`` and a queued run, ``text_embedder`` goes
    through it. All last the one call, so clones of the model without the patch stay
    untouched."""
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
    uncast = [m for m in dit.modules() if getattr(m, "comfy_cast_weights", True) is False]
    for module in uncast:
        module.comfy_cast_weights = True
    sdpa = []
    if comfy.model_management.flash_attention_enabled():
        sdpa = [
            block.comfy_attention
            for block in dit.text_embedder.blocks
            if block.comfy_attention.function is None
        ]
    for attention in sdpa:
        attention.function = attention_pytorch
    dit.forward = forward

    embedder = dit.text_embedder
    embed = embedder.forward
    previous_embed = embedder.__dict__.get("forward")
    context = get_executing_context()
    cached = text_cache is not None and context is not None
    if cached:
        text_cache.begin((id(dit), getattr(base, "current_weight_patches_uuid", None), context.prompt_id))

        def embed_cached(txt, pe, transformer_options={}, mask=None):
            return text_cache.lookup(txt, pe, mask, lambda: embed(txt, pe, transformer_options, mask))

        embedder.forward = embed_cached
    try:
        return executor(x, t, c_concat, *args, **kwargs)
    finally:
        for module in uncast:
            module.comfy_cast_weights = False
        for attention in sdpa:
            attention.function = None
        for hook in hooks:
            hook.remove()
        if cached:
            if previous_embed is None:
                del embedder.forward
            else:
                embedder.forward = previous_embed
        if previous is None:
            del dit.forward
        else:
            dit.forward = previous


def stream_fp32_core(model):
    """A clone of a ComfyUI ``MOSSImage`` MODEL with the fp32 stream."""
    out = model.clone()
    out.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, WRAPPER_KEY)
    out.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.APPLY_MODEL, WRAPPER_KEY, make_apply_model_wrapper(TextCache())
    )
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
        "so it works with dynamic VRAM, LoRAs and the INT8 files."
    )

    def patch(self, model):
        dit = model.get_model_object("diffusion_model")
        if not hasattr(dit, "shared_block_modulation"):
            raise ValueError("MOSS-Image fp32 Stream needs a MOSS-Image MODEL.")
        if is_core_moss(dit):
            return (stream_fp32_core(model),)
        raise ValueError(
            "MOSS-Image fp32 Stream: the moss-image node's DiT is not supported. Load the "
            "file with MOSS-Image Loader or ComfyUI's Load Diffusion Model."
        )


NODE_CLASS_MAPPINGS = {"MOSSStreamFP32": MOSSStreamFP32}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSStreamFP32": "MOSS-Image fp32 Stream"}

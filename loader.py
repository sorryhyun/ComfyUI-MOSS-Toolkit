"""One node that loads MOSS-Image's DiT, Qwen3.5 text encoder and VAE.

Each part is picked on its own, so SFT's DiT (shipped alone) pairs with Base's TE and
VAE, and the TE can be a quantized single file instead of the bf16 shards (ComfyUI's
mixed-precision format): INT8 (``sorryhyun/MOSS-Image-Qwen3.5-9B-int8``, the default when
present) or fp8 (``sorryhyun/MOSS-Image-Qwen3.5-9B-fp8``). The loading itself is the
``moss-image`` node's (``hf.py`` / ``loaders.py``), looked up at execute time: ComfyUI
imports custom nodes alphabetically, so it is not loaded yet when this pack is.

Options:

* DiT — ``moss_image/<name>/transformer/`` dirs, then ``diffusion_models`` files.
* TE — ``moss_image/<name>/text_encoder/`` dirs (bf16), then ``*.safetensors`` beside a
  ``config.json`` in the other ``moss_image/<name>/`` dirs (``int8/qwen35_9b_int8_convrot.safetensors`` /
  ``fp8/qwen35_9b_fp8_e4m3fn.safetensors``, where ``make download-model ARGS=moss-te-int8`` /
  ``moss-te-fp8`` put them), then ``text_encoders`` files.
* VAE — ``moss_image/<name>/vae/`` dirs, then ``vae`` files.

A single-file TE takes its tokenizer from a ``moss_image/<name>/tokenizer/`` dir (the
DiT's own first, else Base's), so every TE tokenizes identically; with no such dir
it falls back to ComfyUI's bundled Qwen3.5 tokenizer, as ``MOSS-Image Load Qwen3.5``
does.

ComfyUI builds text encoders with full-precision matmuls (an INT8 weight is dequantised
per forward) and switches its quantized kernels on only around ``CLIP.generate``.
``int8_matmul`` puts the encode under the same switch, so an INT8 TE runs W8A8
(``comfy_kitchen.int8_linear``), as anima_lora encodes its caches.

``stream`` fp32 runs the residual stream, modulation, gates, norms, embedders and head
in fp32 around the bf16 block GEMMs, as anima_lora's ``GenerateRequest.stream``
(``library/moss/loader.py::stream_fp32_``): every DiT parameter but the block linears'
and ``q_norm`` / ``k_norm`` weights is cast to fp32. The DiT then loads on ComfyUI's
plain ``ModelPatcher`` (not the dynamic one, whose weights stay mmapped in the file's
dtype), and the ``moss-image`` model's ``_rotate`` returns its input's dtype, so the
fp32 RoPE rows do not lift bf16 queries and keys past the bf16 values.
"""

import os
import re
import sys

import comfy.ops
import comfy.sd
import comfy.utils
import folder_paths
import torch

MOSS_FOLDER = "moss_image"
if MOSS_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(MOSS_FOLDER, os.path.join(folder_paths.models_dir, MOSS_FOLDER))

COMPONENTS = ("transformer", "text_encoder", "tokenizer", "vae")
WEIGHT_DTYPES = ["default", "bf16", "fp32", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]
STREAMS = ["bf16", "fp32"]
# Kept in the weight dtype under the fp32 stream: the block GEMMs, and q/k norms whose
# input is a bf16 GEMM output.
BLOCK_KEEP = re.compile(r"layers\.\d+\.(proj\.\w+|[qk]_norm)")


def _moss_image():
    """``(hf, loaders)``, the ``moss-image`` node's modules."""
    for module in list(sys.modules.values()):
        if hasattr(module, "load_hf_clip") and hasattr(module, "HFTokenizer"):
            loaders = sys.modules.get(module.__name__.rsplit(".", 1)[0] + ".loaders")
            if loaders is not None and hasattr(loaders, "load_clip_state"):
                return module, loaders
    raise RuntimeError("MOSS Toolkit loader needs the moss-image node (OpenMOSS) installed")


def _bases():
    return [b for b in folder_paths.get_folder_paths(MOSS_FOLDER) if os.path.isdir(b)]


def _roots(component, marker="config.json"):
    """``{name: root}`` for ``moss_image/<name>/<component>/<marker>``."""
    roots = {}
    for base in _bases():
        for name in sorted(os.listdir(base)):
            if os.path.isfile(os.path.join(base, name, component, marker)):
                roots.setdefault(name, os.path.join(base, name))
    return roots


def _moss_te_files():
    """``{"<name>/<file>": path}``: safetensors beside a ``config.json`` in
    ``moss_image/<name>/`` dirs that hold no HF component, i.e. the fp8 TE's download dir."""
    files = {}
    for base in _bases():
        for name in sorted(os.listdir(base)):
            folder = os.path.join(base, name)
            if not os.path.isfile(os.path.join(folder, "config.json")):
                continue
            if any(os.path.isdir(os.path.join(folder, c)) for c in COMPONENTS):
                continue
            for file in sorted(os.listdir(folder)):
                if file.endswith(".safetensors"):
                    files.setdefault(f"{name}/{file}", os.path.join(folder, file))
    return files


def _dit_options():
    return list(_roots("transformer")) + folder_paths.get_filename_list("diffusion_models")


def _te_options():
    return (
        list(_roots("text_encoder"))
        + list(_moss_te_files())
        + folder_paths.get_filename_list("text_encoders")
    )


def _vae_options():
    return list(_roots("vae")) + folder_paths.get_filename_list("vae")


def _tokenizer_root(prefer):
    roots = _roots("tokenizer", "tokenizer_config.json")
    if prefer in roots:
        return roots[prefer]
    return roots.get("MOSS-Image-Base") or next(iter(roots.values()), None)


def w8a8_encode(clip):
    """Run ``clip``'s encode W8A8 when it holds INT8 linears; anything else is returned
    untouched. ``comfy.ops.use_quantized_matmul`` lifts the full-precision matmul, and
    the INT8 linears' ``comfy_force_cast_weights`` (set because the TE has a compute
    dtype) is lifted with it: either one dequantises the weight before the matmul."""
    model = clip.cond_stage_model
    linears = [m for m in model.modules() if getattr(m, "quant_format", None) == "int8_tensorwise"]
    if not linears:
        return clip
    encode = model.encode_token_weights

    def encode_token_weights(tokens):
        forced = [m.__dict__.get("comfy_force_cast_weights", False) for m in linears]
        try:
            for m in linears:
                m.comfy_force_cast_weights = False
            with comfy.ops.use_quantized_matmul(model, clip.patcher.load_device):
                return encode(tokens)
        finally:
            for m, f in zip(linears, forced):
                m.comfy_force_cast_weights = f

    model.encode_token_weights = encode_token_weights
    return clip


def load_te_file(path, tokenizer_root, device, disable_dynamic=False):
    """A single-file Qwen3.5-9B (bf16, or INT8 / fp8 ``comfy_quant``) → CLIP, tokenized by
    ``tokenizer_root/tokenizer`` when given."""
    hf, loaders = _moss_image()
    if tokenizer_root is None:
        clip = loaders.load_clip(path, device, disable_dynamic)
    else:
        state, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        state, metadata = comfy.utils.convert_old_quants(state, model_prefix="", metadata=metadata)
        tokenizer = os.path.join(tokenizer_root, "tokenizer")
        clip = loaders.load_clip_state(state, device, disable_dynamic, hf.HFTokenizer, {"hf_path": tokenizer})
    clip.patcher.cached_patcher_init = (load_te_file_patcher, (path, tokenizer_root, device))
    return clip


def load_te_file_patcher(path, tokenizer_root, device, disable_dynamic=False):
    return load_te_file(path, tokenizer_root, device, disable_dynamic).patcher


def load_vae_file(path):
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=state, metadata=metadata, dtype=torch.float32)
    vae.throw_exception_if_invalid()
    return vae


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


def load_dit(dit, weight_dtype, stream="bf16", disable_dynamic=False):
    hf, loaders = _moss_image()
    fp32 = stream == "fp32"
    dit_roots = _roots("transformer")
    if dit in dit_roots:
        model = hf.load_hf_model(dit_roots[dit], weight_dtype, disable_dynamic or fp32)
    else:
        path = folder_paths.get_full_path_or_raise("diffusion_models", dit)
        model = loaders.load_model(path, weight_dtype, disable_dynamic or fp32)
    if fp32:
        stream_fp32(model)
        model.cached_patcher_init = (load_dit, (dit, weight_dtype, stream))
    return model


def _default(options, preferred):
    return next((o for o in preferred if o in options), options[0] if options else None)


class MOSSImageLoader:
    """DiT + Qwen3.5 TE + VAE, each from its own source; the TE may be INT8 or fp8."""

    @classmethod
    def INPUT_TYPES(cls):
        dit, te, vae = _dit_options(), _te_options(), _vae_options()
        return {
            "required": {
                "dit": (dit, {"default": _default(dit, ["MOSS-Image-SFT", "MOSS-Image-Base"])}),
                "weight_dtype": (WEIGHT_DTYPES, {"default": "default"}),
                "text_encoder": (te, {
                    "default": _default(te, [
                        "int8/qwen35_9b_int8_convrot.safetensors",
                        "fp8/qwen35_9b_fp8_e4m3fn.safetensors",
                        "MOSS-Image-Base",
                    ]),
                    "tooltip": "bf16: a moss_image/<name>/text_encoder dir. INT8: "
                    "qwen35_9b_int8_convrot.safetensors (moss_image/int8/ or text_encoders/). "
                    "fp8: qwen35_9b_fp8_e4m3fn.safetensors (moss_image/fp8/ or text_encoders/).",
                }),
                "vae": (vae, {"default": _default(vae, ["MOSS-Image-Base"])}),
            },
            "optional": {
                "text_device": (["default", "cpu"], {"default": "default", "advanced": True}),
                "int8_matmul": ("BOOLEAN", {
                    "default": True,
                    "advanced": True,
                    "tooltip": "INT8 TE: int8 activations too (W8A8, faster, as anima_lora "
                    "encodes). Off: the weight is dequantised for a bf16 matmul (ComfyUI's "
                    "default, closer to bf16). No effect on a bf16 / fp8 TE.",
                }),
                "stream": (STREAMS, {
                    "default": "bf16",
                    "advanced": True,
                    "tooltip": "DiT residual stream precision. fp32: the stream, modulation, "
                    "norms, embedders and head run in fp32 around the bf16 block GEMMs "
                    "(+1.5% time, +0.6 GB, ComfyUI's non-dynamic model loading). Renders move "
                    "a few /255 in detail; neither is consistently better.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"
    CATEGORY = "MOSS-Image"

    def load(
        self, dit, weight_dtype, text_encoder, vae, text_device="default", int8_matmul=True, stream="bf16"
    ):
        hf, loaders = _moss_image()
        model = load_dit(dit, weight_dtype, stream)

        te_roots, te_files = _roots("text_encoder"), _moss_te_files()
        if text_encoder in te_roots:
            clip = hf.load_hf_clip(te_roots[text_encoder], text_device)
        else:
            path = te_files.get(text_encoder) or folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
            clip = load_te_file(path, _tokenizer_root(dit), text_device)
            if int8_matmul:
                clip = w8a8_encode(clip)

        vae_roots = _roots("vae")
        if vae in vae_roots:
            vae_out = hf.load_hf_vae(vae_roots[vae])
        else:
            vae_out = load_vae_file(folder_paths.get_full_path_or_raise("vae", vae))
        return (model, clip, vae_out)


NODE_CLASS_MAPPINGS = {"MOSSImageLoader": MOSSImageLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSImageLoader": "MOSS-Image Loader (DiT + TE + VAE)"}

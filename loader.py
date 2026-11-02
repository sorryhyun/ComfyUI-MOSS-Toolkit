"""One node that loads MOSS-Image's DiT, Qwen3.5 text encoder and VAE.

Each part is picked on its own, so SFT's DiT (shipped alone) pairs with Base's TE and
VAE, and the TE can be a quantized single file instead of the bf16 shards (ComfyUI's
mixed-precision format): INT8 (``sorryhyun/MOSS-Image-Qwen3.5-9B-int8``, the default when
present) or fp8 (``sorryhyun/MOSS-Image-Qwen3.5-9B-fp8``). Everything loads through
ComfyUI's own ``MOSSImage`` model and MOSS-Image text encoder: vendor shards (separate
q / k / v) and Comfy-Org's fused or INT8 files alike, the model ``Load Diffusion Model``
gives. It samples with ``ModelSamplingAuraFlow`` (timestep σ); ``ModelSamplingSD3``
(1000·σ) renders noise on it.

Options:

* DiT — ``moss_image/<name>/transformer/`` dirs, then ``diffusion_models`` files.
* TE — ``moss_image/<name>/text_encoder/`` dirs (bf16), then ``*.safetensors`` beside a
  ``config.json`` in the other ``moss_image/<name>/`` dirs (``int8/qwen35_9b_int8_convrot.safetensors`` /
  ``fp8/qwen35_9b_fp8_e4m3fn.safetensors``, where ``make download-model ARGS=moss-te-int8`` /
  ``moss-te-fp8`` put them), then ``text_encoders`` files.
* VAE — ``moss_image/<name>/vae/`` dirs, then ``vae`` files.

ComfyUI builds text encoders with full-precision matmuls (an INT8 weight is dequantised
per forward) and switches its quantized kernels on only around ``CLIP.generate``.
``int8_matmul`` puts the encode under the same switch, so an INT8 TE runs W8A8
(``comfy_kitchen.int8_linear``), as anima_lora encodes its caches.

``stream`` fp32 runs the residual stream, modulation, gates, norms, embedders and head
in fp32 around the block GEMMs, as anima_lora's ``GenerateRequest.stream``: the same
MODEL patch as ``MOSS-Image fp32 Stream`` (:mod:`.stream`).
"""

import json
import logging
import os

import comfy.ops
import comfy.sd
import comfy.utils
import folder_paths
import torch

from .stream import stream_fp32_core

logger = logging.getLogger(__name__)

MOSS_FOLDER = "moss_image"
if MOSS_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(MOSS_FOLDER, os.path.join(folder_paths.models_dir, MOSS_FOLDER))

COMPONENTS = ("transformer", "text_encoder", "tokenizer", "vae")
DTYPES = {
    "default": None,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e4m3fn_fast": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
STREAMS = ["bf16", "fp32"]


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


def read_weights(folder, stem):
    """``folder/<stem>.safetensors``, or the shards its ``.index.json`` names."""
    index = os.path.join(folder, f"{stem}.safetensors.index.json")
    if not os.path.isfile(index):
        return comfy.utils.load_torch_file(os.path.join(folder, f"{stem}.safetensors"), safe_load=True)
    with open(index, encoding="utf-8") as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    state = {}
    for shard in shards:
        state.update(comfy.utils.load_torch_file(os.path.join(folder, shard), safe_load=True))
    return state


def read_te(path):
    """Qwen3.5-9B weights from a ``moss_image/<name>`` root or a single file (bf16, or
    INT8 / fp8 ``comfy_quant``), under the prefixes ComfyUI's detection reads."""
    if os.path.isdir(path):
        state = read_weights(os.path.join(path, "text_encoder"), "model")
    else:
        state, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        state, _ = comfy.utils.convert_old_quants(state, model_prefix="", metadata=metadata)
    state = comfy.utils.state_dict_prefix_replace(
        state, {"language_model.": "model.language_model.", "visual.": "model.visual."}
    )
    if comfy.sd.detect_te_model(state) != comfy.sd.TEModel.QWEN35_9B:
        raise ValueError(f"MOSS-Image Loader: {path} is not a Qwen3.5-9B text encoder")
    return state


def load_te(path, device, disable_dynamic=False):
    """A ``moss_image/<name>`` root or a single TE file → CLIP, on ComfyUI's MOSS-Image
    text encoder and bundled Qwen3.5 tokenizer."""
    options = {}
    if device == "cpu":
        options = {"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")}
    clip = comfy.sd.load_text_encoder_state_dicts(
        [read_te(path)],
        clip_type=comfy.sd.CLIPType.MOSS_IMAGE,
        model_options=options,
        disable_dynamic=disable_dynamic,
    )
    clip.patcher.cached_patcher_init = (load_te_patcher, (path, device))
    return clip


def load_te_patcher(path, device, disable_dynamic=False):
    return load_te(path, device, disable_dynamic).patcher


def load_vae(state, metadata=None):
    vae = comfy.sd.VAE(sd=state, metadata=metadata, dtype=torch.float32)
    vae.throw_exception_if_invalid()
    return vae


def dit_options(weight_dtype):
    """``model_options`` for ``weight_dtype``, as ``Load Diffusion Model``."""
    options = {} if DTYPES[weight_dtype] is None else {"dtype": DTYPES[weight_dtype]}
    if weight_dtype == "fp8_e4m3fn_fast":
        options["fp8_optimizations"] = True
    return options


def load_hf_dit(root, weight_dtype, disable_dynamic=False):
    """A ``moss_image/<name>/transformer`` dir through ComfyUI's loader."""
    state = read_weights(os.path.join(root, "transformer"), "diffusion_pytorch_model")
    model = comfy.sd.load_diffusion_model_state_dict(
        state, model_options=dit_options(weight_dtype), disable_dynamic=disable_dynamic
    )
    if model is None:
        raise RuntimeError(f"MOSS-Image Loader: no MOSS-Image DiT in {root}/transformer")
    model.cached_patcher_init = (load_hf_dit, (root, weight_dtype))
    return model


def fuse_int8_qkv(state):
    """An INT8 file's ``proj.q`` / ``k`` / ``v`` rows as one ``proj.qkv``, the fused layout
    ComfyUI detects: one GEMM and one activation quantization instead of three, with the
    same output. LoRA keys for q / k / v map onto its rows."""
    heads = [k[: -len("q.comfy_quant")] for k in state if k.endswith(".proj.q.comfy_quant")]
    for head in heads:
        for suffix in ("weight", "weight_scale"):
            rows = [state.pop(f"{head}{p}.{suffix}") for p in "qkv"]
            state[f"{head}qkv.{suffix}"] = torch.cat(rows)
        state[f"{head}qkv.comfy_quant"] = state.pop(f"{head}q.comfy_quant")
        del state[f"{head}k.comfy_quant"], state[f"{head}v.comfy_quant"]
    return state


def load_file_dit(path, options, disable_dynamic=False):
    """A single DiT file, as ``Load Diffusion Model`` after :func:`fuse_int8_qkv`."""
    state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    model = comfy.sd.load_diffusion_model_state_dict(
        fuse_int8_qkv(state), model_options=options, metadata=metadata, disable_dynamic=disable_dynamic
    )
    if model is None:
        raise RuntimeError(f"MOSS-Image Loader: no diffusion model in {path}")
    model.cached_patcher_init = (load_file_dit, (path, options))
    return model


def load_dit(dit, weight_dtype, stream="bf16"):
    dit_roots = _roots("transformer")
    if dit in dit_roots:
        model = load_hf_dit(dit_roots[dit], weight_dtype)
    else:
        path = folder_paths.get_full_path_or_raise("diffusion_models", dit)
        model = load_file_dit(path, dit_options(weight_dtype))
    logger.info(
        "MOSS-Image Loader: %s runs on ComfyUI's own MOSS-Image model. Set the flow shift "
        "with ModelSamplingAuraFlow, or leave it at 5: ModelSamplingSD3 samples noise here",
        dit,
    )
    return stream_fp32_core(model) if stream == "fp32" else model


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
                "weight_dtype": (list(DTYPES), {"default": "default"}),
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
                    "norms, embedders and head run in fp32 around the block GEMMs "
                    "(+1.5% time, +0.6 GB). Renders move a few /255 in detail; neither is "
                    "consistently better.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"
    CATEGORY = "MOSS-Image"

    def load(
        self, dit, weight_dtype, text_encoder, vae, text_device="default", int8_matmul=True, stream="bf16"
    ):
        model = load_dit(dit, weight_dtype, stream)

        te_roots, te_files = _roots("text_encoder"), _moss_te_files()
        te_path = (
            te_roots.get(text_encoder)
            or te_files.get(text_encoder)
            or folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
        )
        clip = load_te(te_path, text_device)
        if int8_matmul:
            clip = w8a8_encode(clip)

        vae_roots = _roots("vae")
        if vae in vae_roots:
            vae_out = load_vae(read_weights(os.path.join(vae_roots[vae], "vae"), "diffusion_pytorch_model"))
        else:
            state, metadata = comfy.utils.load_torch_file(
                folder_paths.get_full_path_or_raise("vae", vae), return_metadata=True
            )
            vae_out = load_vae(state, metadata)
        return (model, clip, vae_out)


NODE_CLASS_MAPPINGS = {"MOSSImageLoader": MOSSImageLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSImageLoader": "MOSS-Image Loader (DiT + TE + VAE)"}

"""One node that loads MOSS-Image's DiT, Qwen3.5 text encoder and VAE.

Each part is picked on its own, so SFT's DiT (shipped alone) pairs with Base's TE and
VAE, and the TE can be the fp8 single file (``sorryhyun/MOSS-Image-Qwen3.5-9B-fp8``,
ComfyUI's mixed-precision format) instead of the bf16 shards. The loading itself is the
``moss-image`` node's (``hf.py`` / ``loaders.py``), looked up at execute time: ComfyUI
imports custom nodes alphabetically, so it is not loaded yet when this pack is.

Options:

* DiT — ``moss_image/<name>/transformer/`` dirs, then ``diffusion_models`` files.
* TE — ``moss_image/<name>/text_encoder/`` dirs (bf16), then ``*.safetensors`` beside a
  ``config.json`` in the other ``moss_image/<name>/`` dirs (``fp8/qwen35_9b_fp8_e4m3fn.safetensors``, where
  ``make download-model ARGS=moss-te-fp8`` puts it), then ``text_encoders`` files.
* VAE — ``moss_image/<name>/vae/`` dirs, then ``vae`` files.

A single-file TE takes its tokenizer from a ``moss_image/<name>/tokenizer/`` dir (the
DiT's own first, else Base's), so bf16 and fp8 tokenize identically; with no such dir
it falls back to ComfyUI's bundled Qwen3.5 tokenizer, as ``MOSS-Image Load Qwen3.5``
does.
"""

import os
import sys

import comfy.sd
import comfy.utils
import folder_paths
import torch

MOSS_FOLDER = "moss_image"
if MOSS_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(MOSS_FOLDER, os.path.join(folder_paths.models_dir, MOSS_FOLDER))

COMPONENTS = ("transformer", "text_encoder", "tokenizer", "vae")
WEIGHT_DTYPES = ["default", "bf16", "fp32", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]


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


def load_te_file(path, tokenizer_root, device, disable_dynamic=False):
    """A single-file Qwen3.5-9B (bf16 or fp8 ``comfy_quant``) → CLIP, tokenized by
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


def _default(options, preferred):
    return next((o for o in preferred if o in options), options[0] if options else None)


class MOSSImageLoader:
    """DiT + Qwen3.5 TE + VAE, each from its own source; the TE may be fp8."""

    @classmethod
    def INPUT_TYPES(cls):
        dit, te, vae = _dit_options(), _te_options(), _vae_options()
        return {
            "required": {
                "dit": (dit, {"default": _default(dit, ["MOSS-Image-SFT", "MOSS-Image-Base"])}),
                "weight_dtype": (WEIGHT_DTYPES, {"default": "default"}),
                "text_encoder": (te, {
                    "default": _default(te, ["fp8/qwen35_9b_fp8_e4m3fn.safetensors", "MOSS-Image-Base"]),
                    "tooltip": "bf16: a moss_image/<name>/text_encoder dir. fp8: "
                    "qwen35_9b_fp8_e4m3fn.safetensors (moss_image/fp8/ or text_encoders/).",
                }),
                "vae": (vae, {"default": _default(vae, ["MOSS-Image-Base"])}),
            },
            "optional": {
                "text_device": (["default", "cpu"], {"default": "default", "advanced": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"
    CATEGORY = "MOSS-Image"

    def load(self, dit, weight_dtype, text_encoder, vae, text_device="default"):
        hf, loaders = _moss_image()

        dit_roots = _roots("transformer")
        if dit in dit_roots:
            model = hf.load_hf_model(dit_roots[dit], weight_dtype)
        else:
            model = loaders.load_model(folder_paths.get_full_path_or_raise("diffusion_models", dit), weight_dtype)

        te_roots, te_files = _roots("text_encoder"), _moss_te_files()
        if text_encoder in te_roots:
            clip = hf.load_hf_clip(te_roots[text_encoder], text_device)
        else:
            path = te_files.get(text_encoder) or folder_paths.get_full_path_or_raise("text_encoders", text_encoder)
            clip = load_te_file(path, _tokenizer_root(dit), text_device)

        vae_roots = _roots("vae")
        if vae in vae_roots:
            vae_out = hf.load_hf_vae(vae_roots[vae])
        else:
            vae_out = load_vae_file(folder_paths.get_full_path_or_raise("vae", vae))
        return (model, clip, vae_out)


NODE_CLASS_MAPPINGS = {"MOSSImageLoader": MOSSImageLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSImageLoader": "MOSS-Image Loader (DiT + TE + VAE)"}

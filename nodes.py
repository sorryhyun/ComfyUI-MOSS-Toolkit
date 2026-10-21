"""MOSS i2i nodes: LoRA loader, page-through-the-vision-tower text encode, and the
VAE reference patch.

Preprocessing matches anima_lora's colorize cache (``project/moss_lora/src/
colorize_cache.py`` on ``i2i_probe``): the page is fit (cover + centre crop,
Lanczos) to the target size, which is ``library/moss/cache.py::target_size``; the
vision tower sees it at that pixel count through Qwen2-VL's ``smart_resize``
(factor 32, bicubic), and the ref is the fit page fit again to ``1/ref_scale`` per
side, sides multiples of 16.
"""

import logging
import math

import comfy.patcher_extension
import comfy.sd
import comfy.utils
import folder_paths
import numpy as np
import torch
from PIL import Image

from .reference import make_apply_model_wrapper

logger = logging.getLogger(__name__)

STRIDE = 16
AREA_TOLERANCE = 1.10
VISION_FACTOR = 32  # Qwen3.5 patch 16 × merge 2
VISION_MIN_PIXELS = 65536  # the probe's Qwen2VLImageProcessor shortest_edge
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_PAD = 248056
META_KEY = "moss_i2i"
WRAPPER_KEY = "moss_i2i_ref"


def target_size(width, height, resolution):
    """``(W, H)``: native aspect, multiples of 16, area ≤ ~``resolution²``, never
    larger than the source on either side."""
    cap = resolution * resolution
    if width % STRIDE == 0 and height % STRIDE == 0 and width * height <= cap * AREA_TOLERANCE:
        return width, height
    area = min(cap, width * height)
    aspect = width / height
    out_w = round(math.sqrt(area * aspect) / STRIDE) * STRIDE
    out_h = round(math.sqrt(area / aspect) / STRIDE) * STRIDE
    out_w = max(STRIDE, min(out_w, width // STRIDE * STRIDE))
    out_h = max(STRIDE, min(out_h, height // STRIDE * STRIDE))
    return out_w, out_h


def fit(image, width, height):
    """Resize to cover ``(width, height)`` and centre-crop; identity at that size."""
    if image.size == (width, height):
        return image
    scale = max(width / image.width, height / image.height)
    rw = max(width, math.ceil(image.width * scale))
    rh = max(height, math.ceil(image.height * scale))
    image = image.resize((rw, rh), Image.LANCZOS)
    left, top = (rw - width) // 2, (rh - height) // 2
    return image.crop((left, top, left + width, top + height))


def smart_resize(height, width, max_pixels):
    """Qwen2-VL's ``smart_resize`` at ``factor = 32``."""
    f = VISION_FACTOR
    h_bar = max(f, round(height / f) * f)
    w_bar = max(f, round(width / f) * f)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(f, math.floor(height / beta / f) * f)
        w_bar = max(f, math.floor(width / beta / f) * f)
    elif h_bar * w_bar < VISION_MIN_PIXELS:
        beta = math.sqrt(VISION_MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / f) * f
        w_bar = math.ceil(width * beta / f) * f
    return h_bar, w_bar


def to_pil(image):
    """First frame of a ComfyUI ``IMAGE`` ``(B, H, W, C)`` in ``[0, 1]`` → RGB PIL."""
    array = image[0, ..., :3].clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return Image.fromarray(array, "RGB")


def to_image(pil):
    """RGB PIL → ComfyUI ``IMAGE`` ``(1, H, W, 3)`` in ``[0, 1]``."""
    return torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0)[None]


def page_at_target(image, resolution):
    page = to_pil(image)
    return fit(page, *target_size(page.width, page.height, resolution))


def _ref_timestep(value):
    return None if value in (None, "") or float(value) < 0 else float(value)


def _is_moss(model):
    dit = model.get_model_object("diffusion_model")
    return hasattr(dit, "video_head_modulation") and hasattr(dit, "final_norm_video")


RESOLUTION = (
    "INT",
    {
        "default": 1024,
        "min": 256,
        "max": 2048,
        "step": 16,
        "tooltip": (
            "Target area ≈ resolution², native aspect, sides multiples of 16 (the "
            "training cache's target_size). Keep it equal on the Text Encode and "
            "Reference nodes so the vision tower and the ref see the same page."
        ),
    },
)


class MOSSi2iLoraLoader:
    """A MOSS-Image LoRA from anima_lora's MOSS trainer, model side only. The keys
    are plain ``lora_unet_layers_*`` (channel scaling is folded into the stored
    down weights), so this is ComfyUI's LoRA patch plus a metadata check that the
    Reference node reads."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "MOSS-Image/i2i"
    DESCRIPTION = "Load a MOSS i2i LoRA (e.g. colorize) onto a MOSS-Image MODEL."

    def load(self, model, lora_name, strength):
        if not _is_moss(model):
            raise ValueError("MOSS i2i LoRA Loader needs a MOSS-Image MODEL (moss-image node).")
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        meta = meta or {}
        if meta.get("base_model", "MOSS-Image") != "MOSS-Image":
            logger.warning("MOSS i2i: %s was trained on %s", lora_name, meta["base_model"])
        out = model
        if strength:
            out, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
        else:
            out = model.clone()
        options = dict(out.model_options.get("transformer_options", {}))
        options[META_KEY] = {
            "lora": lora_name,
            "reference_latents": int(meta.get("reference_latents", 0) or 0),
            "ref_scale": meta.get("ref_scale"),
            # Stamped by anima_lora since ref_timestep; negative or absent = shared.
            "ref_timestep": _ref_timestep(meta.get("ref_timestep")),
        }
        out.model_options["transformer_options"] = options
        return (out,)


class MOSSi2iTextEncode:
    """The page through Qwen3.5's vision tower, then the text, in one user turn:
    ``<|vision_start|>[page]<|vision_end|>{text}``. The colorize LoRAs were trained
    on this encode (colour caption as ``text``; empty ``text`` is the page-only
    mode). Use a plain text encode of "" as the negative."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "resolution": RESOLUTION,
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "MOSS-Image/i2i"
    DESCRIPTION = "MOSS-Image text encode with the page through the Qwen3.5 vision tower."

    def encode(self, clip, image, text, resolution):
        page = page_at_target(image, resolution)
        h, w = smart_resize(page.height, page.width, page.width * page.height)
        pixels = to_image(page.resize((w, h), Image.BICUBIC))
        tokens = clip.tokenize(VISION_BLOCK + text)
        placed = 0
        for rows in tokens.values():
            for row in rows:
                for i, token in enumerate(row):
                    if token[0] == IMAGE_PAD:
                        row[i] = ({"type": "image", "data": pixels, "original_type": "image"},) + token[1:]
                        placed += 1
        if placed != 1:
            raise RuntimeError(
                f"MOSS i2i Text Encode placed {placed} images; expected 1 (is this the moss-image CLIP?)"
            )
        return (clip.encode_from_tokens_scheduled(tokens),)


class MOSSi2iReference:
    """The page as a clean VAE latent on RoPE's ``r = 1`` axis, in every forward and
    every CFG branch, for a LoRA trained with references. Works under any sampler
    that calls ``apply_model`` (KSampler, the MOSS Spectrum KSampler). Outputs an
    empty latent at the target size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "resolution": RESOLUTION,
                "ref_scale": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "tooltip": (
                            "Ref side = target side / ref_scale. The colorize ref LoRA "
                            "was trained at 2 (1/4 area)."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "patch"
    CATEGORY = "MOSS-Image/i2i"
    DESCRIPTION = "Feed the page to MOSS-Image as a VAE reference latent (RoPE r = 1)."

    def patch(self, model, vae, image, resolution, ref_scale):
        if not _is_moss(model):
            raise ValueError("MOSS i2i Reference needs a MOSS-Image MODEL (moss-image node).")
        meta = model.model_options.get("transformer_options", {}).get(META_KEY)
        if meta is None:
            logger.warning("MOSS i2i: no MOSS i2i LoRA on this model; the base never saw a reference")
        else:
            if not meta["reference_latents"]:
                logger.warning(
                    "MOSS i2i: %s was trained without references; the ref is off its distribution",
                    meta["lora"],
                )
            if meta["ref_scale"] and int(meta["ref_scale"]) != ref_scale:
                logger.warning(
                    "MOSS i2i: %s was trained at ref_scale %s, running %d",
                    meta["lora"],
                    meta["ref_scale"],
                    ref_scale,
                )
        page = page_at_target(image, resolution)
        w, h = page.size
        rw = max(STRIDE, round(w / ref_scale / STRIDE) * STRIDE)
        rh = max(STRIDE, round(h / ref_scale / STRIDE) * STRIDE)
        ref = vae.encode(to_image(fit(page, rw, rh))).float().cpu()
        ref_timestep = None if meta is None else meta.get("ref_timestep")
        out = model.clone()
        out.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, WRAPPER_KEY)
        out.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.APPLY_MODEL,
            WRAPPER_KEY,
            make_apply_model_wrapper(ref, ref_timestep),
        )
        latent = torch.zeros(1, ref.shape[1], h // STRIDE, w // STRIDE)
        return (out, {"samples": latent})


NODE_CLASS_MAPPINGS = {
    "MOSSi2iLoraLoader": MOSSi2iLoraLoader,
    "MOSSi2iTextEncode": MOSSi2iTextEncode,
    "MOSSi2iReference": MOSSi2iReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSi2iLoraLoader": "MOSS i2i LoRA Loader",
    "MOSSi2iTextEncode": "MOSS i2i Text Encode (page + text)",
    "MOSSi2iReference": "MOSS i2i Reference",
}

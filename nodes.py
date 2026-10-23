"""MOSS i2i nodes: page-through-the-vision-tower text encode, and the VAE reference
patch with its LoRA.

``task = colorize`` on the Reference node runs the colorize recipe the LoRA was
trained with (anima_lora ``project/moss_lora/plan.md``): v3 (``ref_isolate`` in the
metadata) takes the page as a full-size isolated ref with a text-only prompt (plain
CLIP Text Encode), v2 as a half-size joint ref with the page through the vision
tower (MOSS i2i Text Encode).

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
WRAPPER_KEY = "moss_i2i_ref"
TASKS = ("colorize", "custom")
REF_CACHE = ("gpu", "cpu", "off")


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


def load_lora(model, lora_name, strength):
    """``(model, meta)``: an anima_lora MOSS LoRA on ``model``, and the reference
    settings it trained with. The keys are plain ``lora_unet_layers_*`` (channel
    scaling is folded into the stored down weights), so this is ComfyUI's LoRA patch."""
    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora, meta = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    meta = meta or {}
    if meta.get("base_model", "MOSS-Image") != "MOSS-Image":
        logger.warning("MOSS i2i: %s was trained on %s", lora_name, meta["base_model"])
    if strength:
        model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
    return model, {
        "reference_latents": int(meta.get("reference_latents", 0) or 0),
        "ref_scale": meta.get("ref_scale"),
        # Stamped by anima_lora since ref_timestep; negative or absent = shared.
        "ref_timestep": _ref_timestep(meta.get("ref_timestep")),
        # Stamped "1" by anima_lora when the ref queries saw only the ref keys.
        "ref_isolate": meta.get("ref_isolate") in ("1", "True"),
    }


def colorize_recipe(meta):
    """``(ref_scale, label)`` of the colorize recipe ``meta`` trained with: v3
    (``ref_isolate``) full size, v2 half size. A stamped ``ref_scale`` wins."""
    v3 = meta is None or meta["ref_isolate"]
    scale = 1 if v3 else 2
    if meta is not None and meta["ref_scale"]:
        scale = int(meta["ref_scale"])
    if v3:
        return scale, "v3: isolated ref, text-only prompt (CLIP Text Encode, no page)"
    return scale, "v2: joint ref, page through the vision tower (MOSS i2i Text Encode)"


class MOSSi2iReference:
    """A MOSS i2i LoRA (e.g. colorize) plus the page as a clean VAE latent on RoPE's
    ``r = 1`` axis, in every forward and every CFG branch. The LoRA's metadata sets
    the ref rows' timestep and isolation and checks ``ref_scale``; ``task = colorize``
    sets ``ref_scale`` from the recipe the LoRA trained with. Works under any sampler
    that calls ``apply_model`` (KSampler, the MOSS Spectrum KSampler). Outputs an
    empty latent at the target size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "lora_name": (
                    ["None"] + folder_paths.get_filename_list("loras"),
                    {"tooltip": "A LoRA trained with references (reference_latents in its metadata)."},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05},
                ),
                "resolution": RESOLUTION,
                "ref_scale": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "tooltip": (
                            "Ref side = target side / ref_scale (task = custom only). "
                            "Colorize v3 trains at 1 (full size), v2 at 2 (1/4 area)."
                        ),
                    },
                ),
                "task": (
                    list(TASKS),
                    {
                        "default": "colorize",
                        "tooltip": (
                            "colorize: the recipe the LoRA trained with. v3 (ref_isolate "
                            "in its metadata): full-size isolated ref, prompt through a "
                            "plain CLIP Text Encode. v2: half-size ref, prompt through "
                            "MOSS i2i Text Encode. custom: ref_scale as set."
                        ),
                    },
                ),
                "ref_cache": (
                    list(REF_CACHE),
                    {
                        "default": "gpu",
                        "tooltip": (
                            "An isolated ref (colorize v3) depends on the page alone: "
                            "run it through the blocks once and keep its keys / values "
                            "(~2.1 GB bf16 at 1 MP), so each forward runs the image and "
                            "text rows only. cpu: kept in host memory, copied in per "
                            "forward. off: the ref rows in every forward."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "patch"
    CATEGORY = "MOSS-Image/i2i"
    DESCRIPTION = "Load a MOSS i2i LoRA and feed the page to MOSS-Image as a VAE reference latent (RoPE r = 1)."

    def patch(self, model, vae, image, lora_name, strength, resolution, ref_scale, task="colorize", ref_cache="gpu"):
        if not _is_moss(model):
            raise ValueError("MOSS i2i Reference needs a MOSS-Image MODEL (moss-image node).")
        meta = None
        if lora_name == "None":
            logger.warning("MOSS i2i: no LoRA; the base never saw a reference")
        else:
            model, meta = load_lora(model, lora_name, strength)
            if not meta["reference_latents"]:
                logger.warning(
                    "MOSS i2i: %s was trained without references; the ref is off its distribution",
                    lora_name,
                )
        if task == "colorize":
            ref_scale, recipe = colorize_recipe(meta)
            logger.info("MOSS i2i: colorize %s, ref_scale %d", recipe, ref_scale)
        elif meta is not None and meta["ref_scale"] and int(meta["ref_scale"]) != ref_scale:
            logger.warning(
                "MOSS i2i: %s was trained at ref_scale %s, running %d",
                lora_name,
                meta["ref_scale"],
                ref_scale,
            )
        page = page_at_target(image, resolution)
        w, h = page.size
        rw = max(STRIDE, round(w / ref_scale / STRIDE) * STRIDE)
        rh = max(STRIDE, round(h / ref_scale / STRIDE) * STRIDE)
        ref = vae.encode(to_image(fit(page, rw, rh))).float().cpu()
        ref_timestep = None if meta is None else meta["ref_timestep"]
        ref_isolate = meta is not None and meta["ref_isolate"]
        out = model.clone()
        out.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, WRAPPER_KEY)
        out.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.APPLY_MODEL,
            WRAPPER_KEY,
            make_apply_model_wrapper(ref, ref_timestep, ref_isolate, ref_cache),
        )
        latent = torch.zeros(1, ref.shape[1], h // STRIDE, w // STRIDE)
        return (out, {"samples": latent})


NODE_CLASS_MAPPINGS = {
    "MOSSi2iTextEncode": MOSSi2iTextEncode,
    "MOSSi2iReference": MOSSi2iReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSi2iTextEncode": "MOSS i2i Text Encode (page + text)",
    "MOSSi2iReference": "MOSS i2i Reference",
}

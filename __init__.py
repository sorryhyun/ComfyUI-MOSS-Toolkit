"""MOSS Toolkit ComfyUI custom nodes, on ComfyUI's own MOSS-Image support.

Loader (:mod:`.loader`):

* ``MOSSImageLoader`` — DiT + Qwen3.5 TE (bf16 dir, INT8 or fp8 file) + VAE, each from its
  own source -> MODEL, CLIP, VAE.

Stream (:mod:`.stream`):

* ``MOSSStreamFP32`` — MODEL -> MODEL, the fp32 residual stream on ComfyUI's own
  ``MOSSImage`` (bf16 or INT8).

i2i (:mod:`.nodes`, self-contained, pulls no anima_lora code):

* ``MOSSi2iTextEncode`` — CLIP + IMAGE + text -> CONDITIONING, the page through
  Qwen3.5's vision tower ahead of the text.
* ``MOSSi2iReference`` — MODEL + VAE + IMAGE -> MODEL + LATENT, a MOSS i2i LoRA plus
  the page as a clean VAE latent on RoPE's reserved ``r = 1`` axis.

Text (:mod:`.moss_layout`, registered only when ``anime-tools`` is installed):

* ``MOSSCaptionHelper`` — STRING (+ IMAGE) -> STRING, per task: ``layout`` an Anima
  caption as MOSS's layout-JSON prompt,
  ``colorize`` that layout plus the page's OCR lines, the colorize LoRA's caption,
  ``sanitize`` / ``decensor`` that LoRA's fixed caption.
* ``AnimaCaptionToMossLayout`` — deprecated, hidden from the menu; ``layout`` alone,
  kept so saved workflows load.
"""

import logging

from . import loader, nodes, stream

NODE_CLASS_MAPPINGS = {
    **loader.NODE_CLASS_MAPPINGS,
    **stream.NODE_CLASS_MAPPINGS,
    **nodes.NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **loader.NODE_DISPLAY_NAME_MAPPINGS,
    **stream.NODE_DISPLAY_NAME_MAPPINGS,
    **nodes.NODE_DISPLAY_NAME_MAPPINGS,
}

try:
    from . import moss_layout
except ImportError as e:
    logging.getLogger(__name__).warning(
        "MOSS Toolkit: layout node skipped (%s). Install it with "
        "pip install 'anime-tools @ git+https://github.com/sorryhyun/anime_tools'",
        e,
    )
else:
    NODE_CLASS_MAPPINGS.update(moss_layout.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(moss_layout.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

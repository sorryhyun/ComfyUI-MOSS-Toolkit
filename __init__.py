"""MOSS Toolkit ComfyUI custom nodes, on the ``moss-image`` node's MODEL / CLIP / VAE.

Loader (:mod:`.loader`, the ``moss-image`` node's loading code):

* ``MOSSImageLoader`` — DiT + Qwen3.5 TE (bf16 dir, INT8 or fp8 file) + VAE, each from its
  own source -> MODEL, CLIP, VAE.

i2i (:mod:`.nodes`, self-contained, pulls no anima_lora code):

* ``MOSSi2iTextEncode`` — CLIP + IMAGE + text -> CONDITIONING, the page through
  Qwen3.5's vision tower ahead of the text.
* ``MOSSi2iReference`` — MODEL + VAE + IMAGE -> MODEL + LATENT, a MOSS i2i LoRA plus
  the page as a clean VAE latent on RoPE's reserved ``r = 1`` axis.

Text (:mod:`.moss_layout`, registered only when ``anime-tools`` is installed):

* ``AnimaCaptionToMossLayout`` — STRING -> STRING, an Anima caption as MOSS's
  layout-JSON prompt.
"""

import logging

from . import loader, nodes

NODE_CLASS_MAPPINGS = {**loader.NODE_CLASS_MAPPINGS, **nodes.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**loader.NODE_DISPLAY_NAME_MAPPINGS, **nodes.NODE_DISPLAY_NAME_MAPPINGS}

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

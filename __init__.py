"""MOSS Toolkit ComfyUI custom nodes, on the ``moss-image`` node's MODEL / CLIP / VAE.

i2i (:mod:`.nodes`, self-contained, pulls no anima_lora code):

* ``MOSSi2iLoraLoader`` — MODEL -> MODEL, a MOSS i2i LoRA plus a metadata check.
* ``MOSSi2iTextEncode`` — CLIP + IMAGE + text -> CONDITIONING, the page through
  Qwen3.5's vision tower ahead of the text.
* ``MOSSi2iReference`` — MODEL + VAE + IMAGE -> MODEL + LATENT, the page as a clean
  VAE latent on RoPE's reserved ``r = 1`` axis.

Text (:mod:`.moss_layout`, registered only when ``anime-tools`` is installed):

* ``AnimaCaptionToMossLayout`` — STRING -> STRING, an Anima caption as MOSS's
  layout-JSON prompt.
"""

import logging

from . import nodes

NODE_CLASS_MAPPINGS = dict(nodes.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = dict(nodes.NODE_DISPLAY_NAME_MAPPINGS)

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

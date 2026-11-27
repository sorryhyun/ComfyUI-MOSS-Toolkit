"""MOSS Toolkit: ComfyUI nodes on ComfyUI's own MOSS-Image model.

* :mod:`.loader` — ``MOSSImageLoader``: DiT + Qwen3.5 TE + VAE, each from its own source.
* :mod:`.stream` — ``MOSSStreamFP32``: the fp32 residual stream as a MODEL patch.
* :mod:`.nodes` — ``MOSSLoRA``, ``MOSSi2iReference`` (i2i LoRA + the page as a VAE
  reference latent) and ``MOSSi2iTextEncode`` (vision-tower encode, colorize v2 only,
  hidden from the menu).
* :mod:`.moss_layout` — ``MOSSCaptionHelper``, registered only when ``anime-tools`` is
  installed.
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
        "MOSS Toolkit: Caption Helper skipped (%s). Install it with "
        "pip install -r requirements.txt in this node's folder, or "
        "pip install 'anime-tools[ocr] @ git+https://github.com/sorryhyun/anime_tools@v0.16.0'",
        e,
    )
else:
    NODE_CLASS_MAPPINGS.update(moss_layout.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(moss_layout.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

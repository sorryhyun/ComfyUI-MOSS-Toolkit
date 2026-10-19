"""MOSS i2i ComfyUI custom node.

Image-to-image for MOSS-Image (the ``moss-image`` node's MODEL / CLIP / VAE):

* ``MOSSi2iLoraLoader`` — MODEL -> MODEL, a MOSS i2i LoRA plus a metadata check.
* ``MOSSi2iTextEncode`` — CLIP + IMAGE + text -> CONDITIONING, the page through
  Qwen3.5's vision tower ahead of the text.
* ``MOSSi2iReference`` — MODEL + VAE + IMAGE -> MODEL + LATENT, the page as a clean
  VAE latent on RoPE's reserved ``r = 1`` axis.

Self-contained: pulls no anima_lora code.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

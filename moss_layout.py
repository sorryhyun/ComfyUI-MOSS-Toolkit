"""Anima caption → MOSS-Image layout-JSON prompt, and the caption helper.

A MOSS LoRA trained by the anima_lora trainer with ``caption_format=layout_json``
saw every caption through :func:`anime_tools.captions.layout_json.layout_json`;
this node runs the same function so a prompt typed in Anima format reaches the
text encoder as the identical string. The output goes into any STRING input
(``CLIPTextEncode`` with MOSS-Image's text encoder).

``MOSSCaptionHelper`` adds a task switch: ``layout`` is the conversion above,
``colorize`` the colorize LoRA's training caption — the caption through
``caption_to_layout``, then one text item per line the OCR reads off the B&W page
(anima_lora ``project/moss_lora/src/colorize_cache.py``; ``readable`` / ``dedupe``
/ ``_colorize_layout`` mirror it). The OCR models load for the read and are freed
after it.

Torch-free but for the colorize OCR. Needs ``anime-tools``; ``__init__`` skips this node when it is not
installed. The danbooru tag KB (``danbooru_tags_classified.csv``) decides which
tags are characters / series / meta; it is looked up under ``HOME/models`` and
fetched there on first use when absent.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from anime_tools.captions.correction import (
    TAG_CSV_NAME,
    find_tag_csv,
    load_tag_knowledge_base,
)
from anime_tools.captions.layout_json import (
    caption_to_layout,
    centered_box,
    layout_json,
)
from anime_tools.downloads import by_id

from .nodes import RESOLUTION, target_size, to_pil

logger = logging.getLogger(__name__)


def _home() -> Path:
    """``ANIME_TOOLS_HOME`` / ``ANIMA_HOME`` when set, else the ComfyUI base
    directory, else this node's parent dir (not the CWD: under ComfyUI that is
    wherever the server was launched from)."""
    for key in ("ANIME_TOOLS_HOME", "ANIMA_HOME"):
        v = os.environ.get(key)
        if v:
            return Path(v).expanduser().resolve()
    try:
        import folder_paths  # ComfyUI

        return Path(folder_paths.base_path).resolve()
    except Exception:  # noqa: BLE001 — outside ComfyUI
        return Path(__file__).resolve().parents[1]


HOME = _home()


@lru_cache(maxsize=1)
def _kb():
    path = find_tag_csv(HOME)
    if path is None:
        asset = dataclasses.replace(by_id()["danbooru_tags"], dest=HOME / "models")
        logger.info("MOSS layout: fetching %s into %s", TAG_CSV_NAME, asset.dest)
        asset.fetch(log=logger.info)
        path = asset.dest / TAG_CSV_NAME
    logger.info("MOSS layout: tag KB %s", path)
    return load_tag_knowledge_base(path)


class AnimaCaptionToMossLayout:
    """Map an Anima-format caption onto MOSS-Image's layout-JSON prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Anima-format caption: tag bag, then optional "
                            "'On the left, …' position clauses and a "
                            "'Japanese text reads as \"…\"' clause."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "convert"
    CATEGORY = "MOSS-Image/text"
    DESCRIPTION = (
        "Anima caption -> MOSS-Image layout JSON, byte-identical to what the "
        "anima_lora MOSS trainer encodes with caption_format=layout_json. "
        "Rating / meta, characters, series and @artists get their own summary "
        "keys, the remaining tags become summary.caption, each position clause "
        "a character item, each quoted text line a text item. Use it in front of "
        "the text encoder for a LoRA trained on layout-JSON captions."
    )

    def convert(self, caption: str):
        text = layout_json(caption.strip(), _kb())
        logger.info("AnimaCaptionToMossLayout: %s", text)
        return (text,)


# colorize_cache.py's OCR floors: a box under LINE_MIN_DET is kept when the read
# scores LOW_DET_SCORE (the detector scores horizontal English balloons 0.3-0.55),
# and an all-ASCII line's glyph floor is LATIN_MIN_GLYPH.
LOW_DET_SCORE = 0.9
LATIN_MIN_GLYPH = 12.0
OCR_VRAM_GB = 6


def readable(ln) -> bool:
    from anime_tools.captions.ocr_sidecar import DEFAULT_MIN_GLYPH
    from anime_tools.ocr.reread import LINE_MIN_DET

    latin = all(ch.isascii() for ch in ln.text if ch.isalpha())
    floor = LATIN_MIN_GLYPH if latin else DEFAULT_MIN_GLYPH
    sure = ln.det >= LINE_MIN_DET or ln.score >= LOW_DET_SCORE
    return bool(ln.text) and sure and ln.glyph_px >= floor


def dedupe(lines) -> list[tuple[str, list]]:
    """``[(text, box)]``, ``"`` → ``”``, first of each text kept."""
    seen, keep = set(), []
    for text, box in lines:
        text = text.replace('"', "”").strip()
        if text and text not in seen:
            seen.add(text)
            keep.append((text, box))
    return keep


def read_page(page) -> list[tuple[str, list]]:
    """The readable lines of a PIL page, ``(text, [x0, y0, x1, y1])`` in its pixels,
    in reading order: the AnimeText detector, every box read by the manga VL
    reader, ASCII lines kept."""
    import comfy.model_management as mm
    from anime_tools.ocr import load_ocr
    from anime_tools.ocr.animetext import DEFAULT_CONF
    from anime_tools.ocr.reread import RereadEngine
    from anime_tools.ocr.sfx import SfxReader

    device = mm.get_torch_device()
    mm.free_memory(OCR_VRAM_GB * 1024**3, device)
    reader = SfxReader.load(device=str(device))
    engine = RereadEngine(
        engine=load_ocr(device=str(device)),
        read_boxes=reader.read_boxes_scored,
        resized_dir=Path("/"),
        skip_en=False,
        min_det=DEFAULT_CONF,
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            page.save(path)
            lines = engine.read(path)
    finally:
        del engine, reader
        mm.soft_empty_cache()
    return [(ln.text, list(ln.box)) for ln in lines if readable(ln)]


def _colorize_layout(caption, lines, size) -> str:
    layout = caption_to_layout(caption, _kb())
    items = [
        {"label": "text", "caption": "text", "text": t, "bbox": centered_box(b, size)}
        for t, b in lines
    ]
    if items:
        layout["items"] = layout.get("items", []) + items
        layout["bbox_coord_space"] = "abs_centralized"
        layout["bbox_format"] = "yxyx"
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


class MOSSCaptionHelper:
    """An Anima caption as the prompt a MOSS LoRA trained on, per task."""

    TASKS = ("layout", "colorize")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Anima-format caption. For colorize, the colour tags "
                            "the LoRA trained on: copyright, 'comic' for a manga "
                            "page, hair / eye / clothing colours."
                        ),
                    },
                ),
                "task": (
                    cls.TASKS,
                    {
                        "default": "layout",
                        "tooltip": (
                            "layout: the caption as layout JSON. colorize: the "
                            "same, plus a text item per line the OCR reads off "
                            "the page, boxed on the target canvas."
                        ),
                    },
                ),
                "resolution": RESOLUTION,
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {"tooltip": "The B&W page (colorize). The OCR reads it."},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "convert"
    CATEGORY = "MOSS-Image/text"
    DESCRIPTION = (
        "Anima caption -> the prompt a MOSS LoRA trained on. layout: the layout "
        "JSON of 'Anima Caption → MOSS Layout JSON'. colorize: the colorize LoRA's "
        "caption — that layout plus one text item per OCR-read line of the page, "
        "boxed in abs_centralized yxyx on the canvas the i2i nodes generate at "
        "(same resolution). Feed it to MOSS i2i Text Encode."
    )

    def convert(self, caption, task, resolution, image=None):
        caption = caption.strip()
        if task == "layout":
            text = layout_json(caption, _kb())
        else:
            if image is None:
                raise ValueError("MOSS Caption Helper: colorize needs the page image")
            page = to_pil(image)
            w, h = target_size(page.width, page.height, resolution)
            lines = [
                (t, [b[0] * w / page.width, b[1] * h / page.height,
                     b[2] * w / page.width, b[3] * h / page.height])
                for t, b in dedupe(read_page(page))
            ]
            logger.info("MOSSCaptionHelper: %d OCR lines", len(lines))
            text = _colorize_layout(caption, lines, (w, h))
        logger.info("MOSSCaptionHelper (%s): %s", task, text)
        return (text,)


NODE_CLASS_MAPPINGS = {
    "AnimaCaptionToMossLayout": AnimaCaptionToMossLayout,
    "MOSSCaptionHelper": MOSSCaptionHelper,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaCaptionToMossLayout": "Anima Caption → MOSS Layout JSON",
    "MOSSCaptionHelper": "MOSS Caption Helper",
}

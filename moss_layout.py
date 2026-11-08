"""Anima caption → MOSS-Image layout-JSON prompt, and the caption helper.

A MOSS LoRA trained by the anima_lora trainer with ``caption_format=layout_json``
saw every caption through :func:`anime_tools.captions.layout_json.layout_json`;
this node runs the same function so a prompt typed in Anima format reaches the
text encoder as the identical string. The output goes into any STRING input
(``CLIPTextEncode`` with MOSS-Image's text encoder).

``MOSSCaptionHelper`` adds a task switch: ``default`` returns the caption as written,
``layout`` is the conversion above,
``colorize`` the colorize v4 LoRA's training caption — the caption through
``caption_to_layout``, then one text item per line the OCR reads off the B&W page
(anima_lora ``library/moss/colorize.py`` ``captions = "ocr"``; ``dedupe`` /
``_colorize_layout`` mirror it). ``colorize_v5`` is the v5 caption (``captions =
"control"``), no OCR text: a caption with a comic tag is a manga page, ``comic`` (then
a plain caption's other tags) plus the ``character`` items of a layout-JSON caption plus a box-only text item per line
``anime_tools.ocr.floors.keep_line`` keeps (the page connected); any other caption is
an illustration, its series (the tag KB's copyright tags) and colour tags as layout
JSON (``anime_tools.captions.color_tags``). The OCR models load once, each in a ComfyUI-managed
patcher that stays until something else needs the card, and the VL reader only for a
page the detector boxed. ``sanitize`` and
``decensor`` are those LoRAs' one fixed caption each, plain text; the caption input
is ignored. ``swap_text`` is the caption as written (the page's layout JSON) and
encodes Hangul with the syllable split the swap_text LoRA trained on
(:mod:`.token_split`); the split is in the tokenizer, so only the ``conditioning``
output (with ``clip`` connected) carries it.

With ``clip`` connected, the ``conditioning`` output is the prompt encoded as a plain
CLIP Text Encode would, so the helper can feed the sampler directly.

``put_endthink`` returns the whole chat turn with the ``sure`` assistant tail
(``<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n`` +
``\n</think>\n\nSure,``), as anima_lora's ``prompt_tail = sure`` encodes it. ComfyUI's
MOSS tokenizer skips its own template for text starting with ``<|im_start|>``, so the
tail lands after the generation prompt, not inside the user turn.

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
from .token_split import has_hangul, split_clip

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


def is_layout_json(raw: str) -> bool:
    """``raw`` is already a layout-JSON prompt (an object with ``summary``). A copy of
    anima_lora ``library/moss/cache.py::is_layout_json`` — change both together."""
    if not raw.startswith("{"):
        return False
    try:
        obj = json.loads(raw)
    except ValueError:
        return False
    return isinstance(obj, dict) and "summary" in obj


def to_layout_json(caption: str) -> str:
    """``caption`` as the trainer's ``caption_format=layout_json`` encodes it: a
    caption that already is layout JSON (swap_text's staged captions) as written,
    else :func:`layout_json` (converting it again would nest the whole object in
    ``summary.caption``)."""
    return caption if is_layout_json(caption) else layout_json(caption, _kb())


class AnimaCaptionToMossLayout:
    """Map an Anima-format caption onto MOSS-Image's layout-JSON prompt.

    Deprecated: ``MOSSCaptionHelper`` ``task = layout`` is the same conversion.
    Kept registered (hidden from the node menu) so saved workflows still load."""

    DEPRECATED = True

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
        "Deprecated: use MOSS Caption Helper, task layout. "
        "Anima caption -> MOSS-Image layout JSON, byte-identical to what the "
        "anima_lora MOSS trainer encodes with caption_format=layout_json. "
        "Rating / meta, characters, series and @artists get their own summary "
        "keys, the remaining tags become summary.caption, each position clause "
        "a character item, each quoted text line a text item. Use it in front of "
        "the text encoder for a LoRA trained on layout-JSON captions."
    )

    def convert(self, caption: str):
        text = to_layout_json(caption.strip())
        logger.info("AnimaCaptionToMossLayout: %s", text)
        return (text,)


# The OCR floors are anime_tools.ocr.floors (readable, keep_line), the ones the
# colorize and swap_text LoRAs trained on.
OCR_VRAM_GB = 6
# Per task, the anima_lora cache.toml ``fixed_caption`` its LoRA trained on
# (caption_format fixed: every pair encoded this string verbatim) — change both
# together.
FIXED_CAPTIONS = {
    "sanitize": "no sfx, no dialogue, no watermark",
    "decensor": "no bar censor, no mosaic censor, uncensored",
}


def dedupe(lines) -> list[tuple[str, list]]:
    """``[(text, box)]``, ``"`` → ``”``, first of each text kept."""
    seen, keep = set(), []
    for text, box in lines:
        text = text.replace('"', "”").strip()
        if text and text not in seen:
            seen.add(text)
            keep.append((text, box))
    return keep


def _patcher(model, device):
    """``model`` under a patcher of its own: ComfyUI moves it on and off the card and
    evicts it like any other model."""
    import comfy.model_management as mm
    import comfy.model_patcher
    import torch

    class OcrModel(torch.nn.ModuleList):
        pass

    return comfy.model_patcher.ModelPatcher(
        OcrModel([model]), load_device=device, offload_device=mm.unet_offload_device()
    )


@lru_cache(maxsize=1)
def _reader(device):
    """``(patcher, SfxReader)``, loaded on the CPU once per process."""
    from anime_tools.ocr.sfx import SfxReader

    reader = SfxReader.load(device="cpu")
    reader.device = str(device)
    return _patcher(reader.model, device), reader


@lru_cache(maxsize=1)
def _detector(device):
    """``(patcher, RereadEngine)`` over the AnimeText detector, loaded on the CPU once
    per process. The engine calls ``read_boxes`` only for a page with boxes, so a page
    without any never touches the VL reader."""
    import comfy.model_management as mm
    from anime_tools.ocr import load_ocr
    from anime_tools.ocr.animetext import DEFAULT_CONF
    from anime_tools.ocr.reread import RereadEngine

    def read_boxes(bgr, boxes):
        patcher, reader = _reader(device)
        mm.load_models_gpu(
            [patcher],
            memory_required=OCR_VRAM_GB * 1024**3 - patcher.model_size(),
            force_full_load=True,
        )
        return reader.read_boxes_scored(bgr, boxes)

    detected = load_ocr(device="cpu")
    detected.detector.device = str(device)
    engine = RereadEngine(
        engine=detected,
        read_boxes=read_boxes,
        resized_dir=Path("/"),
        skip_en=False,
        min_det=DEFAULT_CONF,
    )
    return _patcher(detected.detector.model, device), engine


def read_page(page, keep="readable") -> list[tuple[str, list]]:
    """The lines of a PIL page ``anime_tools.ocr.floors.<keep>`` keeps,
    ``(text, [x0, y0, x1, y1])`` in its pixels, in reading order: the AnimeText
    detector, every box read by the manga VL reader."""
    import comfy.model_management as mm
    from anime_tools.ocr import floors

    patcher, engine = _detector(mm.get_torch_device())
    mm.load_models_gpu([patcher], force_full_load=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            page.save(path)
            lines = engine.read(path)
    finally:
        mm.soft_empty_cache()
    keeps = getattr(floors, keep)
    return [(ln.text, list(ln.box)) for ln in lines if keeps(ln)]


def _scaled(box, src, dst) -> list[float]:
    """A ``[x0, y0, x1, y1]`` box on a ``src = (W, H)`` page on the ``dst`` canvas."""
    (sw, sh), (dw, dh) = src, dst
    return [box[0] * dw / sw, box[1] * dh / sh, box[2] * dw / sw, box[3] * dh / sh]


def _colorize_layout(caption, lines, size) -> str:
    layout = (
        json.loads(caption)
        if is_layout_json(caption)
        else caption_to_layout(caption, _kb())
    )
    items = [
        {"label": "text", "caption": "text", "text": t, "bbox": centered_box(b, size)}
        for t, b in lines
    ]
    if items:
        layout["items"] = layout.get("items", []) + items
        layout["bbox_coord_space"] = "abs_centralized"
        layout["bbox_format"] = "yxyx"
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def _is_comic(caption: str) -> bool:
    from anime_tools.captions.color_tags import is_comic_tag

    if is_layout_json(caption):
        summary = json.loads(caption).get("summary", {}).get("caption", "")
        return any(is_comic_tag(t) for t in summary.split(","))
    return any(is_comic_tag(t) for t in caption.split(","))


def _illustration_caption(caption: str) -> str:
    """The caption's series + colour tags as layout JSON, as the colorize v5 cache
    built an illustration's draw (series: the tag KB's copyright tags)."""
    from anime_tools.captions.color_tags import (
        _NON_SERIES_COPYRIGHT,
        filter_to_colors_and_protected,
    )
    from anime_tools.captions.correction import tag_kind

    if is_layout_json(caption):
        return caption
    kb = _kb()
    tags = [t.strip() for t in caption.split(",") if t.strip()]
    series = frozenset(
        t.lower()
        for t in tags
        if tag_kind(t, kb) == "copyright" and t.lower() not in _NON_SERIES_COPYRIGHT
    )
    kept = filter_to_colors_and_protected(caption, series, keep_copyright=True)
    dropped = [
        t for t in tags if t.lower() not in {k.strip().lower() for k in kept.split(",")}
    ]
    if dropped:
        logger.info(
            "MOSS Caption Helper colorize_v5: drops %s (not series or colour)", dropped
        )
    return layout_json(kept, kb) if kept else ""


def _manga_v5_layout(caption: str, boxes: list, size) -> str:
    """``comic``, the caption's ``character`` items (a layout-JSON caption), then a
    box-only ``text`` item per OCR line: the colorize v5 cache's manga draw. A plain
    caption's other tags follow ``comic`` in ``summary.caption`` (comic tags dropped)."""
    from anime_tools.captions.color_tags import is_comic_tag

    items = []
    summary = "comic"
    if is_layout_json(caption):
        items = [
            it
            for it in json.loads(caption).get("items", [])
            if it.get("label") != "text"
        ]
    else:
        rest = [
            t.strip() for t in caption.split(",") if t.strip() and not is_comic_tag(t)
        ]
        summary = ", ".join(["comic", *rest])
    items += [{"label": "text", "bbox": centered_box(b, size)} for b in boxes]
    layout: dict = {"summary": {"caption": summary}}
    if items:
        layout.update(
            items=items, bbox_coord_space="abs_centralized", bbox_format="yxyx"
        )
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


TURN = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n"
ENDTHINK = "\n</think>\n\nSure,"


def with_endthink(prompt: str) -> str:
    if prompt.startswith("<|im_start|>"):
        logger.warning(
            "MOSS Caption Helper: the prompt is already a chat turn; put_endthink leaves it as typed"
        )
        return prompt
    return TURN.format(prompt.strip()) + ENDTHINK


class MOSSCaptionHelper:
    """An Anima caption as the prompt a MOSS LoRA trained on, per task."""

    TASKS = (
        "default",
        "layout",
        "colorize",
        "colorize_v5",
        *FIXED_CAPTIONS,
        "swap_text",
    )

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
                            "Anima-format caption. colorize: the colour tags "
                            "the LoRA trained on: copyright, 'comic' for a manga "
                            "page, hair / eye / clothing colours. colorize_v5: "
                            "'comic' for a manga page (other tags follow comic "
                            "in the summary caption; boxed character colours as "
                            "layout-JSON character items), series + colour tags "
                            "for an illustration. layout: a \\n inside a quoted "
                            "text line is a line break in that text item."
                        ),
                    },
                ),
                "task": (
                    cls.TASKS,
                    {
                        "default": "default",
                        "tooltip": (
                            "default: the caption as written. "
                            "layout: the caption as layout JSON (a caption that "
                            "already is layout JSON passes through). colorize: the "
                            "same, plus a text item per line the OCR reads off "
                            "the page, boxed on the target canvas (v4 LoRA). "
                            "colorize_v5: comic (+ the caption's other tags) + a "
                            "box-only text item per OCR line (no text), or an illustration's series + colour "
                            "tags (v5 LoRA). sanitize / "
                            "decensor: that LoRA's fixed caption, plain text; "
                            "the caption is ignored. swap_text: the caption as "
                            "written (the page's layout JSON), Korean encoded "
                            "syllable by syllable as the swap_text LoRA trained; "
                            "connect clip and use the conditioning output."
                        ),
                    },
                ),
                "resolution": RESOLUTION,
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "The B&W page (colorize, colorize_v5 on a manga page). "
                            "The OCR reads it."
                        )
                    },
                ),
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "MOSS-Image text encoder. Connected, the conditioning "
                            "output is the prompt encoded (swap_text: with the "
                            "Hangul syllable split)."
                        )
                    },
                ),
                "put_endthink": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Output the whole chat turn ending in "
                            "'\\n</think>\\n\\nSure,' after the generation prompt "
                            "(anima_lora prompt_tail = sure: the cache default, "
                            "sharper hands and backgrounds on Base). For a plain "
                            "CLIP Text Encode, and for a LoRA cached with "
                            "prompt_tail sure; leave off for MOSS i2i Text "
                            "Encode and LoRAs cached with prompt_tail off."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "CONDITIONING")
    RETURN_NAMES = ("prompt", "conditioning")
    FUNCTION = "convert"
    CATEGORY = "MOSS-Image/text"
    DESCRIPTION = (
        "Anima caption -> the prompt a MOSS LoRA trained on. default: the caption "
        "as written. layout: the layout "
        "JSON the anima_lora MOSS trainer encodes with caption_format=layout_json. colorize: the colorize LoRA's "
        "caption — that layout plus one text item per OCR-read line of the page, "
        "boxed in abs_centralized yxyx on the canvas the i2i nodes generate at "
        "(same resolution). Feed it to MOSS i2i Text Encode. colorize_v5: the v5 "
        "LoRA's caption, comic + box-only text items for a manga page, series + "
        "colour tags for an illustration. sanitize / decensor: "
        "that LoRA's fixed caption, verbatim, for a plain CLIP Text Encode. "
        "swap_text: the page's layout JSON with Korean split by syllable at "
        "encode time; take the conditioning output. With clip connected, "
        "conditioning is the prompt encoded."
    )

    def convert(
        self, caption, task, resolution, image=None, put_endthink=False, clip=None
    ):
        text = self._prompt(caption, task, resolution, image)
        if put_endthink:
            if task == "swap_text":
                logger.warning(
                    "MOSS Caption Helper: swap_text was cached with prompt_tail off; "
                    "put_endthink adds a tail it never saw"
                )
            text = with_endthink(text)
        logger.info("MOSSCaptionHelper (%s): %s", task, text)
        split = task == "swap_text" and has_hangul(text)
        if clip is None:
            if split:
                logger.warning(
                    "MOSS Caption Helper: swap_text without clip; the prompt string "
                    "encodes Korean without the syllable split"
                )
            return (text, None)
        encoder = split_clip(clip) if split else clip
        return (text, encoder.encode_from_tokens_scheduled(encoder.tokenize(text)))

    def _prompt(self, caption, task, resolution, image):
        if task == "default":
            return caption
        if task == "swap_text":
            return caption.strip()
        caption = caption.strip()
        if task == "layout":
            text = to_layout_json(caption)
        elif task in FIXED_CAPTIONS:
            text = FIXED_CAPTIONS[task]
            if caption and caption != text:
                logger.warning(
                    "MOSS Caption Helper: %s ignores the caption %r", task, caption
                )
        elif task == "colorize_v5":
            if not _is_comic(caption):
                return _illustration_caption(caption)
            boxes = []
            if image is not None:
                page = to_pil(image)
                w, h = target_size(page.width, page.height, resolution)
                boxes = [
                    _scaled(b, page.size, (w, h))
                    for _, b in read_page(page, keep="keep_line")
                ]
                logger.info("MOSSCaptionHelper: %d text boxes", len(boxes))
            else:
                w = h = 0
            text = _manga_v5_layout(caption, boxes, (w, h))
        else:
            if image is None:
                raise ValueError("MOSS Caption Helper: colorize needs the page image")
            page = to_pil(image)
            w, h = target_size(page.width, page.height, resolution)
            lines = [
                (t, _scaled(b, page.size, (w, h))) for t, b in dedupe(read_page(page))
            ]
            logger.info("MOSSCaptionHelper: %d OCR lines", len(lines))
            text = _colorize_layout(caption, lines, (w, h))
        return text


NODE_CLASS_MAPPINGS = {
    "AnimaCaptionToMossLayout": AnimaCaptionToMossLayout,
    "MOSSCaptionHelper": MOSSCaptionHelper,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaCaptionToMossLayout": "Anima Caption → MOSS Layout JSON",
    "MOSSCaptionHelper": "MOSS Caption Helper",
}

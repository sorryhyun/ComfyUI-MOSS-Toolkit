"""Hangul syllable split for the MOSS-Image Qwen3.5 tokenizer.

A copy of anima_lora ``library/moss/token_split.py`` (``syllable``): every Hangul
syllable is tokenized alone, and a space before one is its own token. Text without
Hangul tokenizes as it does by default. A LoRA trained with ``token_split =
syllable`` (stamped in its metadata) learned its Korean glyphs on these tokens, and
the default tokenizer's multi-syllable tokens (``인가요``) reach the DiT as vectors it
never saw. Change both copies together.

The split lives in the tokenizer, so a prompt string cannot carry it:
:func:`split_clip` returns a CLIP whose tokenizer splits, for one encode.
"""

from __future__ import annotations

import copy
import re

_SYLLABLE = "가-힣"
_HANGUL = re.compile(f"[{_SYLLABLE}]")
_PIECES = re.compile(f"([{_SYLLABLE}]| (?=[{_SYLLABLE}]))")


def has_hangul(text: str) -> bool:
    return bool(_HANGUL.search(text))


def syllable_ids(tok, text: str) -> list[int]:
    pieces = [p for p in _PIECES.split(text) if p]
    encoded = tok(pieces, add_special_tokens=False)["input_ids"]
    return [i for ids in encoded for i in ids]


class SyllableTokenizer:
    """The HF tokenizer an ``SDTokenizer`` calls as ``tokenizer(word)``, split."""

    def __init__(self, tok):
        self._tok = tok

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __call__(self, text, **kwargs):
        if kwargs or not isinstance(text, str):
            raise ValueError("MOSS token split: only tokenizer(text) is split")
        return {"input_ids": syllable_ids(self._tok, text)}


def split_clip(clip):
    """A clone of ``clip`` whose tokenizer applies the syllable split; ``clip`` and
    its tokenizer stay as they are."""
    outer = copy.copy(clip.tokenizer)
    name = getattr(outer, "clip", None)
    inner = getattr(outer, name, None) if name else None
    if inner is None or not hasattr(inner, "tokenizer"):
        raise ValueError(
            "MOSS token split: the CLIP is not a MOSS-Image text encoder "
            f"(tokenizer {type(clip.tokenizer).__name__})"
        )
    inner = copy.copy(inner)
    inner.tokenizer = SyllableTokenizer(inner.tokenizer)
    setattr(outer, name, inner)
    n = clip.clone()
    n.tokenizer = outer
    return n

# MOSS Caption Helper

Turns an Anima-format caption (tag bag, optional `On the left, …` position clauses and a
`… text reads as "…"` clause) into the prompt a MOSS LoRA trained on. Connect `latent` to
the Reference's LATENT (the canvas boxes are placed on) and `clip` to the MOSS-Image text
encoder, then wire `conditioning` to the KSampler's positive.

## Tasks

- **`default`**: the caption as written.
- **`layout`**: `anime_tools.captions.layout_json`, the string the anima_lora MOSS trainer
  encodes with `caption_format=layout_json`. Rating / meta, characters, series and
  `@artists` get their own `summary` keys, the other tags become `summary.caption`, each
  position clause a character item, each quoted text line a text item (`\n` inside the
  quotes is a line break). Meta or unknown tags ahead of the first count tag (`1girl`, …) go
  to `summary.meta`. A caption that already is layout JSON (an object with `summary`)
  passes through.
- **`colorize`** (v4 LoRA): `layout` plus one
  `{"label":"text","caption":"text","text":…,"bbox":…}` item per line the OCR reads off
  the page, boxes in `abs_centralized` `yxyx` on the latent's canvas. Type the colour tags
  the LoRA trained on: copyright, `comic` for a manga page, hair / eye / clothing colours.
- **`colorize_v5`**: no OCR text. A caption with a comic tag is a manga page:
  `summary.caption` is `comic` followed by a plain caption's other tags, then a layout-JSON
  caption's `character` items, then one box-only `{"label":"text","bbox":…}` item per line
  `anime_tools.ocr.floors.keep_line` keeps on the page. `comic` alone with no page also
  works. Any other caption is an illustration: its series (the tag KB's copyright tags) and
  colour tags as layout JSON, the rest dropped.
- **`sanitize`** / **`decensor`**: that LoRA's one fixed caption verbatim
  (`no sfx, no dialogue, no watermark` / `no bar censor, no mosaic censor, uncensored`);
  the caption input is ignored.
- **`swap_text`**: the caption as written (the page's layout JSON, one text item per line
  to letter, `bbox` in `abs_centralized` `yxyx` pixels of the canvas; a balloon left out
  stays as it is). With `clip` connected, Hangul encodes one syllable per token and a space
  before a syllable as its own token, as anima_lora's `token_split = syllable`. The split is
  in the tokenizer, so only `conditioning` carries it, not the `prompt` string. For English,
  or a swap_text LoRA without the split, `layout` into a plain CLIP Text Encode is the same.

## put_endthink

Outputs the whole chat turn,
`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n` +
`\n</think>\n\nSure,`, as anima_lora's `prompt_tail = sure` encodes it. The MOSS tokenizer
skips its own template for a prompt starting with `<|im_start|>`, so the tail lands after
the generation prompt. Use it for LoRAs cached on `sure`; leave it off for ones cached on
`off` (the i2i LoRAs).

## Notes

- Write parentheses plain (`@channel (caststation)`). Inside layout JSON the backslash of
  `\(` is stored doubled and reaches the text encoder; the helper warns on it.
- ComfyUI's `(tag:1.2)` weights do not apply: the MOSS-Image text encoder tokenizes with
  weights off.
- OCR: anime-tools' AnimeText detector, with the manga VL reader on every box. Both load
  in ComfyUI-managed patchers that stay until something else needs the card; the VL
  reader loads only for a page the detector boxed. Weights download on first use.
- The danbooru tag KB (`danbooru_tags_classified.csv`) decides characters / series / meta.
  It is looked up under `$ANIME_TOOLS_HOME` / `$ANIMA_HOME` / the ComfyUI folder
  (`models/`) and downloaded there on first use.

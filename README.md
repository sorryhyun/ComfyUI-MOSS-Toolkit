# MOSS Toolkit for ComfyUI

Nodes for [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image), on top of the
`moss-image` node's MODEL / CLIP / VAE (formerly **MOSS i2i**, `comfyui-moss-i2i`):

- **Loader** — DiT, Qwen3.5 text encoder and VAE in one node, each from its own
  source; the TE can be the INT8 or fp8 file.
- **i2i** — image-to-image. First task: colorizing a B&W manga page with the colorize
  LoRA trained in anima_lora.
- **Layout JSON / Caption Helper** — an Anima-format caption as MOSS's layout-JSON
  prompt, for LoRAs trained on `caption_format=layout_json` captions; for colorize,
  with the page's OCR lines added.

## Loader

| Node | In → out | Notes |
|---|---|---|
| **MOSS-Image Loader (DiT + TE + VAE)** | → MODEL, CLIP, VAE | Picks each part on its own, so SFT's DiT pairs with Base's TE and VAE |

The `moss_image` model folder is the one the `moss-image` node registers
(`models/moss_image/`, or `moss_image:` in `extra_model_paths.yaml`).

| Input | Options |
|---|---|
| `dit` | `moss_image/<name>/transformer/` dirs, then `diffusion_models` files |
| `text_encoder` | `moss_image/<name>/text_encoder/` dirs (bf16), then `*.safetensors` beside a `config.json` in the other `moss_image/<name>/` dirs, then `text_encoders` files |
| `vae` | `moss_image/<name>/vae/` dirs, then `vae` files |

A quantized TE is one 9.2 GB file instead of the 16.8 GB bf16 shards. Download
[`sorryhyun/MOSS-Image-Qwen3.5-9B-int8`](https://huggingface.co/sorryhyun/MOSS-Image-Qwen3.5-9B-int8)
into `models/moss_image/int8/` (it lists as `int8/qwen35_9b_int8_convrot.safetensors`, the
default when present) or
[`sorryhyun/MOSS-Image-Qwen3.5-9B-fp8`](https://huggingface.co/sorryhyun/MOSS-Image-Qwen3.5-9B-fp8)
into `models/moss_image/fp8/` (`fp8/qwen35_9b_fp8_e4m3fn.safetensors`), or put the file in
`models/text_encoders/`.

INT8 runs W8A8 by default (`int8_matmul`, advanced): ComfyUI otherwise dequantizes a text
encoder's weights for a bf16 matmul. RTX 5070 Ti, 20 prompts, embedding error vs the bf16 TE:

| TE | Encode, tokens/s | Rel. L2 vs bf16 |
|---|---|---|
| INT8, `int8_matmul` on (W8A8) | 1321 | 0.047 |
| INT8, `int8_matmul` off | 873 | 0.036 |
| fp8 | 883 | 0.045 |

W8A8 is what anima_lora encodes MOSS captions with by default. A
single-file TE is tokenized with `moss_image/<name>/tokenizer/`: the DiT's own, else
`MOSS-Image-Base`'s, else ComfyUI's bundled Qwen3.5 tokenizer. The vision tower is in the
file, so the i2i text encode works with it. The loading is the `moss-image` node's.

## i2i

MOSS-Image has no image conditioning. This node adds two paths:

- **Vision tower.** The page goes through Qwen3.5's vision tower ahead of the text in
  the same user turn. This brings back the page's content, but not its line placement
  (one token per 32×32 px).
- **VAE reference.** The page also goes in as a clean VAE latent: an extra token block
  `[image | ref | text]` on RoPE's reserved `r = 1` axis. Its `(h, w)` positions are
  stretched onto the target grid. A LoRA trained with these refs keeps the page's line
  work. A LoRA stamped with `ref_timestep` (colorize v2 / v3: 0) gets the ref rows
  modulated at that timestep in every block; without it (v1) they share the step's.
  A LoRA stamped with `ref_isolate` (colorize v3) has the ref queries attend only to
  the ref keys; image and text still read the ref.

Colorize v3 uses the VAE reference alone: the page at full size, isolated, and a
text-only prompt (a plain CLIP Text Encode). v2 uses both paths, the ref at half size.

| Colorize | Ref | Prompt encode |
|---|---|---|
| v3 (`ref_isolate` in the metadata) | full size, isolated | CLIP Text Encode, no page |
| v2 | half size (`ref_scale 2`), joint | MOSS i2i Text Encode (page + text) |

### Nodes

| Node | In → out | Notes |
|---|---|---|
| **MOSS i2i Text Encode (page + text)** | CLIP, IMAGE, text → CONDITIONING | `<|vision_start|>[page]<|vision_end|>{text}`. Empty text = page only. The page is fit to the target size and seen at that pixel count (~1000 tokens at 1 MP) |
| **MOSS i2i Reference** | MODEL, VAE, IMAGE → MODEL, LATENT | Loads the i2i LoRA (`lora_name`, `strength`; plain `lora_unet_layers_*`) and reads `reference_latents`, `ref_scale`, `ref_timestep` and `ref_isolate` from its metadata. `task = auto` sets the ref size from the recipe the LoRA trained with: full size for an isolated ref (colorize v3, sanitize, swap_text), half for colorize v2. Workflows saved with `task = colorize` run as `auto`; ones saved with `task = custom` still load and use `ref_scale` (ref at `1/ref_scale` per side, 2 = 1/4 area). The ref goes into every forward and every CFG branch. `ref_cache`: see below. Also outputs an empty latent at the target size |

### Ref K/V cache

An isolated ref with its own timestep (colorize v3) depends only on the page: not on
the step, the prompt or the CFG branch. With `ref_cache = gpu` (default) the first
forward runs it through the 42 blocks once and keeps each block's ref keys and
values (~2.1 GB bf16 at 1 MP). After that, each forward runs only the image and text
rows, so the full-size ref costs about as much as extra keys. The cache lasts until
the model's weight patches or the target size change, so a reseed of the same page
reuses it. `cpu` keeps it in pinned host memory and copies one block's K/V in at a
time (for low VRAM); `off` runs the ref rows in every forward. v1 / v2 LoRAs
(joint ref) always run the ref rows.

### Wiring

Colorize v3 (`example_workflows/moss-i2i-colorize-v3.json`):

```
MOSS-Image Loader ─ MODEL ─ MOSS i2i Reference (LoRA, task auto) ────── MODEL ─┐
                  ─ CLIP ─┬ CLIP Text Encode (Caption Helper's prompt) ─ positive ─┤
                          └ CLIP Text Encode ("") ─────────────────── negative ─┤ KSampler /
                  ─ VAE ──── MOSS i2i Reference ───────────────────── LATENT ───┘ MOSS Spectrum KSampler
Load Image (page) → Reference.image, Caption Helper.image (task colorize, colour tags)
```

Colorize v2 (`example_workflows/moss-i2i-colorize.json`):

```
MOSS-Image Loader ─ MODEL ─ MOSS i2i Reference (LoRA) ──────────────── MODEL ─┐
                  ─ CLIP ─┬ MOSS i2i Text Encode (page, colour tags) ─ positive ─┤
                          └ CLIP Text Encode ("") ─────────────────── negative ─┤ KSampler /
                  ─ VAE ──── MOSS i2i Reference ───────────────────── LATENT ───┘ MOSS Spectrum KSampler
Load Image (page) → Text Encode.image, Reference.image
```

Sanitize (text / bubble removal, `example_workflows/moss-i2i-sanitize.json`): the
colorize v3 wiring (the LoRA's `ref_isolate` comes from its metadata), with the
positive from Caption Helper `task sanitize`, the LoRA's fixed caption (the helper
needs no image).

Swap text (JA lettering → translated): the same wiring, with the positive a layout-JSON
prompt that lists one text item per line to letter, its `bbox` in `abs_centralized`
`yxyx` pixels of the page at the Reference's `resolution` (a balloon left out stays
Japanese). Caption Helper `task layout` passes a caption that already is layout JSON
through unchanged, so the prompt can be typed there or straight into `CLIP Text Encode`.

Use the same `resolution` on Text Encode, Caption Helper and Reference. The negative is a plain text
encode of `""`, which is what the LoRA was evaluated with. For CFG 3 with the uncond
block-22 skip, use the MOSS Spectrum KSampler. anima_lora's colorize eval ran 30 steps,
CFG 3, ER-SDE, Spectrum at its defaults, with the fp8 TE. That is the v3 example
workflow, except that ComfyUI's σ grid is not snapped to the training grid and its
ER-SDE noise differs, so a seed does not reproduce the eval's image.

### Compatibility

- **Spectrum (MOSS).** Cached steps never call the DiT, so they cost no ref rows. On
  full steps the capture hook sees the image rows only.
- **DAVE.** Its hook reads the leading image rows. The ref rows come after them (or,
  with the ref cache, are not in the block's rows at all). The block functions this
  node runs in place of `layer(...)` call the block's forward hooks themselves.
- **SLG / uncond skip.** The `patches_replace` dispatch is kept, so it still works.
  With `ref_isolate`, a skipped block still runs the ref rows (they depend only on
  themselves there), so the uncond branch reads the same ref as the cond, as in
  training. With the ref cache this needs no extra work: the cache holds every
  block's pass.

The patch is an `APPLY_MODEL` wrapper. For the length of each `apply_model` call, the
live `diffusion_model` runs a ref-aware copy of the `moss-image` forward.

## Layout JSON

| Node | In → out | Notes |
|---|---|---|
| **MOSS Caption Helper** | STRING, task (+ IMAGE) → STRING | `task = layout`: the node below; a caption that already is layout JSON passes through as typed. `task = colorize`: that layout plus one `{"label":"text","caption":"text","text":…,"bbox":…}` item per line the OCR reads off the page, the caption the colorize LoRA trained on. Boxes are `abs_centralized` `yxyx` on the canvas the i2i nodes generate at, so use the same `resolution`. `task = sanitize`: `no sfx, no dialogue, no watermark` verbatim, the one caption every sanitize pair trained on. `task = decensor`: `no bar censor, no mosaic censor, uncensored`, likewise for the decensor LoRA. Both ignore the caption input |
| **Anima Caption → MOSS Layout JSON** | STRING → STRING | `anime_tools.captions.layout_json`, byte-identical to what the anima_lora MOSS trainer encodes with `caption_format=layout_json`. Rating / meta, characters, series and `@artists` get their own `summary` keys, the other tags become `summary.caption`, each position clause a character item, each quoted text line a text item. A caption that already is layout JSON (an object with `summary`) passes through as typed, as in the trainer |

Feed the layout output into `CLIP Text Encode` with MOSS-Image's text encoder, the
colorize output into **MOSS i2i Text Encode** as its `text` (v2) or a plain `CLIP Text
Encode` (v3), the sanitize and decensor outputs into a plain `CLIP Text Encode`. For colorize, type the
colour tags the LoRA trained on (copyright, `comic` for a manga page, hair / eye /
clothing colours). The OCR is anime-tools' AnimeText detector with the manga VL
reader on every box; both load for the read (~6 GB VRAM freed first) and unload
after it, and their weights download on first use. Both nodes need
`anime-tools`; without it the toolkit loads the i2i nodes only and logs a warning:

```
pip install 'anime-tools @ git+https://github.com/sorryhyun/anime_tools'
```

The danbooru tag KB (`danbooru_tags_classified.csv`) is looked up under
`$ANIME_TOOLS_HOME` / `$ANIMA_HOME` / the ComfyUI folder (`models/`) and downloaded
there on first use. The node used to ship in the Anima Tagger pack; the node id
(`AnimaCaptionToMossLayout`) is unchanged, so old workflows load as-is.

## License

Apache-2.0. `reference.py` derives from the `moss-image` node's `MOSSImageTransformer`
(OpenMOSS, Apache-2.0).

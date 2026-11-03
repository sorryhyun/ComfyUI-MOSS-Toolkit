# MOSS Toolkit for ComfyUI

Nodes for [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image), on ComfyUI's own
MOSS-Image support (formerly **MOSS i2i**, `comfyui-moss-i2i`). The `moss-image` node is
not required:

- **Loader** — DiT, Qwen3.5 text encoder and VAE in one node, each from its own
  source; the DiT and the TE can be INT8 files.
- **fp32 Stream** — a MODEL patch that runs the residual stream in fp32 around bf16 or
  INT8 block GEMMs.
- **i2i** — image-to-image. First task: colorizing a B&W manga page with the colorize
  LoRA trained in anima_lora.
- **Caption Helper** — an Anima-format caption as MOSS's layout-JSON
  prompt, for LoRAs trained on `caption_format=layout_json` captions; for colorize,
  with the page's OCR lines added.

## Loader

| Node | In → out | Notes |
|---|---|---|
| **MOSS-Image Loader (DiT + TE + VAE)** | → MODEL, CLIP, VAE | Picks each part on its own, so SFT's DiT pairs with Base's TE and VAE |

The Loader registers the `moss_image` model folder (`models/moss_image/`, or
`moss_image:` in `extra_model_paths.yaml`).

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

W8A8 is what anima_lora encodes MOSS captions with by default. Every TE is tokenized with
ComfyUI's bundled Qwen3.5 tokenizer. The vision tower is in the file, so the i2i text
encode works with it.

Everything loads through ComfyUI's own loaders and `MOSSImage` model, the same as `Load
Diffusion Model`: vendor shards (separate q/k/v), Comfy-Org's `moss_image_*` files with
fused `proj.qkv`, and INT8 files. An INT8 file with separate q/k/v is fused at load, one GEMM
instead of three, with the same output. That model takes its timestep as σ, not 1000·σ, so a
saved workflow's `ModelSamplingSD3` samples noise there: remove it (the shift defaults
to 5) or use `ModelSamplingAuraFlow`. **KSampler (Spectrum for MOSS)** sets this itself
(Spectrum pack 2.13.1 or later).

`stream` (advanced, `bf16` default) set to `fp32` runs the DiT's residual stream,
modulation, norms, embedders and head in fp32 around the block GEMMs, as anima_lora's
`generate --stream fp32`. The prediction error against an fp32 forward drops 2.4–3.5× for
+1.5% time and +0.6 GB. Renders change in detail, and neither precision is consistently
better. It is the MODEL patch below.

## LoRA on an INT8 DiT

| Node | In → out | Notes |
|---|---|---|
| **MOSS-Image LoRA** | MODEL → MODEL | Either q/k/v layout; `bypass` on by default; chains |

ComfyUI applies a LoRA to an INT8 weight by requantizing the whole linear, which loses a GPTQ
file's weights. `bypass` adds the LoRA in the forward and leaves them alone, including on the
MOSS-Image Loader's fused `proj.qkv`, where ComfyUI's own `Load LoRA (Bypass)` drops the
q / k / v adapters. **MOSS i2i Reference** has the same `bypass` (advanced, off by default).

On a ConvRot INT8 linear, bypass runs the LoRA's `down` matrices as one small INT8 GEMM and only
`up` in float, so `ffn_down` keeps its fused activation. One forward on real latents (about 4100
image tokens), RTX 5070 Ti, FlashAttention, fp32 stream, a rank 32 LoRA on all 252 block linears,
at σ 0.2 / 0.5 / 0.9:

| | Rel. L2 vs bf16 with the same LoRA | Forward |
|---|---|---|
| no LoRA | – | 282 ms |
| patch | 1.15 / 1.74 / 3.17% | 282 ms |
| bypass | 0.85 / 1.31 / 2.18% | 359 ms |

bf16 linears and LoKr / LoHa keep ComfyUI's float bypass hook. On an INT8 DiT that path takes the
eager `ffn_down` path as well (486 ms with the LoRA above). Any change to a forward re-rolls its
bf16 rounding, so two bypass setups that should match can differ by about 1.5% in the prediction.

## fp32 stream patch

| Node | In → out | Notes |
|---|---|---|
| **MOSS-Image fp32 Stream** | MODEL → MODEL | `stream = fp32` for a MODEL from `Load Diffusion Model`, bf16 or INT8 |

No weight changes and nothing reloads, so dynamic VRAM stays on and the node can sit
anywhere in the MODEL chain: the sampled latent stays fp32 going in, and only the blocks'
GEMM inputs and the attention stay in the model's compute dtype (INT8 GEMMs stay INT8).
An isolated ref's keys and values stay on a stream in the compute dtype, a joint ref runs
in fp32. The node raises on a `moss-image` node DiT.

Prediction error of MOSS-Image Base against an fp32 forward (relative L2, one latent,
RTX 5070 Ti). 30 steps at 1024² with CFG took 14 s against 13 s:

| σ | bf16 | bf16 + stream | INT8 | INT8 + stream |
|---|---|---|---|---|
| 0.9 | 0.0204 | 0.0042 | 0.0265 | 0.0157 |
| 0.5 | 0.0182 | 0.0073 | 0.0293 | 0.0247 |
| 0.2 | 0.0324 | 0.0125 | 0.0527 | 0.0435 |

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
| **MOSS i2i Reference** | MODEL, VAE, IMAGE → MODEL, LATENT | Loads the i2i LoRA (`lora_name`, `strength`; any key layout, see [LoRA files](#lora-files)) and reads `reference_latents`, `ref_scale`, `ref_timestep` and `ref_isolate` from its metadata. `task = auto` sets the ref size from the recipe the LoRA trained with: full size for an isolated ref, half for a joint ref. Workflows saved with `task = colorize` run as `auto`; ones saved with `task = custom` still load and use `ref_scale` (ref at `1/ref_scale` per side, 2 = 1/4 area). The ref goes into every forward and every CFG branch. `ref_cache`: see below. Also outputs an empty latent at the target size |

### Ref K/V cache

An isolated ref with its own timestep depends only on the page: not on
the step, the prompt or the CFG branch. With `ref_cache = gpu` (default) the first
forward runs it through the 42 blocks once and keeps each block's ref keys and
values (~2.1 GB bf16 at 1 MP). After that, each forward runs only the image and text
rows, so the full-size ref costs about as much as extra keys. The cache is rebuilt
once per prompt (and when the model's weight patches or the target size change). A
K/V kept from the previous prompt broke ComfyUI's dynamic-VRAM allocation graph in
every block on a reseed (sampling 17 → 25 s), and the rebuild costs about a second. `cpu` keeps it in pinned host memory and copies one block's K/V in at a
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
Japanese). Type it into Caption Helper with `task swap_text`, connect the CLIP to the
helper's `clip`, and take its `conditioning` output as the positive. A LoRA trained
with anima_lora's `token_split = syllable` (stamped `token_split` in its metadata)
learned Korean on one token per Hangul syllable. The default Qwen3.5 tokenizer merges
syllables (`인가요`) into tokens that LoRA never saw, and the split happens in the
tokenizer, so a prompt string through a plain `CLIP Text Encode` loses it. For
English, or a swap_text LoRA without the split, `task layout` into `CLIP Text Encode`
is the same prompt.

Use the same `resolution` on Text Encode, Caption Helper and Reference. The negative is a plain text
encode of `""`, which is what the LoRA was evaluated with. For CFG 3 with the uncond
block-22 skip, use the MOSS Spectrum KSampler. anima_lora's colorize eval ran 30 steps,
CFG 3, ER-SDE, Spectrum at its defaults, with the fp8 TE. That is the v3 example
workflow, except that ComfyUI's σ grid is not snapped to the training grid and its
ER-SDE noise differs, so a seed does not reproduce the eval's image.

### LoRA files

MOSS-Image DiTs come in two layouts: the vendor's separate `layers.N.proj.{q,k,v}` (the
`moss-image` node; `moss_image/<name>/transformer` dirs) and the Comfy-Org repack's fused
`layers.N.proj.qkv` (ComfyUI's own MOSS-Image model, which ai-toolkit trains on). ComfyUI
maps a separate-q/k/v LoRA onto a fused model by row slices, but not the reverse, so a
fused LoRA (ai-toolkit's LoKr, ai-toolkit saves before 2026-10-17) patched nothing on the
`moss-image` model. The Reference node splits a fused group by rows first: exact for
LoRA, LoHa and LoKr (a LoKr whose `w1` rows do not divide by 3 becomes three full
`diff` patches; DoRA groups stay fused and are reported). `transformer.` and bare
module paths take the `diffusion_model.` prefix. Every load logs how many weights it
patched, names the non-block ones (`shared_block_modulation`, a `mod_rank` LoRA's
modulation adapter), warns with the modules this model has no weight for, and fails
if nothing matched.

An ai-toolkit edit LoRA (`ss_base_model_version = moss_image`) carries no ref metadata.
`task = auto` runs it as ai-toolkit trains it: the page at full size, its rows at t = 0,
isolated (ai-toolkit's `kv_cache`, on in its edit presets); `task = ai-toolkit,
kv_cache off` runs the joint ref instead. ai-toolkit also shows the reference to the
text encoder (`Picture 1:` and the image at ≤ 384² px), which ComfyUI's **Text Encode
MOSS-Image Edit** builds (leave its VAE unconnected so it adds no second reference).

### Model layouts

On ComfyUI's own MOSS-Image model (Load Diffusion Model on the Comfy-Org repack or the
vendor shards) the node does not run its copy of the forward; it passes the ref to the
model's built-in reference path. An isolated ref at t = 0 goes in as `ref_kv`, built
once per prompt and weight patch set (`ref_cache` gpu / cpu; off: recomputed each forward). A
joint ref goes in as `index_timestep_zero`, a shared-timestep one as `index`. The
outputs match the `moss-image` path for a full-size ref (colorize v3, sanitize,
swap_text, ai-toolkit). That model puts a ref on its own grid, not stretched onto the
target, so a half-size ref (colorize v1 / v2) stops with an error there and needs
the `moss-image` node's model: install that node and load with its **MOSS-Image HF
Loader**, then use `ModelSamplingSD3` as before.

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
| **MOSS Caption Helper** | STRING, task (+ IMAGE, CLIP) → STRING, CONDITIONING | `task = default`: the caption as written. `task = layout`: `anime_tools.captions.layout_json`, byte-identical to what the anima_lora MOSS trainer encodes with `caption_format=layout_json`. Rating / meta, characters, series and `@artists` get their own `summary` keys, the other tags become `summary.caption`, each position clause a character item, each quoted text line a text item. Every meta or unknown tag ahead of the first count tag (`1girl`, `2boys`, …) is front matter and goes to `summary.meta`. A caption that already is layout JSON (an object with `summary`) passes through as typed, as in the trainer. `task = colorize`: that layout plus one `{"label":"text","caption":"text","text":…,"bbox":…}` item per line the OCR reads off the page, the caption the colorize LoRA trained on. Boxes are `abs_centralized` `yxyx` on the canvas the i2i nodes generate at, so use the same `resolution`. `task = sanitize`: `no sfx, no dialogue, no watermark` verbatim, the one caption every sanitize pair trained on. `task = decensor`: `no bar censor, no mosaic censor, uncensored`, likewise for the decensor LoRA. Both ignore the caption input. `task = swap_text`: the caption as written (the page's layout JSON); with `clip` connected, Hangul encodes one syllable per token, a space before a syllable as its own token, as anima_lora's `token_split = syllable` (`token_split.py`, a copy of `library/moss/token_split.py`), identical token ids to the trainer on its eval pages. The `prompt` string carries no split. With `clip` connected, every task also outputs `conditioning`, the prompt encoded as a plain `CLIP Text Encode` would. `put_endthink` (off by default) outputs the whole chat turn, `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n` + `\n</think>\n\nSure,`, byte-identical to anima_lora's `prompt_tail = sure` (the trainer's cache default; on Base it sharpens hands and backgrounds). The MOSS tokenizer skips its own template for a prompt starting with `<|im_start|>`, so the tail lands after the generation prompt. Use it with a plain `CLIP Text Encode` and LoRAs cached on `sure`; leave it off for LoRAs cached on `off` (the i2i ones) and for MOSS i2i Text Encode. The trainer tails the uncond too, but a plain `""` negative renders nearly the same |

Feed the layout output into `CLIP Text Encode` with MOSS-Image's text encoder, the
colorize output into **MOSS i2i Text Encode** as its `text` (v2) or a plain `CLIP Text
Encode` (v3), the sanitize and decensor outputs into a plain `CLIP Text Encode`. For colorize, type the
colour tags the LoRA trained on (copyright, `comic` for a manga page, hair / eye /
clothing colours). The OCR is anime-tools' AnimeText detector with the manga VL
reader on every box; both load for the read (~6 GB VRAM freed first) and unload
after it, and their weights download on first use. The helper needs
`anime-tools`; without it the toolkit loads the i2i nodes only and logs a warning:

```
pip install 'anime-tools @ git+https://github.com/sorryhyun/anime_tools'
```

The danbooru tag KB (`danbooru_tags_classified.csv`) is looked up under
`$ANIME_TOOLS_HOME` / `$ANIMA_HOME` / the ComfyUI folder (`models/`) and downloaded
there on first use. The old **Anima Caption → MOSS Layout JSON** node
(`AnimaCaptionToMossLayout`) is the helper's `task = layout`; it is deprecated and hidden
from the node menu, but stays registered so saved workflows load as-is.

## License

Apache-2.0. `reference.py` derives from the `moss-image` node's `MOSSImageTransformer`
(OpenMOSS, Apache-2.0).

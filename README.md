# MOSS Toolkit for ComfyUI

Nodes for [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image), on top of the
`moss-image` node's MODEL / CLIP / VAE (formerly **MOSS i2i**, `comfyui-moss-i2i`):

- **Loader** — DiT, Qwen3.5 text encoder and VAE in one node, each from its own
  source; the TE can be the fp8 file.
- **i2i** — image-to-image. First task: colorizing a B&W manga page with the colorize
  LoRA trained in anima_lora.
- **Layout JSON** — an Anima-format caption as MOSS's layout-JSON prompt, for LoRAs
  trained on `caption_format=layout_json` captions.

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

For the fp8 TE, download
[`sorryhyun/MOSS-Image-Qwen3.5-9B-fp8`](https://huggingface.co/sorryhyun/MOSS-Image-Qwen3.5-9B-fp8)
into `models/moss_image/fp8/` (it lists as `fp8/qwen35_9b_fp8_e4m3fn.safetensors`), or put
the file in `models/text_encoders/`. It is 9.2 GB instead of the 16.8 GB bf16 shards. A
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
  work. A LoRA stamped with `ref_timestep` (colorize v2: 0) gets the ref rows
  modulated at that timestep in every block; without it (v1) they share the step's.

### Nodes

| Node | In → out | Notes |
|---|---|---|
| **MOSS i2i Text Encode (page + text)** | CLIP, IMAGE, text → CONDITIONING | `<|vision_start|>[page]<|vision_end|>{text}`. Empty text = page only. The page is fit to the target size and seen at that pixel count (~1000 tokens at 1 MP) |
| **MOSS i2i Reference** | MODEL, VAE, IMAGE → MODEL, LATENT | Loads the i2i LoRA (`lora_name`, `strength`; plain `lora_unet_layers_*`) and reads `reference_latents`, `ref_scale` and `ref_timestep` from its metadata. Ref at `1/ref_scale` per side (2 = 1/4 area). The ref goes into every forward and every CFG branch. Also outputs an empty latent at the target size |

### Wiring

```
MOSS-Image Loader ─ MODEL ─ MOSS i2i Reference (LoRA) ──────────────── MODEL ─┐
                  ─ CLIP ─┬ MOSS i2i Text Encode (page, colour tags) ─ positive ─┤
                          └ CLIP Text Encode ("") ─────────────────── negative ─┤ KSampler /
                  ─ VAE ──── MOSS i2i Reference ───────────────────── LATENT ───┘ MOSS Spectrum KSampler
Load Image (page) → Text Encode.image, Reference.image
```

Use the same `resolution` on Text Encode and Reference. The negative is a plain text
encode of `""`, which is what the LoRA was evaluated with. For CFG 3 with the uncond
block-22 skip, use the MOSS Spectrum KSampler.

### Compatibility

- **Spectrum (MOSS).** Cached steps never call the DiT, so they cost no ref rows. On
  full steps the capture hook sees the image rows only.
- **DAVE.** Its hook reads the leading image rows. The ref rows come after them.
- **SLG / uncond skip.** The `patches_replace` dispatch is kept, so it still works.

The patch is an `APPLY_MODEL` wrapper. For the length of each `apply_model` call, the
live `diffusion_model` runs a ref-aware copy of the `moss-image` forward.

## Layout JSON

| Node | In → out | Notes |
|---|---|---|
| **Anima Caption → MOSS Layout JSON** | STRING → STRING | `anime_tools.captions.layout_json`, byte-identical to what the anima_lora MOSS trainer encodes with `caption_format=layout_json`. Rating / meta, characters, series and `@artists` get their own `summary` keys, the other tags become `summary.caption`, each position clause a character item, each quoted text line a text item |

Feed its output into `CLIP Text Encode` with MOSS-Image's text encoder. It needs
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

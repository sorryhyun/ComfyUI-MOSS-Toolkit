# MOSS Toolkit for ComfyUI

Nodes for [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) on ComfyUI's own MOSS-Image
model. i2i needs ComfyUI 0.43.0 or later. The Caption Helper needs `anime-tools[ocr]`,
which `requirements.txt` installs (ComfyUI Manager and the registry do this for you):

```
pip install -r ComfyUI/custom_nodes/comfyui-moss-toolkit/requirements.txt
```

Without it the other nodes load and a warning is logged.

| Node | In → out | What it does |
|---|---|---|
| **MOSS-Image Loader (DiT + TE + VAE)** | → MODEL, CLIP, VAE | Each part from its own source; DiT and TE can be INT8 files |
| **MOSS-Image fp32 Stream** | MODEL → MODEL | Residual stream, modulation, norms, embedders and head in fp32 around bf16 / INT8 GEMMs |
| **MOSS-Image LoRA** | MODEL → MODEL | Either q/k/v layout; `bypass` (on by default) keeps INT8 weights intact; chains |
| **MOSS i2i Reference** | MODEL, VAE, IMAGE → MODEL, LATENT | Loads an i2i LoRA and feeds the page as a clean VAE reference latent |
| **MOSS i2i Text Encode (page + text)** | CLIP, IMAGE, text → CONDITIONING | Page through Qwen3.5's vision tower; colorize v2 only, hidden from the menu |
| **MOSS Caption Helper** | STRING, task, LATENT (+ IMAGE, CLIP) → STRING, CONDITIONING | The prompt each MOSS LoRA trained on |

## Loader

| Input | Lists |
|---|---|
| `dit` | `models/moss_image/<name>/transformer/` dirs, then `diffusion_models` files |
| `text_encoder` | `moss_image/<name>/text_encoder/` dirs (bf16), then `*.safetensors` beside a `config.json` in other `moss_image/<name>/` dirs, then `text_encoders` files |
| `vae` | `moss_image/<name>/vae/` dirs, then `vae` files |

`moss_image:` in `extra_model_paths.yaml` adds more roots. A quantized TE is one 9.2 GB
file instead of 16.8 GB of bf16 shards:
[`sorryhyun/MOSS-Image-Qwen3.5-9B-int8`](https://huggingface.co/sorryhyun/MOSS-Image-Qwen3.5-9B-int8)
into `models/moss_image/int8/` (the default when present) or
[`sorryhyun/MOSS-Image-Qwen3.5-9B-fp8`](https://huggingface.co/sorryhyun/MOSS-Image-Qwen3.5-9B-fp8)
into `models/moss_image/fp8/`. INT8 runs W8A8 (`int8_matmul`, advanced); off, ComfyUI
dequantizes to a bf16 matmul. Both files carry the vision tower.

Everything loads through ComfyUI's own `MOSSImage` model: vendor shards, Comfy-Org's fused
`proj.qkv` files and INT8 files (separate q/k/v is fused at load). That model takes σ as its
timestep, so drop `ModelSamplingSD3` (shift defaults to 5) or use `ModelSamplingAuraFlow`;
the MOSS Spectrum KSampler (Spectrum pack 2.13.1+) sets it itself. `stream = fp32`
(advanced) on the Loader is the fp32 Stream node built in.

## LoRA on an INT8 DiT

ComfyUI patches an INT8 weight by requantizing the whole linear, which loses a GPTQ file's
weights. `bypass` adds the LoRA in the forward instead, including on the fused `proj.qkv`
where ComfyUI's `Load LoRA (Bypass)` drops the q/k/v adapters. On a ConvRot INT8 linear the
LoRA's `down` runs as an INT8 GEMM. bf16 linears and LoKr / LoHa use ComfyUI's float bypass.
Each load logs how many weights it patched and fails if nothing matched.

## i2i

The page goes in as a clean VAE latent on RoPE's reserved `r = 1` axis. MOSS i2i Reference
reads the LoRA's metadata (`reference_latents`, `ref_scale`, `ref_timestep`, `ref_isolate`,
`ref_adapters`) and sets the ref size, timestep and isolation from it (`task = auto`).
`ref_cache = gpu` builds an isolated ref's keys / values once per prompt. Mechanics, kv
LoRAs, ai-toolkit LoRAs and model layouts: [docs/i2i.md](docs/i2i.md).

| LoRA | Ref | Prompt | Example workflow |
|---|---|---|---|
| colorize v5 | full size, isolated | Caption Helper `colorize_v5` | `moss-i2i-colorize-v5.json` |
| colorize v3 / v4 | full size, isolated | v3: colour tags; v4: Caption Helper `colorize` | `moss-i2i-colorize-v3.json` |
| colorize v2 | half size, joint | MOSS i2i Text Encode | `moss-i2i-colorize.json` |
| sanitize / decensor | full size, isolated | Caption Helper `sanitize` / `decensor` | `moss-i2i-sanitize.json` |
| swap_text | full size, isolated | Caption Helper `swap_text`, `conditioning` output | — |

```
MOSS-Image Loader ─ MODEL ─ MOSS i2i Reference (LoRA) ───────────── MODEL ──┐
                  ─ VAE ──── MOSS i2i Reference ───────────────────── LATENT ─┤
                  ─ CLIP ─┬ MOSS Caption Helper (latent, clip) ─ conditioning ─┤ KSampler /
                          └ CLIP Text Encode ("") ───────────────── negative ─┘ MOSS Spectrum KSampler
Load Image (page) → Reference.image, Caption Helper.image
Reference.LATENT → KSampler.latent_image and Caption Helper.latent
```

The Reference's `resolution` sets the canvas. The negative is a plain encode of `""`; for
CFG with the uncond block skip use the MOSS Spectrum KSampler.

## Caption Helper

| Task | Output |
|---|---|
| `default` | The caption as written |
| `layout` | Layout JSON, the same string anima_lora encodes with `caption_format=layout_json` |
| `colorize` | `layout` plus one text item per line the OCR reads off the page |
| `colorize_v5` | Manga page: `comic` + box-only text items; illustration: series + colour tags |
| `sanitize` / `decensor` | That LoRA's fixed caption; the caption input is ignored |
| `swap_text` | The caption as written, Hangul split one syllable per token (needs `clip`) |

With `clip` connected, `conditioning` is the prompt encoded; wire it to the positive.
`put_endthink` appends anima_lora's `prompt_tail = sure` chat tail. Write parentheses
plain: inside layout JSON `\(` reaches the text encoder as typed. Details:
[docs/caption-helper.md](docs/caption-helper.md).

## License

Apache-2.0. `reference.py` derives from the `moss-image` node's `MOSSImageTransformer`
(OpenMOSS, Apache-2.0).

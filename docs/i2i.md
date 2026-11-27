# MOSS i2i Reference

MOSS-Image has no image conditioning. The Reference node adds the page as a clean VAE
latent: an extra token block `[image | ref | text]` on RoPE's reserved `r = 1` axis, its
`(h, w)` positions stretched onto the target grid. The page is fit (cover + centre crop) to
`resolution²` at native aspect, sides multiples of 16, then fit again to `1/ref_scale` per
side for the ref.

## What the LoRA metadata sets

| Key | Effect |
|---|---|
| `reference_latents` | 0 or absent: the LoRA never saw a ref; the node warns |
| `ref_scale` | Ref size; absent: 1 for an isolated ref, 2 for a joint ref |
| `ref_timestep` | The ref rows are modulated at this timestep in every block; absent: the step's own |
| `ref_isolate` | The ref queries attend only to the ref keys; image and text still read the ref |
| `ref_adapters = kv` | A per-block K/V bank for the ref (below); `all` raises |

## Ref K/V cache

An isolated ref with its own timestep depends only on the page, not on the step, the prompt
or the CFG branch. `ref_cache = gpu` runs it through the blocks once per prompt and keeps
each block's keys and values; every forward then runs only the image and text rows. The
cache is rebuilt per prompt and when the weight patches or the target size change. `cpu`
keeps it in pinned host memory and copies one block in at a time; `off` runs the ref rows
in every forward. A joint ref always runs its rows.

## kv LoRAs

A LoRA trained with anima_lora's `ref_adapters = kv` (`ref_layers_N_proj_{k,v}` keys) ran its
ref rows on the Base weights; only the keys and values the image and text rows read carry a
per-block adapter. The node builds the ref K/V that way: the LoRA goes on in bypass (forced),
every bypass adapter this toolkit loaded is off for the ref pass, and the bank's deltas go on
`proj.k` / `proj.v` before `k_norm`. Loaded any other way (MOSS-Image LoRA, ComfyUI's Load
LoRA) the bank is dropped and the ref rows run the LoRA. A LoRA added upstream by another
node stays on in the ref pass.

## LoRA files

MOSS-Image DiTs come with separate `layers.N.proj.{q,k,v}` (vendor, the `moss-image` node)
or fused `layers.N.proj.qkv` (Comfy-Org repack, ComfyUI's own model, ai-toolkit). The node
splits a fused group by rows, exact for LoRA, LoHa and LoKr (a LoKr whose `w1` rows do not
divide by 3 becomes three full `diff` patches; DoRA stays fused and is reported).
`transformer.` and bare module paths take the `diffusion_model.` prefix.

An ai-toolkit edit LoRA (`ss_base_model_version = moss_image`) has no ref metadata.
`task = auto` runs it as ai-toolkit trains it: full-size page at t = 0, isolated (its
`kv_cache`); `task = ai-toolkit, kv_cache off` runs a joint ref. ai-toolkit also shows the
reference to the text encoder, which ComfyUI's **Text Encode MOSS-Image Edit** builds
(leave its VAE unconnected so it adds no second reference).

## Model layouts

On ComfyUI's own MOSS-Image model the node passes the ref to the model's built-in reference
path: an isolated ref at t = 0 as `ref_kv`, a joint ref as `index_timestep_zero`, a
shared-timestep one as `index`. That model puts a ref on its own grid instead of stretching
it, so a half-size ref (colorize v2) stops with an error there; load it with the `moss-image`
node's **MOSS-Image HF Loader** and `ModelSamplingSD3`. On the `moss-image` model the node
runs its own ref-aware copy of the forward for each `apply_model` call (an `APPLY_MODEL`
wrapper).

## With other nodes

- **Spectrum (MOSS)**: cached steps skip the DiT and the ref rows; on full steps the capture
  hook sees the image rows only.
- **DAVE**: its hook reads the leading image rows; the ref rows come after them or sit in
  the cache. The block functions call the block's forward hooks themselves.
- **SLG / uncond skip**: the `patches_replace` dispatch is kept. Under `ref_isolate` a
  skipped block still runs the ref rows, so the uncond branch reads the same ref as the cond.

# Example workflows

Five small workflows that show how Luna-H3-Sampler fits into a normal MiniMax H3 graph. Apart from this node they
use only ComfyUI core nodes and one KJNodes node (`MiniMaxChunkFeedForward`, memory saving). The model side is the
same as in the official Comfy MiniMax H3 templates; only the sampler section is replaced by Luna-H3-Sampler.

ComfyUI lists them under Templates › Custom Nodes › Luna-MinimaxH3. Pick your own files in the loaders, the image in
`LoadImage`, and write your own prompt.

| Workflow | What it shows |
|---|---|
| `Luna-H3-Sampler - Reference to Video.json` | Reference image to video with sound (`MiniMaxH3ReferenceToVideo`). |
| `Luna-H3-Sampler - Image to Video.json` | First frame to video with sound (`MiniMaxH3ImageToVideo`, fl2va model). |
| `Luna-H3-Sampler - LoRA on pass 1 only.json` | A LoRA that only pass 1 gets: `LoraLoaderModelOnly` on `model`, the plain model on `high_res_model`. The example uses Kijai's Singularity V2 LoRA as a motion dial (1.0 = more, faster motion; 0 = none). |
| `Luna-H3-Sampler - 1 MP to 2K.json` | Text to video at 1344x768 with Luna-H3-Sampler, then a 2K refine stage to 2560x1440 (see below). The fast route. |
| `Luna-H3-Sampler - 2 MP to 2K.json` | The same at 1920x1088 first. The best quality of our tests. |

## 2K workflows

Stage 1 is Luna-H3-Sampler with the 2 MP recipe below, at 1344x768 or 1920x1088. Stage 2 is the 2K refine stage
from Kijai's two-stage Open-H3-Regenerate workflow, unchanged:

```
Luna-H3-Sampler latent -> MiniMaxH3LatentUpscaleWithModel (2560x1440, learned upscaler)
model -> LoraLoaderModelOnly (DMAD 4-step) -> LoraLoaderModelOnly (Open-H3-Regenerate-2K refiner, rank 16)
      -> BlockSparseAttention (sol-attn) -> BasicGuider
RandomNoise + KSamplerSelect (er_sde) + ManualSigmas "0.7, 0.494118, 0" -> SamplerCustomAdvanced
-> VAEDecodeTiled -> CreateVideo (sound from stage 1, VAEDecodeAudio) -> SaveVideo
```

- Needs ComfyUI 0.43 nightly from 2026-10-28 or newer (`MiniMaxH3LatentUpscaleWithModel`, `BlockSparseAttention`).
- `ModelAttentionBackend` is set to `comfy kitchen attention` (INT8 attention, Nvidia and AMD only); pick
  `pytorch attention` elsewhere.
- Only the 2K video is decoded, tiled. Decoding extra videos (the first stage, a side-by-side) costs about 3 to 11 GB
  of RAM each at this size.

Tested on one RTX 4090 (24 GB) with 64 GB RAM, 124 frames, same prompt and seed (2026-10-29):

| First stage | Total time | Lowest free RAM | Verdict |
|---|---|---|---|
| Kijai's own chain, 1344x768, DMAD 4 steps | 182 s | 1.7 GB (int8 text encoder, extra decodes) | reference |
| Luna-H3-Sampler 1344x768 (1 MP) | 166 s | 11.2 GB | better than Kijai's chain at the same size |
| Luna-H3-Sampler 1664x928 (1.5 MP) | 202 s | 15.4 GB | below 2 MP |
| Luna-H3-Sampler 1920x1088 (2 MP) | 259 s | 15.8 GB | best quality |

Peak VRAM was 22 to 24 GB in every run (dynamic VRAM fills the card).

### Limit: clip length at 2K (24 GB card)

On the RTX 4090 the 2K stage works for 5 s clips only so far:

| Clip | First stage (2 MP) | 2K stage (2 steps) |
|---|---|---|
| 124 frames (5.2 s), 2560x1440 | about 1.5 min | 66 s |
| 124 frames (5.2 s), vertical 1440x2560, 3 reference images | 2.6 min | 85 s |
| 345 frames (14.4 s), 2560x1440, 3 reference images | 5.9 min (works) | did not finish its first step in 12.5 min; stopped |

The first stage handles full scene length; the 2K stage does not on 24 GB. Most likely the 2K latent at that length no
longer fits next to the model, so ComfyUI keeps swapping weights (inferred, not measured). Lengths between 5 and 14 s
are untested. A windowed 2K pass (overlapping 5 s windows) is the planned way around it; until then, use the 2K
workflows for clips of about 5 s, or a card with more memory.

## Credits

- **Kijai** ([github.com/kijai](https://github.com/kijai)): the two-stage 2K workflow these follow (its stage 2 is
  used as he made it), the `MiniMaxH3LatentUpscaleWithModel` node in ComfyUI, the rank-16 refiner and DMAD LoRAs
  in [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental), the turbo LoRAs in
  [Kijai/MiniMax-H3_comfy](https://huggingface.co/Kijai/MiniMax-H3_comfy), and KJNodes.
- **NVIDIA (NVlabs / Efficient-Large-Model)**: the Open-H3-Regenerate-2K refiner
  ([model](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K),
  [project page](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/)) and sol-attn.
- **LBH-123-AI**: the H3 latent upscaler model and code
  ([Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)).
- **LightX2V**: the turbo LoRAs. **MiniMax**: the H3 model.

These workflows contain settings, not code, from those projects. Each model keeps its own licence.

## The graph

```
UNETLoader -> MiniMaxChunkFeedForward -> model ┐
CLIPLoader -> MiniMaxH3ReferenceToVideo / ImageToVideo -> positive, latent_image ┤
                 └-> ConditioningZeroOut -> negative ┤
VAELoader (video) -> vae ┤
KSamplerSelect x2 -> sampler_low, sampler_high ┘-> Luna-H3-Sampler -> latent
latent -> VAEDecode (video VAE) + VAEDecodeAudio (audio VAE) -> CreateVideo -> SaveVideo
```

- The conditioning node builds the latent at the final size; the sampler makes the half-size pass itself.
- `negative` is `ConditioningZeroOut` of the positive: with CFG 1 it is not used, but the input needs one.
- Two `KSamplerSelect` nodes, so each pass can take its own sampler (for example `er_sde`, or a Detail Daemon sampler).
- The sampler settings are the measured 2 MP recipe: turbo LoRA 0.6 in pass 1 and 0.8 in pass 2, Euler, `simple`,
  12 steps, transition 9, shift 12 / 3, 1920x1088, 124 frames (5 s at 24 fps).

## Models

From [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):
`diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` (reference to video),
`diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` (image to video),
`text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, `vae/minimax_h3_video_vae_int8_convrot.safetensors`,
`vae/minimax_h3_audio_vae_fp32.safetensors`.

From [Kijai/MiniMax-H3_comfy](https://huggingface.co/Kijai/MiniMax-H3_comfy) (`loras/`): the turbo LoRAs
`minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors` and
`minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors`.

From [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental) (`loras/`), only for the
LoRA example: `minimax_h3_singularity_v2_lora_avg_rank_351_bf16.safetensors` (3.3 GB; Kijai's card says it keeps
roughly 40-50 % of the Singularity V2 fine-tune).

The latent upscaler `minimax_h3_latent_upscaler_3d_fp16.safetensors` is optional: set `upscaler_model` to `none` if
you do not have it.

Node packs: [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) for `MiniMaxChunkFeedForward`. Without it,
connect `UNETLoader` straight to `model` (more VRAM at 2 MP).

For the 2K workflows also: `minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors` and
`minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors` from Kijai/MiniMax-H3-experimental (`loras/`), and
the latent upscaler (required there).

The workflows are built by `tools/make_example_workflows.py`.

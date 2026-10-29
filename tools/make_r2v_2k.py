"""Reference to video in two stages: Luna-H3-Sampler at 2 MP, then Kijai's 2K refine stage (API graph).

Made 2026-10-29 for Peter: "Build  a portrait video with Elira using her image and background. Waist up, clear face
and skin." (vertical, 2 MP to 2K, 5 s). Stage 1 = Luna-H3-Sampler with the measured 2 MP recipe (ref2v lightx2v
turbo rank-20 LoRA 0.6 / 0.8, Euler, simple, 12 steps, transition 9, shift 12 / 3, learned upscaler). Stage 2 =
`make_example_workflows.to_2k` (MiniMaxH3LatentUpscaleWithModel, DMAD + Open-H3-Regenerate-2K refiner rank 16,
sol-attn, er_sde, sigmas 0.7, 0.494118, 0). The references are encoded twice: at the first-stage size and at the 2K
size, because H3's reference latents are sized to the video. Only the 2K video is decoded (tiled); the sound comes
from stage 1.

Usage: python make_r2v_2k.py <out_api.json> <prompt.txt> <width>x<height> <prefix> <image> [<image> ...]
`width`x`height` is the 2K size (1440x2560 vertical, 2560x1440 landscape); stage 1 is that size x 0.75 on the
32-pixel grid (1088x1920 / 1920x1088). Images are copied into ComfyUI's input folder as `luna_r2v_<name>` (LoadImage lists only top-level files), in order
(<Picture 1>, <Picture 2>, ...).
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_example_workflows import (AUDIO_VAE, CLIP, DMAD, FPS, LENGTH, REFINER, UPSCALER_BF16, VIDEO_VAE,  # noqa: E402
                                    node, sampler_inputs)

UNET = "MiniMaxH3\\Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
TURBO_R2V = "MiniMaxH3/minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors"
COMFY_INPUT = Path("E:/ComfyUI-Easy-Install/ComfyUI/input")


def build(prompt, size_2k, prefix, images):
    w2, h2 = size_2k
    w1, h1 = (round(v * 0.75 / 32) * 32 for v in size_2k)
    g = {}
    base = node(g, 1, "UNETLoader", unet_name=UNET, weight_dtype="default")
    chunked = node(g, 2, "MiniMaxChunkFeedForward", model=base, chunks=2, seq_threshold=4096)
    model = node(g, 7, "ModelAttentionBackend", model=chunked, attention="comfy kitchen attention")
    clip = node(g, 3, "CLIPLoader", clip_name="MiniMaxH3\\" + CLIP, type="minimax", device="default")
    vae = node(g, 4, "VAELoader", vae_name="MiniMaxH3\\" + VIDEO_VAE)
    audio_vae = node(g, 5, "VAELoader", vae_name="MiniMaxH3\\" + AUDIO_VAE)
    refs = {}
    for i, image in enumerate(images):
        refs[f"ref_images.ref_image_{i}"] = node(g, 60 + i, "LoadImage", image=f"luna_r2v_{Path(image).name}")
    cond1 = node(g, 11, "MiniMaxH3ReferenceToVideo", clip=clip, prompt=prompt, width=w1, height=h1, length=LENGTH,
                 ref_image_size="max", vae=vae, audio_vae=audio_vae, **refs)
    negative = node(g, 12, "ConditioningZeroOut", conditioning=cond1)
    node(g, 20, "KSamplerSelect", sampler_name="euler")
    node(g, 21, "KSamplerSelect", sampler_name="euler")
    stage1 = node(g, 30, "Luna-H3-Sampler", **sampler_inputs(model, cond1, negative, vae, ["11", 1], TURBO_R2V,
                                                              noise_seed=101))
    cond2 = node(g, 13, "MiniMaxH3ReferenceToVideo", clip=clip, prompt=prompt, width=w2, height=h2, length=LENGTH,
                 ref_image_size="max", vae=vae, audio_vae=audio_vae, **refs)
    dmad = node(g, 50, "LoraLoaderModelOnly", model=model, lora_name="MiniMaxH3\\" + DMAD, strength_model=1.0)
    refiner = node(g, 51, "LoraLoaderModelOnly", model=dmad, lora_name="MiniMaxH3\\" + REFINER, strength_model=1.0)
    sparse = node(g, 52, "BlockSparseAttention", model=refiner, selection="sol-attn", **{"selection.tau": 1.3},
                  start_percent=0.0, end_percent=1.0, dense_blocks="", min_tokens=12288, extra_tokens=256,
                  sink_conditioning="exact_kv_and_rows", verbose=False)
    guider = node(g, 53, "BasicGuider", model=sparse, conditioning=cond2)
    noise = node(g, 54, "RandomNoise", noise_seed=101)
    sampler = node(g, 55, "KSamplerSelect", sampler_name="er_sde")
    sigmas = node(g, 56, "ManualSigmas", sigmas="0.7, 0.494118, 0")
    upscaler = node(g, 57, "LatentUpscaleModelLoader", model_name=UPSCALER_BF16)
    lifted = node(g, 58, "MiniMaxH3LatentUpscaleWithModel", model=upscaler, samples=stage1, width=w2, height=h2)
    stage2 = node(g, 59, "SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas,
                  latent_image=lifted)
    images_out = node(g, 40, "VAEDecodeTiled", samples=stage2, vae=vae, tile_size=512, overlap=64, temporal_size=64,
                      temporal_overlap=8)
    audio = node(g, 41, "VAEDecodeAudio", samples=stage1, vae=audio_vae)
    video = node(g, 42, "CreateVideo", images=images_out, audio=audio, fps=FPS)
    node(g, 43, "SaveVideo", video=video, filename_prefix=prefix, format="auto", codec="auto")
    g["30"]["inputs"]["lora_high"] = TURBO_R2V
    return g


def main(out, prompt_file, size, prefix, *images):
    size_2k = tuple(int(v) for v in size.lower().split("x"))
    for image in images:
        shutil.copy2(image, COMFY_INPUT / f"luna_r2v_{Path(image).name}")
    graph = build(Path(prompt_file).read_text(encoding="utf-8").strip(), size_2k, prefix, images)
    with Path(out).open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=1)
    print(f"{len(graph)} nodes -> {out}; stage 1 {graph['11']['inputs']['width']}x{graph['11']['inputs']['height']}")


if __name__ == "__main__":
    main(*sys.argv[1:])

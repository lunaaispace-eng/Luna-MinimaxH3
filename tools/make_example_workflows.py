"""Build the graphs behind example_workflows/ (API format).

To UI files (2026-10-29): open ComfyUI, run `await comfyAPI.app.app.loadApiJson(graph, title)` in the page, place
the nodes, set `noise_seed` to fixed, then save `JSON.stringify(app.rootGraph.serialize())` as
`example_workflows/<title>.json` (json.dump, indent 2).

Peter, 2026-10-29: simple user workflows that show how Luna-H3-Sampler fits with ComfyUI core nodes and KJNodes, none
of our other custom nodes. The model side follows the official Comfy MiniMax H3 templates (UNETLoader, CLIPLoader
type minimax, two VAELoaders, MiniMaxH3ReferenceToVideo / MiniMaxH3ImageToVideo, VAEDecode + VAEDecodeAudio,
CreateVideo, SaveVideo); the sampler section is Luna-H3-Sampler with the measured 2 MP recipe.

Usage: python make_example_workflows.py <out_dir> [object_info.json]
With object_info.json (from a running ComfyUI's /object_info), every file name is mapped to the installed name with
the same base name (for example `MiniMaxH3\\x.safetensors`), so the graphs validate on that machine. Without it the
plain official names are written.
"""

import json
import sys
from pathlib import Path

UNET_R2V = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
UNET_I2V = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_R2V = "minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors"
TURBO_I2V = "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors"
V2_LORA = "minimax_h3_singularity_v2_lora_avg_rank_351_bf16.safetensors"
UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"

WIDTH, HEIGHT, LENGTH, FPS = 1920, 1088, 124, 24

PROMPT_R2V = (
    "Cinematic, natural light, shallow depth of field. <Picture 1> stands in a sunlit courtyard and turns toward the "
    "camera, then speaks calmly: \"We start again tomorrow.\" A light breeze moves her hair and the leaves behind "
    "her. Medium shot, slow push-in. Audio: her voice, soft wind, distant birds."
)
PROMPT_I2V = (
    "The person in the first frame looks up, smiles and says: \"Ready when you are.\" Natural light, steady camera, "
    "gentle movement. Audio: the voice, quiet room tone."
)


def node(graph, node_id, class_type, **inputs):
    graph[str(node_id)] = {"class_type": class_type, "inputs": inputs}
    return [str(node_id), 0]


def sampler_inputs(model, positive, negative, vae, latent, lora, **extra):
    """The measured 2 MP recipe: turbo LoRA 0.6 / 0.8, Euler, simple, 12 steps, transition 9, shift 12 / 3."""
    inputs = dict(
        model=model, positive=positive, negative=negative, vae=vae, latent_image=latent,
        lora_low=lora, strength_low=0.6, shift_low=12.0, audio_shift_low=3.0, sampler_low=["20", 0], cfg_low=1.0,
        lora_high=lora, strength_high=0.8, shift_high=12.0, audio_shift_high=3.0, sampler_high=["21", 0],
        cfg_high=1.0, scheduler="simple", total_steps=12, transition=9, highres_denoise=1.0, noise_seed=1111,
        lowres_scale=0.5, upscaler_model=UPSCALER, rho=0.0, w_min=0.5, w_max=1.0,
    )
    inputs.update(extra)
    return inputs


def common(graph, unet):
    """Loaders, KJNodes memory patch, the two samplers, decode and save."""
    base = node(graph, 1, "UNETLoader", unet_name=unet, weight_dtype="default")
    model = node(graph, 2, "MiniMaxChunkFeedForward", model=base, chunks=2, seq_threshold=4096)
    clip = node(graph, 3, "CLIPLoader", clip_name=CLIP, type="minimax", device="default")
    vae = node(graph, 4, "VAELoader", vae_name=VIDEO_VAE)
    audio_vae = node(graph, 5, "VAELoader", vae_name=AUDIO_VAE)
    node(graph, 20, "KSamplerSelect", sampler_name="euler")
    node(graph, 21, "KSamplerSelect", sampler_name="euler")
    images = node(graph, 40, "VAEDecode", samples=["30", 0], vae=vae)
    audio = node(graph, 41, "VAEDecodeAudio", samples=["30", 0], vae=audio_vae)
    video = node(graph, 42, "CreateVideo", images=images, audio=audio, fps=FPS)
    node(graph, 43, "SaveVideo", video=video, filename_prefix="video/Luna_H3_Sampler", format="auto", codec="auto")
    return base, model, clip, vae, audio_vae


def reference_to_video():
    g = {}
    _base, model, clip, vae, audio_vae = common(g, UNET_R2V)
    ref = node(g, 10, "LoadImage", image="example.png")
    cond = node(g, 11, "MiniMaxH3ReferenceToVideo", clip=clip, prompt=PROMPT_R2V, width=WIDTH, height=HEIGHT,
                length=LENGTH, ref_image_size="match", vae=vae, audio_vae=audio_vae,
                **{"ref_images.ref_image_0": ref})
    negative = node(g, 12, "ConditioningZeroOut", conditioning=cond)
    node(g, 30, "Luna-H3-Sampler", **sampler_inputs(model, cond, negative, vae, ["11", 1], TURBO_R2V))
    return g


def image_to_video():
    g = {}
    _base, model, clip, vae, _audio_vae = common(g, UNET_I2V)
    first = node(g, 10, "LoadImage", image="example.png")
    cond = node(g, 11, "MiniMaxH3ImageToVideo", clip=clip, vae=vae, prompt=PROMPT_I2V, width=WIDTH, height=HEIGHT,
                length=LENGTH, first_frame=first)
    negative = node(g, 12, "ConditioningZeroOut", conditioning=cond)
    node(g, 30, "Luna-H3-Sampler", **sampler_inputs(model, cond, negative, vae, ["11", 1], TURBO_I2V))
    return g


def lora_on_pass_1():
    """Kijai's Singularity V2 LoRA on pass 1 only: LoRA on `model`, the plain model on `high_res_model`."""
    g = reference_to_video()
    pass_1 = node(g, 6, "LoraLoaderModelOnly", model=["2", 0], lora_name=V2_LORA, strength_model=1.0)
    g["30"]["inputs"]["model"] = pass_1
    g["30"]["inputs"]["high_res_model"] = ["2", 0]
    return g


DMAD = "minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors"
REFINER = "minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors"
UPSCALER_BF16 = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
PROMPT_T2V = (
    "Cinematic, natural evening light, shallow depth of field. A woman in a dark green jacket and a cream sweater "
    "stands on a city sidewalk, tucks her hair behind her ear, smiles and says: \"I knew you would come back.\" Cars "
    "and passers-by move in the soft background. Medium close-up, steady handheld camera. Audio: her warm voice, "
    "city traffic, footsteps."
)


def to_2k(width, height):
    """Text to video in two stages, tested 2026-10-29 on an RTX 4090 (Peter: ours beats Kijai's chain at 1 MP; 2 MP
    is the best quality). Stage 1 is Luna-H3-Sampler at `width` x `height`. Stage 2 is Kijai's 2K refine from his
    two-stage workflow, unchanged: ComfyUI's MiniMaxH3LatentUpscaleWithModel to 2560x1440, then DMAD + the Open
    H3-Regenerate-2K refiner LoRA (rank 16) + sol-attn sparse attention, er_sde, sigmas 0.7, 0.494118, 0. Only the
    2K video is decoded (tiled); the sound comes from stage 1. Needs ComfyUI 0.43 nightly (2026-10-28) or newer."""
    g = {}
    base = node(g, 1, "UNETLoader", unet_name=UNET_I2V, weight_dtype="default")
    chunked = node(g, 2, "MiniMaxChunkFeedForward", model=base, chunks=2, seq_threshold=4096)
    model = node(g, 7, "ModelAttentionBackend", model=chunked, attention="comfy kitchen attention")
    clip = node(g, 3, "CLIPLoader", clip_name=CLIP, type="minimax", device="default")
    vae = node(g, 4, "VAELoader", vae_name=VIDEO_VAE)
    audio_vae = node(g, 5, "VAELoader", vae_name=AUDIO_VAE)
    cond = node(g, 11, "CLIPTextEncode", clip=clip, text=PROMPT_T2V)
    negative = node(g, 12, "ConditioningZeroOut", conditioning=cond)
    latent = node(g, 10, "EmptyMiniMaxH3LatentAV", width=width, height=height, length=LENGTH)
    node(g, 20, "KSamplerSelect", sampler_name="euler")
    node(g, 21, "KSamplerSelect", sampler_name="euler")
    stage1 = node(g, 30, "Luna-H3-Sampler", **sampler_inputs(model, cond, negative, vae, latent, TURBO_I2V,
                                                              noise_seed=101))
    dmad = node(g, 50, "LoraLoaderModelOnly", model=model, lora_name=DMAD, strength_model=1.0)
    refiner = node(g, 51, "LoraLoaderModelOnly", model=dmad, lora_name=REFINER, strength_model=1.0)
    sparse = node(g, 52, "BlockSparseAttention", model=refiner, selection="sol-attn", **{
        "selection.tau": 1.3}, start_percent=0.0, end_percent=1.0, dense_blocks="", min_tokens=12288,
        extra_tokens=256, sink_conditioning="exact_kv_and_rows", verbose=False)
    guider = node(g, 53, "BasicGuider", model=sparse, conditioning=cond)
    noise = node(g, 54, "RandomNoise", noise_seed=101)
    sampler = node(g, 55, "KSamplerSelect", sampler_name="er_sde")
    sigmas = node(g, 56, "ManualSigmas", sigmas="0.7, 0.494118, 0")
    upscaler = node(g, 57, "LatentUpscaleModelLoader", model_name=UPSCALER_BF16)
    lifted = node(g, 58, "MiniMaxH3LatentUpscaleWithModel", model=upscaler, samples=stage1, width=2560, height=1440)
    stage2 = node(g, 59, "SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas,
                  latent_image=lifted)
    images = node(g, 40, "VAEDecodeTiled", samples=stage2, vae=vae, tile_size=512, overlap=64, temporal_size=64,
                  temporal_overlap=8)
    audio = node(g, 41, "VAEDecodeAudio", samples=stage1, vae=audio_vae)
    video = node(g, 42, "CreateVideo", images=images, audio=audio, fps=FPS)
    node(g, 43, "SaveVideo", video=video, filename_prefix="video/Luna_H3_Sampler_2K", format="auto", codec="auto")
    return g


WORKFLOWS = {
    "Luna-H3-Sampler - Reference to Video": reference_to_video,
    "Luna-H3-Sampler - Image to Video": image_to_video,
    "Luna-H3-Sampler - LoRA on pass 1 only": lora_on_pass_1,
    "Luna-H3-Sampler - 1 MP to 2K": lambda: to_2k(1344, 768),
    "Luna-H3-Sampler - 2 MP to 2K": lambda: to_2k(1920, 1088),
}


def installed_names(object_info):
    """Base name -> installed name for every file combo in /object_info."""
    names = {}
    for spec in object_info.values():
        for group in ("required", "optional"):
            for item in spec.get("input", {}).get(group, {}).values():
                options = item[0] if item and isinstance(item[0], list) else (
                    item[1].get("options") if len(item) > 1 and isinstance(item[1], dict) else None)
                for option in options or []:
                    if isinstance(option, str) and option.endswith(".safetensors"):
                        names.setdefault(option.replace("\\", "/").rsplit("/", 1)[-1], option)
    return names


def localise(graph, names):
    for spec in graph.values():
        for key, value in spec["inputs"].items():
            if isinstance(value, str) and value in names:
                # Luna-H3-Sampler lists LoRAs with forward slashes; core loaders use the OS separator.
                spec["inputs"][key] = (names[value].replace("\\", "/") if spec["class_type"] == "Luna-H3-Sampler"
                                       else names[value])
    return graph


def main(out_dir, object_info=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = installed_names(json.loads(Path(object_info).read_text(encoding="utf-8"))) if object_info else {}
    for title, build in WORKFLOWS.items():
        graph = localise(build(), names)
        with (out / f"{title}.api.json").open("w", encoding="utf-8") as f:
            json.dump(graph, f, indent=1)
        print(out / f"{title}.api.json")


if __name__ == "__main__":
    main(*sys.argv[1:3])

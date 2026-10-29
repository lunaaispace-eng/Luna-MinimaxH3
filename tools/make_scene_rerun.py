"""Rerun an archived long-video SelfLift scene with Luna-H3-Sampler, every other setting kept.

Scenes (the archived API graphs in Luna-Research `long-video/2026-10-25-long-video`, same template):
- `dojo2`: dojo-motion DOJO2 (original SelfLift, 2026-10-26), chained from scene 1.
- `es1`: Elira and Shadow ES1 (accepted chain v2; the s10_es12 Loop Scene take used the same recipe), fresh scene.

Changes to the graph:
- SelfLiftAvatarH3Sampler 102 -> Luna-H3-Sampler with the same values: ref2v lightx2v v0.1 rank-20 LoRA 0.6 on pass 1
  and 0.8 on pass 2 (the old chains 95 and 131), Euler both passes, simple, 12 steps, transition 9, CFG 1, the graph's
  seed, shift 12 / 3 (the model default; the old graph had no shift node), same learned upscaler. low_res_head from
  the previous scene only when the graph's continuation switch (node 2) is on.
- One model chain for the node: base 30 -> attention backend -> sol-attn (as nodes 96/97) -> MiniMaxChunkFeedForward
  2 / 4096 (pass-2 memory, as in the Luna-H3-Sampler test workflow) -> preview override (as 98). The old high pass
  used SelfLift's own global_stream path instead.
- Project folder (previous-scene latents in, this scene's latents out) -> a scratch folder (with copies of the
  previous scene's latents when chained), so archived latents are never overwritten.
- Reference images -> the archived originals copied into ComfyUI input under the graph's own names (Picture 1, 2, 3
  as in the prompt's subject definitions). Delete `input/luna_chain_refs` after the run.
- LunaAssetLoader -> Luna-H3-AssetLoader (renamed by the node merge, same inputs and outputs).

- Optional `low_unet` (2026-10-29, Peter: "Yes and document it"): pass 1 runs on that unet with the same chain
  (nodes 510-514), pass 2 on the graph's own unet through `high_res_model`; the prefix gets `_<unet stem>_low`.
  The two-fine-tune test: `dojo2` with the Singularity V2 pruned unet for pass 1, v1.3 for pass 2.
- Or `lora:<lora name>@<strength>` (2026-10-29): pass 1 = the graph's unet + that LoRA (LoraLoaderModelOnly 520, chain
  521-524), pass 2 = the plain unet through `high_res_model`. The test of Kijai's Singularity V2 LoRA on v1.3.

- Or `2k` (2026-10-29): the scene as above, then Kijai's 2K refine stage on its finished latent (`add_2k_stage`).

Usage: python make_scene_rerun.py <dojo2|es1> <scratch_dir> <out_api.json> [low_unet | lora:<name>@<s> | 2k]
"""

import json
import shutil
import sys
from pathlib import Path

LONG = Path("D:/Claude/Luna-Research/long-video/2026-10-25-long-video")
MEDIA = Path("D:/Claude/Luna-Research/media/long-video/2026-10-25-long-video")
CHARS = Path("D:/Claude/DA-Music-Videos/assets/characters")
COMFY_INPUT = Path("E:/ComfyUI-Easy-Install/ComfyUI/input")
LORA = "MiniMaxH3/minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors"
UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"

SCENES = {
    "dojo2": {
        "graph": LONG / "dojo-motion/renders/selflift_pair/scene_002_api.json",
        "latents": MEDIA / "dojo-motion/renders/selflift_pair/latents",
        "previous": "scene_001",
        "refs": {"14": LONG / "dojo-motion/bg_outside_dojo.png",
                 "149": LONG / "dojo-motion/shadow_training_sheet.png",
                 "150": LONG / "dojo-motion/flame_training_sheet.png"},
        "prefix": "Selflift/DOJO2_Luna_H3_Sampler_2MP",
    },
    "es1": {
        "graph": LONG / "elira-shadow/renders/chain_v2/scene_001_api.json",
        "latents": None,
        "previous": None,
        "refs": {"14": LONG / "elira-shadow/bg_dressing_room.png",
                 "149": CHARS / "Elira_sheet.png",
                 "150": CHARS / "The_Shadow_sheet.png"},
        "prefix": "Selflift/ES1_Luna_H3_Sampler_2MP",
    },
}


def main(name, scratch, out, low_unet=None):
    sc = SCENES[name]
    scratch = Path(scratch)
    g = json.loads(sc["graph"].read_text(encoding="utf-8"))
    chained = g["2"]["inputs"]["value"]

    (scratch / "latents").mkdir(parents=True, exist_ok=True)
    if chained:
        for ext in (".latent", ".latent.json", ".pass1.latent", ".pass1.latent.json"):
            shutil.copy2(sc["latents"] / (sc["previous"] + ext), scratch / "latents" / (sc["previous"] + ext))
    folder = scratch.as_posix()
    g["6"]["inputs"]["value"] = folder
    g["68"]["inputs"]["project_folder"] = folder
    g["69"]["inputs"]["project_folder"] = folder

    for node_id, src in sc["refs"].items():
        dst = COMFY_INPUT / g[node_id]["inputs"]["image_paths"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for node in g.values():
        if node["class_type"] == "LunaAssetLoader":
            node["class_type"] = "Luna-H3-AssetLoader"
    g["16"]["inputs"]["value"] = sc["prefix"]

    def chain(first, unet):
        g[str(first)] = {"class_type": "ModelAttentionBackend",
                         "inputs": {"model": [unet, 0], "attention": "comfy kitchen attention"}}
        g[str(first + 1)] = {"class_type": "BlockSparseAttention",
                             "inputs": dict(g["97"]["inputs"], model=[str(first), 0])}
        g[str(first + 2)] = {"class_type": "MiniMaxChunkFeedForward",
                             "inputs": {"model": [str(first + 1), 0], "chunks": 2, "seq_threshold": 4096}}
        g[str(first + 3)] = {"class_type": "ModelPreviewOverrideKJ",
                             "inputs": dict(g["98"]["inputs"], model=[str(first + 2), 0])}
        return [str(first + 3), 0]

    model = high_model = chain(500, "30")
    if low_unet and low_unet.startswith("lora:"):
        name, strength = low_unet[5:].rsplit("@", 1)
        g["520"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": ["30", 0], "lora_name": name, "strength_model": float(strength)}}
        model = chain(521, "520")
        g["16"]["inputs"]["value"] = (sc["prefix"] + "_" + Path(name.replace("\\", "/")).stem + "_"
                                      + strength.replace(".", "") + "_low")
    elif low_unet and low_unet != "2k":
        g["510"] = {"class_type": "UNETLoader", "inputs": dict(g["30"]["inputs"], unet_name=low_unet)}
        model = chain(511, "510")
        g["16"]["inputs"]["value"] = sc["prefix"] + "_" + Path(low_unet.replace("\\", "/")).stem + "_low"

    old = g.pop("102")["inputs"]
    g["102"] = {"class_type": "Luna-H3-Sampler", "inputs": {
        "model": model, "positive": old["positive"], "negative": old["negative"], "vae": old["vae"],
        "latent_image": old["latent_image"],
        "lora_low": LORA, "strength_low": 0.6, "shift_low": 12.0, "audio_shift_low": 3.0, "sampler_low": ["100", 0],
        "cfg_low": 1.0,
        "lora_high": LORA, "strength_high": 0.8, "shift_high": 12.0, "audio_shift_high": 3.0,
        "sampler_high": ["100", 0], "cfg_high": 1.0,
        "scheduler": "simple", "total_steps": 12, "transition": 9, "highres_denoise": 1.0,
        "noise_seed": old["seed"], "lowres_scale": old["lowres_scale"], "upscaler_model": UPSCALER,
        "rho": old["rho"], "w_min": old["w_min"], "w_max": old["w_max"]}}
    if chained:
        g["102"]["inputs"]["low_res_head"] = old["low_res_head"]
    if low_unet and low_unet != "2k":
        g["102"]["inputs"]["high_res_model"] = high_model
    del g["99"]
    if low_unet == "2k":
        add_2k_stage(g, old["seed"])
        g["16"]["inputs"]["value"] = sc["prefix"] + "_to2K"

    Path(out).write_text(json.dumps(g, indent=1), encoding="utf-8")


def add_2k_stage(g, seed):
    """`2k` (2026-10-29, Peter: "Full chain, 2mp to 2k so we have it as reference for the normal 2 k"): after the
    scene's finished latent (VRGDG save node 105), Kijai's 2K refine stage as in his two-stage workflow:
    MiniMaxH3LatentUpscaleWithModel to 2560x1440, the scene's unet + DMAD + Open-H3-Regenerate-2K refiner rank 16 +
    attention backend + chunked feed-forward + sol-attn, er_sde, sigmas 0.7, 0.494118, 0. The reference conditioning
    is encoded again at 2560x1440 (a copy of MiniMaxH3ReferenceToVideo 66), because the reference latents are sized to
    the video. The video is decoded tiled from the 2K latent; the sound stays the scene's (stage 1)."""
    g["620"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": dict(g["66"]["inputs"], width=2560, height=1440)}
    g["630"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["30", 0], "lora_name": "MiniMaxH3\\minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors",
        "strength_model": 1.0}}
    g["631"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["630", 0],
        "lora_name": "MiniMaxH3\\minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors", "strength_model": 1.0}}
    g["632"] = {"class_type": "ModelAttentionBackend", "inputs": {"model": ["631", 0],
                                                                  "attention": "comfy kitchen attention"}}
    g["633"] = {"class_type": "MiniMaxChunkFeedForward", "inputs": {"model": ["632", 0], "chunks": 2,
                                                                    "seq_threshold": 4096}}
    g["634"] = {"class_type": "BlockSparseAttention", "inputs": {
        "model": ["633", 0], "selection": "sol-attn", "selection.tau": 1.3, "start_percent": 0.0, "end_percent": 1.0,
        "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows",
        "verbose": False}}
    g["640"] = {"class_type": "BasicGuider", "inputs": {"model": ["634", 0], "conditioning": ["620", 0]}}
    g["641"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}
    g["642"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}}
    g["643"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.7, 0.494118, 0"}}
    g["644"] = {"class_type": "LatentUpscaleModelLoader",
                "inputs": {"model_name": "minimax_h3_latent_upscaler_3d_bf16.safetensors"}}
    g["645"] = {"class_type": "MiniMaxH3LatentUpscaleWithModel",
                "inputs": {"model": ["644", 0], "samples": ["105", 0], "width": 2560, "height": 1440}}
    g["646"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["641", 0], "guider": ["640", 0], "sampler": ["642", 0], "sigmas": ["643", 0],
        "latent_image": ["645", 0]}}
    g["650"] = {"class_type": "VAEDecodeTiled", "inputs": {
        "samples": ["646", 0], "vae": ["32", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64,
        "temporal_overlap": 8}}
    g["108"]["inputs"]["images"] = ["650", 0]
    g.pop("107", None)  # the 2 MP decode is not needed (RAM)


if __name__ == "__main__":
    main(*sys.argv[1:5])

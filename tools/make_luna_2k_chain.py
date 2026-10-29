"""Our sampler in Kijai's 2K chain: Luna-H3-Sampler at 2 MP, then Kijai's latent upscale to 2560x1440 and his
refiner stage 2, unchanged.

Peter, 2026-10-29: "until it runs what do we do to use our sampler ... i have a feeling it will be better".
Starts from Kijai's two-stage workflow as an API graph (`ui_to_api.py` of
`D:\\Claude\\Luna-Research\\h3\\2026-10-29-kijai2stage_compare.json`) and replaces only his stage 1:
- `EmptyMiniMaxH3LatentAV` becomes 1920x1088x124 (our 2 MP size); the text encoder becomes our nvfp4 one (Peter,
  2026-10-29: "Ok, proceed with recomandation");
- Luna-H3-Sampler with the measured 2 MP recipe (turbo LoRA 0.6 / 0.8, Euler, simple, 12 steps, transition 9,
  shift 12 / 3, learned upscaler fp16, seed 101) on the model chain before Kijai's DMAD LoRA
  (UNETLoader -> MiniMaxChunkFeedForward -> ModelAttentionBackend);
- its latent feeds his VAEDecode / VAEDecodeAudio (stage-1 video and the sound) and `MiniMaxH3LatentUpscaleWithModel`.
Stage 2 (DMAD + refiner rank 16 + sol-attn, er_sde, sigmas node `sigmas`) stays as Kijai made it.

Usage: python make_luna_2k_chain.py <kijai_api.json> <out_api.json> [sigmas_node=150] [object_info.json] [lean]
`lean`: only the 2K video is decoded (VAEDecodeTiled) and saved. A 7th argument `WxH` sets the first clip's size
(default 1920x1088; 1664x928 = 1.54 MP for the 1.5 MP test).
"""

import json
import sys
from pathlib import Path

TURBO_FL2V = "MiniMaxH3/minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors"
UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
# Our recipe's text encoder (15.7 GB). Kijai's int8 one (27 GB) left 0.5 GB of RAM free on the 0.909 run.
TEXT_ENCODER = "MiniMaxH3\\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"


def build(g, sigmas_node="150", lean=False, size=(1920, 1088)):
    g = json.loads(json.dumps(g))
    for nid in ("14", "16", "9", "120", "121"):  # Kijai's stage-1 sampler, guider, scheduler and sigma preview
        g.pop(nid, None)
    g["80"]["inputs"].update(width=size[0], height=size[1], length=124)
    g["13"]["inputs"]["clip_name"] = TEXT_ENCODER
    g["200"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["110", 1]}}
    g["201"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["202"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["210"] = {"class_type": "Luna-H3-Sampler", "inputs": dict(
        model=["126", 0], positive=["110", 1], negative=["200", 0], vae=["11", 0], latent_image=["80", 0],
        lora_low=TURBO_FL2V, strength_low=0.6, shift_low=12.0, audio_shift_low=3.0, sampler_low=["201", 0],
        cfg_low=1.0, lora_high=TURBO_FL2V, strength_high=0.8, shift_high=12.0, audio_shift_high=3.0,
        sampler_high=["202", 0], cfg_high=1.0, scheduler="simple", total_steps=12, transition=9,
        highres_denoise=1.0, noise_seed=101, lowres_scale=0.5, upscaler_model=UPSCALER, rho=0.0, w_min=0.5,
        w_max=1.0)}
    for nid, key in (("10", "samples"), ("23", "samples"), ("138", "samples")):
        g[nid]["inputs"][key] = ["210", 0]
    g["135"]["inputs"]["sigmas"] = [sigmas_node, 0]
    for nid, prefix in (("146", "video/luna2k_2mp"), ("113", "video/luna2k_2560"), ("149", "video/luna2k_compare")):
        g[nid]["inputs"]["filename_prefix"] = prefix
    if lean:
        # Only the 2K video is decoded (tiled) and saved: no 2 MP decode/save, no stacked comparison
        # (Peter, 2026-10-29: "Yes" to measuring RAM with a lean graph).
        for nid in ("10", "21", "145", "146", "147", "148", "149"):
            g.pop(nid, None)
        g["141"] = {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["135", 0], "vae": ["11", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64,
            "temporal_overlap": 8}}
        g["113"]["inputs"]["filename_prefix"] = "video/luna2k_lean_2560"
        if size != (1920, 1088):
            g["113"]["inputs"]["filename_prefix"] = f"video/luna2k_lean_from{size[0]}x{size[1]}_2560"
    return g


def check(graph, object_info):
    """Offline check against /object_info: known classes, required inputs present, links to existing nodes and
    outputs, combo values in their options. Returns the problems."""
    problems = []
    for nid, spec in graph.items():
        info = object_info.get(spec["class_type"])
        if info is None:
            problems.append(f"{nid}: unknown class {spec['class_type']}")
            continue
        declared = {**info["input"].get("required", {}), **info["input"].get("optional", {})}
        for key in info["input"].get("required", {}):
            if key not in spec["inputs"]:
                problems.append(f"{nid} {spec['class_type']}: missing {key}")
        for key, value in spec["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                src = graph.get(value[0])
                if src is None:
                    problems.append(f"{nid}.{key}: link to missing node {value[0]}")
                elif value[1] >= len(object_info[src["class_type"]]["output"]):
                    problems.append(f"{nid}.{key}: output {value[1]} of {src['class_type']} does not exist")
            elif key in declared:
                kind = declared[key][0]
                options = kind if isinstance(kind, list) else (
                    declared[key][1].get("options") if kind == "COMBO" and len(declared[key]) > 1 else None)
                if options is not None and value not in options:
                    problems.append(f"{nid}.{key}: {value!r} not among the options")
    return problems


def main(src, out, sigmas_node="150", object_info=None, lean=None, size="1920x1088"):
    width, height = (int(v) for v in size.lower().split("x"))
    graph = build(json.loads(Path(src).read_text(encoding="utf-8")), sigmas_node, lean == "lean", (width, height))
    with Path(out).open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=1)
    print(f"{len(graph)} nodes -> {out}")
    if object_info:
        problems = check(graph, json.loads(Path(object_info).read_text(encoding="utf-8")))
        print("check:", "ok" if not problems else "")
        for line in problems:
            print("  ", line)


if __name__ == "__main__":
    main(*sys.argv[1:7])

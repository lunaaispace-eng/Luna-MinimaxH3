"""Build the Luna-H3-Sampler test workflow from Peter's repaired "H3 SelfLift audio switch - Luna" workflow.

The original SelfLift chain (LoraLoaderModelOnly 56, MiniMaxH3SigmaShift 151, BasicScheduler 51,
SelfLiftAvatarH3Sampler 149) is replaced by one Luna-H3-Sampler node with the Luna workflow's own values: fl2v turbo
4-step 768p LoRA 0.8 on both arms, shift 12 / 3, Euler (KSamplerSelect 50) on both passes, simple, 12 steps with
transition 9, CFG 1, the workflow's seed, learned upscaler minimax_h3_latent_upscaler_3d_fp16. MiniMaxChunkFeedForward
(58) is switched on for pass 2's memory, as in Snow. Everything else (prompt, Elira RefMods, native sound, sol-attn,
decode, save) stays as in the Luna workflow. Node titles are left as they are.

Usage: python make_luna_test_workflow.py <Luna.json> <out.json> [seed]
"""

import json
import sys
from pathlib import Path

LORA = "MiniMaxH3/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
REMOVE = {56, 151, 51, 149}


def main(src, dst, seed=598558112803321):
    wf = json.loads(Path(src).read_text(encoding="utf-8"))
    links = {l[0]: l for l in wf["links"]}
    for lid, l in list(links.items()):
        if l[1] in REMOVE or l[3] in REMOVE:
            del links[lid]
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in REMOVE]
    nodes = {n["id"]: n for n in wf["nodes"]}
    for n in wf["nodes"]:
        for i in n.get("inputs", []):
            if i.get("link") is not None and i["link"] not in links:
                i["link"] = None
        for o in n.get("outputs", []):
            if o.get("links"):
                o["links"] = [x for x in o["links"] if x in links]
    nodes[58]["mode"] = 0

    next_link = max([0] + list(links)) + 1

    def slot(node_id, name, outputs=False):
        items = nodes[node_id]["outputs" if outputs else "inputs"]
        return next(i for i, s in enumerate(items) if s["name"] == name or s.get("type") == name)

    def link(src_id, src_slot, dst_id, dst_slot, kind):
        nonlocal next_link
        lid = next_link
        next_link += 1
        links[lid] = [lid, src_id, src_slot, dst_id, dst_slot, kind]
        out = nodes[src_id]["outputs"][src_slot]
        out["links"] = (out.get("links") or []) + [lid]
        nodes[dst_id]["inputs"][dst_slot]["link"] = lid

    selflift = 400
    x, y = nodes[62]["pos"]
    nodes[selflift] = {
        "id": selflift, "type": "Luna-H3-Sampler", "pos": [x + 450, y], "size": [420, 800], "flags": {}, "order": 0,
        "mode": 0,
        "inputs": [{"name": n, "type": t, "link": None} for n, t in (
            ("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("vae", "VAE"),
            ("latent_image", "LATENT"), ("sampler_low", "SAMPLER"), ("sampler_high", "SAMPLER"))],
        "outputs": [{"name": "latent", "type": "LATENT", "links": []},
                    {"name": "low_res_latent", "type": "LATENT", "links": []}],
        "properties": {"Node name for S&R": "Luna-H3-Sampler"},
        "widgets_values": [LORA, 0.8, 12.0, 3.0, 1.0, LORA, 0.8, 12.0, 3.0, 1.0,
                           "simple", 12, 9, 1.0, int(seed), "fixed", 0.5, UPSCALER, 0.0, 0.5, 1.0],
    }
    wf["nodes"].append(nodes[selflift])

    link(55, 0, 57, slot(57, "model"), "MODEL")      # UNET -> attention backend (the LoRA moves into the node)
    link(59, 0, 113, slot(113, "model"), "MODEL")    # chunk FF / low-VRAM chain -> sol-attn (the shift moves in)
    link(62, 0, selflift, 0, "MODEL")
    link(104, 0, selflift, 1, "CONDITIONING")
    link(77, 0, selflift, 2, "CONDITIONING")
    link(53, 0, selflift, 3, "VAE")
    link(121, 0, selflift, 4, "LATENT")
    link(50, 0, selflift, 5, "SAMPLER")
    link(50, 0, selflift, 6, "SAMPLER")
    link(selflift, 0, 63, slot(63, "av_latent"), "LATENT")

    wf["links"] = sorted(links.values(), key=lambda l: l[0])
    wf["last_link_id"] = max(links)
    wf["last_node_id"] = max(nodes)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:3], *(sys.argv[3:4]))

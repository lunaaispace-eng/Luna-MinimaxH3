"""Build the first Luna-H3-Sampler GPU test workflow from Peter's Snow workflow.

Snow's sampler chain (MiniMaxH3SigmaShift, Power Lora Loader, BasicScheduler, BasicGuider, RandomNoise,
SamplerCustomAdvanced) is replaced by one Luna-H3-Sampler node with Snow's own values: DMAD 4-step LoRA 0.8 on both
arms, shift 12 / 3, er_sde wrapped by Detail Daemon on both passes, simple, 12 steps = 8 low + 4 high, CFG 1, seed 86.
Everything else stays as in Snow; node ids renamed by the 2026-10-28 merge are updated (Luna-MinimaxH3).
Node titles are left as they are (Peter: original node names).

Usage: python make_test_workflow.py <Snow.json> <out.json>
"""

import copy
import json
import sys
from pathlib import Path

DMAD = "MiniMaxH3/minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors"
REMOVE = {221, 302, 124, 126, 129, 125}
RENAMED = {"LunaMiniMaxH3Canvas": "Luna-H3-Canvas", "LunaAssetLoader": "Luna-H3-AssetLoader",
           "LunaShotList": "Luna-H3-ShotList", "SaveVideoSimple": "Luna-SaveVideo"}


def main(src, dst):
    wf = json.loads(Path(src).read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf["links"]}
    snow = {"vae_get": 259, "model_get": 263, "attn": 223, "cond": 307, "latent": 136, "sampler": 260,
            "decode": 234, "decode_audio": 235, "save": 311}

    # Drop the replaced nodes and every link touching them.
    for lid, l in list(links.items()):
        if l[1] in REMOVE or l[3] in REMOVE:
            del links[lid]
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in REMOVE]
    nodes = {n["id"]: n for n in wf["nodes"]}
    for n in wf["nodes"]:
        n["type"] = RENAMED.get(n["type"], n["type"])
        for i in n.get("inputs", []):
            if i.get("link") is not None and i["link"] not in links:
                i["link"] = None
        for o in n.get("outputs", []):
            if o.get("links"):
                o["links"] = [x for x in o["links"] if x in links]

    next_link = max([0] + list(links)) + 1
    selflift, zero, vae_get = 400, 401, 402

    def link(src_id, src_slot, dst_id, dst_slot, kind):
        nonlocal next_link
        lid = next_link
        next_link += 1
        links[lid] = [lid, src_id, src_slot, dst_id, dst_slot, kind]
        out = nodes[src_id]["outputs"][src_slot]
        out["links"] = (out.get("links") or []) + [lid]
        nodes[dst_id]["inputs"][dst_slot]["link"] = lid
        return lid

    x, y = nodes[snow["latent"]]["pos"]
    nodes[selflift] = {
        "id": selflift, "type": "Luna-H3-Sampler", "pos": [x + 450, y], "size": [420, 760], "flags": {}, "order": 0,
        "mode": 0,
        "inputs": [{"name": n, "type": t, "link": None} for n, t in (
            ("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("vae", "VAE"),
            ("latent_image", "LATENT"), ("sampler_low", "SAMPLER"), ("sampler_high", "SAMPLER"))],
        "outputs": [{"name": "latent", "type": "LATENT", "links": []},
                    {"name": "low_res_latent", "type": "LATENT", "links": []}],
        "properties": {"Node name for S&R": "Luna-H3-Sampler"},
        "widgets_values": [DMAD, 0.8, 12.0, 3.0, 1.0, DMAD, 0.8, 12.0, 3.0, 1.0,
                           "simple", 12, 8, 1.0, 86, "fixed", 0.5, 0.0, 0.5, 1.0],
    }
    nodes[zero] = {
        "id": zero, "type": "ConditioningZeroOut", "pos": [x + 150, y + 420], "size": [250, 30], "flags": {},
        "order": 0, "mode": 0, "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": None}],
        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}],
        "properties": {"Node name for S&R": "ConditioningZeroOut"}, "widgets_values": []}
    get = copy.deepcopy(nodes[snow["vae_get"]])
    get.update({"id": vae_get, "pos": [x + 150, y + 500]})
    get["outputs"][0]["links"] = []
    nodes[vae_get] = get
    wf["nodes"] += [nodes[selflift], nodes[zero], nodes[vae_get]]

    attn_in = next(i for i, s in enumerate(nodes[snow["attn"]]["inputs"]) if s["name"] == "model")
    link(snow["model_get"], 0, snow["attn"], attn_in, "MODEL")
    link(snow["attn"], 0, selflift, 0, "MODEL")
    link(snow["cond"], 0, selflift, 1, "CONDITIONING")
    link(snow["cond"], 0, zero, 0, "CONDITIONING")
    link(zero, 0, selflift, 2, "CONDITIONING")
    link(vae_get, 0, selflift, 3, "VAE")
    latent_out = next(i for i, o in enumerate(nodes[snow["latent"]]["outputs"]) if o["type"] == "LATENT")
    link(snow["latent"], latent_out, selflift, 4, "LATENT")
    link(snow["sampler"], 0, selflift, 5, "SAMPLER")
    link(snow["sampler"], 0, selflift, 6, "SAMPLER")
    for dec in (snow["decode"], snow["decode_audio"]):
        slot = next(i for i, s in enumerate(nodes[dec]["inputs"]) if s["name"] == "samples")
        link(selflift, 0, dec, slot, "LATENT")
    nodes[snow["save"]]["widgets_values"][8] = "Luna_H3_SelfLift"

    wf["links"] = sorted(links.values(), key=lambda l: l[0])
    wf["last_link_id"] = max(links)
    wf["last_node_id"] = max(nodes)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

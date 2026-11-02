"""Make `H3 Luna Sampler_Chain.json` from Peter's `H3 Luna Sampler_Main.json`: Main plus the clip chain, nothing else changed.

    E:\\ComfyUI-Easy-Install\\python_embeded\\python.exe -I tools\\make_chain_workflow.py [--force]

Reads Main, never writes it. Adds a VHS Meta Batch Manager, Luna-H3-Chain, -ChainContinue, -ChainRemember, -ChainJoin
and a VHS Video Combine in a group "11 · Chain", and rewires:
  Canvas frames -> Batch Manager frames_per_batch -> Chain meta_batch (and Video Combine meta_batch)
  Chain song_start, shot -> Asset Loader song_start, shot; Chain seed -> Sampler noise_seed (the Seed node stays, unwired)
  "If/Else: latent" -> ChainContinue latent -> Sampler latent_image; ChainContinue low_res_head -> Sampler low_res_head
  Sampler latent, low_res_latent -> ChainRemember -> ChainJoin saved
  Watermark images and "If/Else: saved audio" -> ChainJoin -> Video Combine
Luna-SaveVideo keeps its wires, so every clip is also saved whole (the raw take, head included).
Node ids of Main are found by type and title, not by number. Prints every change.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

WORKFLOWS = Path("E:/ComfyUI-Easy-Install/ComfyUI/user/default/workflows")
MAIN = WORKFLOWS / "H3 Luna Sampler_Main.json"
OUT = WORKFLOWS / "H3 Luna Sampler_Chain.json"


def one(nodes, type_, title=None):
    found = [n for n in nodes if n["type"] == type_ and (title is None or n.get("title") == title)]
    if len(found) != 1:
        sys.exit(f"Main: expected one {type_} {title or ''}, found {len(found)}")
    return found[0]


def socket(name, type_, link=None, optional=False):
    s = {"localized_name": name, "name": name, "type": type_, "link": link}
    if optional:
        s["shape"] = 7
    return s


def widget_in(name, type_):
    return {"localized_name": name, "name": name, "type": type_, "widget": {"name": name}, "link": None}


def out(name, type_):
    return {"localized_name": name, "name": name, "type": type_, "links": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing H3 Luna Sampler_Chain.json")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        sys.exit(f"{OUT} exists; pass --force to overwrite it")

    wf = json.loads(MAIN.read_text(encoding="utf-8"))
    nodes, links = wf["nodes"], wf["links"]
    canvas = one(nodes, "Luna-H3-Canvas")
    loader = one(nodes, "Luna-H3-AssetLoader")
    sampler = one(nodes, "Luna-H3-Sampler")
    latent_sw = one(nodes, "ComfySwitchNode", "If/Else: latent")
    audio_sw = one(nodes, "ComfySwitchNode", "If/Else: saved audio")
    mark = one(nodes, "Luna-Watermark")
    song_on = one(nodes, "PrimitiveBoolean", "Audio: ON = external song, OFF = native")
    card = json.loads(loader["widgets_values"][3])["audio"][0]
    start = card["start"]
    if card.get("position") == "shot":
        # The card's window in "By shot number" mode: shot n starts at song_start + (n - 1) x (length - overlap / 24).
        seconds = canvas["widgets_values"][6]
        start = card.get("song_start", 0.0) + (max(1, int(card.get("shot", 1))) - 1) * (
            seconds - int(card.get("overlap", 22)) / 24)
    if any(n["mode"] == 0 for n in nodes if n["type"] == "Luna-RTXFrameGen") and \
            one(nodes, "PrimitiveBoolean", "Interpolation")["widgets_values"][0]:
        sys.exit("Main has Interpolation on: frame generation changes the frame count the chain cuts. Turn it off first.")

    next_node = max(n["id"] for n in nodes) + 1
    next_link = max([l[0] for l in links] + [wf.get("last_link_id", 0)]) + 1
    by_id = {n["id"]: n for n in nodes}
    log = []

    def new_id():
        nonlocal next_node
        next_node += 1
        return next_node - 1

    def unlink(node, name):
        inp = next(i for i in node["inputs"] if i["name"] == name)
        lid = inp["link"]
        if lid is None:
            return
        l = next(l for l in links if l[0] == lid)
        src = by_id[l[1]]["outputs"][l[2]]
        src["links"] = [x for x in (src.get("links") or []) if x != lid]
        links.remove(l)
        inp["link"] = None
        log.append(f"unwired {by_id[l[1]]['type']} #{l[1]} -> {node['type']} #{node['id']}.{name}")

    def link(src, slot, dst, name):
        nonlocal next_link
        inp = next(i for i in dst["inputs"] if i["name"] == name)
        if inp["link"] is not None:
            unlink(dst, name)
        lid = next_link
        next_link += 1
        type_ = src["outputs"][slot]["type"]
        links.append([lid, src["id"], slot, dst["id"], dst["inputs"].index(inp), type_])
        src["outputs"][slot]["links"] = (src["outputs"][slot].get("links") or []) + [lid]
        inp["link"] = lid
        log.append(f"wired {src['type']} #{src['id']}.{src['outputs'][slot]['name']} -> {dst['type']} #{dst['id']}.{name}")

    # Under the "10 · Save" group, clear of every node of Main.
    save_group = next(gr for gr in wf.get("groups", []) if gr.get("title", "").startswith("10"))
    x0, y0 = save_group["bounding"][0] + 20, save_group["bounding"][1] + save_group["bounding"][3] + 80

    def add(type_, inputs, outputs, widgets, pos, size, title=None):
        n = {"id": new_id(), "type": type_, "pos": list(pos), "size": list(size), "flags": {}, "order": 0, "mode": 0,
             "inputs": inputs, "outputs": outputs, "properties": {"Node name for S&R": type_},
             "widgets_values": widgets}
        if title:
            n["title"] = title
        nodes.append(n)
        by_id[n["id"]] = n
        log.append(f"added {type_} #{n['id']}")
        return n

    manager = add("VHS_BatchManager", [widget_in("frames_per_batch", "INT")], [out("meta_batch", "VHS_BatchManager")],
                  {"frames_per_batch": 362, "count": 1}, (x0, y0 + 20), (380, 58), "VHS Meta Batch Manager (frames = Canvas)")
    chain = add("Luna-H3-Chain",
                [socket("meta_batch", "VHS_BatchManager"), socket("first_prompt", "STRING", optional=True),
                 socket("later_prompt", "STRING", optional=True), widget_in("clips", "INT"), widget_in("seed", "INT"),
                 widget_in("overlap", "COMBO"), widget_in("song_locked", "BOOLEAN"), widget_in("song_offset", "FLOAT"),
                 widget_in("clips_already_made", "INT")],
                [out("chain", "LUNA_H3_CHAIN"), out("seed", "INT"), out("song_start", "FLOAT"), out("shot", "INT"),
                 out("overlap", "INT"), out("frames", "INT"), out("seconds", "FLOAT"), out("prompt", "STRING"),
                 out("is_first_clip", "BOOLEAN")],
                [3, 12345, "fixed", "22" if song_on["widgets_values"][0] else "39", bool(song_on["widgets_values"][0]),
                 float(start), 0],
                (x0, y0 + 120), (380, 362))
    cont = add("Luna-H3-ChainContinue",
               [socket("chain", "LUNA_H3_CHAIN"), socket("latent", "LATENT"), socket("first_image", "IMAGE", optional=True),
                socket("vae", "VAE", optional=True), widget_in("name", "STRING"), widget_in("keep_look_of_clip_1", "BOOLEAN")],
               [out("latent", "LATENT"), out("low_res_head", "LATENT"), out("start_image", "IMAGE")],
               ["chain", True], (x0 + 420, y0 + 120), (330, 142))
    rem = add("Luna-H3-ChainRemember",
              [socket("chain", "LUNA_H3_CHAIN"), socket("latent", "LATENT"),
               socket("low_res_latent", "LATENT", optional=True), widget_in("name", "STRING")],
              [out("saved", "LUNA_H3_CHAIN_SAVED")], ["chain"], (x0 + 420, y0 + 310), (330, 100))
    join = add("Luna-H3-ChainJoin",
               [socket("chain", "LUNA_H3_CHAIN"), socket("images", "IMAGE"), socket("audio", "AUDIO"),
                socket("saved", "LUNA_H3_CHAIN_SAVED")],
               [out("images", "IMAGE"), out("audio", "AUDIO")], [], (x0 + 420, y0 + 460), (330, 100))
    combine = add("VHS_VideoCombine",
                  [socket("images", "IMAGE"), socket("audio", "AUDIO", optional=True),
                   socket("meta_batch", "VHS_BatchManager", optional=True), socket("vae", "VAE", optional=True)]
                  + [widget_in(n, t) for n, t in (("frame_rate", "FLOAT"), ("loop_count", "INT"),
                                                  ("filename_prefix", "STRING"), ("format", "COMBO"),
                                                  ("pingpong", "BOOLEAN"), ("save_output", "BOOLEAN"))],
                  [out("Filenames", "VHS_FILENAMES")],
                  {"frame_rate": 24, "loop_count": 0, "filename_prefix": "h3-chain/H3_Luna_Sampler_Chain",
                   "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 14, "save_metadata": True,
                   "trim_to_audio": False, "pingpong": False, "save_output": True},
                  (x0 + 800, y0 + 120), (320, 330))

    canvas_frames = next(i for i, o in enumerate(canvas["outputs"]) if o["name"] == "frames")
    link(canvas, canvas_frames, manager, "frames_per_batch")
    link(manager, 0, chain, "meta_batch")
    link(manager, 0, combine, "meta_batch")
    link(chain, 2, loader, "song_start")
    link(chain, 3, loader, "shot")
    link(chain, 1, sampler, "noise_seed")
    link(chain, 0, cont, "chain")
    link(latent_sw, 0, cont, "latent")
    link(cont, 0, sampler, "latent_image")
    link(cont, 1, sampler, "low_res_head")
    link(chain, 0, rem, "chain")
    link(sampler, 0, rem, "latent")
    link(sampler, 1, rem, "low_res_latent")
    link(chain, 0, join, "chain")
    link(mark, 0, join, "images")
    link(audio_sw, 0, join, "audio")
    link(rem, 0, join, "saved")
    link(join, 0, combine, "images")
    link(join, 1, combine, "audio")

    wf.setdefault("groups", []).append({"title": "11 · Chain", "bounding": [x0 - 20, y0 - 40, 1180, 600],
                                        "color": "#3f789e", "font_size": 24, "flags": {}})
    wf["last_node_id"] = next_node - 1
    wf["last_link_id"] = next_link - 1
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False)
    print("\n".join(log))
    print(f"song_offset {start} s (Main's song start), song_locked {song_on['widgets_values'][0]}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()

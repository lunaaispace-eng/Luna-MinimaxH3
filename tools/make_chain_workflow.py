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

    E:\\ComfyUI-Easy-Install\\python_embeded\\python.exe -I tools\\make_chain_workflow.py --add-clip-saves

Stop and continue (plans\\chain-stop-continue.md): adds to the existing H3 Luna Sampler_Chain.json, keeping Peter's own edits
(a fresh build adds the same): a second Luna-SaveVideo "Clip video", ChainJoin images + clip_audio -> one finished video per
clip, `clip_00001.mp4` ..., in the chain's own folder `output/luna_h3_chain/chain` (cleared with the endings at a fresh clip 1);
and ChainRemember save_whole_clip on. The file is copied to the pack's git-ignored `.work\\<name>.backup-<time>.json` first.

    E:\\ComfyUI-Easy-Install\\python_embeded\\python.exe -I tools\\make_chain_workflow.py --song-on-loader

The song settings on the Asset Loader (2026-11-03, Peter: "it needs to be the other way around, set it in asset loader"):
moves an existing H3 Luna Sampler_Chain.json to the new layout (a fresh build ends in the same one). Luna-H3-Chain loses
overlap and song_offset (outputs: chain, seed, clip, frames, seconds, prompt, is_first_clip); Chain chain -> Asset Loader
chain replaces the song_start and shot wires; Asset Loader overlap -> ChainContinue, ChainRemember, ChainJoin overlap. The
sound card goes to 'By shot number' (from its own start when it was free) and keeps its Start and Shot; an overlap the chain
cannot use becomes 22. Backup first, as above.
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


CLIP_FOLDER = "luna_h3_chain/chain"


def named(node, *drop, **values):
    """Keep the frontend's by-name copy of the widget values (widgets_values_named) in step with widgets_values: the
    frontend restores from it when it is there, so a value changed only positionally would come back as it was."""
    if "widgets_values_named" not in node:
        return
    for k in drop:
        node["widgets_values_named"].pop(k, None)
    node["widgets_values_named"].update(values)


def add_clip_saves(wf):
    """Join's clip_audio output, Remember's save_whole_clip on, and a Luna-SaveVideo per clip. Returns the change log."""
    nodes, links = wf["nodes"], wf["links"]
    by_id = {n["id"]: n for n in nodes}
    log = []
    if any(n.get("title") == "Clip video" for n in nodes):
        sys.exit("the workflow already has the Clip video node")
    join = one(nodes, "Luna-H3-ChainJoin")
    rem = one(nodes, "Luna-H3-ChainRemember")
    raw = one(nodes, "Luna-SaveVideo")
    canvas = one(nodes, "Luna-H3-Canvas")
    next_node = max(n["id"] for n in nodes) + 1
    next_link = max([l[0] for l in links] + [wf.get("last_link_id", 0)]) + 1

    if not any(o["name"] == "clip_audio" for o in join["outputs"]):
        join["outputs"].append(out("clip_audio", "AUDIO"))
        log.append(f"Luna-H3-ChainJoin #{join['id']}: output clip_audio")
    if not any(i["name"] == "save_whole_clip" for i in rem["inputs"]):
        rem["inputs"].append(widget_in("save_whole_clip", "BOOLEAN"))
    rem["widgets_values"] = [rem["widgets_values"][0], True]
    named(rem, save_whole_clip=True)
    log.append(f"Luna-H3-ChainRemember #{rem['id']}: save_whole_clip on")

    save = copy.deepcopy(raw)
    save["id"], save["title"] = next_node, "Clip video"
    save["pos"] = [join["pos"][0] + 380, join["pos"][1]]
    for i in save["inputs"]:
        i["link"] = None
    for o in save["outputs"]:
        o["links"] = []
    names = [i["name"] for i in save["inputs"] if i.get("widget")]
    values = dict(zip(names, save["widgets_values"]))
    values.update({"quality": 10, "filename_prefix": "clip", "output_path": CLIP_FOLDER, "save_output": True,
                   "crop_to_audio": False, "audio_codec": "AAC", "audio_bitrate": "320k", "save_first_frame": False,
                   "save_last_frame": False, "save_prompt": False})
    save["widgets_values"] = [values[n] for n in names] + save["widgets_values"][len(names):]
    named(save, **values)
    nodes.append(save)
    by_id[save["id"]] = save
    log.append(f"added Luna-SaveVideo #{save['id']} \"Clip video\": {CLIP_FOLDER}/clip_NNNNN.mp4, quality 10, AAC 320k")

    def link(src, slot, dst, name):
        nonlocal next_link
        inp = next(i for i in dst["inputs"] if i["name"] == name)
        links.append([next_link, src["id"], slot, dst["id"], dst["inputs"].index(inp), src["outputs"][slot]["type"]])
        src["outputs"][slot]["links"] = (src["outputs"][slot].get("links") or []) + [next_link]
        inp["link"] = next_link
        log.append(f"wired {src['type']} #{src['id']}.{src['outputs'][slot]['name']} -> {dst['type']} #{dst['id']}.{name}")
        next_link += 1

    rate_link = next(i["link"] for i in raw["inputs"] if i["name"] == "frame_rate")
    rate_src = next(l for l in links if l[0] == rate_link) if rate_link is not None else None
    link(join, 0, save, "images")
    link(join, next(k for k, o in enumerate(join["outputs"]) if o["name"] == "clip_audio"), save, "audio")
    if rate_src is not None:
        link(by_id[rate_src[1]], rate_src[2], save, "frame_rate")
    else:
        link(canvas, next(k for k, o in enumerate(canvas["outputs"]) if o["name"] == "fps"), save, "frame_rate")
    for gr in wf.get("groups", []):
        if gr.get("title") == "11 · Chain":
            gr["bounding"][2] = max(gr["bounding"][2], save["pos"][0] + 420 - gr["bounding"][0])
    wf["last_node_id"] = max(n["id"] for n in nodes)
    wf["last_link_id"] = next_link - 1
    return log


OVERLAPS = (22, 39, 90, 141, 192)
OLD_CHAIN_OUTS = ("chain", "seed", "song_start", "shot", "overlap", "frames", "seconds", "prompt", "is_first_clip")
NEW_CHAIN_OUTS = ("chain", "seed", "clip", "frames", "seconds", "prompt", "is_first_clip")


def song_on_loader(wf):
    """The old layout (song start and overlap on Luna-H3-Chain) -> the new one (on the Asset Loader). Returns the log."""
    nodes, links = wf["nodes"], wf["links"]
    by_id = {n["id"]: n for n in nodes}
    log = []
    chain = one(nodes, "Luna-H3-Chain")
    loader = one(nodes, "Luna-H3-AssetLoader")
    if [o["name"] for o in chain["outputs"]] != list(OLD_CHAIN_OUTS):
        sys.exit("Luna-H3-Chain is not in the old layout; nothing to move")
    next_link = max([l[0] for l in links] + [wf.get("last_link_id", 0)]) + 1

    def drop_link(lid):
        l = next(l for l in links if l[0] == lid)
        src, dst = by_id[l[1]], by_id[l[3]]
        src["outputs"][l[2]]["links"] = [x for x in (src["outputs"][l[2]].get("links") or []) if x != lid]
        dst["inputs"][l[4]]["link"] = None
        links.remove(l)
        log.append(f"unwired {src['type']} #{src['id']}.{src['outputs'][l[2]]['name']} -> "
                   f"{dst['type']} #{dst['id']}.{dst['inputs'][l[4]]['name']}")

    def link(src, slot, dst, name, type_, after):
        """Wire src.slot -> dst.name; a new socket goes right after the socket `after`, where the node defines it,
        and the links into the sockets behind it move one slot on."""
        nonlocal next_link
        inp = next((i for i in dst["inputs"] if i["name"] == name), None)
        if inp is None:
            inp = {"localized_name": name, "name": name, "type": type_, "link": None}
            at = next((k + 1 for k, i in enumerate(dst["inputs"]) if i["name"] == after), len(dst["inputs"]))
            dst["inputs"].insert(at, inp)
            for l in links:
                if l[3] == dst["id"] and l[4] >= at:
                    l[4] += 1
        if inp["link"] is not None:
            drop_link(inp["link"])
        links.append([next_link, src["id"], slot, dst["id"], dst["inputs"].index(inp), type_])
        src["outputs"][slot]["links"] = (src["outputs"][slot].get("links") or []) + [next_link]
        inp["link"] = next_link
        log.append(f"wired {src['type']} #{src['id']}.{src['outputs'][slot]['name']} -> {dst['type']} #{dst['id']}.{name}")
        next_link += 1

    # Luna-H3-Chain: the song start and overlap go, and so do their wires.
    old = dict(zip(["clips", "seed", "control", "overlap", "song_locked", "song_offset", "clips_already_made"],
                   chain["widgets_values"]))
    for slot in (2, 4):
        for lid in list(chain["outputs"][slot].get("links") or []):
            drop_link(lid)
    keep = {0: 0, 1: 1, 3: 2, 5: 3, 6: 4, 7: 5, 8: 6}
    for l in links:
        if l[1] == chain["id"]:
            l[2] = keep[l[2]]
    outs = [chain["outputs"][k] for k in sorted(keep)]
    for o, name in zip(outs, NEW_CHAIN_OUTS):
        o["name"] = o["localized_name"] = name
    chain["outputs"] = outs
    for i in [i for i in chain["inputs"] if i["name"] in ("overlap", "song_offset")]:
        if i.get("link") is not None:
            drop_link(i["link"])
        chain["inputs"].remove(i)
    for l in links:   # input slots after the removed ones moved up
        if l[3] == chain["id"]:
            l[4] = next(k for k, i in enumerate(chain["inputs"]) if i.get("link") == l[0])
    old.update({k: v for k, v in chain.get("widgets_values_named", {}).items() if k in old})
    chain["widgets_values"] = [old["clips"], old["seed"], old["control"], old["song_locked"], old["clips_already_made"]]
    named(chain, "overlap", "song_offset")
    log.append(f"Luna-H3-Chain #{chain['id']}: overlap ({old['overlap']}) and song_offset ({old['song_offset']}) removed")

    # The Asset Loader's sound card owns the song: start, shot, overlap.
    media = json.loads(loader.get("widgets_values_named", {}).get("media_state") or loader["widgets_values"][3])
    card = media["audio"][0]
    if card.get("position") != "shot":
        card.update({"position": "shot", "song_start": card.get("start", 0.0), "shot": 1})
        log.append(f"sound card: Position 'By shot number' from its start {card['song_start']}")
    if int(card.get("overlap", 0)) not in OVERLAPS:
        log.append(f"sound card: overlap {card.get('overlap')} -> 22")
        card["overlap"] = 22
    loader["widgets_values"][3] = json.dumps(media, ensure_ascii=False)
    named(loader, media_state=loader["widgets_values"][3])
    log.append(f"sound card: song start {card['song_start']} s, shot {card['shot']}, overlap {card['overlap']}")
    for name in ("song_start", "shot"):
        inp = next((i for i in loader["inputs"] if i["name"] == name), None)
        if inp is not None and inp.get("link") is not None:
            drop_link(inp["link"])
    link(chain, 0, loader, "chain", "LUNA_H3_CHAIN", "song_start")
    overlap_slot = next(k for k, o in enumerate(loader["outputs"]) if o["name"] == "overlap")
    for type_ in ("Luna-H3-ChainContinue", "Luna-H3-ChainRemember", "Luna-H3-ChainJoin"):
        link(loader, overlap_slot, one(nodes, type_), "overlap", "INT", "chain")
    wf["last_link_id"] = next_link - 1
    return log


def patch_existing(change):
    import shutil
    import time
    wf = json.loads(OUT.read_text(encoding="utf-8"))
    log = change(wf)
    scratch = Path(__file__).resolve().parents[1] / ".work"
    scratch.mkdir(exist_ok=True)
    backup = scratch / f"{OUT.stem}.backup-{time.strftime('%Y%m%d-%H%M%S')}.json"
    shutil.copy2(OUT, backup)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False)
    print("\n".join(log))
    print(f"backup {backup}\nwritten {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing H3 Luna Sampler_Chain.json")
    ap.add_argument("--add-clip-saves", action="store_true",
                    help="add the per-clip save to the existing H3 Luna Sampler_Chain.json (backup first)")
    ap.add_argument("--song-on-loader", action="store_true",
                    help="move the song settings of the existing H3 Luna Sampler_Chain.json to the Asset Loader (backup first)")
    args = ap.parse_args()
    if args.add_clip_saves:
        return patch_existing(add_clip_saves)
    if args.song_on_loader:
        return patch_existing(song_on_loader)
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
               [out("images", "IMAGE"), out("audio", "AUDIO"), out("clip_audio", "AUDIO")], [],
               (x0 + 420, y0 + 460), (330, 100))
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
    log += song_on_loader(wf)
    log += add_clip_saves(wf)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False)
    print("\n".join(log))
    print(f"song_offset {start} s (Main's song start), song_locked {song_on['widgets_values'][0]}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()

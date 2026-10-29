"""Repair a saved workflow after the 2026-10-28 node-pack merge, and drop the old SelfLift chain-patch sockets.

- Node ids renamed by the merge (no aliases in the new packs): LunaMiniMaxH3Canvas -> Luna-H3-Canvas,
  LunaAssetLoader -> Luna-H3-AssetLoader, LunaShotList -> Luna-H3-ShotList, SaveVideoSimple -> Luna-SaveVideo,
  LLMPrompt -> Luna-LLMPrompt. Widgets are unchanged (checked against the running server's object_info).
- SelfLiftAvatarH3Sampler: the `low_res_head` input and `low_res_latent` output came from our old chain patch; the
  installed upstream node has neither. They are removed when unlinked.
Titles are not touched (Peter: original node names).

Usage: python repair_luna_workflow.py <in.json> <out.json>
"""

import json
import sys
from pathlib import Path

RENAMED = {"LunaMiniMaxH3Canvas": "Luna-H3-Canvas", "LunaAssetLoader": "Luna-H3-AssetLoader",
           "LunaShotList": "Luna-H3-ShotList", "SaveVideoSimple": "Luna-SaveVideo", "LLMPrompt": "Luna-LLMPrompt"}


def repair(wf):
    changed = []
    for n in wf["nodes"]:
        new = RENAMED.get(n["type"])
        if new:
            changed.append(f"{n['id']} {n['type']} -> {new}")
            n["type"] = new
            props = n.setdefault("properties", {})
            if props.get("Node name for S&R") in RENAMED:
                props["Node name for S&R"] = new
        if n["type"] == "SelfLiftAvatarH3Sampler":
            before = len(n.get("inputs", [])), len(n.get("outputs", []))
            n["inputs"] = [i for i in n.get("inputs", []) if not (i["name"] == "low_res_head" and i.get("link") is None)]
            n["outputs"] = [o for o in n.get("outputs", []) if not (o["name"] == "low_res_latent" and not o.get("links"))]
            if before != (len(n["inputs"]), len(n["outputs"])):
                changed.append(f"{n['id']} SelfLiftAvatarH3Sampler: old patch sockets removed")
    return changed


def main(src, dst):
    wf = json.loads(Path(src).read_text(encoding="utf-8"))
    for line in repair(wf):
        print(line)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

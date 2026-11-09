"""Copy a chain workflow with Luna-MetaBatch and Luna-SaveVideo in place of VHS (Peter 2026-11-09, "Port it too").

    E:\\ComfyUI-Easy-Install\\python_embeded\\python.exe -I tools\\chain_to_luna_metabatch.py SRC.json DST.json

What changes, nothing else:
  - every VHS_BatchManager becomes Luna-MetaBatch (same socket type, same links, same frames_per_batch and its link);
  - every VHS_VideoCombine fed by a meta_batch becomes a Luna-SaveVideo with that meta_batch, the same images, audio and
    frame rate, H.264 MP4, quality = the old crf, AAC 320k, the old filename_prefix split into folder and name;
  - that Luna-SaveVideo also gets the sampler's latent (the node feeding Luna-H3-ChainRemember.latent), so each clip's
    whole latent is saved beside the film.
The widget values are written to `widgets_values` and `widgets_values_named`: the frontend restores from the second one
(lesson of 2026-11-03, plans\\chain-stop-continue.md). The source file is never written. Built for
`H3 Luna Sampler_Chain.json`, whose per-clip "Clip video" Luna-SaveVideo is the template for the new node's layout.
"""

import copy
import json
import sys
from pathlib import Path

SAVE_WIDGETS = ["frame_rate", "codec", "container", "bit_depth", "quality", "log_level", "pingpong", "save_metadata",
                "filename_prefix", "output_path", "save_output", "pass_frames", "crop_to_audio", "audio_codec",
                "audio_bitrate", "save_first_frame", "save_last_frame", "save_prompt"]


def _links(workflow):
    return {link[0]: link for link in workflow["links"]}


def _input(node, name):
    return next((slot for slot in node.get("inputs", []) if slot["name"] == name), None)


def _vhs_values(node):
    values = node.get("widgets_values")
    return values if isinstance(values, dict) else dict(node.get("widgets_values_named") or {})


def convert(workflow):
    workflow = copy.deepcopy(workflow)
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = _links(workflow)
    template = next((n for n in workflow["nodes"] if n["type"] == "Luna-SaveVideo"), None)
    if template is None:
        raise SystemExit("No Luna-SaveVideo in the workflow to copy the layout from.")
    remember = next((n for n in workflow["nodes"] if n["type"] == "Luna-H3-ChainRemember"), None)
    latent_from = None
    if remember is not None and _input(remember, "latent") and _input(remember, "latent")["link"] is not None:
        source = links[_input(remember, "latent")["link"]]
        latent_from = (source[1], source[2])
    changed = []

    for node in workflow["nodes"]:
        if node["type"] != "VHS_BatchManager":
            continue
        frames = int(_vhs_values(node).get("frames_per_batch", 16))
        node["type"] = "Luna-MetaBatch"
        node["title"] = "Luna-MetaBatch (frames = Canvas)"
        node["properties"] = {"Node name for S&R": "Luna-MetaBatch"}
        node["widgets_values"] = [frames]
        node["widgets_values_named"] = {"frames_per_batch": frames}
        changed.append(f"{node['id']} VHS_BatchManager -> Luna-MetaBatch")

    next_link = max(links) + 1
    for index, node in enumerate(list(workflow["nodes"])):
        if node["type"] != "VHS_VideoCombine" or not _input(node, "meta_batch") or _input(node, "meta_batch")["link"] is None:
            continue
        old = _vhs_values(node)
        folder, _, name = str(old.get("filename_prefix", "chain")).rpartition("/")
        values = dict(zip(SAVE_WIDGETS, template["widgets_values"]))
        values.update(frame_rate=float(old.get("frame_rate", 24)), codec="H.264", container="MP4", bit_depth="Auto",
                      quality=int(old.get("crf", 14)), pingpong=False, save_metadata=bool(old.get("save_metadata", True)),
                      filename_prefix=name or "chain", output_path=folder, save_output=bool(old.get("save_output", True)),
                      pass_frames=False, crop_to_audio=bool(old.get("trim_to_audio", False)), audio_codec="AAC",
                      audio_bitrate="320k", save_first_frame=False, save_last_frame=False, save_prompt=False)
        new = copy.deepcopy(template)
        new.update(id=node["id"], pos=node["pos"], order=node.get("order", new.get("order")), mode=node.get("mode", 0))
        new["title"] = "Luna-SaveVideo (meta_batch: the whole chain)"
        wired = {name: _input(node, name)["link"] for name in ("images", "audio", "meta_batch") if _input(node, name)}
        inputs = [slot for slot in copy.deepcopy(template["inputs"]) if slot["name"] not in ("latent", "meta_batch")]
        for slot in inputs:
            slot["link"] = wired.get(slot["name"]) if slot["name"] in ("images", "audio") else None
        at = next(i for i, slot in enumerate(inputs) if slot["name"] == "prompt_text") + 1
        inputs[at:at] = [
            {"localized_name": "latent", "name": "latent", "shape": 7, "type": "LATENT", "link": None},
            {"localized_name": "meta_batch", "name": "meta_batch", "shape": 7, "type": "VHS_BatchManager",
             "link": wired.get("meta_batch")},
        ]
        new["inputs"] = inputs
        for slot in new["outputs"]:
            slot["links"] = []
        full = list(template["widgets_values"])
        full[:len(SAVE_WIDGETS)] = [values[name] for name in SAVE_WIDGETS]
        new["widgets_values"] = full
        new["widgets_values_named"] = {**(template.get("widgets_values_named") or {}), **values}
        for slot_index, slot in enumerate(inputs):
            if slot["link"] is not None:
                links[slot["link"]][3], links[slot["link"]][4] = new["id"], slot_index
        if latent_from is not None:
            slot_index = next(i for i, slot in enumerate(inputs) if slot["name"] == "latent")
            inputs[slot_index]["link"] = next_link
            workflow["links"].append([next_link, latent_from[0], latent_from[1], new["id"], slot_index, "LATENT"])
            nodes[latent_from[0]]["outputs"][latent_from[1]]["links"].append(next_link)
            next_link += 1
        workflow["nodes"][index] = new
        changed.append(f"{node['id']} VHS_VideoCombine -> Luna-SaveVideo (meta_batch"
                       f"{', latent from node %d' % latent_from[0] if latent_from else ''})")

    workflow["last_link_id"] = max(next_link - 1, workflow.get("last_link_id", 0))
    return workflow, changed


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    source, target = Path(argv[1]), Path(argv[2])
    if source.resolve() == target.resolve():
        raise SystemExit("Write to a new file; the source workflow stays as it is.")
    workflow, changed = convert(json.loads(source.read_text(encoding="utf-8")))
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(workflow, handle, indent=2)
    print("\n".join(changed) or "nothing to change")
    print(f"written: {target}")


if __name__ == "__main__":
    main(sys.argv)

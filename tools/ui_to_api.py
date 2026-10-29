"""Turn a ComfyUI UI workflow (with `widgets_values_named`) into an API graph, with file names mapped to this PC.

Made 2026-10-29 to run Kijai's two-stage Open-H3-Regenerate workflow unchanged on Peter's PC (Peter: "pull kijai
node, make a test, analise, see if it works, then we decide the next step"). Copy of that workflow:
`D:\\Claude\\Luna-Research\\h3\\2026-10-29-kijai2stage_compare.json`.

Usage: python ui_to_api.py <ui.json> <object_info.json> <out_api.json> [old=new ...]
- Muted / bypassed nodes and nodes the server does not know (notes, reroutes) are left out.
- Every `.safetensors` value is mapped to the installed name with the same base name (for example
  `MiniMaxH3\\x.safetensors`). `old=new` pairs rename a base name first (a file stored under another name).
- Widget values come from `widgets_values_named`; `control_after_generate` and UI-only keys are dropped.
"""

import json
import sys
from pathlib import Path

UI_ONLY = {"control_after_generate", "Update inputs", "preview"}


def installed_names(object_info):
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


def convert(ui, object_info, renames):
    names = installed_names(object_info)
    link_from = {link[0]: (str(link[1]), link[2]) for link in ui["links"]}
    graph, missing = {}, []
    for n in ui["nodes"]:
        if n.get("mode", 0) != 0 or n["type"] not in object_info:
            continue
        inputs = {}
        for key, value in (n.get("widgets_values_named") or {}).items():
            if key in UI_ONLY:
                continue
            if isinstance(value, str) and value.endswith(".safetensors"):
                base = value.replace("\\", "/").rsplit("/", 1)[-1]
                base = renames.get(base, base)
                if base not in names:
                    missing.append(f"{n['id']} {n['type']}.{key}: {value}")
                value = names.get(base, value)
                if n["type"] == "Luna-H3-Sampler":  # it lists LoRAs with forward slashes
                    value = value.replace("\\", "/")
            inputs[key] = value
        for socket in n.get("inputs", []):
            if socket.get("link") is not None and socket["link"] in link_from:
                inputs[socket["name"]] = list(link_from[socket["link"]])
        graph[str(n["id"])] = {"class_type": n["type"], "inputs": inputs}
    # Drop links to nodes that were left out.
    for spec in graph.values():
        for key, value in list(spec["inputs"].items()):
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] not in graph:
                del spec["inputs"][key]
    return graph, missing


def main(ui_path, info_path, out_path, *pairs):
    renames = dict(pair.split("=", 1) for pair in pairs)
    ui = json.loads(Path(ui_path).read_text(encoding="utf-8"))
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    graph, missing = convert(ui, info, renames)
    with Path(out_path).open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=1)
    print(f"{len(graph)} nodes -> {out_path}")
    for line in missing:
        print("MISSING", line)


if __name__ == "__main__":
    main(*sys.argv[1:])

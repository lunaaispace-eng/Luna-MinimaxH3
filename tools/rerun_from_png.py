"""Queue a finished run again from the graph saved in its PNG (VHS Video Combine's metadata), with a few inputs changed.

Every other node and value stays as it ran. For one-variable tests (Peter, 2026-10-31: "not we change one variable").

Usage:
  python rerun_from_png.py <run.png> [--set NODE.INPUT=VALUE ...] [--set-file NODE.INPUT=PATH ...] [--dry-run]
VALUE is read as JSON when it parses (1.3, true, "x"), else as text. --set-file puts a file's text into a STRING input,
for example a prompt saved beside a raw take. Queues to ComfyUI on 127.0.0.1:8188 and prints the prompt id.

Example (chain clip 1 with Sol-Attn tau 1.3 and the prompt pinned, 2026-11-03):
  python rerun_from_png.py H3_Luna_Sampler_Chain_00001.png --set 113.selection.tau=1.3 --set 231.clips=1 \
    --set 230.requeue=0 --set 232.name=tau13 --set 233.name=tau13 --set 182.value=false \
    --set-file 99.positive=M1_..._00039.txt --set 222.filename_prefix=chain_clip1_tau13
"""

import argparse
import json
import urllib.request
from pathlib import Path

from PIL import Image


def parse_value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def split_key(key):
    node, _, name = key.partition(".")
    if not node or not name:
        raise SystemExit(f"bad key {key!r}: use NODE.INPUT")
    return node, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("png", type=Path)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--set-file", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    info = Image.open(args.png).info
    prompt = json.loads(info["prompt"])
    workflow = json.loads(info["workflow"]) if "workflow" in info else None

    # The PNG keeps list values bare (LoRA Manager's `loras`); the API reads a bare list as a link unless it is
    # [node_id, slot], so wrap the rest as the frontend does.
    for node in prompt.values():
        for name, value in node["inputs"].items():
            is_link = isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)
            if isinstance(value, list) and not is_link:
                node["inputs"][name] = {"__value__": value}

    changes = [(k, parse_value(v)) for k, _, v in (s.partition("=") for s in args.set)]
    changes += [(k, Path(v).read_text(encoding="utf-8")) for k, _, v in (s.partition("=") for s in args.set_file)]
    for key, value in changes:
        node, name = split_key(key)
        if node not in prompt:
            raise SystemExit(f"node {node} is not in the graph")
        inputs = prompt[node]["inputs"]
        if name not in inputs:
            raise SystemExit(f"node {node} ({prompt[node]['class_type']}) has no input {name!r}")
        old = inputs[name]
        if isinstance(old, list):
            raise SystemExit(f"node {node} input {name!r} is a link, not a value")
        inputs[name] = value
        shown = value if not isinstance(value, str) or len(value) < 60 else value[:57] + "..."
        print(f"{node} {prompt[node]['class_type']}.{name}: {str(old)[:60]!r} -> {shown!r}")

    if args.dry_run:
        return
    body = {"prompt": prompt}
    if workflow is not None:
        body["extra_data"] = {"extra_pnginfo": {"workflow": workflow}}
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        print(json.loads(r.read())["prompt_id"])


if __name__ == "__main__":
    main()

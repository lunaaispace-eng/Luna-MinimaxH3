"""Luna-H3-LoopStart / Luna-H3-LoopEnd — one queue renders N scenes, each saved to disk.

Place the pair around an H3 graph. Luna-H3-LoopStart reads a JSON scene list and hands the
body the current scene (index, JSON, prompt). Luna-H3-LoopEnd saves that scene's frames,
audio and latent to `<ComfyUI output>/h3_loops/<run_name>/`, then either expands the
body again for the next scene or, after the last scene, writes `run.manifest.json`.

This is S9 of plans/S9-S14-loop-nodes.md: the skeleton. It carries nothing between
scenes except the small state dict (run name, scene list, index) — the carried latent
head, the song-lock window, the join analyser and the guide are later steps.

How the loop works. Luna-H3-LoopEnd takes `flow` as a *raw link*, so it receives the link
itself ([start_node_id, slot]) instead of the value. From that it walks the graph
backwards from itself to find the body (every node between Start and End that End
depends on, plus output nodes hanging off the body), clones the body with ComfyUI's
GraphBuilder, hands the next state to the cloned Start as `initial_state`, and
returns the clone as `expand`. A node that is *outside* the body keeps feeding the
clone through its original link.

Licence: GPL-3.0 since 2026-10-26. The expansion functions below are adapted from
Context Loop (ethanfel/ComfyUI-MiniMaxH3-Context-Loop, `chain_nodes.py`,
MiniMaxH3ChainLoopEnd), GPL-3.0, Copyright (C) 2026 NikoDemon80; its recursion is
itself adapted from Ethanfel's SxCP loop nodes. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import shutil
import struct
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from comfy_api.latest import io

try:  # ComfyUI core. Absent only in a bare unit-test environment.
    from comfy_execution.graph_utils import GraphBuilder, is_link
except ImportError:  # pragma: no cover
    GraphBuilder = None

    def is_link(value):
        return isinstance(value, list) and len(value) == 2

LOG = "[Luna-H3-Loop]"
logger = logging.getLogger("Luna.H3Loop")

START_ID = "Luna-H3-LoopStart"
END_ID = "Luna-H3-LoopEnd"
FLOW_TYPE = "LUNA_H3_LOOP_FLOW"
STATE_TYPE = "LUNA_H3_LOOP_STATE"
MEMORY_POLICIES = ("off", "unload_models", "fresh_scene")
OUTPUT_SUBDIR = "h3_loops"
FORMAT_VERSION = 1

LoopFlow = io.Custom(FLOW_TYPE)
LoopState = io.Custom(STATE_TYPE)


# --------------------------------------------------------------------------- scenes

def parse_scenes(text: str) -> list[dict]:
    """The scene list: a JSON array of objects. Raises ValueError saying what is wrong."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{LOG} scenes is empty; give a JSON list like "
                         '[{"name": "S1", "prompt": "..."}].')
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{LOG} scenes is not valid JSON: {exc.msg} "
                         f"(line {exc.lineno}, column {exc.colno}).") from exc
    if not isinstance(data, list):
        raise ValueError(f"{LOG} scenes must be a JSON list of scene objects, "
                         f"got {type(data).__name__}.")
    if not data:
        raise ValueError(f"{LOG} scenes is an empty list; need at least one scene.")
    for i, scene in enumerate(data):
        if not isinstance(scene, dict):
            raise ValueError(f"{LOG} scene {i} must be a JSON object, "
                             f"got {type(scene).__name__}.")
        if "prompt" in scene and not isinstance(scene["prompt"], str):
            raise ValueError(f"{LOG} scene {i}: 'prompt' must be a string.")
    return data


_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,79}$")


# Windows treats these as devices whatever the extension (CON.txt is still CON).
_DEVICE_NAMES = frozenset({"CON", "PRN", "AUX", "NUL",
                           *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))})


def check_run_name(name: str) -> str:
    name = name or ""
    if (not _RUN_NAME.match(name) or ".." in name or name.endswith((".", " "))
            or name.split(".")[0].rstrip(" ").upper() in _DEVICE_NAMES):
        raise ValueError(f"{LOG} run_name {name!r} is not usable as a folder name: start with a "
                         "letter or digit, then letters, digits, space . _ - (80 characters at most), "
                         "no trailing dot or space, and not a Windows device name (CON, PRN, AUX, NUL, "
                         "COM1-9, LPT1-9).")
    return name


def _output_root() -> Path:
    import folder_paths
    return Path(folder_paths.get_output_directory())


def run_dir(run_name: str) -> Path:
    return _output_root() / OUTPUT_SUBDIR / check_run_name(run_name)


def _scene_stem(index: int) -> str:
    return f"scene_{int(index):04d}"


# --------------------------------------------------------------------------- hashing / files

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _part(path: Path) -> Path:
    """Temp name a file is written under; os.replace moves it into place once complete."""
    return path.with_name(path.name + ".part")


def _write_json(path: Path, data: Any) -> None:
    """Atomic: a half-written manifest would read as a finished scene."""
    tmp = _part(path)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _file_entry(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
            "bytes": path.stat().st_size}


# --------------------------------------------------------------------------- staged reference copies
# Luna-H3-AssetLoader accepts only input-relative names, so a reference image outside ComfyUI's input folder is
# copied there (as chain_driver.py's stage_references does) and the copy is removed when the run ends. The list
# of copies and of the folders made for them lives in <run>/staged_refs.json, so it survives the cloning of the
# body and a restart.

STAGED_FILE = "staged_refs.json"
STAGE_SUBDIR = "luna_chain_refs"


def _input_root() -> Path:
    import folder_paths
    return Path(folder_paths.get_input_directory())


def staged_name(folder: Path, source: Path) -> str:
    """chain_driver.reference_input_name: input-relative, forward slashes, one folder per project."""
    import hashlib as _h
    namespace = _h.sha256(Path(folder).as_posix().casefold().encode()).hexdigest()[:12]
    name = _h.sha256(Path(source).as_posix().casefold().encode()).hexdigest()[:24]
    return (Path(STAGE_SUBDIR) / namespace / name).with_suffix(Path(source).suffix.lower()).as_posix()


def _read_registry(root: Path) -> dict:
    path = root / STAGED_FILE
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": [], "directories": []}


def stage_reference(source: Path, run_root: Path) -> str:
    """Copy `source` under ComfyUI's input folder; return the input-relative name (forward slashes).

    A copy that is already there with the same bytes is reused and not claimed. One with different bytes is
    replaced if this run's registry lists it (a crashed earlier run, an edited reference or a rewritten
    last frame); an unlisted file is somebody else's, and is an error. Ownership of a new file or folder is written to the registry before the
    copy, so a failed write is still cleaned up.
    """
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{LOG} reference image not found: {source}")
    input_root = _input_root().resolve()
    relative = staged_name(run_root, source)
    target = (input_root / relative).resolve()
    if not target.is_relative_to(input_root):
        raise ValueError(f"{LOG} staged reference escaped ComfyUI's input folder.")
    registry = _read_registry(run_root)
    if target.exists():
        if sha256_file(target) == sha256_file(source):
            return relative
        if target.as_posix() not in registry["files"]:
            raise FileExistsError(f"{LOG} staged reference differs from its source and is not this run's; "
                                  f"refusing to replace {target}")
        shutil.copyfile(source, target)
        return relative
    for directory in (target.parent.parent, target.parent):
        if not directory.exists():
            registry["directories"].append(directory.as_posix())
            run_root.mkdir(parents=True, exist_ok=True)
            _write_json(run_root / STAGED_FILE, registry)
            directory.mkdir()
    registry["files"].append(target.as_posix())
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / STAGED_FILE, registry)
    shutil.copyfile(source, target)
    return relative


def cleanup_staged_refs(run_root: Path) -> int:
    """Remove the copies (and folders) this run staged; leave anything it did not create. Returns files removed."""
    registry = _read_registry(run_root)
    removed = 0
    for name in reversed(registry["files"]):
        path = Path(name)
        if path.is_file():
            path.unlink()
            removed += 1
    for name in reversed(registry["directories"]):
        try:
            Path(name).rmdir()
        except OSError:
            pass  # not empty or already gone: not ours to force
    try:
        (run_root / STAGED_FILE).unlink()
    except FileNotFoundError:
        pass
    return removed


# --------------------------------------------------------------------------- saving a scene

def _frames_to_uint8(images: torch.Tensor):
    arr = (images.detach().cpu().float().clamp(0, 1) * 255.0).round().to(torch.uint8)
    return arr.numpy()  # [N,H,W,C]


def save_frames(images: torch.Tensor, root: Path, stem: str, fps: float) -> tuple[list[Path], str]:
    """Frames to `<stem>.mp4` through PyAV (libx264, yuv420p, crf 12) when it is there
    and the size is even; otherwise one PNG per frame in `<stem>_frames/`.

    The mp4 is for looking at and for joining; it is lossy. The latent is the exact record.
    Returns (files written, "mp4" | "png").
    """
    arr = _frames_to_uint8(images)
    n, h, w, c = arr.shape
    if c not in (3, 4):
        raise ValueError(f"{LOG} images must have 3 or 4 channels, got {c}.")
    arr = arr[..., :3]

    try:
        import av
    except ImportError:
        av = None
    if av is not None and h % 2 == 0 and w % 2 == 0:
        path = root / f"{stem}.mp4"
        try:
            container = av.open(str(_part(path)), "w", format="mp4")
            stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
            stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
            stream.options = {"crf": "12", "preset": "medium"}
            for frame_arr in arr:
                frame = av.VideoFrame.from_ndarray(frame_arr, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            os.replace(_part(path), path)
            return [path], "mp4"
        except Exception as exc:  # codec missing, bad build: fall back, do not lose the scene
            logger.warning("%s mp4 write failed (%s); saving PNGs instead.", LOG, exc)
            try:
                _part(path).unlink()
            except OSError:
                pass

    from PIL import Image
    folder = root / f"{stem}_frames"
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for i, frame_arr in enumerate(arr):
        p = folder / f"frame_{i:05d}.png"
        Image.fromarray(frame_arr).save(_part(p), format="PNG")
        os.replace(_part(p), p)
        files.append(p)
    return files, "png"


def save_audio_wav(audio: dict, path: Path) -> dict:
    """AUDIO dict ({"waveform": [B,C,T], "sample_rate"}) to a 32-bit float WAV.

    Float so nothing is rounded. Only batch item 0 is written (a loop scene has one soundtrack).
    """
    wave = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if wave.ndim == 3:
        wave = wave[0]
    if wave.ndim != 2:
        raise ValueError(f"{LOG} audio waveform must be [B,C,T] or [C,T], got {tuple(wave.shape)}.")
    channels, length = int(wave.shape[0]), int(wave.shape[1])
    data = wave.detach().cpu().float().t().contiguous().numpy().astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHHH", 3, channels, sample_rate, sample_rate * channels * 4,
                      channels * 4, 32, 0)  # IEEE float, cbSize 0
    fact = struct.pack("<I", length)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"fact" + struct.pack("<I", len(fact)) + fact
            + b"data" + struct.pack("<I", len(data)) + data)
    with open(_part(path), "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", len(body)) + body)
    os.replace(_part(path), path)
    return {"channels": channels, "samples": length, "sample_rate": sample_rate}


def _streams(x) -> tuple[bool, list[torch.Tensor]]:
    """(nested?, streams) for a tensor or a NestedTensor."""
    if getattr(x, "is_nested", False):
        return True, list(x.unbind())
    return False, [x]


def save_latent(latent: dict, root: Path, stem: str) -> list[Path]:
    """Every stream of a LATENT dict to `<stem>.latent.safetensors` plus `<stem>.latent.json`.

    H3 latents are a NestedTensor of a video and an audio tensor. Each stream is stored under
    its own key; the sidecar records which keys belong to `samples` / `noise_mask`, whether
    they were nested, and each shape and dtype, so `load_latent` can rebuild the same thing.
    Other tensor entries (batch_index, ...) are not saved.
    """
    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    sidecar: dict[str, Any] = {"format_version": FORMAT_VERSION, "entries": {}}
    for key in ("samples", "noise_mask"):
        value = latent.get(key)
        if value is None:
            continue
        nested, streams = _streams(value)
        names = []
        for i, t in enumerate(streams):
            name = f"{key}.{i}"
            tensors[name] = t.detach().cpu().contiguous()
            names.append({"key": name, "shape": list(t.shape), "dtype": str(t.dtype)})
        sidecar["entries"][key] = {"nested": nested, "streams": names}
    if "samples" not in sidecar["entries"]:
        raise ValueError(f"{LOG} latent has no 'samples'.")
    st = root / f"{stem}.latent.safetensors"
    js = root / f"{stem}.latent.json"
    save_file(tensors, str(_part(st)))
    os.replace(_part(st), st)
    _write_json(js, sidecar)
    return [st, js]


def load_latent(safetensors_path: Path) -> dict:
    """Inverse of save_latent (reads the sidecar next to the file)."""
    from safetensors.torch import load_file

    path = Path(safetensors_path)
    sidecar = json.loads(path.with_name(path.name.replace(".safetensors", ".json"))
                         .read_text(encoding="utf-8"))
    tensors = load_file(str(path))
    out: dict[str, Any] = {}
    for key, entry in sidecar["entries"].items():
        parts = [tensors[s["key"]] for s in entry["streams"]]
        if entry["nested"]:
            import comfy.nested_tensor
            out[key] = comfy.nested_tensor.NestedTensor(tuple(parts))
        else:
            out[key] = parts[0]
    return out


def trim_to_plan(images: torch.Tensor, plan: dict) -> torch.Tensor:
    """Cut the hidden head and the alignment tail: frames [trim, trim + keep) of the decoded batch.

    The decoded batch must be exactly the planned render length, or the plan and the render disagree
    and the trim would cut the wrong frames.
    """
    if int(images.shape[0]) != int(plan["render_frames"]):
        raise ValueError(f"{LOG} the scene decoded to {int(images.shape[0])} frames but the plan rendered "
                         f"{plan['render_frames']}; refusing to trim.")
    first = int(plan["trim_frames"])
    return images[first:first + int(plan["keep_frames"])]


def trim_audio_to_plan(audio: dict, plan: dict) -> dict:
    """Generated audio covers the whole render; keep the same window as the frames (chain_driver trims the
    raw mp4's video and audio with one -ss / -t). A shortfall of up to two frames at the end is zero-filled;
    more than that means the audio and the render disagree."""
    wave = audio["waveform"]
    rate = int(audio["sample_rate"])
    per_frame = rate / 24.0
    have = int(wave.shape[-1])
    expected = round(int(plan["render_frames"]) * per_frame)
    if abs(have - expected) > 2 * per_frame:
        raise ValueError(f"{LOG} the generated audio is {have} samples but {plan['render_frames']} frames are "
                         f"{expected} at {rate} Hz; refusing to trim.")
    first = round(int(plan["trim_frames"]) * per_frame)
    length = round(int(plan["keep_frames"]) * per_frame)
    piece = wave[..., first:first + length]
    if piece.shape[-1] < length:
        pad = torch.zeros(*piece.shape[:-1], length - piece.shape[-1], dtype=piece.dtype)
        piece = torch.cat((piece, pad), dim=-1)
    return {"waveform": piece, "sample_rate": rate}


def check_beta_save(root: Path, plan: dict) -> list[Path]:
    """Beta's Save Latent wrote this scene as planned; return its files for the manifest.

    The final latent and its sidecar are required, and the sidecar must carry the planned token count
    and tail padding (chain_driver's verify_saved). Pass-1 files are listed when present.
    """
    n = int(plan["scene_number"])
    folder = root / "latents"
    files = []
    for suffix, required in ((".latent", True), (".latent.json", True),
                             (".pass1.latent", False), (".pass1.latent.json", False)):
        path = folder / f"scene_{n:03d}{suffix}"
        if path.is_file():
            files.append(path)
        elif required:
            raise FileNotFoundError(f"{LOG} Beta's Save Latent did not write {path}.")
    meta = json.loads((folder / f"scene_{n:03d}.latent.json").read_text(encoding="utf-8-sig"))
    if (int(meta.get("token_count", -1)) != int(plan["token_count"])
            or int(meta.get("tail_padding_frames", -1)) != int(plan["tail_padding_frames"])):
        raise ValueError(f"{LOG} scene {n}'s saved latent has {meta.get('token_count')} tokens / "
                         f"{meta.get('tail_padding_frames')} tail padding; the plan said "
                         f"{plan['token_count']} / {plan['tail_padding_frames']}.")
    return files


def head_audio_of(audio: dict, plan: dict) -> dict:
    """The audio of the hidden head: samples before the kept window (the head frames' song time)."""
    rate = int(audio["sample_rate"])
    first = round(int(plan["trim_frames"]) * rate / 24.0)
    return {"waveform": audio["waveform"][..., :first], "sample_rate": rate}


def save_scene(root: Path, index: int, scene: dict, *, images=None, audio=None, latent=None,
               fps: float = 24.0, memory_policy: str = "off", plan: dict | None = None,
               head_images=None, head_audio=None) -> dict:
    """Write one scene's files, then its manifest (last, so a manifest means a finished scene).

    With a `plan` (from Luna-H3-LoopScene) the last kept frame is also saved as `<stem>_last.png`
    (the next scene's previous_last_frame reference) and Beta's latent files for the scene are
    checked and listed in the manifest. A carried scene's hidden head (`head_images`, and its audio) is kept
    as `<stem>_head.*` with its frame-time mapping, so the join can cut anywhere in the overlap.
    """
    root.mkdir(parents=True, exist_ok=True)
    stem = _scene_stem(index)
    files: list[Path] = []
    info: dict[str, Any] = {}

    if images is not None:
        written, fmt = save_frames(images, root, stem, fps)
        files += written
        info["frames"] = {"format": fmt, "count": int(images.shape[0]),
                          "height": int(images.shape[1]), "width": int(images.shape[2]),
                          "fps": float(fps)}
    if images is not None and plan is not None:
        from PIL import Image
        last = root / f"{stem}_last.png"
        Image.fromarray(_frames_to_uint8(images[-1:])[0][..., :3]).save(_part(last), format="PNG")
        os.replace(_part(last), last)
        files.append(last)
    if head_images is not None and plan is not None and plan.get("overlap_map"):
        written, fmt = save_frames(head_images, root, f"{stem}_head", fps)
        files += written
        om = plan["overlap_map"]
        info["head"] = {"format": fmt, "frames": int(head_images.shape[0]), "count": int(head_images.shape[0]),
                        "fps": float(fps), "mapping": {
            **om, "song_time_of_head_frame_0": round(plan["start_seconds"] - int(plan["trim_frames"]) / 24.0, 6),
            "rule": "head frame h shows the same song time as the previous scene's kept frame offset + h"}}
    if head_audio is not None and "head" in info:
        head_wav = root / f"{stem}_head.wav"
        save_audio_wav(head_audio, head_wav)
        files.append(head_wav)
    if audio is not None:
        wav = root / f"{stem}.wav"
        info["audio"] = save_audio_wav(audio, wav)
        files.append(wav)
    if latent is not None:
        files += save_latent(latent, root, stem)
        info["latent"] = json.loads((root / f"{stem}.latent.json").read_text(encoding="utf-8"))
    if plan is not None:
        files += check_beta_save(root, plan)
        info["plan"] = plan

    manifest = {
        "format_version": FORMAT_VERSION,
        "scene_index": int(index),
        "scene": scene,
        "memory_policy": memory_policy,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "info": info,
        "files": [_file_entry(root, p) for p in files],
    }
    path = root / f"{stem}.manifest.json"
    _write_json(path, manifest)
    return {"manifest": path, "files": len(files)}


def verify_scene(root: Path, index: int) -> list[str]:
    """Problems with one saved scene: manifest missing or wrong, a listed file missing or changed."""
    stem = _scene_stem(index)
    mpath = root / f"{stem}.manifest.json"
    if not mpath.is_file():
        return [f"scene {index}: {mpath.name} is missing"]
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"scene {index}: {mpath.name} is unreadable ({exc})"]
    problems = []
    if manifest.get("scene_index") != index:
        problems.append(f"scene {index}: {mpath.name} says scene_index {manifest.get('scene_index')}")
    for entry in manifest.get("files", []):
        p = root / entry["path"]
        if not p.is_file():
            problems.append(f"scene {index}: {entry['path']} is missing")
        elif p.stat().st_size != entry["bytes"] or sha256_file(p) != entry["sha256"]:
            problems.append(f"scene {index}: {entry['path']} does not match its manifest (size or SHA-256)")
    return problems


def verify_resume(root: Path, start_scene: int) -> None:
    """Scenes 0..start_scene-1 must be saved and intact, or resuming would build on nothing."""
    problems = []
    for i in range(start_scene):
        problems += verify_scene(root, i)
    if problems:
        raise ValueError(f"{LOG} cannot resume at scene {start_scene} in {root}:\n  "
                         + "\n  ".join(problems))


def _beta_latent_files(root: Path, start_scene: int) -> dict[int, list[Path]]:
    """Beta's latent files (latents/scene_NNN.latent, .pass1.latent and their sidecars) of scenes at or after
    start_scene, by Beta scene number (index + 1). Only files directly inside this run's latents folder."""
    found: dict[int, list[Path]] = {}
    folder = root / "latents"
    if not folder.is_dir():
        return found
    for p in folder.iterdir():
        m = re.fullmatch(r"scene_(\d{3,})\.(?:pass1\.)?latent(?:\.json)?", p.name)
        if m and p.is_file() and int(m.group(1)) - 1 >= start_scene:
            found.setdefault(int(m.group(1)), []).append(p)
    return found


def _scene_files(root: Path, index: int) -> list[Path]:
    stem = _scene_stem(index)
    return list(root.glob(stem + ".*")) + list(root.glob(stem + "_frames"))


def prepare_run_dir(root: Path, start_scene: int, overwrite: bool) -> None:
    """A fresh start must not leave an old run's record behind.

    Deletes run.manifest.json. Saved scenes (a manifest on disk) at index >= start_scene are
    refused unless `overwrite`, in which case their files are deleted first.
    """
    if not root.is_dir():
        return
    beta = _beta_latent_files(root, start_scene)
    stale = []
    for p in root.glob("scene_*.manifest.json"):
        m = re.fullmatch(r"scene_(\d+)\.manifest\.json", p.name)
        if m and int(m.group(1)) >= start_scene:
            stale.append(int(m.group(1)))
    stale.sort()
    if (stale or beta) and not overwrite:
        found = stale or sorted({n - 1 for n in beta})
        raise ValueError(f"{LOG} {root} already holds saved scene(s) {found} at or after "
                         f"start_scene {start_scene}. Pick another run_name, or turn on "
                         "overwrite_run to delete them.")
    for files in beta.values():                         # Beta's latents: a skipped write must not pass on an old file
        for p in files:
            p.unlink()
    for i in stale:
        for p in _scene_files(root, i):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    try:
        (root / "run.manifest.json").unlink()
    except FileNotFoundError:
        pass


def write_run_manifest(root: Path, run_name: str, scene_count: int) -> Path:
    scenes = []
    problems = []
    for i in range(scene_count):
        problems += verify_scene(root, i)
        mpath = root / f"{_scene_stem(i)}.manifest.json"
        if mpath.is_file():
            m = json.loads(mpath.read_text(encoding="utf-8"))
            scenes.append({"scene_index": i, "manifest": mpath.name, "manifest_sha256": sha256_file(mpath),
                           "scene": m.get("scene"), "files": [e["path"] for e in m.get("files", [])]})
    if problems:
        raise ValueError(f"{LOG} run {run_name!r} is not complete:\n  " + "\n  ".join(problems))
    path = root / "run.manifest.json"
    _write_json(path, {"format_version": FORMAT_VERSION, "run_name": run_name,
                       "scene_count": scene_count,
                       "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "scenes": scenes})
    return path


# --------------------------------------------------------------------------- memory policy

def release_between_scenes(policy: str, scene_index: int) -> dict:
    """Best-effort memory release once a scene is safely on disk.

    off: nothing, not even an import. unload_models: unload every model and empty the CUDA
    allocator. fresh_scene: also evict ComfyUI's execution cache, collect garbage and release
    pinned pages, so the next scene reloads what it needs. Adapted in shape from Context Loop's
    `_release_loop_boundary_resources`, using the public model manager only.
    """
    if policy not in MEMORY_POLICIES:
        raise ValueError(f"{LOG} memory_policy must be one of {MEMORY_POLICIES}, got {policy!r}.")
    result: dict[str, Any] = {"policy": policy}
    if policy == "off":
        return result
    try:
        import comfy.model_management as mm
    except ImportError as exc:
        logger.warning("%s %s cleanup after scene %d skipped: %s", LOG, policy, scene_index, exc)
        return result

    if policy == "fresh_scene":
        try:
            import comfy.memory_management as mem
            release = getattr(mem, "extra_ram_release", None)
            if callable(release):
                result["cache_bytes"] = int(release(sys.maxsize, free_active=True))
        except Exception as exc:
            logger.warning("%s could not evict the execution cache: %s", LOG, exc)
        result["collected_objects"] = int(gc.collect())
        for name, kwargs in (("cleanup_models_gc", {}),):
            fn = getattr(mm, name, None)
            if callable(fn):
                try:
                    fn(**kwargs)
                except Exception as exc:
                    logger.warning("%s %s failed: %s", LOG, name, exc)
        free_pins = getattr(mm, "free_pins", None)
        if callable(free_pins):
            try:
                result["pinned_bytes"] = int(free_pins(sys.maxsize, evict_active=True))
            except Exception as exc:
                logger.warning("%s could not release pinned pages: %s", LOG, exc)

    try:
        mm.unload_all_models()
    except Exception as exc:
        logger.warning("%s could not unload models: %s", LOG, exc)
    empty = getattr(mm, "soft_empty_cache", None)
    if callable(empty):
        try:
            empty(force=True)
        except TypeError:
            empty()
        except Exception as exc:
            logger.warning("%s could not empty the device allocator: %s", LOG, exc)
    logger.info("%s scene %d boundary cleanup=%s: %s", LOG, scene_index, policy, result)
    return result


# --------------------------------------------------------------------------- body detection and clone
# Adapted from Context Loop, MiniMaxH3ChainLoopEnd._explore_dependencies /
# _collect_contained / _recurse (GPL-3.0, Copyright (C) 2026 NikoDemon80).

def _output_node_types() -> set[str]:
    """class_type names ComfyUI marks OUTPUT_NODE. Empty when `nodes` cannot be imported."""
    try:
        import nodes as comfy_nodes
        mappings = comfy_nodes.NODE_CLASS_MAPPINGS
    except Exception:
        return set()
    return {name for name, cls in mappings.items() if getattr(cls, "OUTPUT_NODE", False)}


def _explore_dependencies(node_id: str, dynprompt: Any, upstream: dict[str, list[str]],
                          parent_ids: list[str]) -> None:
    """upstream[x] = the nodes that take x as an input, for every ancestor x of node_id."""
    node_info = dynprompt.get_node(node_id)
    for value in node_info.get("inputs", {}).values():
        if not is_link(value):
            continue
        parent_id = value[0]
        display_id = dynprompt.get_display_node_id(parent_id)
        if dynprompt.get_node(display_id)["class_type"] != END_ID:
            parent_ids.append(display_id)
        if parent_id not in upstream:
            upstream[parent_id] = []
            _explore_dependencies(parent_id, dynprompt, upstream, parent_ids)
        upstream[parent_id].append(node_id)


def _mirror_id(reference_id: str, original_id: str) -> str:
    """The id `original_id` has in the same expansion as `reference_id`.

    Clones are named prefix + original id, so inside an expansion every node shares the
    reference's prefix (everything up to its last dot).
    """
    if "." in reference_id:
        return reference_id.rsplit(".", 1)[0] + "." + original_id
    return original_id


def _explore_output_paths(dynprompt: Any, open_node: str, end_node: str) -> list[str]:
    """Ids of the nodes that belong to the body only because they lead to an output node.

    End never depends on them, so the backwards walk cannot see them: a Save node hanging off
    the body, or a whole branch such as VAEDecode -> CreateVideo -> SaveVideo next to
    VAEDecode -> End. Every node downstream of Start that reaches an output node is cloned with
    the body, intermediates included, except End and whatever depends on End. Differs from
    Context Loop, which only follows output nodes linked directly to a body node.
    """
    types = _output_node_types() - {END_ID}
    if not types:
        return []
    prompt = dynprompt.get_original_prompt()
    start_orig = dynprompt.get_display_node_id(open_node)
    end_orig = dynprompt.get_display_node_id(end_node)
    children: dict[str, set[str]] = {}
    parents: dict[str, set[str]] = {}
    for nid, node in prompt.items():
        for v in node.get("inputs", {}).values():
            if is_link(v):
                children.setdefault(v[0], set()).add(nid)
                parents.setdefault(nid, set()).add(v[0])

    def reach(seeds, edges):
        seen, stack = set(), list(seeds)
        while stack:
            n = stack.pop()
            for m in edges.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return seen

    outputs = {nid for nid, node in prompt.items() if node.get("class_type") in types}
    keep = (reach([start_orig], children)
            & (reach(outputs, parents) | outputs)) - reach([end_orig], children) - {end_orig, start_orig}
    found = []
    for orig in sorted(keep):
        current = _mirror_id(open_node, orig)
        try:
            dynprompt.get_node(current)
        except KeyError:
            continue
        found.append(current)
    return found


def _collect_contained(node_id: str, upstream: dict[str, list[str]], contained: dict[str, bool]) -> None:
    for child_id in upstream.get(node_id, []):
        if child_id in contained:
            continue
        contained[child_id] = True
        _collect_contained(child_id, upstream, contained)


def find_body(flow: Any, dynprompt: Any, unique_id: str) -> dict[str, bool]:
    """Ids of every node in the loop body: Start, End and what lies between (plus outputs)."""
    unique_id = str(unique_id)
    if not is_link(flow):
        raise ValueError(f"{LOG} Luna-H3-LoopEnd must receive `flow` straight from Luna-H3-LoopStart's flow output.")
    open_node = str(flow[0])
    if dynprompt.get_node(open_node)["class_type"] != START_ID:
        raise ValueError(f"{LOG} Luna-H3-LoopEnd's `flow` must come directly from Luna-H3-LoopStart.")
    upstream: dict[str, list[str]] = {}
    parent_ids: list[str] = []
    _explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
    contained: dict[str, bool] = {unique_id: True, open_node: True}
    _collect_contained(open_node, upstream, contained)
    for node_id in _explore_output_paths(dynprompt, open_node, unique_id):
        contained[node_id] = True
    return contained


def expand_body(flow: Any, next_state: dict, dynprompt: Any, unique_id: str, n_outputs: int) -> dict:
    """Clone the body with GraphBuilder for the next scene.

    Links between body nodes are re-pointed at the clones; links to nodes outside the body are
    left as they are. The cloned Start gets `next_state` as `initial_state`; the cloned End is
    called "Recurse" and its outputs are what this End returns.
    """
    if GraphBuilder is None:
        raise RuntimeError(f"{LOG} needs ComfyUI's GraphBuilder to loop.")
    unique_id = str(unique_id)
    open_node = str(flow[0])
    contained = find_body(flow, dynprompt, unique_id)

    graph = GraphBuilder()
    for node_id in contained:
        original = dynprompt.get_node(node_id)
        node = graph.node(original["class_type"], "Recurse" if node_id == unique_id else node_id)
        node.set_override_display_id(node_id)
    for node_id in contained:
        original = dynprompt.get_node(node_id)
        node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
        for key, value in original.get("inputs", {}).items():
            if is_link(value) and value[0] in contained:
                node.set_input(key, graph.lookup_node(value[0]).out(value[1]))
            else:
                node.set_input(key, value)
    graph.lookup_node(open_node).set_input("initial_state", next_state)
    recurse = graph.lookup_node("Recurse")
    return {"outputs": tuple(recurse.out(i) for i in range(n_outputs)), "graph": graph.finalize()}


# --------------------------------------------------------------------------- nodes

class LunaH3LoopStart(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Luna-H3-LoopStart",
            display_name="Luna-H3-LoopStart",
            category="Luna-MinimaxH3",
            description=(
                "Start of a scene loop. Give it a JSON list of scenes; it outputs the current "
                "scene (index, JSON, prompt). Wire `flow` straight to Luna-H3-LoopEnd, put the "
                "graph that renders one scene between the two, and one queue renders every scene.\n\n"
                "start_scene resumes a run: the earlier scenes must already be saved in "
                "output/h3_loops/<run_name>/ with intact manifests. Never cached."
            ),
            inputs=[
                io.String.Input(
                    "scenes", multiline=True, default='[{"name": "S1", "prompt": ""}]',
                    tooltip='JSON list of scene objects, e.g. [{"name": "ES1", "prompt": "..."}]. '
                            "Each object is free-form; `prompt` (if present) goes to the prompt output.",
                ),
                io.String.Input(
                    "run_name", default="run",
                    tooltip="Folder under output/h3_loops/ that holds the saved scenes and manifests.",
                ),
                io.Int.Input(
                    "start_scene", default=0, min=0, max=100000,
                    tooltip="0-based scene to start at. Above 0 resumes: scenes before it are "
                            "verified on disk first.",
                ),
                io.Boolean.Input(
                    "overwrite_run", default=False,
                    tooltip="Off: refuse to start if run_name already holds saved scenes at or after "
                            "start_scene. On: delete those scenes first. Starting always deletes "
                            "run.manifest.json, so an old finished-run record cannot outlive new scenes.",
                ),
                LoopState.Input(
                    "initial_state", optional=True,
                    tooltip="Set only by the clones Luna-H3-LoopEnd creates. Leave unconnected.",
                ),
            ],
            outputs=[
                LoopFlow.Output(id="flow", display_name="flow",
                                tooltip="Connect to Luna-H3-LoopEnd's flow, directly."),
                LoopState.Output(id="state", display_name="state",
                                 tooltip="Connect to Luna-H3-LoopEnd's state."),
                io.Int.Output(id="scene_index", display_name="scene_index"),
                io.Int.Output(id="scene_count", display_name="scene_count"),
                io.String.Output(id="scene", display_name="scene",
                                 tooltip="The current scene object as JSON."),
                io.String.Output(id="prompt", display_name="prompt",
                                 tooltip="The scene's `prompt` field, or empty."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # NaN never equals itself: this node runs on every queue, so a changed file on disk
        # (resume) or an edited scene list is always seen.
        return float("nan")

    @classmethod
    def validate_inputs(cls, scenes=None, run_name=None, start_scene=None, **kwargs):
        # The executor passes None for an input that is wired to another node (its value
        # is not known yet): validate constants only; execute checks the real values.
        try:
            parsed = parse_scenes(scenes) if scenes is not None else None
            if run_name is not None:
                check_run_name(run_name)
        except ValueError as exc:
            return str(exc)
        if parsed is not None and start_scene is not None and not 0 <= int(start_scene) < len(parsed):
            return f"{LOG} start_scene {start_scene} is outside the {len(parsed)} scenes (0..{len(parsed) - 1})."
        return True

    @classmethod
    def execute(cls, scenes, run_name, start_scene, overwrite_run=False, initial_state=None):
        if initial_state is None:
            parsed = parse_scenes(scenes)
            name = check_run_name(run_name)
            start = int(start_scene)
            if not 0 <= start < len(parsed):
                raise ValueError(f"{LOG} start_scene {start} is outside the {len(parsed)} scenes "
                                 f"(0..{len(parsed) - 1}).")
            root = run_dir(name)
            if start > 0:
                verify_resume(root, start)
            prepare_run_dir(root, start, bool(overwrite_run))
            state = {"run_name": name, "scenes": parsed, "scene_index": start,
                     "scene_count": len(parsed), "start_scene": start}
        else:
            state = initial_state
            if not isinstance(state, dict) or "scenes" not in state or "scene_index" not in state:
                raise ValueError(f"{LOG} initial_state is not a Loop state; leave it unconnected.")
        index = int(state["scene_index"])
        count = len(state["scenes"])
        if not 0 <= index < count:
            raise ValueError(f"{LOG} scene index {index} is outside the {count} scenes.")
        scene = state["scenes"][index]
        flow = {"run_name": state["run_name"], "scene_index": index}
        logger.info("%s scene %d of %d%s", LOG, index + 1, count,
                    f" ({scene['name']})" if isinstance(scene.get("name"), str) else "")
        return io.NodeOutput(flow, state, index, count, json.dumps(scene, ensure_ascii=False),
                             scene.get("prompt", ""))


def _join_module():
    try:
        from . import h3_loop_join
    except ImportError:  # flat import in the unit tests
        import h3_loop_join
    return h3_loop_join


def _join_inputs() -> list:
    """Luna-H3-LoopEnd's join settings (defaults in brackets in the tooltips)."""
    j = _join_module()
    return [
        io.Combo.Input("cut_rule", options=list(j.CUT_RULES), default=j.DEFAULTS["cut_rule"], optional=True,
                       tooltip="Where inside the overlap a carried join is cut [best_match]. breath: the lowest energy "
                               "in the breath_source audio. lowest_motion: the lowest optical-flow magnitude. "
                               "best_match: the least picture difference between scene k's frame and scene k+1's "
                               "head frame at the same song time. manual: take manual_cuts."),
        io.Boolean.Input("time_lock", default=True, optional=True,
                         tooltip="[on] No time shift: only the cut frame is chosen, and every frame keeps its song "
                                 "time (use for song-locked runs). Off (dialogue / generated): the cut also looks "
                                 "for an offset of up to search_window frames between scene k's tail and scene k+1's head."),
        io.Int.Input("search_window", default=0, min=0, max=192, optional=True,
                     tooltip="Frames, counted back from the end of scene k, in which the cut may fall; also the "
                             "largest offset when time_lock is off. [0 = the whole overlap]"),
        io.Combo.Input("breath_source", options=list(j.BREATH_SOURCES), default=j.DEFAULTS["breath_source"],
                       optional=True,
                       tooltip="Audio the breath rule listens to [scene_audio]. scene_audio: the generated audio of "
                               "scene k (generated runs) or the lock file (song-locked runs). breath_audio: the file "
                               "below, on the song's timeline. none: no breath rule."),
        io.String.Input("breath_audio", default="", optional=True,
                        tooltip="Path to a vocal stem (same timeline as the lock file) for breath_source breath_audio."),
        io.Float.Input("match_threshold", default=j.DEFAULTS["match_threshold"], min=0.0, max=255.0, step=0.1,
                       optional=True,
                       tooltip="Largest mean picture difference (0-255, grey, at 160 px) between the two frames at "
                               "the cut that still counts as seamless [8.0; not calibrated yet, read the join "
                               "reports]. Above it the fallback applies."),
        io.Combo.Input("fallback", options=list(j.FALLBACKS), default=j.DEFAULTS["fallback"], optional=True,
                       tooltip="What a join above match_threshold gets [crossfade]: a crossfade of crossfade_frames "
                               "over the overlap, a hard cut, or stop (the scenes stay saved; re-roll one)."),
        io.Int.Input("crossfade_frames", default=j.DEFAULTS["crossfade_frames"], min=0, max=48, optional=True,
                     tooltip="[6] Length of the fallback crossfade; the audio crossfade has the same length."),
        io.Combo.Input("color_match", options=list(j.COLOR_MODES), default=j.DEFAULTS["color_match"], optional=True,
                       tooltip="Match scene k+1 to scene k's tail [carried_only]. One constant per-channel mean/std "
                               "correction fitted on the overlap (the same song time in both scenes) and applied to "
                               "every frame of scene k+1, so there is no step inside it. all: also fresh joins "
                               "(fitted on the last and first frames, which show different things)."),
        io.Float.Input("color_strength", default=j.DEFAULTS["color_strength"], min=0.0, max=1.0, step=0.05,
                       optional=True, tooltip="[1.0] How much of the correction is applied."),
        io.String.Input("manual_cuts", default="", optional=True,
                        tooltip='JSON like {"1": 330}: join number (1 = between scenes 0 and 1) to the frame of the '
                                "JOINED video where the next scene takes over. Overrides the rule."),
        io.Boolean.Input("report", default=True, optional=True,
                         tooltip="[on] Write join_NN.json and join_NN_sheet.jpg (frames around the cut and the "
                                 "x8 difference maps) per join, and join_report.json."),
    ]


class LunaH3LoopEnd(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Luna-H3-LoopEnd",
            display_name="Luna-H3-LoopEnd",
            category="Luna-MinimaxH3",
            description=(
                "End of a scene loop. Saves the scene's frames, audio and latent to "
                "output/h3_loops/<run_name>/scene_NNNN.* with a manifest (SHA-256 of each file), "
                "applies the memory policy, then repeats the graph between Luna-H3-LoopStart and here "
                "for the next scene. After the last scene it writes run.manifest.json.\n\n"
                "Frames are an mp4 when PyAV is available and the size is even, otherwise PNGs."
            ),
            inputs=[
                LoopFlow.Input(
                    "flow", raw_link=True,
                    tooltip="Connect directly from Luna-H3-LoopStart's flow. This raw link marks "
                            "the loop body that gets repeated.",
                ),
                LoopState.Input("state", tooltip="From Luna-H3-LoopStart's state."),
                io.Image.Input("images", optional=True, tooltip="This scene's frames."),
                io.Audio.Input("audio", optional=True, tooltip="This scene's soundtrack."),
                io.Latent.Input("latent", optional=True,
                                tooltip="This scene's sampled latent (an H3 video + audio pair is "
                                        "saved stream by stream)."),
                io.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.01, optional=True,
                               tooltip="Frame rate of `images`, for the mp4 and the manifest."),
                io.Combo.Input(
                    "memory_policy", options=list(MEMORY_POLICIES), default="off", optional=True,
                    tooltip="Applied once the scene is on disk, before the next one starts. "
                            "off: nothing. unload_models: unload all models and empty the CUDA "
                            "allocator. fresh_scene: also evict the execution cache, collect "
                            "garbage and release pinned pages (the next scene reloads its models).",
                ),
                *_join_inputs(),
            ],
            outputs=[
                io.String.Output(id="manifest_path", display_name="manifest_path",
                                 tooltip="run.manifest.json, once the last scene is saved."),
                io.Int.Output(id="scene_count", display_name="scene_count"),
                io.String.Output(id="joined_video_path", display_name="joined_video_path",
                                 tooltip="The whole run as one mp4 with sound (empty if the run could not be joined)."),
                io.String.Output(id="report_path", display_name="report_path",
                                 tooltip="join_report.json; join_NN.json and join_NN_sheet.jpg are beside it."),
            ],
            hidden=[io.Hidden.dynprompt, io.Hidden.unique_id],
            is_output_node=True,
            enable_expand=True,
        )

    @classmethod
    def validate_inputs(cls, memory_policy="off", **kwargs):
        if memory_policy is not None and memory_policy not in MEMORY_POLICIES:
            return f"{LOG} memory_policy must be one of {MEMORY_POLICIES}."
        join = _join_module()
        try:
            join.check_settings(**{k: v for k, v in kwargs.items() if k in join.DEFAULTS})
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def execute(cls, flow, state, images=None, audio=None, latent=None, fps=24.0,
                memory_policy="off", **join_settings):
        index = int(state["scene_index"])
        count = int(state["scene_count"])
        root = run_dir(state["run_name"])
        scene = state["scenes"][index]

        plan = state.get("plan")
        join_cfg = _join_module().check_settings(**join_settings)       # a bad setting stops before the render is lost
        head_images = head_audio = None
        if plan is not None and images is not None:
            if plan.get("continuation") and plan.get("overlap_map") and int(plan["trim_frames"]) > 0:
                head_images = images[:int(plan["trim_frames"])]
                if audio is not None:
                    head_audio = head_audio_of(audio, plan)
            images = trim_to_plan(images, plan)
        if plan is not None and audio is not None:
            audio = trim_audio_to_plan(audio, plan)       # whatever audio is wired covers the whole render
        saved = save_scene(root, index, scene, images=images, audio=audio, latent=latent,
                           fps=float(fps), memory_policy=memory_policy, plan=plan,
                           head_images=head_images, head_audio=head_audio)
        logger.info("%s scene %d/%d saved: %d file(s), %s", LOG, index + 1, count,
                    saved["files"], saved["manifest"].name)

        release_between_scenes(memory_policy, index)

        if index + 1 < count:
            # The plan belongs to this scene; the next Luna-H3-LoopScene makes its own.
            next_state = {k: v for k, v in state.items() if k != "plan"}
            next_state["scene_index"] = index + 1
            built = expand_body(flow, next_state, cls.hidden.dynprompt, cls.hidden.unique_id, 4)
            return io.NodeOutput(*built["outputs"], expand=built["graph"])

        path = write_run_manifest(root, state["run_name"], count)
        try:
            joined = _join_module().join_run(root, state["run_name"], count, join_cfg)
        finally:
            removed = cleanup_staged_refs(root)
            if removed:
                logger.info("%s removed %d staged reference cop%s from ComfyUI's input folder", LOG, removed,
                            "y" if removed == 1 else "ies")
        video = report = ""
        if joined is not None:
            video = joined["video"].as_posix()
            report = joined["report"].as_posix() if joined["report"] else ""
            run = json.loads(path.read_text(encoding="utf-8"))
            run.update(joined_video=joined["video"].name, join_report=Path(report).name if report else None)
            _write_json(path, run)
        logger.info("%s run %r finished: %d scene(s), %s", LOG, state["run_name"], count, path)
        return io.NodeOutput(path.as_posix(), count, video, report)

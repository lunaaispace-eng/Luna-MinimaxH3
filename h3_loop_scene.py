"""Luna-H3-LoopScene / Luna-H3-LoopProfile — what one scene of the SelfLift graph needs, from the scene list.

S10 of plans/S9-S14-loop-nodes.md. Both nodes sit in the loop body (so they are cloned for every
scene) and take Luna-H3-LoopStart's `state`:

- **Luna-H3-LoopScene** computes the values chain_driver.py (music-video-from-audio skill) used to
  patch into the API graph by hand: scene number, project folder, continuation, the render frame
  count and trim, the song-lock audio window, seeds, the reference images. Its `state` output
  carries the scene's plan to Luna-H3-LoopEnd, which trims the hidden head and checks Beta's sidecar.
- **Luna-H3-LoopProfile** turns a scene's `profile` ("normal" / "fast") into the model, LoRA, step
  and shift values for the SelfLift nodes.

Timing is Beta's own: `plan_masked_context` and `calculate_minimax_h3_timing` are taken from the
installed vrgdg pack's `minimax/latent_manager.py` (loaded from the running pack, or from its file),
never copied. The head itself travels through Beta's own latent files (`latents/scene_NNN.latent`),
which Beta's loaders read; nothing is passed as a tensor.

Audio modes: song_lock (the song's window conditions H3 and is locked) or generated (H3 makes the audio; the
lock file gives only timing, as chain_driver's audio_mode generated). Reference images are staged as input-relative
names for the Luna-H3-AssetLoaders, not loaded here.

Scene fields: start, end (seconds in the song, required), prompt, overlap (39 | 90 | 141 | 192,
default 39), join ("carried" default | "fresh"; scene 0 is always fresh), preroll_frames (fresh scenes,
default 0), seeds ([a, b], default [1111, 2222]), selflift_seed (default seeds[0]), profile,
references (1 to 4 image paths, or {"previous_last_frame": true}).

Licence: GPL-3.0. The decisions and formulas follow chain_driver.py / timing_check.py of the
music-video-from-audio skill (this project's own code).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
from comfy_api.latest import io

try:
    from .h3_loop import LOG, LoopState, run_dir, stage_reference
    from .h3_loop_guide import GuideType
except ImportError:  # imported flat by the unit tests
    from h3_loop import LOG, LoopState, run_dir, stage_reference
    from h3_loop_guide import GuideType

logger = logging.getLogger("Luna.H3LoopScene")

FPS = 24
OVERLAPS = (39, 90, 141, 192)
AUDIO_MODES = ("song_lock", "generated")
MAX_REFS = 4
DEFAULT_SEEDS = (1111, 2222)

# --------------------------------------------------------------------------- Beta's timing

_BETA = None


def _beta_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        import folder_paths
        dirs += [Path(p) for p in folder_paths.get_folder_paths("custom_nodes")]
        dirs.append(Path(folder_paths.__file__).resolve().parent / "custom_nodes")
    except Exception:
        pass
    return dirs


def load_beta():
    """Beta 2.0's latent_manager module: the running pack's, else its file loaded on its own."""
    global _BETA
    if _BETA is not None:
        return _BETA
    module = None
    for name, mod in list(sys.modules.items()):
        if name.endswith("minimax.latent_manager") and hasattr(mod, "plan_masked_context"):
            module = mod
            break
    if module is None:
        for base in _beta_search_dirs():
            path = base / "comfyui-vrgamedevgirl" / "minimax" / "latent_manager.py"
            if path.is_file():
                spec = importlib.util.spec_from_file_location("luna_beta_latent_manager", path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module  # dataclasses look the module up by name
                spec.loader.exec_module(module)
                break
    if module is None:
        raise RuntimeError(f"{LOG} Beta 2.0 (comfyui-vrgamedevgirl) is not installed: Luna-H3-LoopScene takes its "
                           "timing from minimax/latent_manager.py.")
    _check_beta(module)
    _BETA = module
    return module


def _check_beta(module) -> None:
    """Fail loudly if an update of Beta changed what the numbers here assume."""
    if tuple(module.MASKED_CONTEXT_FRAMES) != OVERLAPS:
        raise RuntimeError(f"{LOG} Beta's MASKED_CONTEXT_FRAMES is {tuple(module.MASKED_CONTEXT_FRAMES)}, "
                           f"expected {OVERLAPS}; update h3_loop_scene.py.")
    first = calculate(module, 0, 5, 300)
    if first["timing"]["h3_frame_count"] != 124 or first["tail_padding_frames"] != 4:
        raise RuntimeError(f"{LOG} Beta's timing no longer gives 124 frames / 4 padding for a 5 s opening "
                           "scene; update h3_loop_scene.py.")


def calculate(beta, start: float, end: float, source_duration: float,
              predecessor: dict | None = None, overlap: int = 39, preroll: int = 0) -> dict:
    """chain_driver's timing_check.calculate, with the overlap as a parameter (the driver fixed it at 39)."""
    plan = None
    if predecessor is not None:
        if predecessor.get("tail_padding_frames") is None:
            raise ValueError("Predecessor tail padding must be known, not guessed.")
        plan = beta.plan_masked_context(int(predecessor["token_count"]),
                                        int(predecessor["tail_padding_frames"]), int(overlap))
    warm = plan["warmup_frames"] if plan else int(preroll)
    t = beta.calculate_minimax_h3_timing(
        start, end, warmup_frames=warm, cooldown_frames=0, source_start_seconds=start,
        source_duration_seconds=source_duration, pad_warmup=plan is not None).to_dict()
    padding = max(0, t["h3_frame_count"] - round(t["actual_warmup_seconds"] * FPS) - t["final_frame_count"])
    return {"context_plan": plan, "timing": t, "tail_padding_frames": padding}


# --------------------------------------------------------------------------- scene fields

def scene_settings(scene: dict, index: int) -> dict:
    """The scene's fields with defaults, validated. Raises ValueError naming the scene."""
    where = f"{LOG} scene {index}"
    try:
        start, end = float(scene["start"]), float(scene["end"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{where}: 'start' and 'end' (seconds in the song) are required numbers.") from None
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError(f"{where}: need 0 <= start < end, got {start}, {end}.")
    overlap = scene.get("overlap", 39)
    if overlap not in OVERLAPS:
        raise ValueError(f"{where}: overlap must be one of {OVERLAPS}, got {overlap!r}.")
    join = scene.get("join", "carried")
    if join not in ("carried", "fresh"):
        raise ValueError(f"{where}: join must be 'carried' or 'fresh', got {join!r}.")
    seeds = scene.get("seeds", list(DEFAULT_SEEDS))
    if (not isinstance(seeds, list) or len(seeds) != 2
            or any(type(s) is not int or not 0 <= s < 2 ** 64 for s in seeds)):
        raise ValueError(f"{where}: seeds must be two integers.")
    sl_seed = scene.get("selflift_seed", seeds[0])
    if type(sl_seed) is not int or not 0 <= sl_seed < 2 ** 64:
        raise ValueError(f"{where}: selflift_seed must be an integer.")
    preroll = scene.get("preroll_frames", 0)
    if type(preroll) is not int or preroll < 0:
        raise ValueError(f"{where}: preroll_frames must be a whole number >= 0.")
    refs = scene.get("references")
    if not isinstance(refs, list) or not 1 <= len(refs) <= MAX_REFS:
        raise ValueError(f"{where}: references must be a list of 1 to {MAX_REFS} images.")
    for r in refs:
        if not (isinstance(r, str) and r.strip()) and r != {"previous_last_frame": True}:
            raise ValueError(f"{where}: each reference is an image path or {{\"previous_last_frame\": true}}.")
        if r == {"previous_last_frame": True} and index == 0:
            raise ValueError(f"{where}: the first scene has no previous last frame.")
    return {"start": start, "end": end, "overlap": overlap, "join": join, "seeds": seeds,
            "selflift_seed": sl_seed, "preroll_frames": preroll, "references": refs,
            "profile": scene.get("profile")}


def check_contiguous(scenes: list[dict], audio_mode: str) -> None:
    """A song-locked run is muxed from one slice of the master, so the scenes must follow each other in song time
    (chain_driver checks the same, to 1e-6 s). Generated runs may leave gaps. Raises before anything renders."""
    if audio_mode != "song_lock":
        return
    for i in range(1, len(scenes)):
        prev_end, start = float(scenes[i - 1]["end"]), float(scenes[i]["start"])
        if abs(start - prev_end) > 1e-6:
            raise ValueError(f"{LOG} scene {i} starts at {start} s but scene {i - 1} ends at {prev_end} s: a "
                             "song-locked run needs contiguous scenes (end of one = start of the next), because "
                             "the master is muxed once. Fix the scene list, or use audio_mode generated.")


def check_run_scenes(scenes: list[dict]) -> None:
    """Whole-run checks: every scene valid, one reference count (the graph has a fixed number of slots)."""
    counts = set()
    for i, scene in enumerate(scenes):
        counts.add(len(scene_settings(scene, i)["references"]))
    if len(counts) > 1:
        raise ValueError(f"{LOG} every scene needs the same number of references (the graph wires a fixed "
                         f"number of slots); found {sorted(counts)}.")


# --------------------------------------------------------------------------- Beta's files

def latents_dir(root: Path) -> Path:
    return root / "latents"


def sidecar_path(root: Path, scene_number: int, pass1: bool = False) -> Path:
    return latents_dir(root) / f"scene_{int(scene_number):03d}{'.pass1' if pass1 else ''}.latent.json"


def read_sidecar(root: Path, scene_number: int, pass1: bool = False) -> dict:
    path = sidecar_path(root, scene_number, pass1)
    if not path.is_file() or not path.with_name(path.name[:-5]).is_file():
        raise FileNotFoundError(f"{LOG} the carried head needs {path.name[:-5]} and its sidecar in "
                                f"{latents_dir(root)}; scene {scene_number} did not save it.")
    meta = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("token_count", "tail_padding_frames"):
        if meta.get(key) in (None, ""):
            raise ValueError(f"{LOG} {path.name} has no {key}; the predecessor was not saved by this graph.")
    return {"token_count": int(meta["token_count"]), "tail_padding_frames": int(meta["tail_padding_frames"]),
            "timestamp": float(meta.get("timestamp", 0)), "pass": meta.get("pass")}


def pass1_matches(root: Path, scene_number: int, final: dict) -> bool:
    """Beta's own rule for a usable pass-1 latent: same render as the final one (length and time)."""
    try:
        p = read_sidecar(root, scene_number, pass1=True)
    except (OSError, ValueError, KeyError):
        return False
    return (p["pass"] == "1" and p["token_count"] == final["token_count"]
            and p["tail_padding_frames"] == final["tail_padding_frames"]
            and 0 <= final["timestamp"] - p["timestamp"] < 4 * 3600)


# --------------------------------------------------------------------------- song-lock audio

_AUDIO_CACHE: dict[tuple, tuple[torch.Tensor, int]] = {}


def load_audio_file(path: Path) -> tuple[torch.Tensor, int]:
    """(waveform [C, T] float32, sample rate) at the file's own rate. Decoded once per file version."""
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    if key in _AUDIO_CACHE:
        return _AUDIO_CACHE[key]
    import av
    chunks = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        rate = int(stream.rate)
        resampler = av.AudioResampler(format="fltp", layout=stream.layout, rate=rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None) or []:
            chunks.append(out.to_ndarray())
    if not chunks:
        raise ValueError(f"{LOG} {path} has no audio.")
    wave = torch.from_numpy(__import__("numpy").concatenate(chunks, axis=1)).float()
    _AUDIO_CACHE.clear()  # one lock file at a time
    _AUDIO_CACHE[key] = (wave, rate)
    return wave, rate


def audio_window(wave: torch.Tensor, rate: int, timing: dict) -> dict:
    """The song-lock window for one scene, as an AUDIO dict at the file's sample rate.

    The same slice chain_driver's pad_audio cuts with ffmpeg: the source from audio_trim_start for
    (audio_trim_duration - leading padding) seconds, preceded by `leading` seconds of silence, the
    whole being audio_trim_duration long. Sample counts are rounded once, from seconds * rate.
    """
    lead = timing["audio_leading_padding_seconds"]
    total_s = timing["audio_trim_duration_seconds"]
    selected_s = total_s - lead
    if selected_s <= 0:
        raise ValueError(f"{LOG} no source audio remains after the leading padding.")
    first = round(timing["audio_trim_start_seconds"] * rate)
    take = round(selected_s * rate)
    lead_n = round(lead * rate)
    total = round(total_s * rate)
    segment = wave[:, first:first + take]
    out = torch.zeros(wave.shape[0], total, dtype=wave.dtype)
    n = min(segment.shape[1], total - lead_n)
    out[:, lead_n:lead_n + n] = segment[:, :n]
    return {"waveform": out.unsqueeze(0), "sample_rate": int(rate)}


# --------------------------------------------------------------------------- the overlap

def _guide_module():
    try:
        from . import h3_loop_guide
    except ImportError:  # flat import in the unit tests
        import h3_loop_guide
    return h3_loop_guide


def _join_module():
    try:
        from . import h3_loop_join
    except ImportError:  # flat import in the unit tests
        import h3_loop_join
    return h3_loop_join


def _earlier_keeps(root: Path, index: int) -> list[int] | None:
    """keep_frames of scenes 0..index-1 from their manifests, or None if one is missing (a run from before S10)."""
    keeps = []
    for j in range(index):
        path = root / f"scene_{j:04d}.manifest.json"
        if not path.is_file():
            return None
        plan = json.loads(path.read_text(encoding="utf-8")).get("info", {}).get("plan")
        if not plan:
            return None
        keeps.append(int(plan["keep_frames"]))
    return keeps


def overlap_mapping(root: Path, index: int, context_plan: dict, timing: dict) -> dict | None:
    """Where this scene's hidden head lies in the previous scene's kept frames, proved against Beta's plan.

    Beta copies the previous latent's frames [start_frame, end_frame) into this scene's head. The previous
    scene's kept frames begin at its trim, so head frame 0 is the previous scene's kept frame
    `start_frame - previous_trim`. Luna-H3-LoopEnd also needs `previous_keep - warm` for the same thing; the two must
    agree exactly (they say the head ends where the previous visible frames end), or the saved overlap would be
    misplaced. Returns None when the previous scene's plan is not on disk (a run saved before S10).
    """
    path = root / f"scene_{index - 1:04d}.manifest.json"
    if not path.is_file():
        return None
    prev = json.loads(path.read_text(encoding="utf-8")).get("info", {}).get("plan")
    if not prev:
        return None
    warm = round(timing["final_trim_start_seconds"] * FPS)
    if warm != int(context_plan["warmup_frames"]):
        raise ValueError(f"{LOG} the hidden head ({warm} frames) is not Beta's warm-up "
                         f"({context_plan['warmup_frames']}).")
    from_beta = int(context_plan["start_frame"]) - int(prev["trim_frames"])
    from_keep = int(prev["keep_frames"]) - warm
    if from_beta != from_keep:
        raise ValueError(f"{LOG} the frame-time mapping of the overlap disagrees with Beta's plan: head frame 0 is "
                         f"the previous scene's frame {from_beta} by Beta (start_frame {context_plan['start_frame']} "
                         f"- trim {prev['trim_frames']}) but {from_keep} by keep - head ({prev['keep_frames']} - "
                         f"{warm}). Refusing to render.")
    return {"offset": from_beta, "head_frames": warm, "predecessor_scene": index - 1,
            "predecessor_keep_frames": int(prev["keep_frames"]), "predecessor_trim_frames": int(prev["trim_frames"]),
            "beta_start_frame": int(context_plan["start_frame"]), "beta_end_frame": int(context_plan["end_frame"])}


# --------------------------------------------------------------------------- reference images

def last_frame_path(root: Path, index: int) -> Path:
    return root / f"scene_{int(index):04d}_last.png"


def reference_names(settings: dict, index: int, root: Path) -> list[str]:
    """The scene's references in order as input-relative names (staged copies under ComfyUI's input folder,
    forward slashes), padded with "" to MAX_REFS (unwired slots). previous_last_frame is the previous
    scene's `scene_NNNN_last.png` that Luna-H3-LoopEnd wrote, staged the same way."""
    names = []
    for r in settings["references"]:
        source = last_frame_path(root, index - 1) if isinstance(r, dict) else Path(r)
        names.append(stage_reference(source, root))
    return names + [""] * (MAX_REFS - len(names))


# --------------------------------------------------------------------------- Luna-H3-LoopScene

class LunaH3LoopScene(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Luna-H3-LoopScene",
            display_name="Luna-H3-LoopScene",
            category="Luna-MinimaxH3",
            description=(
                "Everything the SelfLift graph needs for the current scene, from the scene list: scene "
                "number, project folder, continuation, render frames, trim, the song-lock audio window, "
                "seeds and reference images. Put it in the loop body and wire Luna-H3-LoopStart's state in; "
                "wire its `state` out to Luna-H3-LoopEnd. The carried head is read by Beta's loaders from "
                "Beta's own latent files, so nothing is passed here as a tensor.\n\n"
                "Expect one WARNING per scene from Beta's Load Latent nodes: their IS_CHANGED calls int() on "
                "the linked scene_number (None there). Harmless; ComfyUI treats the node as changed, so it "
                "re-runs every scene.\n\n"
                "Audio modes: song_lock (the song's window conditions H3 and is locked) or generated (H3 makes "
                "the audio; the lock file gives only timing). Reference images are staged as input-relative "
                "names for the Luna-H3-AssetLoaders.\n\n"
                "Scene fields: start, end, overlap (39/90/141/192), join (carried/fresh), seeds, "
                "selflift_seed, preroll_frames, references (image paths, or {\"previous_last_frame\": true})."
            ),
            inputs=[
                LoopState.Input("state", tooltip="From Luna-H3-LoopStart."),
                io.String.Input(
                    "lock_audio", default="",
                    tooltip="Path to the song-lock WAV on the full master timeline. Decoded once per file; "
                            "each scene gets its window at the file's own sample rate."),
                io.Combo.Input(
                    "audio_mode", options=list(AUDIO_MODES), default="song_lock",
                    tooltip="song_lock: the song's window conditions H3 and is locked into the result. "
                            "generated: H3 makes the audio (dialogue, sound); the lock file only gives the "
                            "timing, and Luna-H3-LoopEnd keeps the generated audio of the kept frames. The workflow "
                            "builder wires the graph for the chosen mode (chain.json audio_mode)."),
                io.String.Input(
                    "manual_cuts", default="", optional=True,
                    tooltip="Copy Luna-H3-LoopEnd's manual_cuts here so a manual cut outside the overlap is refused before "
                            "this scene renders, not after. Leave empty if Luna-H3-LoopEnd has none."),
                io.Int.Input(
                    "search_window", default=0, min=0, max=192, optional=True,
                    tooltip="Copy of Luna-H3-LoopEnd's search_window (0 = the whole overlap), for the manual-cut check."),
                io.Boolean.Input(
                    "time_lock", default=True, optional=True,
                    tooltip="Copy of Luna-H3-LoopEnd's time_lock. With it off the manual-cut range cannot be checked here."),
                io.String.Input(
                    "loader_folder", default="", optional=True,
                    tooltip="The project folder written into Beta's Load Latent nodes by the workflow "
                            "builder (they must hold a constant, not a link). Checked against this run's "
                            "folder; empty skips the check."),
                io.Combo.Input(
                    "guide_mode", options=["off", "add_guide", "add_guide_vlm"], default="off", optional=True,
                    tooltip="[off] Default for scenes without a guide_mode field. add_guide: clean keyframe rows at "
                            "the guide's frame, aligned to target time, VLM-blind (core Add Guide). add_guide_vlm: "
                            "the same rows plus the same frames shown to the VLM (the FL2VA pairing)."),
                io.Combo.Input(
                    "guide_source", options=["previous_tail", "clip", "images"], default="previous_tail", optional=True,
                    tooltip="[previous_tail] Where the guide frames come from. previous_tail: the previous scene's "
                            "last guide_length delivered frames (fresh scenes only: a carried scene already starts "
                            "from them). clip: guide_clip, a video file. images: guide_images, key images each "
                            "with a frame. Scene fields guide_source / guide_clip / guide_images override per scene."),
                io.Combo.Input(
                    "guide_length", options=[5, 22, 39], default=22, optional=True,
                    tooltip="[22] Frames of a previous_tail or clip guide: exactly 5, 22 or 39, because core Add "
                            "Guide keeps the first 17k+5 frames of a longer batch."),
                io.Int.Input(
                    "guide_frame_idx", default=0, min=-9999, max=9999, optional=True,
                    tooltip="[0] Where the guide starts, counted from the scene's first VISIBLE frame (negative: "
                            "back from the end of the render). It never lands in the carried head."),
                io.Boolean.Input(
                    "guide_audio", default=False, optional=True,
                    tooltip="[off] Also anchor the guide's audio (generated runs only): the previous scene's saved "
                            "audio tail, or the clip's own soundtrack."),
            ],
            outputs=[
                LoopState.Output(id="state", display_name="state",
                                 tooltip="State plus this scene's plan. Wire to Luna-H3-LoopEnd's state."),
                io.String.Output(id="project_folder", display_name="project_folder",
                                 tooltip="The run folder: Beta's Save Latent writes latents/ in it."),
                io.Int.Output(id="scene_number", display_name="scene_number",
                              tooltip="1-based scene number, for Beta's latent files."),
                io.Boolean.Output(id="continuation", display_name="continuation"),
                io.Boolean.Output(id="pass1", display_name="pass1",
                                  tooltip="The predecessor's pass-1 latent matches its final latent."),
                io.Int.Output(id="context_frames", display_name="context_frames"),
                io.Int.Output(id="render_frames", display_name="render_frames"),
                io.Float.Output(id="trim_start", display_name="trim_start"),
                io.Int.Output(id="keep_frames", display_name="keep_frames"),
                io.Float.Output(id="visible_duration", display_name="visible_duration"),
                io.Int.Output(id="seed_1", display_name="seed_1"),
                io.Int.Output(id="seed_2", display_name="seed_2"),
                io.Int.Output(id="selflift_seed", display_name="selflift_seed"),
                io.Audio.Output(id="audio", display_name="audio",
                                tooltip="The song-lock window for this scene (unused in generated mode)."),
                io.String.Output(id="ref_1", display_name="ref_1", tooltip="Input-relative name, to a Luna-H3-AssetLoader's image_paths."),
                io.String.Output(id="ref_2", display_name="ref_2"),
                io.String.Output(id="ref_3", display_name="ref_3"),
                io.String.Output(id="ref_4", display_name="ref_4"),
                # Added last, like every output since S10: saved workflows link outputs by slot number.
                GuideType.Output(id="guide", display_name="guide",
                                 tooltip="The scene's guide for Luna-H3-GuideApply / Luna-H3-GuideClip (off: nothing)."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")  # reads files that the previous scene has just written

    @classmethod
    def execute(cls, state, lock_audio, audio_mode="song_lock", manual_cuts="", search_window=0, time_lock=True,
                loader_folder="", guide_mode="off", guide_source="previous_tail", guide_length=22,
                guide_frame_idx=0, guide_audio=False):
        if audio_mode not in AUDIO_MODES:
            raise ValueError(f"{LOG} audio_mode must be one of {AUDIO_MODES}, got {audio_mode!r}.")
        index = int(state["scene_index"])
        scenes = state["scenes"]
        check_run_scenes(scenes)
        check_contiguous(scenes, audio_mode)
        guide_defaults = {"guide_mode": guide_mode, "guide_source": guide_source, "guide_length": guide_length,
                          "guide_frame_idx": guide_frame_idx, "guide_audio": guide_audio}
        guide_mod = _guide_module()
        guide_mod.check_guides(scenes, guide_defaults, audio_mode)       # every scene's guide, before any render
        s = scene_settings(scenes[index], index)
        root = run_dir(state["run_name"])
        if loader_folder and Path(loader_folder).resolve() != root.resolve():
            raise ValueError(f"{LOG} Beta's Load Latent nodes hold {loader_folder} but this run's folder is "
                             f"{root}. Rebuild the workflow with run_name {state['run_name']!r}.")
        lock = Path(lock_audio)
        if not lock_audio or not lock.is_file():
            raise FileNotFoundError(f"{LOG} lock_audio not found: {lock_audio!r}")
        wave, rate = load_audio_file(lock)
        duration = wave.shape[1] / rate

        if s["end"] > duration + 1e-6:
            raise ValueError(f"{LOG} scene {index} ends at {s['end']} s but the lock audio is {duration:.3f} s.")

        scene_number = index + 1
        continuation = index > 0 and s["join"] != "fresh"
        predecessor = None
        pass1 = False
        if continuation:
            predecessor = read_sidecar(root, scene_number - 1)
            pass1 = pass1_matches(root, scene_number - 1, predecessor)
        beta = load_beta()
        calc = calculate(beta, s["start"], s["end"], duration, predecessor, s["overlap"], s["preroll_frames"])
        t = calc["timing"]
        overlap_map = None
        if continuation:
            overlap_map = overlap_mapping(root, index, calc["context_plan"], t)
        join_mod = _join_module()
        for note in join_mod.check_manual_early(
                join_mod.parse_manual_cuts(manual_cuts), index, len(scenes), continuation,
                round(t["final_trim_start_seconds"] * FPS), _earlier_keeps(root, index), int(search_window),
                bool(time_lock)):
            logger.warning("%s %s", LOG, note)
        plan = {
            "start_seconds": s["start"], "end_seconds": s["end"], "lock_audio": lock.as_posix(),
            "context_plan": calc["context_plan"], "overlap_map": overlap_map,
            "scene_number": scene_number, "project_folder": root.as_posix(), "continuation": continuation,
            "pass1": pass1, "overlap": s["overlap"], "audio_mode": audio_mode, "timing": t,
            "tail_padding_frames": calc["tail_padding_frames"],
            "token_count": beta._frames_to_tokens(t["h3_frame_count"]),
            "render_frames": int(t["h3_frame_count"]), "trim_frames": round(t["final_trim_start_seconds"] * FPS),
            "keep_frames": int(t["final_frame_count"]),
        }
        if plan["trim_frames"] + plan["keep_frames"] > plan["render_frames"]:
            raise ValueError(f"{LOG} scene {index}: trim {plan['trim_frames']} + keep {plan['keep_frames']} "
                             f"is more than the {plan['render_frames']} rendered frames.")
        audio = audio_window(wave, rate, t)
        refs = reference_names(s, index, root)
        logger.info("%s scene %d: %s, %d frames (trim %d, keep %d), overlap %d, pass1=%s", LOG, index,
                    "carried" if continuation else "fresh", plan["render_frames"], plan["trim_frames"],
                    plan["keep_frames"], s["overlap"], pass1)
        out_state = dict(state, plan=plan)
        guide = guide_mod.build_guide(guide_mod.guide_fields(scenes[index], guide_defaults, index), root=root,
                                      index=index, trim=plan["trim_frames"], render_frames=plan["render_frames"])
        return io.NodeOutput(
            out_state, root.as_posix(), scene_number, continuation, pass1, s["overlap"],
            plan["render_frames"], float(t["final_trim_start_seconds"]), plan["keep_frames"],
            float(t["final_trim_duration_seconds"]), s["seeds"][0], s["seeds"][1], s["selflift_seed"],
            audio, *refs, guide)


# --------------------------------------------------------------------------- Luna-H3-LoopProfile

# Defaults come from accepted recipes (a recipe is not a Peter decision unless he said so):
#  normal (v1.3): the accepted Elira and Shadow chain v2, Luna-Research long-video/2026-10-25-long-video/
#    elira-shadow/chain.json: nodes 95 strength 0.6, 136 0.0, 131 0.8, 132 0.0, 99 steps 12 simple,
#    102 transition_step 9; the unet, LoRA file and shifts 12 / 3 are the workflow defaults. Agrees with
#    root SPEC.md "Fixed for every song": SelfLift schedule 12 steps, 9 low + 3 high.
#  fast (V2): dojo-motion/renders/v2_std_dmad/scene_002_api.json, the take Peter accepted ("Sound is good,
#    motion is better. So for fast moving use V2, for normal use 1.3"): V2 unet, DMAD full critic 0.8 in
#    node 95 as the only LoRA, 8 steps, transition 5, no high model, shifts 12 / 3.
_RANK20 = "MiniMaxH3\\minimax_h3_ref2v_lightx2v_turbo_4step_v0.1_resized_avg_rank_20_bf16.safetensors"
DEFAULT_PROFILES = {
    "normal": {
        "unet_name": "MiniMaxH3\\Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
        "low_lora": [{"name": _RANK20, "strength": 0.6}, {"name": _RANK20, "strength": 0.0}],
        "high_lora": [{"name": _RANK20, "strength": 0.8}, {"name": _RANK20, "strength": 0.0}],
        "steps": 12, "transition_step": 9, "use_high_model": True,
        "shift_video": 12.0, "shift_audio": 3.0,
    },
    "fast": {
        "unet_name": "MiniMaxH3\\Minimax-h3_Singularity_ref2va_v2_pruned_int8_convrot.safetensors",
        "low_lora": [{"name": "MiniMaxH3\\dmad_minimax_h3_4step_full_critic_comfyui.safetensors",
                      "strength": 0.8}, {"name": _RANK20, "strength": 0.0}],
        "high_lora": [{"name": _RANK20, "strength": 0.8}, {"name": _RANK20, "strength": 0.0}],
        "steps": 8, "transition_step": 5, "use_high_model": False,
        "shift_video": 12.0, "shift_audio": 3.0,
    },
}
DEFAULT_PROFILES_JSON = json.dumps(DEFAULT_PROFILES, indent=2)


def parse_profiles(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{LOG} profiles is not valid JSON: {exc.msg} (line {exc.lineno}).") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{LOG} profiles must be a JSON object of named profiles.")
    for name, p in data.items():
        where = f"{LOG} profile {name!r}"
        if not isinstance(p, dict):
            raise ValueError(f"{where} must be an object.")
        if not isinstance(p.get("unet_name"), str) or not p["unet_name"]:
            raise ValueError(f"{where}: unet_name must be a file name.")
        for line in ("low_lora", "high_lora"):
            slots = p.get(line)
            if (not isinstance(slots, list) or len(slots) != 2
                    or any(not isinstance(x, dict) or not isinstance(x.get("name"), str)
                           or not isinstance(x.get("strength"), (int, float)) for x in slots)):
                raise ValueError(f"{where}: {line} must be two {{name, strength}} slots (strength 0 = off).")
        steps, trans = p.get("steps"), p.get("transition_step")
        if type(steps) is not int or type(trans) is not int or not 1 <= trans < steps:
            raise ValueError(f"{where}: need whole numbers 1 <= transition_step < steps.")
        if not isinstance(p.get("use_high_model"), bool):
            raise ValueError(f"{where}: use_high_model must be true or false.")
        for k in ("shift_video", "shift_audio"):
            if not isinstance(p.get(k), (int, float)) or isinstance(p.get(k), bool):
                raise ValueError(f"{where}: {k} must be a number.")
    return data


def pick_profile(profiles: dict, scene: dict, default: str) -> tuple[str, dict]:
    name = scene.get("profile") or default
    if name not in profiles:
        raise ValueError(f"{LOG} scene asks for profile {name!r}; profiles are {sorted(profiles)}.")
    return name, profiles[name]


class LunaH3LoopProfile(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Luna-H3-LoopProfile",
            display_name="Luna-H3-LoopProfile",
            category="Luna-MinimaxH3",
            description=(
                "Model, LoRAs, steps, transition and shifts for the current scene's profile (the scene's "
                "`profile` field: normal or fast). The name outputs are Any-type so they can feed the "
                "loaders' file-name combos. use_high_model false: feed SelfLift's high_res_model through "
                "a switch from the low model, which is what a disconnected input does.\n\n"
                "Defaults: normal = the accepted Elira and Shadow chain v2 recipe (elira-shadow/chain.json) "
                "and root SPEC 'SelfLift schedule 12 steps, 9 low + 3 high'; fast = the accepted dojo V2 take "
                "(dojo-motion/renders/v2_std_dmad): V2, DMAD full critic 0.8 alone, 8 steps, transition 5, "
                "no high model."
            ),
            inputs=[
                LoopState.Input("state", tooltip="From Luna-H3-LoopStart."),
                io.String.Input(
                    "profiles", multiline=True, default=DEFAULT_PROFILES_JSON,
                    tooltip="JSON object of named profiles. normal: elira-shadow/chain.json (nodes 95/136/131/"
                            "132/99/102) + root SPEC 12 steps 9+3. fast: dojo-motion v2_std_dmad "
                            "scene_002_api.json. Edit freely; strength 0 turns a LoRA slot off."),
                io.String.Input("default_profile", default="normal",
                                tooltip="Used for scenes that have no `profile` field."),
            ],
            outputs=[
                io.String.Output(id="profile_name", display_name="profile_name"),
                io.AnyType.Output(id="unet_name", display_name="unet_name", tooltip="To node 30."),
                io.AnyType.Output(id="low_lora_1_name", display_name="low_lora_1_name", tooltip="To node 95."),
                io.Float.Output(id="low_lora_1_strength", display_name="low_lora_1_strength"),
                io.AnyType.Output(id="low_lora_2_name", display_name="low_lora_2_name", tooltip="To node 136."),
                io.Float.Output(id="low_lora_2_strength", display_name="low_lora_2_strength"),
                io.AnyType.Output(id="high_lora_1_name", display_name="high_lora_1_name", tooltip="To node 131."),
                io.Float.Output(id="high_lora_1_strength", display_name="high_lora_1_strength"),
                io.AnyType.Output(id="high_lora_2_name", display_name="high_lora_2_name", tooltip="To node 132."),
                io.Float.Output(id="high_lora_2_strength", display_name="high_lora_2_strength"),
                io.Int.Output(id="steps", display_name="steps", tooltip="To node 99."),
                io.Int.Output(id="transition_step", display_name="transition_step", tooltip="To node 102."),
                io.Boolean.Output(id="use_high_model", display_name="use_high_model"),
                io.Float.Output(id="shift_video", display_name="shift_video", tooltip="To nodes 203 and 204."),
                io.Float.Output(id="shift_audio", display_name="shift_audio", tooltip="To nodes 203 and 204."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def validate_inputs(cls, profiles=None, **kwargs):
        if profiles is None:
            return True
        try:
            parse_profiles(profiles)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def execute(cls, state, profiles, default_profile="normal"):
        table = parse_profiles(profiles)
        scene = state["scenes"][int(state["scene_index"])]
        name, p = pick_profile(table, scene, default_profile)
        lo, hi = p["low_lora"], p["high_lora"]
        logger.info("%s profile %s: %s, %d steps, transition %d, high model %s", LOG, name, p["unet_name"],
                    p["steps"], p["transition_step"], p["use_high_model"])
        return io.NodeOutput(
            name, p["unet_name"], lo[0]["name"], float(lo[0]["strength"]), lo[1]["name"], float(lo[1]["strength"]),
            hi[0]["name"], float(hi[0]["strength"]), hi[1]["name"], float(hi[1]["strength"]),
            p["steps"], p["transition_step"], p["use_high_model"], float(p["shift_video"]), float(p["shift_audio"]))

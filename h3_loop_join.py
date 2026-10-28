"""Luna-H3-LoopEnd's join analyser: find the cut, match the colour, join the run into one video.

S11 of plans/S9-S14-loop-nodes.md. After the last scene, Luna-H3-LoopEnd reads the saved scenes (frames, and for a
carried scene its hidden head), decides for every join where inside the overlap to cut, optionally matches
scene k+1's colour to scene k's tail, and encodes one mp4 with sound. Per join it writes `join_NN.json` and
`join_NN_sheet.jpg`; the run gets `join_report.json`.

THE TIME MAPPING (why a cut can fall anywhere in the overlap)
A carried scene k+1 is rendered with the last frames of scene k as its head. Scene k+1's render frame j
(`F[j]`, head first, then its kept frames) therefore shows the same song time as scene k's kept frame
`off + j`, where `off = keep_k - warm_{k+1}` and `warm_{k+1}` is the hidden head Luna-H3-LoopEnd trims off scene k+1
(`trim_frames`). Scene k+1's kept frame 0 is `F[warm]`, which is scene k's frame `keep_k`: the original
boundary. Luna-H3-LoopScene proves this against Beta's own plan before the render (`plan_masked_context`'s start_frame
minus scene k's trim must equal `keep_k - warm`), so the head is a second picture of the last `warm` frames of
scene k, and a cut at k's frame `c` (`keep_k - warm < c <= keep_k`) shows k[0:c] and then F[c - off:].
With `time_lock` off the continuation may start `delta` frames earlier or later: F[c - off + delta:].

Frames are compared on small copies (160 px grey for scoring, 320 px for colour), then the encode streams the
full-size frames scene by scene, so memory stays at one scene.

Licence: GPL-3.0. Written for this pack; the join checks follow the skill scripts join_align.py /
join_diff_map.py (this project's own code).
"""

from __future__ import annotations

import json
import logging
import math
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    from .h3_loop import LOG, FORMAT_VERSION, _part, _scene_stem, _write_json, sha256_file  # noqa: F401
except ImportError:  # flat import in the unit tests
    from h3_loop import LOG, FORMAT_VERSION, _part, _scene_stem, _write_json, sha256_file  # noqa: F401

logger = logging.getLogger("Luna.H3LoopJoin")

FPS = 24
CUT_RULES = ("breath", "lowest_motion", "best_match", "manual")
BREATH_SOURCES = ("scene_audio", "breath_audio", "none")
FALLBACKS = ("crossfade", "hard_cut", "stop")
COLOR_MODES = ("off", "carried_only", "all")
AUDIO_XFADE_S = 0.010          # audio crossfade at every join (a crossfade fallback uses its own length)
SCORE_WIDTH = 160              # picture-difference and motion copies
COLOR_WIDTH = 320
CRF = 13
STD_RATIO = (0.8, 1.25)        # a colour match never stretches contrast by more than this

DEFAULTS = {
    "cut_rule": "best_match", "time_lock": True, "search_window": 0, "breath_source": "scene_audio",
    "breath_audio": "", "match_threshold": 8.0, "fallback": "crossfade", "crossfade_frames": 6,
    "color_match": "carried_only", "color_strength": 1.0, "manual_cuts": "", "report": True,
}


# --------------------------------------------------------------------------- settings

def parse_manual_cuts(text: str) -> dict[int, int]:
    """{join_index: frame} from JSON. A join is numbered by the scene that follows it (join 1 = scene 0 -> 1);
    the frame is the frame of the JOINED video where the next scene takes over."""
    if not text or not str(text).strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{LOG} manual_cuts is not valid JSON: {exc.msg}.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{LOG} manual_cuts must be a JSON object like {{\"1\": 330}}.")
    out = {}
    for k, v in data.items():
        try:
            join_no, frame = int(k), int(v)
        except (TypeError, ValueError):
            raise ValueError(f"{LOG} manual_cuts entry {k!r}: {v!r} must be join number: frame number.") from None
        if join_no < 1:
            raise ValueError(f"{LOG} manual_cuts join number {join_no} is not valid: joins are numbered from 1 "
                             "(1 = between scenes 0 and 1).")
        if frame < 0:
            raise ValueError(f"{LOG} manual_cuts frame {frame} for join {join_no} is not valid: frames start at 0.")
        out[join_no] = frame
    return out


def check_manual_early(manual: dict[int, int], index: int, count: int, carried: bool, warm: int,
                       keeps: list[int] | None, search_window: int, time_lock: bool) -> list[str]:
    """What Luna-H3-LoopScene can say about manual_cuts before scene `index` renders. Raises ValueError; returns notes.

    A join number beyond the run is an error. For join `index` (between scenes index-1 and index): on a fresh join
    the entry is ignored (a note); on a carried join with time_lock on, the joined-video frame must lie inside the
    overlap: sum(keep of scenes before index-1) + c with c in [max(1, keep_{index-1} - window + 1), keep_{index-1}]
    (with time_lock off earlier cuts shift the positions, so only the join analysis can check it).
    """
    notes = []
    for j in manual:
        if j > count - 1:
            raise ValueError(f"{LOG} manual_cuts names join {j} but the run has {max(count - 1, 0)} join(s).")
    frame = manual.get(index) if index >= 1 else None
    if frame is None:
        return notes
    if not carried:
        return [f"manual cut {frame} ignored: join {index} is a fresh join, a plain cut at the boundary"]
    if not time_lock or keeps is None or len(keeps) < index:
        return notes
    w = warm if int(search_window) <= 0 else min(int(search_window), warm)
    before = sum(keeps[:index - 1])
    lo, hi = before + max(1, keeps[index - 1] - w + 1), before + keeps[index - 1]
    if not lo <= frame <= hi:
        raise ValueError(f"{LOG} manual cut {frame} for join {index} falls outside the overlap: with this scene's "
                         f"{warm}-frame head and search_window {search_window} the joined-video frame must be "
                         f"{lo}..{hi}.")
    return notes


def check_settings(**kw) -> dict:
    """The settings with defaults, validated. Raises ValueError saying what is wrong."""
    s = {**DEFAULTS, **{k: v for k, v in kw.items() if v is not None}}
    unknown = set(s) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"{LOG} unknown join setting(s): {sorted(unknown)}")
    for key, options in (("cut_rule", CUT_RULES), ("breath_source", BREATH_SOURCES), ("fallback", FALLBACKS),
                         ("color_match", COLOR_MODES)):
        if s[key] not in options:
            raise ValueError(f"{LOG} {key} must be one of {options}, got {s[key]!r}.")
    if not isinstance(s["time_lock"], bool) or not isinstance(s["report"], bool):
        raise ValueError(f"{LOG} time_lock and report must be true or false.")
    if int(s["search_window"]) < 0 or int(s["crossfade_frames"]) < 0:
        raise ValueError(f"{LOG} search_window and crossfade_frames must be >= 0.")
    if not (math.isfinite(float(s["match_threshold"])) and float(s["match_threshold"]) >= 0):
        raise ValueError(f"{LOG} match_threshold must be a number >= 0.")
    if not 0.0 <= float(s["color_strength"]) <= 1.0:
        raise ValueError(f"{LOG} color_strength must be between 0 and 1.")
    s["manual"] = parse_manual_cuts(s["manual_cuts"])
    if s["cut_rule"] == "manual" and not s["manual"]:
        raise ValueError(f"{LOG} cut_rule manual needs an entry in manual_cuts, e.g. {{\"1\": 330}}.")
    if s["cut_rule"] == "breath" and s["breath_source"] == "none":
        raise ValueError(f"{LOG} cut_rule breath needs a breath_source (scene_audio or breath_audio).")
    if s["breath_source"] == "breath_audio" and not str(s["breath_audio"]).strip():
        raise ValueError(f"{LOG} breath_source breath_audio needs a breath_audio file path.")
    return s


# --------------------------------------------------------------------------- reading frames

def frame_source(root: Path, stem: str, frames_info: dict) -> dict:
    return {"root": Path(root), "stem": stem, "format": frames_info["format"], "count": int(frames_info["count"])}


def iter_frames(src: dict, start: int = 0) -> Iterator[np.ndarray]:
    """Full-size frames [start:] as uint8 HxWx3, from the mp4 or the PNG folder a scene was saved as."""
    if src["format"] == "mp4":
        import av
        with av.open(str(src["root"] / f"{src['stem']}.mp4")) as container:
            for k, frame in enumerate(container.decode(video=0)):
                if k >= start:
                    yield frame.to_ndarray(format="rgb24")
    else:
        from PIL import Image
        folder = src["root"] / f"{src['stem']}_frames"
        for k in range(start, src["count"]):
            with Image.open(folder / f"frame_{k:05d}.png") as im:
                yield np.asarray(im.convert("RGB"))


def _resize(frame: np.ndarray, width: int) -> np.ndarray:
    import cv2
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def read_small(src: dict, width: int = COLOR_WIDTH) -> np.ndarray:
    frames = [_resize(f, width) for f in iter_frames(src)]
    if len(frames) != src["count"]:
        raise ValueError(f"{LOG} {src['stem']} holds {len(frames)} frames but its manifest says {src['count']}.")
    return np.stack(frames)


def to_grey(frames: np.ndarray, width: int = SCORE_WIDTH) -> np.ndarray:
    """[N,h,w,3] uint8 -> [N,h',w'] float32 grey at `width`."""
    import cv2
    out = []
    for f in frames:
        g = cv2.cvtColor(_resize(f, width) if f.shape[1] != width else f, cv2.COLOR_RGB2GRAY)
        out.append(g)
    return np.stack(out).astype(np.float32)


# --------------------------------------------------------------------------- colour match

def apply_color(frames: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """y = a * x + b per channel, clipped; any shape ending in 3 channels."""
    out = frames.astype(np.float32) * a.astype(np.float32) + b.astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def fit_color(source: np.ndarray, target: np.ndarray, strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std transfer: the (a, b) that bring `source` frames to `target` frames' statistics.

    Mean/std rather than a histogram match: a histogram match is exact on the frames it was fitted to but
    reshapes the whole tone curve, which bands and posterises on a scene whose content differs (a fresh join,
    or the rest of scene k+1 after the overlap). Mean/std is two numbers per channel, moves everything
    smoothly, and the std ratio is bounded to 0.8-1.25 so a flat overlap cannot stretch the contrast.
    `strength` blends the correction in: 0 leaves the frames as they are.
    """
    s = source.reshape(-1, 3).astype(np.float64)
    t = target.reshape(-1, 3).astype(np.float64)
    ratio = np.clip(t.std(axis=0) / np.maximum(s.std(axis=0), 1e-3), *STD_RATIO)
    a = ratio
    b = t.mean(axis=0) - ratio * s.mean(axis=0)
    st = float(strength)
    return 1.0 + st * (a - 1.0), st * b


def channel_mean_gap(x: np.ndarray, y: np.ndarray) -> float:
    """Mean over channels of |mean(x) - mean(y)|, 0-255: how far apart two sets of frames look in colour."""
    return float(np.abs(x.reshape(-1, 3).mean(axis=0) - y.reshape(-1, 3).mean(axis=0)).mean())


# --------------------------------------------------------------------------- the analysis

def picture_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def score_table(k_grey: np.ndarray, f_grey: np.ndarray, off: int, lo: int, hi: int, deltas: range) -> dict:
    """D[(c, delta)] = mean |k[c-1] - F[c-1-off+delta]|: the frame k shows last against the one k+1 would show
    at the same song time (shifted by delta)."""
    table = {}
    n = len(f_grey)
    for c in range(lo, hi + 1):
        js = [(c - 1 - off + d, d) for d in deltas if 0 <= c - 1 - off + d < n]
        if not js:
            continue
        stack = f_grey[[j for j, _ in js]]
        diffs = np.abs(stack - k_grey[c - 1][None]).reshape(len(js), -1).mean(axis=1)
        for (j, d), v in zip(js, diffs):
            table[(c, d)] = float(v)
    return table


def best_of(table: dict, c: int | None = None) -> tuple[int, int, float]:
    """Lowest score (optionally at one cut); ties go to the later cut, then the smaller shift."""
    items = [(k, v) for k, v in table.items() if c is None or k[0] == c]
    if not items:
        raise ValueError(f"{LOG} no frame pair to compare at this cut.")
    (cc, dd), v = min(items, key=lambda kv: (round(kv[1], 6), -kv[0][0], abs(kv[0][1])))
    return cc, dd, v


def motion_by_cut(k_grey: np.ndarray, lo: int, hi: int) -> dict[int, float]:
    """Optical-flow magnitude (Farneback, mean over the frame) of the step into k's frame c-1."""
    import cv2
    out = {}
    for c in range(lo, hi + 1):
        if c < 2:
            out[c] = float("inf")          # no earlier frame to measure from: chosen only if it is the only cut
            continue
        flow = cv2.calcOpticalFlowFarneback(k_grey[c - 2].astype(np.uint8), k_grey[c - 1].astype(np.uint8),
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        out[c] = float(np.linalg.norm(flow, axis=2).mean())
    return out


def argmin_late(values: dict[int, float]) -> int:
    """The key with the lowest value; ties go to the later one."""
    if not values:
        raise ValueError(f"{LOG} no candidate cut to choose from.")
    return min(values, key=lambda c: (round(values[c], 9), -c))


def choose_cut(k_grey, f_grey, geom: dict, cfg: dict, energy=None, motion=None, manual_c=None) -> dict:
    """Decide (c, delta) for one carried join. `geom`: off, keep, warm. Returns the choice and its evidence."""
    off, keep, warm = geom["off"], geom["keep"], geom["warm"]
    w = warm if int(cfg["search_window"]) <= 0 else min(int(cfg["search_window"]), warm)
    lo, hi = max(1, keep - w + 1), keep
    deltas = range(0, 1) if cfg["time_lock"] else range(-w, w + 1)
    table = score_table(k_grey, f_grey, off, lo, hi, deltas)
    zero = {c: v for (c, d), v in table.items() if d == 0}
    if manual_c is not None:
        if not lo <= manual_c <= hi:
            raise ValueError(f"{LOG} manual cut falls outside this join's overlap (scene frames {lo}..{hi}).")
        c, rule = manual_c, "manual"
    elif cfg["cut_rule"] == "best_match":
        c, rule = best_of(table)[0], "best_match"
    elif cfg["cut_rule"] == "lowest_motion":
        motion = motion if motion is not None else motion_by_cut(k_grey, lo, hi)
        c, rule = argmin_late({c: m for c, m in motion.items() if lo <= c <= hi}), "lowest_motion"
    elif cfg["cut_rule"] == "breath":
        values = {c: energy(c) for c in range(lo, hi + 1)}
        c, rule = argmin_late(values), "breath"
    else:
        raise ValueError(f"{LOG} cut_rule manual needs an entry in manual_cuts for this join.")
    c, d, score = best_of(table, c)
    natural = zero.get(keep)
    return {"c": c, "delta": d, "score": score, "rule": rule, "window": [lo, hi],
            "natural_score": natural, "scores": {str(cc): round(v, 3) for cc, v in sorted(zero.items())}}


# --------------------------------------------------------------------------- audio helpers

def mono(wave: np.ndarray) -> np.ndarray:
    return wave.mean(axis=0) if wave.ndim == 2 else wave


def rms(wave: np.ndarray, first: int, last: int) -> float:
    seg = mono(wave)[max(0, first):last]
    return float("inf") if seg.size == 0 else float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))


def crossfade_audio(prev_tail: np.ndarray, next_before: np.ndarray | None) -> np.ndarray:
    """Join two [C, L] pieces. With `next_before` (the same song time from the next scene) a linear blend;
    without it (a fresh join) a cosine fade out and in, so nothing clicks."""
    n = prev_tail.shape[1]
    if n == 0:
        return prev_tail
    if next_before is not None:
        ramp = (np.arange(n, dtype=np.float32) + 1) / (n + 1)
        return prev_tail * (1 - ramp) + next_before * ramp
    return prev_tail


def fade_edges(piece: np.ndarray, n: int, out: bool) -> np.ndarray:
    n = min(n, piece.shape[1])
    if n <= 0:
        return piece
    ramp = 0.5 * (1 + np.cos(np.linspace(0, np.pi, n, dtype=np.float32)))      # 1 -> 0
    piece = piece.copy()
    if out:
        piece[:, -n:] *= ramp
    else:
        piece[:, :n] *= ramp[::-1]
    return piece


# --------------------------------------------------------------------------- loading a run

def load_run(root: Path, count: int) -> list[dict] | None:
    """The saved scenes with everything the join needs, or None if the run cannot be joined (a scene with no
    frames or no plan: an S9 body, or a scene saved before S10)."""
    scenes = []
    for i in range(count):
        path = root / f"{_scene_stem(i)}.manifest.json"
        m = json.loads(path.read_text(encoding="utf-8"))
        info = m.get("info", {})
        if "frames" not in info or "plan" not in info:
            return None
        plan = info["plan"]
        sc = {"index": i, "manifest": m, "plan": plan, "stem": _scene_stem(i),
              "keep": int(info["frames"]["count"]), "fps": float(info["frames"].get("fps", FPS)),
              "src": frame_source(root, _scene_stem(i), info["frames"]),
              "carried": bool(plan.get("continuation")), "head": None, "warm": 0}
        if sc["keep"] != int(plan["keep_frames"]):
            raise ValueError(f"{LOG} scene {i} saved {sc['keep']} frames but its plan kept {plan['keep_frames']}.")
        head = info.get("head")
        if sc["carried"] and head:
            sc["head"] = frame_source(root, f"{_scene_stem(i)}_head", head)
            sc["warm"] = int(head["frames"])
            sc["map"] = head["mapping"]
        scenes.append(sc)
    return scenes


# --------------------------------------------------------------------------- the join

def _scene_frames_small(sc: dict) -> np.ndarray:
    """F: head (if any) followed by the kept frames, at COLOR_WIDTH."""
    kept = read_small(sc["src"])
    if sc["head"] is not None:
        return np.concatenate([read_small(sc["head"]), kept])
    return kept


def _load_audio(path):
    """([C, T] float32 numpy, rate) of an audio file."""
    try:
        from .h3_loop_scene import load_audio_file
    except ImportError:
        from h3_loop_scene import load_audio_file
    wave, rate = load_audio_file(Path(path))
    return wave.numpy(), int(rate)


def _scene_audio(root: Path, sc: dict):
    """([C, T] audio on the scene's F timeline: head then kept, rate), or None if the scene saved no audio."""
    w = root / f"{sc['stem']}.wav"
    if not w.is_file():
        return None
    kept, rate = _load_audio(w)
    hw = root / f"{sc['stem']}_head.wav"
    if sc["head"] is not None:
        if not hw.is_file():
            return None
        head, _ = _load_audio(hw)
        kept = np.concatenate([head, kept], axis=1)
    return kept, rate


def _energy_fn(cfg: dict, root: Path, k: dict, audio_mode: str):
    """energy(c): RMS of the breath source around the cut after scene k's kept frame c-1 (two frames wide)."""
    source = cfg["breath_source"]
    cache: dict = {}

    def window(wave, rate, centre_s):
        return rms(wave, round((centre_s - 1 / FPS) * rate), round((centre_s + 1 / FPS) * rate))

    def fn(c: int) -> float:
        t = k["plan"]["start_seconds"] + c / FPS
        if source == "breath_audio":
            wave, rate = cache.setdefault("breath", _load_audio(cfg["breath_audio"]))
            return window(wave, rate, t)
        if audio_mode == "generated":
            got = cache.setdefault("scene", _scene_audio(root, k))
            if got is None:
                raise ValueError(f"{LOG} cut_rule breath needs scene {k['index']}'s audio saved.")
            wave, rate = got
            return window(wave, rate, (k["warm"] + c) / FPS)
        wave, rate = cache.setdefault("lock", _load_audio(k["plan"]["lock_audio"]))
        return window(wave, rate, t)
    return fn


def frame_round(seconds: float) -> int:
    """Seconds to a frame number the way Beta rounds a timeline boundary (half up, on the decimal string)."""
    from decimal import Decimal, ROUND_HALF_UP
    return int((Decimal(str(seconds)) * FPS).to_integral_value(rounding=ROUND_HALF_UP))


def song_drift(scenes: list[dict], i: int) -> dict:
    """How far scene i's song time is from where the joined video's single master slice puts it.

    The master is cut from scene 0's start rounded to the frame grid (the picture's own grid), and scene i's
    first kept frame sits sum(keep_j, j < i) frames in. Beta keeps keep_j = round(end_j * 24) - round(start_j * 24),
    which telescopes for contiguous scenes: the position is exactly round(start_i * 24) - round(start_0 * 24), so the
    drift is only the rounding of start_i itself, at most half a frame, however many scenes there are.
    """
    start = [sc["plan"]["start_seconds"] for sc in scenes]
    keeps = [sc["keep"] for sc in scenes]
    placed = (frame_round(start[0]) + sum(keeps[:i])) / FPS
    local = start[i] - (start[i - 1] + keeps[i - 1] / FPS) if i else 0.0
    return {"frames": round((start[i] - placed) * FPS, 4), "local_frames": round(local * FPS, 4)}


def analyse_run(root: Path, scenes: list[dict], cfg: dict) -> dict:
    """Decide every join and plan the emission. Returns joins, per-scene colour transforms and, for scene i, the
    F_i index range [start[i], end[i]) it contributes to the joined video (pos[i] = where it starts there)."""
    n = len(scenes)
    audio_mode = scenes[0]["plan"].get("audio_mode", "song_lock")
    if audio_mode == "song_lock" and not cfg["time_lock"]:
        raise ValueError(f"{LOG} a song-locked run cannot shift time: turn time_lock on, or render with "
                         "audio_mode generated.")
    if cfg["breath_source"] == "breath_audio" and audio_mode == "generated":
        raise ValueError(f"{LOG} breath_audio is a file on the song's timeline; a generated run has none.")
    for k in cfg["manual"]:
        if k > n - 1:
            raise ValueError(f"{LOG} manual_cuts names join {k} but the run has {n - 1} join(s).")
    identity = (np.ones(3), np.zeros(3))
    transforms = [identity] * n
    start_idx, end_idx, pos = [0] * n, [0] * n, [0] * n
    joins: list[dict] = []
    small_prev = _scene_frames_small(scenes[0])
    grey_prev = None
    for i in range(1, n):
        k, k1 = scenes[i - 1], scenes[i]
        carried = k1["carried"] and k1["head"] is not None
        small = _scene_frames_small(k1)
        wk = k["warm"] if k["head"] is not None else 0
        kept_k = small_prev[wk:]
        rec: dict[str, Any] = {
            "join": i, "scene_before": i - 1, "scene_after": i, "type": "carried" if carried else "fresh",
            "notes": [] if carried or not k1["carried"] else ["scene saved without its head: plain cut at the boundary"]}
        rec["drift"] = song_drift(scenes, i)
        if audio_mode == "song_lock" and abs(rec["drift"]["frames"]) > 0.5 + 1e-6:
            raise ValueError(f"{LOG} scene {i} starts {rec['drift']['frames']:+.2f} frames away from where the single "
                             "master slice puts it (the scenes are not contiguous in song time); the sound would not "
                             "match the picture. Render a contiguous scene list, or use audio_mode generated.")
        if not carried and i in cfg["manual"]:
            rec["notes"].append(f"manual cut {cfg['manual'][i]} ignored: a fresh join is a plain cut at the boundary")
        # ---- colour: scene k+1 is matched to scene k's tail, one constant correction for the whole scene
        do_color = cfg["color_match"] == "all" or (cfg["color_match"] == "carried_only" and carried)
        rec["color"] = None
        if do_color and cfg["color_strength"] > 0:
            if carried:
                off = int(k1["map"]["offset"])
                src, dst = small[:k1["warm"]], kept_k[off:off + k1["warm"]]
            else:
                m = min(12, len(kept_k), len(small))
                src, dst = small[:m], kept_k[-m:]
            m = min(len(src), len(dst))
            idx = np.linspace(0, m - 1, min(m, 24)).astype(int)
            a, b = fit_color(src[idx], dst[idx], cfg["color_strength"])
            rec["color"] = {"a": [round(float(x), 5) for x in a], "b": [round(float(x), 3) for x in b],
                            "gap_before": round(channel_mean_gap(src[idx], dst[idx]), 3),
                            "gap_after": round(channel_mean_gap(apply_color(src[idx], a, b), dst[idx]), 3)}
            small = apply_color(small, a, b)
            transforms[i] = (a, b)
        grey = to_grey(small)
        if grey_prev is None:
            grey_prev = to_grey(small_prev)
        # ---- the cut
        if carried:
            off = int(k1["map"]["offset"])
            geom = {"off": off, "keep": k["keep"], "warm": k1["warm"]}
            manual_frame = cfg["manual"].get(i)
            manual_c = None if manual_frame is None else manual_frame - pos[i - 1] + start_idx[i - 1] - wk
            energy = _energy_fn(cfg, root, k, audio_mode) if cfg["cut_rule"] == "breath" and manual_c is None else None
            dec = choose_cut(grey_prev[wk:], grey, geom, cfg, energy=energy, manual_c=manual_c)
            c, delta = dec["c"], dec["delta"]
            seamless = dec["score"] <= cfg["match_threshold"]
            action = "cut" if seamless else cfg["fallback"]
            rec.update(decision=dec, seamless=bool(seamless), action=action, delta=delta, cut_scene_frame=c,
                       overlap={"offset": off, "head_frames": k1["warm"], "mapping": k1["map"]})
            if action == "stop":
                joins.append(rec)
                return {"joins": joins, "stopped": i}
            nfade = 0
            if action == "crossfade":
                nfade = max(0, min(int(cfg["crossfade_frames"]), c - 1, c - off + delta))
            rec["crossfade_frames"] = nfade
            start_idx[i], end_idx[i - 1] = c - off + delta, wk + c
        else:
            gap = float(np.abs(grey[0] - grey_prev[-1]).mean())
            rec.update(decision={"c": k["keep"], "delta": 0, "score": round(gap, 3), "rule": "boundary"},
                       seamless=gap <= cfg["match_threshold"], action="cut", crossfade_frames=0,
                       cut_scene_frame=k["keep"], delta=0)
            start_idx[i], end_idx[i - 1] = 0, wk + k["keep"]
        pos[i] = pos[i - 1] + (end_idx[i - 1] - start_idx[i - 1])
        rec["joined_frame"] = pos[i]
        rec["song_time_seconds"] = round(k["plan"]["start_seconds"] + rec["cut_scene_frame"] / FPS, 4)
        joins.append(rec)
        small_prev, grey_prev = small, grey
    end_idx[n - 1] = scenes[n - 1]["warm"] + scenes[n - 1]["keep"]
    total = pos[n - 1] + (end_idx[n - 1] - start_idx[n - 1])
    return {"joins": joins, "transforms": transforms, "start": start_idx, "end": end_idx, "pos": pos,
            "total_frames": total, "stopped": None}


# --------------------------------------------------------------------------- encode

def blend_frames(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return np.clip(np.rint(a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t), 0, 255).astype(np.uint8)


def emit_scene(read_from, s: int, e: int, nin: int, nout: int, pending, emit, blend):
    """Emit one scene's part of the joined picture and return the frames its successor will blend with.

    The scene contributes F[s:e). The `nin` frames F[s-nin:s) are the partners of the previous scene's `pending`
    tail (blended and emitted first); F[s:e-nout) go out as they are; F[e-nout:e) are returned, not emitted, for the
    next scene to blend. The three windows may overlap when the scene is short (e - s < nout, possible with a time
    shift), so frames are read once, in order, from the earliest one needed, and the tail is a sliding window.
    """
    from collections import deque
    lo = min(s - nin, e - nout)
    if lo < 0 or e <= s:
        raise ValueError(f"{LOG} cannot plan frames {s}..{e} with {nin} blended in and {nout} blended out.")
    it = read_from(lo)
    pre: list = []
    tail: deque = deque(maxlen=max(nout, 1))
    blended = nin == 0
    for idx in range(lo, e):
        frame = next(it)
        if idx >= s and not blended:
            for k, (a_, b_) in enumerate(zip(pending, pre)):
                emit(blend(a_, b_, (k + 1) / (nin + 1)))
            blended = True
        if s - nin <= idx < s:
            pre.append(frame)
        if s <= idx < e - nout:
            emit(frame)
        if nout:
            tail.append(frame)
    return list(tail) if nout else None


def _scene_iter(sc: dict, start: int) -> Iterator[np.ndarray]:
    """F_i frames from index `start` on: the head (if any), then the kept frames."""
    head_n = sc["warm"] if sc["head"] is not None else 0
    if start < head_n:
        yield from iter_frames(sc["head"], start)
        yield from iter_frames(sc["src"])
    else:
        yield from iter_frames(sc["src"], start - head_n)


class _AudioFeed:
    """Feeds the aac stream in step with the video, so the mp4 is interleaved."""

    def __init__(self, container, astream, wave: np.ndarray | None, rate: int):
        self.container, self.stream, self.wave, self.rate = container, astream, wave, rate
        self.done = 0

    def advance(self, frames: int, final: bool = False) -> None:
        if self.wave is None:
            return
        import av
        target = self.wave.shape[1] if final else min(self.wave.shape[1], round(frames / FPS * self.rate))
        layout = "stereo" if self.wave.shape[0] == 2 else "mono"
        while self.done < target:
            n = min(4096, target - self.done)
            chunk = np.ascontiguousarray(self.wave[:, self.done:self.done + n].astype(np.float32))
            frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=layout)
            frame.sample_rate = self.rate
            frame.pts = self.done
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            self.done += n
        if final:
            for packet in self.stream.encode():
                self.container.mux(packet)


def _encode(root: Path, scenes: list[dict], plan: dict, out_path: Path, wave, rate) -> None:
    import av
    n = len(scenes)
    first = next(iter_frames(scenes[0]["src"]))
    h, w = first.shape[:2]
    if h % 2 or w % 2:
        raise ValueError(f"{LOG} the frames are {w}x{h}; yuv420p needs even sizes.")
    container = av.open(str(_part(out_path)), "w", format="mp4")
    vstream = container.add_stream("libx264", rate=Fraction(FPS))
    vstream.width, vstream.height, vstream.pix_fmt = w, h, "yuv420p"
    vstream.options = {"crf": str(CRF), "preset": "medium"}
    astream = None
    if wave is not None:
        astream = container.add_stream("aac", rate=rate, layout="stereo" if wave.shape[0] == 2 else "mono")
    feed = _AudioFeed(container, astream, wave, rate)
    written = 0

    def emit(arr: np.ndarray) -> None:
        nonlocal written
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in vstream.encode(frame):
            container.mux(packet)
        written += 1
        if written % 24 == 0:
            feed.advance(written)

    fade_in = [0] * n
    for j in plan["joins"]:
        fade_in[j["scene_after"]] = int(j.get("crossfade_frames", 0))

    def corrected(it, i):
        a, b = plan["transforms"][i]
        neutral = not ((a != 1).any() or (b != 0).any())
        for arr in it:
            yield arr if neutral else apply_color(arr, a, b)

    pending = None
    for i, sc in enumerate(scenes):
        nout = fade_in[i + 1] if i + 1 < n else 0
        pending = emit_scene(lambda lo, sc=sc, i=i: corrected(_scene_iter(sc, lo), i),
                             plan["start"][i], plan["end"][i], fade_in[i], nout, pending, emit, blend_frames)
    for packet in vstream.encode():
        container.mux(packet)
    feed.advance(written, final=True)
    container.close()
    import os
    os.replace(_part(out_path), out_path)


def build_audio(root: Path, scenes: list[dict], plan: dict):
    """The joined soundtrack as ([C, T] float32, rate), or (None, None) when it cannot be built.

    song_lock: the lock file from the run's first scene start for exactly the joined length: the cut never moves
    time, so the master stays in sync. generated: each scene's own audio, cut at the picture's cut, with a ~10 ms
    crossfade at every join (a crossfade fallback uses the same length in audio as in frames)."""
    total = plan["total_frames"]
    if scenes[0]["plan"].get("audio_mode", "song_lock") == "song_lock":
        lock = scenes[0]["plan"].get("lock_audio")
        if not lock or not Path(lock).is_file():
            logger.warning("%s the lock audio is not available; the joined video has no sound.", LOG)
            return None, None
        wave, rate = _load_audio(lock)
        # From the first scene's start on the frame grid, the grid the picture uses: the audio then sits on the same
        # quantised time as the frames, and drift stays within half a frame (see song_drift).
        first, length = round(frame_round(scenes[0]["plan"]["start_seconds"]) / FPS * rate), round(total / FPS * rate)
        out = np.zeros((wave.shape[0], length), dtype=np.float32)
        seg = wave[:, first:first + length]
        out[:, :seg.shape[1]] = seg
        return out, rate
    got = [_scene_audio(root, sc) for sc in scenes]
    if any(g is None for g in got):
        logger.warning("%s a scene has no saved audio; the joined video has no sound.", LOG)
        return None, None
    rate = got[0][1]
    audios = [g[0] for g in got]
    per = rate / FPS
    xf = max(1, round(AUDIO_XFADE_S * rate))
    s0 = [round(plan["start"][i] * per) for i in range(len(scenes))]
    e0 = [round(plan["end"][i] * per) for i in range(len(scenes))]
    modes = []                                    # per join: (kind, L)
    for j in plan["joins"]:
        i = j["scene_after"]
        carried = j["type"] == "carried"
        L = max(xf, round(int(j.get("crossfade_frames", 0)) * per)) if carried else xf
        L = max(1, min(L, e0[i - 1] - s0[i - 1]))
        modes.append(("blend" if carried and s0[i] >= L else "fade", L))
    chunks = []
    for i, a in enumerate(audios):
        mode_in = modes[i - 1] if i > 0 else None
        mode_out = modes[i] if i + 1 < len(audios) else None
        end = e0[i] - (mode_out[1] if mode_out and mode_out[0] == "blend" else 0)
        body = a[:, s0[i]:end]
        if mode_in and mode_in[0] == "fade":
            body = fade_edges(body, mode_in[1], out=False)
        if mode_out and mode_out[0] == "fade":
            body = fade_edges(body, mode_out[1], out=True)
        chunks.append(body)
        if mode_out and mode_out[0] == "blend":
            L = mode_out[1]
            chunks.append(crossfade_audio(a[:, e0[i] - L:e0[i]], audios[i + 1][:, s0[i + 1] - L:s0[i + 1]]))
    return np.concatenate(chunks, axis=1).astype(np.float32), rate


# --------------------------------------------------------------------------- the sheet

def _read_output_frames(path: Path, wanted: set[int], width: int = 640) -> dict[int, np.ndarray]:
    import av
    out = {}
    last = max(wanted) if wanted else -1
    with av.open(str(path)) as container:
        for k, frame in enumerate(container.decode(video=0)):
            if k in wanted:
                out[k] = _resize(frame.to_ndarray(format="rgb24"), width)
            if k >= last:
                break
    return out


def write_sheet(video: Path, joined_frame: int, path: Path, total: int, label: str) -> None:
    """Frames J-3..J+2 around the cut and the x8 difference maps between neighbours (inferno), like
    join_diff_map.py. J is the first frame of the new scene."""
    import cv2
    ids = [min(max(k, 0), total - 1) for k in range(joined_frame - 3, joined_frame + 3)]
    frames = _read_output_frames(video, set(ids))
    tiles = []
    for k in ids:
        t = cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR).copy()
        cv2.putText(t, f"{k}" + ("  <- new scene" if k == joined_frame else ""), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(t)
    maps = []
    for a, b in zip(ids[:-1], ids[1:]):
        d = np.abs(frames[a].astype(np.int16) - frames[b].astype(np.int16)).mean(axis=2)
        m = cv2.applyColorMap(np.clip(d * 8, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.putText(m, f"diff {a}->{b} x8 mean {d.mean():.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        maps.append(m)
    blank = np.zeros_like(tiles[0])
    cv2.putText(blank, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    sheet = np.vstack([np.hstack(tiles), np.hstack(maps + [blank])])
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError(f"{LOG} could not encode {path.name}")
    _part(path).write_bytes(buf.tobytes())
    import os
    os.replace(_part(path), path)


# --------------------------------------------------------------------------- the whole join

def join_run(root: Path, run_name: str, count: int, cfg: dict) -> dict | None:
    """Join a finished run. Returns {"video", "report", "joins", ...}; None if the run cannot be joined (scenes
    without frames or plan). Raises ValueError if a join falls back to `stop`."""
    started = time.time()
    scenes = load_run(root, count)
    if scenes is None:
        logger.info("%s the run has scenes without frames or a plan; nothing to join.", LOG)
        return None
    plan = analyse_run(root, scenes, cfg)
    if cfg["report"]:
        _write_join_files(root, plan["joins"], None, cfg)
    if plan["stopped"] is not None:
        raise ValueError(f"{LOG} join {plan['stopped']} is above match_threshold {cfg['match_threshold']} "
                         f"(score {plan['joins'][-1]['decision']['score']:.2f}) and fallback is stop; the scenes "
                         f"are saved in {root}. Re-roll scene {plan['stopped']} or change the join settings.")
    wave, rate = build_audio(root, scenes, plan)
    out = root / "joined.mp4"
    _encode(root, scenes, plan, out, wave, rate)
    result = {"video": out, "report": None, "joins": plan["joins"], "total_frames": plan["total_frames"],
              "audio": None if wave is None else {"samples": int(wave.shape[1]), "rate": rate,
                                                  "mode": scenes[0]["plan"].get("audio_mode", "song_lock")}}
    if cfg["report"]:
        for rec in plan["joins"]:
            write_sheet(out, rec["joined_frame"], root / f"join_{rec['join']:02d}_sheet.jpg", plan["total_frames"],
                        f"join {rec['join']} {rec['type']} {rec['action']} score {rec['decision']['score']}")
        _write_join_files(root, plan["joins"], out, cfg)
        report = root / "join_report.json"
        _write_json(report, {
            "format_version": FORMAT_VERSION, "run_name": run_name, "scenes": count,
            "joined_video": out.name, "joined_sha256": sha256_file(out), "total_frames": plan["total_frames"],
            "duration_seconds": round(plan["total_frames"] / FPS, 4), "audio": result["audio"],
            "crf": CRF, "encoder": "PyAV libx264 yuv420p + aac", "elapsed_seconds": round(time.time() - started, 1),
            "settings": {k: cfg[k] for k in DEFAULTS}, "joins": [
                {k: r[k] for k in ("join", "type", "action", "seamless", "joined_frame", "song_time_seconds")}
                | {"score": r["decision"]["score"], "file": f"join_{r['join']:02d}.json"} for r in plan["joins"]]})
        result["report"] = report
    logger.info("%s joined %d scene(s) into %s: %d frames, %d join(s)", LOG, count, out.name, plan["total_frames"],
                len(plan["joins"]))
    return result


def _write_join_files(root: Path, joins: list[dict], video: Path | None, cfg: dict) -> None:
    for rec in joins:
        body = dict(rec, settings={k: cfg[k] for k in DEFAULTS},
                    sheet=f"join_{rec['join']:02d}_sheet.jpg" if video is not None else None)
        _write_json(root / f"join_{rec['join']:02d}.json", body)

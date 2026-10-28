"""Unit tests for Luna-H3-LoopEnd's join analyser (S11), on synthetic frames and audio, CPU only.

The world is a textured picture with a white square that moves one pixel per frame: the square's x gives the
song time of any frame, in the sources and in the joined video, so a cut that lands on the wrong frame or a
time shift shows as a wrong position. Scenes are saved through the real `save_scene` (PNG frames, float WAV),
with the plan and head mapping Luna-H3-LoopScene writes, then joined with `join_run`.

Run from this folder: `pytest . --basetemp=D:/t/ComfyUI-LunaMiniMaxH3`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("comfy_api.latest")
pytest.importorskip("cv2")
pytest.importorskip("av")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import h3_loop  # noqa: E402
import h3_loop_join as hj  # noqa: E402

W, H = 96, 64
RATE = 48000
PER = RATE // 24                 # samples per frame


# --------------------------------------------------------------------------- the synthetic world

def world(t: int, tint=(0, 0, 0), bump: float = 0.0, scene_noise: int = 0) -> np.ndarray:
    """Frame at global time t: texture + white square at x = t (+ noise that depends on t only)."""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([40 + (xx * 2) % 60, 40 + (yy * 3) % 50, np.full((H, W), 70)], axis=-1).astype(np.float32)
    rng = np.random.default_rng(1000 + t + 7919 * scene_noise)
    img += rng.normal(0, 1.5, img.shape)
    x = int(t) % (W - 12)
    img[20:32, x:x + 12] = 255
    img += np.array(tint, dtype=np.float32) + bump
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def square_x(frame: np.ndarray) -> int:
    cols = np.where((frame[20:32] > 190).all(axis=2).all(axis=0))[0]
    return int(cols[0]) if len(cols) else -1


def assert_positions(frames, first_t: int, tol: int = 1) -> None:
    """Output frame k shows song time first_t + k: the square is at x = (first_t + k) % (W - 12), within `tol`
    pixels (the mp4 is lossy)."""
    bad = [(k, square_x(f), (first_t + k) % (W - 12)) for k, f in enumerate(frames)
           if abs(square_x(f) - (first_t + k) % (W - 12)) > tol]
    assert not bad, f"frames on the wrong song time (frame, found x, expected x): {bad[:5]}"


def tone(freq: float, n: int, amp: float = 0.3) -> np.ndarray:
    t = np.arange(n) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def stereo(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack([x, x]))[None]               # [1, 2, T]


def fake_beta(root: Path, number: int, token_count=37, tail=4) -> None:
    folder = root / "latents"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"scene_{number:03d}.latent").write_bytes(b"x")
    (folder / f"scene_{number:03d}.latent.json").write_text(
        json.dumps({"token_count": str(token_count), "tail_padding_frames": str(tail)}), encoding="utf-8")


def build_run(root: Path, scenes: list[dict], audio_mode="song_lock", lock: Path | None = None) -> int:
    """Save `scenes` as Luna-H3-LoopEnd would. Each spec: keep, warm (0 = fresh), tint, bump (callable t -> float),
    shift (frames the picture runs ahead of its nominal time), tone (generated mode). Returns the total frames."""
    T = 24                                          # the run starts 1 s into the song
    prev_keep = None
    total = 0
    for i, sp in enumerate(scenes):
        T += sp.get("gap", 0)                       # song time skipped before this scene (a non-contiguous list)
        keep, warm = sp["keep"], sp.get("warm", 0) if i else 0
        tint, bump, shift = sp.get("tint", (0, 0, 0)), sp.get("bump", lambda t: 0.0), sp.get("shift", 0)
        kept = np.stack([world(T + m + shift, tint) for m in range(keep)])
        head = np.stack([world(T - warm + h + shift, tint, bump(T - warm + h)) for h in range(warm)]) if warm else None
        carried = warm > 0
        plan = {
            "scene_number": i + 1, "project_folder": root.as_posix(), "continuation": carried, "pass1": False,
            "overlap": 39, "audio_mode": audio_mode, "timing": {}, "tail_padding_frames": 4, "token_count": 37,
            "render_frames": warm + keep, "trim_frames": warm, "keep_frames": keep,
            "start_seconds": (T + sp.get("start_frac", 0)) / 24,
            "end_seconds": (T + keep) / 24, "lock_audio": lock.as_posix() if lock else "", "context_plan": {},
            "overlap_map": ({"offset": prev_keep - warm, "head_frames": warm, "predecessor_scene": i - 1,
                             "predecessor_keep_frames": prev_keep, "predecessor_trim_frames": 0} if carried else None)}
        fake_beta(root, i + 1)
        audio = head_audio = None
        if audio_mode == "generated":
            f = sp.get("tone", 440.0 * (i + 1))
            full = sp["audio"] if sp.get("audio") is not None else tone(f, (warm + keep) * PER)
            audio = {"waveform": stereo(full[warm * PER:]), "sample_rate": RATE}
            head_audio = {"waveform": stereo(full[:warm * PER]), "sample_rate": RATE} if carried else None
        with mock.patch.dict(sys.modules, {"av": None}):                 # PNG frames: exact pixels
            h3_loop.save_scene(root, i, {"name": f"S{i}"}, images=torch.from_numpy(kept.astype(np.float32) / 255.0),
                               audio=audio, plan=plan,
                               head_images=None if head is None else torch.from_numpy(head.astype(np.float32) / 255.0),
                               head_audio=head_audio)
        prev_keep = keep
        T += keep
        total += keep
    return total


def join(root: Path, n: int, **opts):
    return hj.join_run(root, "t", n, hj.check_settings(**opts))


def decode_video(path: Path) -> list[np.ndarray]:
    import av
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def decode_audio(path: Path):
    import av
    chunks = []
    with av.open(str(path)) as c:
        s = c.streams.audio[0]
        for f in c.decode(s):
            chunks.append(f.to_ndarray())
        return np.concatenate(chunks, axis=1), int(s.rate)


def dominant_freq(x: np.ndarray, a: int, b: int) -> float:
    seg = x[a:b] * np.hanning(b - a)
    spec = np.abs(np.fft.rfft(seg))
    return float(np.argmax(spec) * RATE / (b - a))


@pytest.fixture
def root(tmp_path):
    return tmp_path / "run"


# --------------------------------------------------------------------------- settings

def test_settings_defaults_and_errors():
    s = hj.check_settings()
    assert (s["cut_rule"], s["time_lock"], s["search_window"], s["match_threshold"], s["fallback"],
            s["crossfade_frames"], s["color_match"], s["color_strength"], s["report"]) == (
        "best_match", True, 0, 8.0, "crossfade", 6, "carried_only", 1.0, True)
    for kw, fragment in (({"cut_rule": "x"}, "cut_rule"), ({"fallback": "x"}, "fallback"),
                         ({"color_strength": 2.0}, "color_strength"), ({"crossfade_frames": -1}, "crossfade"),
                         ({"manual_cuts": "[1]"}, "manual_cuts"), ({"manual_cuts": "{"}, "valid JSON"),
                         ({"cut_rule": "breath", "breath_source": "none"}, "needs a breath_source"),
                         ({"breath_source": "breath_audio"}, "breath_audio file"), ({"bogus": 1}, "unknown")):
        with pytest.raises(ValueError, match=fragment):
            hj.check_settings(**kw)
    assert hj.check_settings(manual_cuts='{"1": 40}')["manual"] == {1: 40}
    assert hj.check_settings(time_lock=None)["time_lock"] is True        # a linked (None) input keeps the default


def test_end_validates_join_settings_and_schema():
    from h3_loop import LunaH3LoopEnd
    assert LunaH3LoopEnd.validate_inputs(cut_rule="best_match", manual_cuts='{"1": 5}') is True
    assert "cut_rule" in LunaH3LoopEnd.validate_inputs(cut_rule="nope")
    schema = LunaH3LoopEnd.GET_SCHEMA()
    ids = [i.id for i in schema.inputs]
    assert ids[-12:] == ["cut_rule", "time_lock", "search_window", "breath_source", "breath_audio",
                         "match_threshold", "fallback", "crossfade_frames", "color_match", "color_strength",
                         "manual_cuts", "report"]
    defaults = {i.id: getattr(i, "default", None) for i in schema.inputs[-12:]}
    assert defaults["cut_rule"] == "best_match" and defaults["time_lock"] is True and defaults["report"] is True
    assert defaults["match_threshold"] == 8.0 and defaults["crossfade_frames"] == 6 and defaults["color_strength"] == 1.0


# --------------------------------------------------------------------------- the mapping and the plain join

def test_a_join_keeps_every_frame_on_its_song_time(root):
    """Two carried scenes, no colour step: the joined video shows the square at x = t for every output frame, at
    the cut and everywhere else, and its length is the sum of the kept frames (the cut never shifts time)."""
    total = build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12}])
    res = join(root, 2, color_match="off", cut_rule="best_match", fallback="hard_cut", match_threshold=50)
    assert res["total_frames"] == total == 72
    frames = decode_video(res["video"])
    assert len(frames) == 72
    assert_positions(frames, 24)


def test_best_match_cuts_where_the_two_pictures_agree(root):
    exact = 24 + 30                                           # the one overlap time where head == scene k
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12,
                                    "bump": lambda t: 0.0 if t == exact else 30.0}])
    res = join(root, 2, color_match="off", cut_rule="best_match")
    j = res["joins"][0]
    # k's frame c-1 is the matching one: c - 1 = exact - 24
    assert j["decision"]["c"] == exact - 24 + 1 and j["decision"]["delta"] == 0 and j["decision"]["score"] < 1
    assert j["seamless"] and j["action"] == "cut" and j["joined_frame"] == exact - 24 + 1
    assert j["decision"]["window"] == [25, 36]


def test_search_window_limits_where_the_cut_may_fall(root):
    exact = 24 + 26                                           # matches at k frame 26 ...
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12, "bump": lambda t: 0.0 if t == exact else 30.0}])
    j = join(root, 2, color_match="off", search_window=5, fallback="hard_cut")["joins"][0]
    assert j["decision"]["window"] == [32, 36] and 32 <= j["decision"]["c"] <= 36   # ... which is outside the window


def test_lowest_motion_cuts_at_the_stillest_step(root):
    # Scene k holds still for one step: the picture at global time 24+28 repeats the one before it.
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12}])
    pause_t = 24 + 28
    frames = np.stack([world(24 + m if 24 + m != pause_t else pause_t - 1) for m in range(36)])
    grey = hj.to_grey(frames)
    motion = hj.motion_by_cut(grey, 25, 36)
    best = hj.argmin_late(motion)
    assert best == 28 + 1                                     # step 27->28 is the still one: c - 1 = 28
    assert motion[best] < min(v for c, v in motion.items() if c != best)


def test_lowest_motion_rule_end_to_end(root):
    # Make scene k's own frames pause: rebuild scene 0 with a repeated frame, then join with the rule.
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12}])
    k_dir = root
    pause_m = 28
    for m in (pause_m,):
        from PIL import Image
        Image.fromarray(world(24 + pause_m - 1)).save(k_dir / "scene_0000_frames" / f"frame_{m:05d}.png")
    # keep the manifest honest: re-hash the edited file
    mpath = k_dir / "scene_0000.manifest.json"
    manifest = json.loads(mpath.read_text())
    for e in manifest["files"]:
        e["sha256"] = h3_loop.sha256_file(k_dir / e["path"])
        e["bytes"] = (k_dir / e["path"]).stat().st_size
    mpath.write_text(json.dumps(manifest))
    j = join(root, 2, color_match="off", cut_rule="lowest_motion", fallback="hard_cut", match_threshold=255)["joins"][0]
    assert j["decision"]["rule"] == "lowest_motion" and j["decision"]["c"] == pause_m + 1


def test_breath_cuts_where_the_scene_audio_is_quietest(root):
    gap = 30                                                   # silence over frames 29-30: only boundary 30 is fully inside
    audio = tone(440, 36 * PER)
    audio[29 * PER:31 * PER] = 0
    build_run(root, [{"keep": 36, "audio": audio}, {"keep": 36, "warm": 12}], audio_mode="generated")
    j = join(root, 2, color_match="off", cut_rule="breath", fallback="hard_cut", match_threshold=255)["joins"][0]
    assert j["decision"]["rule"] == "breath" and j["decision"]["c"] == gap


def test_breath_audio_file_on_the_song_timeline(root, tmp_path):
    lock = tmp_path / "lock.wav"
    n = (24 + 80) * PER                                       # the whole timeline from song time 0
    h3_loop.save_audio_wav({"waveform": stereo(tone(220, n)), "sample_rate": RATE}, lock)
    breath = tone(330, n)
    cut_t = 24 + 31                                           # silence around global frame 55
    breath[(cut_t - 1) * PER:(cut_t + 1) * PER] = 0
    stem = tmp_path / "vocals.wav"
    h3_loop.save_audio_wav({"waveform": stereo(breath), "sample_rate": RATE}, stem)
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12}], lock=lock)
    j = join(root, 2, color_match="off", cut_rule="breath", breath_source="breath_audio",
             breath_audio=stem.as_posix(), fallback="hard_cut", match_threshold=255)["joins"][0]
    assert j["decision"]["c"] == 31
    with pytest.raises(ValueError, match="needs a breath_source"):
        hj.check_settings(cut_rule="breath", breath_source="none")


def test_manual_cuts_override_the_rule(root):
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12}])
    j = join(root, 2, color_match="off", manual_cuts='{"1": 30}', fallback="hard_cut", match_threshold=255)["joins"][0]
    assert j["joined_frame"] == 30 and j["decision"]["rule"] == "manual" and j["decision"]["c"] == 30
    with pytest.raises(ValueError, match="outside this join's overlap"):
        join(root, 2, color_match="off", manual_cuts='{"1": 10}')
    with pytest.raises(ValueError, match="needs an entry in manual_cuts"):
        join(root, 2, color_match="off", cut_rule="manual")


def test_manual_cut_on_a_later_join_counts_in_joined_frames(root):
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12}, {"keep": 36, "warm": 12}])
    res = join(root, 3, color_match="off", manual_cuts='{"1": 30, "2": 62}', fallback="hard_cut", match_threshold=255)
    assert [j["joined_frame"] for j in res["joins"]] == [30, 62]
    frames = decode_video(res["video"])
    assert len(frames) == 108
    assert_positions(frames, 24)


# --------------------------------------------------------------------------- time shift

def test_time_lock_off_finds_a_known_offset(root):
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12, "shift": 3}], audio_mode="generated")
    off = join(root, 2, color_match="off", time_lock=False, search_window=6, fallback="hard_cut")
    j = off["joins"][0]
    assert j["delta"] == -3 and j["decision"]["score"] < 1 and j["action"] == "cut"
    assert off["total_frames"] == 72 + 3                      # the continuation starts 3 frames earlier
    on = join(root, 2, color_match="off", time_lock=True, fallback="hard_cut", match_threshold=255)
    assert on["joins"][0]["delta"] == 0 and on["joins"][0]["decision"]["score"] > 3
    assert on["total_frames"] == 72


def test_a_song_locked_run_refuses_a_time_shift(root):
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12}])
    with pytest.raises(ValueError, match="cannot shift time"):
        join(root, 2, time_lock=False)


# --------------------------------------------------------------------------- threshold and fallback

def test_above_the_threshold_the_fallback_applies(root):
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12, "bump": lambda t: 60.0}])
    base = dict(color_match="off")
    cf = join(root, 2, **base, fallback="crossfade", crossfade_frames=6)["joins"][0]
    assert not cf["seamless"] and cf["action"] == "crossfade" and cf["crossfade_frames"] == 6
    hc = join(root, 2, **base, fallback="hard_cut")["joins"][0]
    assert not hc["seamless"] and hc["action"] == "hard_cut" and hc["crossfade_frames"] == 0
    ok = join(root, 2, **base, fallback="stop", match_threshold=255)["joins"][0]
    assert ok["seamless"] and ok["action"] == "cut"
    with pytest.raises(ValueError, match="fallback is stop"):
        join(root, 2, **base, fallback="stop")
    assert (root / "join_01.json").is_file()                  # the evidence is written before the stop


def test_crossfade_blends_the_two_pictures_over_the_overlap(root):
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12, "bump": lambda t: 60.0}])
    res = join(root, 2, color_match="off", fallback="crossfade", crossfade_frames=6)
    frames = decode_video(res["video"])
    assert len(frames) == 72
    j = res["joins"][0]["joined_frame"]
    means = [float(f.mean()) for f in frames[j - 8:j + 2]]
    # The head is 60 brighter; over the 6 frames before the cut the picture ramps up to it, then stays.
    ramp = means[2:8]
    assert all(b > a for a, b in zip(ramp, ramp[1:])) and means[0] < means[-1] - 40 < means[-1]


def test_crossfade_is_shortened_when_the_cut_leaves_no_room(root):
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12, "bump": lambda t: 60.0}])
    j = join(root, 2, color_match="off", fallback="crossfade", crossfade_frames=40, manual_cuts='{"1": 26}')["joins"][0]
    assert j["crossfade_frames"] == 2          # c - off = 26 - 24: only two frames of the head lie before the cut


# --------------------------------------------------------------------------- colour

STEP = (26, -4, -16)


def _step_run(root, fresh=False):
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 0 if fresh else 12, "tint": STEP}])


def test_colour_match_removes_a_known_step_in_a_carried_join(root):
    _step_run(root)
    same_cut = dict(manual_cuts='{"1": 30}', fallback="hard_cut", match_threshold=255)   # the cut frame must not differ
    off = join(root, 2, color_match="off", **same_cut)
    f_off = decode_video(off["video"])                          # joined.mp4 is rewritten by the next join
    on = join(root, 2, color_match="carried_only", **same_cut)
    c = on["joins"][0]["color"]
    assert c["gap_before"] > 12 and c["gap_after"] < 1.0
    f_on = decode_video(on["video"])
    j = on["joins"][0]["joined_frame"]
    step_off = np.abs(f_off[j].mean(axis=(0, 1)) - f_off[j - 1].mean(axis=(0, 1))).mean()
    step_on = np.abs(f_on[j].mean(axis=(0, 1)) - f_on[j - 1].mean(axis=(0, 1))).mean()
    assert step_off > 12 and step_on < 3
    # One constant correction for the whole scene: no step inside scene k+1 (its first and last kept frames).
    k1_first, k1_last = f_on[36], f_on[-1]
    assert abs(float(k1_first.mean()) - float(world(24 + 36).mean())) < 3
    assert abs(float(k1_last.mean()) - float(world(24 + 71).mean())) < 3


def test_colour_strength_scales_the_correction(root):
    _step_run(root)
    half = join(root, 2, color_match="carried_only", color_strength=0.5, fallback="hard_cut", match_threshold=255)
    zero = join(root, 2, color_match="carried_only", color_strength=0.0, fallback="hard_cut", match_threshold=255)
    assert zero["joins"][0]["color"] is None
    gb = half["joins"][0]["color"]
    assert 0.4 * gb["gap_before"] < gb["gap_after"] < 0.6 * gb["gap_before"]


def test_fresh_join_colour_only_with_all(root):
    _step_run(root, fresh=True)
    co = join(root, 2, color_match="carried_only", fallback="hard_cut", match_threshold=255)
    assert co["joins"][0]["type"] == "fresh" and co["joins"][0]["color"] is None
    f0 = decode_video(co["video"])
    al = join(root, 2, color_match="all", fallback="hard_cut", match_threshold=255)
    assert al["joins"][0]["color"] is not None
    f = decode_video(al["video"])
    j = al["joins"][0]["joined_frame"]
    # The fresh scene shows other content, so the match is on the first/last frames' statistics: the step shrinks.
    gap = lambda fr: np.abs(fr[j].mean(axis=(0, 1)) - fr[j - 1].mean(axis=(0, 1))).mean()
    assert gap(f) < 0.5 * gap(f0)


def test_colour_fit_is_bounded_and_exact_on_a_pure_gain():
    rng = np.random.default_rng(0)
    target = rng.integers(60, 180, (4, 20, 20, 3)).astype(np.uint8)
    source = np.clip(target.astype(np.float32) * 0.9 + 5, 0, 255).astype(np.uint8)
    a, b = hj.fit_color(source, target)
    assert np.abs(hj.apply_color(source, a, b).astype(int) - target).mean() < 1.0
    flat = np.full((2, 8, 8, 3), 100, np.uint8)                          # no contrast to measure: ratio stays bounded
    a2, _ = hj.fit_color(flat, target)
    assert (a2 <= hj.STD_RATIO[1] + 1e-9).all() and (a2 >= hj.STD_RATIO[0] - 1e-9).all()


# --------------------------------------------------------------------------- fresh joins

def test_a_fresh_join_is_a_plain_cut_at_the_boundary(root):
    build_run(root, [{"keep": 36}, {"keep": 36}])
    res = join(root, 2, color_match="off", fallback="crossfade", cut_rule="best_match")
    j = res["joins"][0]
    assert j["type"] == "fresh" and j["joined_frame"] == 36 and j["crossfade_frames"] == 0 and j["delta"] == 0
    assert j["decision"]["rule"] == "boundary" and res["total_frames"] == 72
    assert len(decode_video(res["video"])) == 72


# --------------------------------------------------------------------------- audio

def _lock(tmp_path, click_t: float) -> Path:
    lock = tmp_path / "lock.wav"
    n = (24 + 100) * PER
    wave = tone(220, n, 0.05)
    c = round(click_t * RATE)
    wave[c:c + 40] = 1.0
    h3_loop.save_audio_wav({"waveform": stereo(wave), "sample_rate": RATE}, lock)
    return lock


def test_song_lock_muxes_the_master_for_exactly_the_joined_length_and_in_sync(root, tmp_path):
    click_t = (24 + 50) / 24                                   # in scene 1's kept frames
    lock = _lock(tmp_path, click_t)
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12}], lock=lock)
    res = join(root, 2, color_match="off", fallback="hard_cut", match_threshold=255)
    audio, rate = decode_audio(res["video"])
    assert rate == RATE
    mono = audio.mean(axis=0)
    assert abs(len(mono) - 72 * PER) <= 2048                    # aac pads to a whole frame of 1024
    peak = int(np.argmax(np.abs(mono)))
    expected = round((click_t - 1.0) * RATE)                    # the run starts at song time 1.0 s
    assert abs(peak - expected) < 2500                          # aac delay is ~1024 samples (21 ms)
    # And it is the master itself, not a re-cut: the same tone, no seam at the join.
    j = res["joins"][0]["joined_frame"] * PER
    assert abs(float(np.abs(mono[j - 200:j - 100]).max()) - float(np.abs(mono[j + 100:j + 200]).max())) < 0.05


def test_generated_audio_is_cut_with_the_picture_and_crossfaded(root):
    build_run(root, [{"keep": 36, "tone": 440.0}, {"keep": 36, "warm": 12, "tone": 880.0}], audio_mode="generated")
    res = join(root, 2, color_match="off", manual_cuts='{"1": 30}', fallback="hard_cut", match_threshold=255)
    audio, rate = decode_audio(res["video"])
    mono = audio.mean(axis=0)
    assert abs(len(mono) - 72 * PER) <= 2048
    cut = 30 * PER
    assert abs(dominant_freq(mono, 5 * PER, 9 * PER) - 440) < 15          # scene k before the cut
    assert abs(dominant_freq(mono, 34 * PER, 38 * PER) - 880) < 15       # scene k+1 after it
    # The change happens at the cut: 880 Hz energy is absent a few frames before it and present a few after.
    assert abs(dominant_freq(mono, cut - 6 * PER, cut - 3 * PER) - 440) < 25
    assert abs(dominant_freq(mono, cut + 3 * PER, cut + 6 * PER) - 880) < 25


def test_audio_crossfade_has_the_length_of_the_picture_crossfade(root):
    build_run(root, [{"keep": 36, "tone": 440.0}, {"keep": 36, "warm": 12, "tone": 880.0, "bump": lambda t: 60.0}],
              audio_mode="generated")
    res = join(root, 2, color_match="off", manual_cuts='{"1": 30}', fallback="crossfade", crossfade_frames=6)
    j = res["joins"][0]
    assert j["crossfade_frames"] == 6
    mono = decode_audio(res["video"])[0].mean(axis=0)
    cut = 30 * PER
    # Inside the 6-frame blend both tones are present (the ramp runs 440 -> 880), outside only one.
    spec = lambda a, b: np.abs(np.fft.rfft(mono[a:b] * np.hanning(b - a)))
    bins = lambda a, b, f: spec(a, b)[int(f * (b - a) / RATE)]
    inside = (cut - 4 * PER, cut - 2 * PER)
    assert bins(*inside, 440) > 5 and bins(*inside, 880) > 5
    far = (cut - 12 * PER, cut - 9 * PER)
    assert bins(*far, 880) < 0.1 * bins(*far, 440)


def test_missing_scene_audio_gives_a_silent_video_not_a_failure(root):
    build_run(root, [{"keep": 24}, {"keep": 24, "warm": 12}], audio_mode="generated")
    (root / "scene_0001.wav").unlink()
    res = join(root, 2, color_match="off", fallback="hard_cut", match_threshold=255)
    import av
    with av.open(str(res["video"])) as c:
        assert not c.streams.audio and res["audio"] is None


# --------------------------------------------------------------------------- reports and encode

def test_report_files_exist_and_say_what_was_decided(root):
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12, "tint": STEP}, {"keep": 36}])
    res = join(root, 3, color_match="carried_only", fallback="hard_cut", match_threshold=255)
    for name in ("join_01.json", "join_01_sheet.jpg", "join_02.json", "join_02_sheet.jpg", "join_report.json", "joined.mp4"):
        assert (root / name).is_file(), name
    rep = json.loads((root / "join_report.json").read_text())
    assert rep["scenes"] == 3 and rep["total_frames"] == 108 and rep["audio"] is None    # no lock file in this run
    assert [j["type"] for j in rep["joins"]] == ["carried", "fresh"]
    j1 = json.loads((root / "join_01.json").read_text())
    assert j1["settings"]["cut_rule"] == "best_match" and j1["overlap"]["offset"] == 24 and j1["color"]["gap_before"] > 10
    assert j1["sheet"] == "join_01_sheet.jpg" and j1["decision"]["window"] == [25, 36]
    import cv2
    sheet = cv2.imread(str(root / "join_01_sheet.jpg"))
    assert sheet is not None and sheet.shape[0] >= 2 * 64 and sheet.shape[1] >= 6 * 96
    assert res["report"] == root / "join_report.json"
    assert rep["joined_sha256"] == h3_loop.sha256_file(root / "joined.mp4")


def test_report_off_writes_no_join_files(root):
    build_run(root, [{"keep": 24}, {"keep": 24, "warm": 12}])
    res = join(root, 2, report=False, color_match="off")
    assert res["video"].is_file() and res["report"] is None
    assert not list(root.glob("join_*"))


def test_the_joined_file_is_h264_yuv420p_24fps(root):
    build_run(root, [{"keep": 24}, {"keep": 24, "warm": 12}])
    res = join(root, 2, color_match="off", report=False)
    import av
    with av.open(str(res["video"])) as c:
        v = c.streams.video[0]
        assert v.codec_context.name == "h264" and v.codec_context.pix_fmt == "yuv420p"
        assert float(v.average_rate) == 24.0 and v.frames == 48


def test_a_run_without_frames_or_plan_is_left_alone(root):
    h3_loop.save_scene(root, 0, {"name": "x"}, audio={"waveform": stereo(tone(440, PER)), "sample_rate": RATE})
    assert hj.join_run(root, "t", 1, hj.check_settings()) is None


def test_a_carried_scene_saved_without_its_head_joins_with_a_plain_cut(root):
    build_run(root, [{"keep": 24}, {"keep": 24, "warm": 12}])
    # Pretend the head was never saved (a run from before S11): drop it from the manifest.
    path = root / "scene_0001.manifest.json"
    m = json.loads(path.read_text())
    m["info"].pop("head")
    m["files"] = [e for e in m["files"] if "_head" not in e["path"]]
    path.write_text(json.dumps(m))
    res = join(root, 2, color_match="off")
    j = res["joins"][0]
    assert j["type"] == "fresh" and j["notes"] and res["total_frames"] == 48


# --------------------------------------------------------------------------- Luna-H3-LoopEnd saves the head

def test_loop_end_keeps_the_hidden_head_with_its_mapping(root, tmp_path, monkeypatch):
    monkeypatch.setattr(h3_loop, "_output_root", lambda: tmp_path)
    run_root = tmp_path / "h3_loops" / "t"
    fake_beta(run_root, 2)
    plan = {"scene_number": 2, "project_folder": run_root.as_posix(), "continuation": True, "pass1": False,
            "overlap": 39, "audio_mode": "generated", "timing": {}, "tail_padding_frames": 4, "token_count": 37,
            "render_frames": 30, "trim_frames": 12, "keep_frames": 18, "start_seconds": 2.0, "end_seconds": 2.75,
            "lock_audio": "", "context_plan": {},
            "overlap_map": {"offset": 24, "head_frames": 12, "predecessor_scene": 0, "predecessor_keep_frames": 36,
                            "predecessor_trim_frames": 0}}
    state = {"run_name": "t", "scenes": [{"name": "a"}, {"name": "b"}], "scene_index": 1, "scene_count": 3, "plan": plan}
    images = torch.from_numpy(np.stack([world(40 + k) for k in range(30)]).astype(np.float32) / 255.0)
    audio = {"waveform": torch.rand(1, 2, 30 * PER), "sample_rate": RATE}
    from types import SimpleNamespace
    from test_h3_loop import FakeDynPrompt
    prompt = {"1": {"class_type": h3_loop.START_ID, "inputs": {"scenes": "[]", "run_name": "t", "start_scene": 0}},
              "4": {"class_type": h3_loop.END_ID, "inputs": {"flow": ["1", 0], "state": ["1", 1]}}}
    monkeypatch.setattr(h3_loop.LunaH3LoopEnd, "hidden",
                        SimpleNamespace(dynprompt=FakeDynPrompt(prompt), unique_id="4"), raising=False)
    h3_loop.LunaH3LoopEnd.execute(["1", 0], state, images=images, audio=audio)
    m = json.loads((run_root / "scene_0001.manifest.json").read_text())
    head = m["info"]["head"]
    assert head["frames"] == 12 and m["info"]["frames"]["count"] == 18
    assert head["mapping"]["offset"] == 24 and abs(head["mapping"]["song_time_of_head_frame_0"] - 1.5) < 1e-9
    assert {e["path"] for e in m["files"]} >= {"scene_0001_head.mp4" if head["format"] == "mp4" else
                                               "scene_0001_head_frames/frame_00000.png", "scene_0001_head.wav"}
    assert h3_loop.verify_scene(run_root, 1) == []
    # The head frames are the first 12 decoded frames, the kept ones the next 18.
    kept = hj.read_small(hj.frame_source(run_root, "scene_0001", m["info"]["frames"]), 96)
    heads = hj.read_small(hj.frame_source(run_root, "scene_0001_head", head), 96)
    assert len(heads) == 12 and len(kept) == 18
    mid = (images[12:14].numpy() * 255).round().astype(int)
    assert np.abs(kept[:2].astype(int) - mid).mean() < 8        # mp4 is lossy; the mapping is what is checked


# --------------------------------------------------------------------------- S11 review fixes

def test_manual_cuts_validate_the_join_and_the_frame_separately():
    for text, fragment in (('{"0": 5}', "join number 0 is not valid"), ('{"-2": 5}', "join number -2 is not valid"),
                           ('{"1": -3}', "frame -3 for join 1 is not valid"), ('{"1": "x"}', "must be join number")):
        with pytest.raises(ValueError, match=fragment):
            hj.parse_manual_cuts(text)
    assert hj.parse_manual_cuts('{"1": 0, "3": 10}') == {1: 0, 3: 10}


def test_cut_rule_manual_needs_manual_cuts_at_queue_time():
    from h3_loop import LunaH3LoopEnd
    with pytest.raises(ValueError, match="cut_rule manual needs an entry in manual_cuts"):
        hj.check_settings(cut_rule="manual")
    assert "needs an entry in manual_cuts" in LunaH3LoopEnd.validate_inputs(cut_rule="manual", manual_cuts="")
    assert LunaH3LoopEnd.validate_inputs(cut_rule="manual", manual_cuts='{"1": 30}') is True
    assert LunaH3LoopEnd.validate_inputs(cut_rule="manual") is not True


def test_manual_cut_checks_before_the_render():
    early = hj.check_manual_early
    keeps = [36]
    assert early({1: 30}, 1, 2, True, 12, keeps, 0, True) == []                      # c = 30 in [25, 36]
    assert early({1: 36}, 1, 2, True, 12, keeps, 0, True) == []
    for bad in (24, 37, 5):
        with pytest.raises(ValueError, match="outside the overlap.*25..36"):
            early({1: bad}, 1, 2, True, 12, keeps, 0, True)
    with pytest.raises(ValueError, match="30..36"):                                  # search_window 7 narrows it
        early({1: 25}, 1, 2, True, 12, keeps, 7, True)
    assert early({1: 5}, 1, 2, True, 12, keeps, 0, False) == []                      # time_lock off: not checkable here
    assert early({1: 5}, 1, 2, True, 12, None, 0, True) == []                        # earlier plans not on disk
    with pytest.raises(ValueError, match="names join 3 but the run has 1 join"):
        early({3: 5}, 1, 2, True, 12, keeps, 0, True)
    notes = early({1: 20}, 1, 2, False, 0, keeps, 0, True)
    assert len(notes) == 1 and "fresh join" in notes[0]
    assert early({2: 80}, 1, 3, True, 12, keeps, 0, True) == []                      # another join's entry: not this scene's
    # three scenes: join 2 counts from the end of scene 0's kept frames: 36 + c, c in [25, 36]
    assert early({2: 36 + 30}, 2, 3, True, 12, [36, 36], 0, True) == []
    with pytest.raises(ValueError, match="61..72"):
        early({2: 40}, 2, 3, True, 12, [36, 36], 0, True)


def test_a_manual_cut_on_a_fresh_join_is_noted_in_the_report(root):
    build_run(root, [{"keep": 36}, {"keep": 36}])
    j = join(root, 2, color_match="off", manual_cuts='{"1": 20}')["joins"][0]
    assert j["joined_frame"] == 36 and any("ignored" in n for n in j["notes"])
    assert any("ignored" in n for n in json.loads((root / "join_01.json").read_text())["notes"])


def test_a_manual_entry_for_a_join_the_run_does_not_have_is_refused(root):
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12}])
    with pytest.raises(ValueError, match="names join 4"):
        join(root, 2, color_match="off", manual_cuts='{"4": 30}', fallback="hard_cut")


def test_motion_by_cut_keeps_cut_one_as_a_last_resort():
    grey = hj.to_grey(np.stack([world(24 + m) for m in range(6)]))
    motion = hj.motion_by_cut(grey, 1, 4)
    assert motion[1] == float("inf") and all(np.isfinite(motion[c]) for c in (2, 3, 4))
    assert hj.argmin_late(motion) in (2, 3, 4)
    assert hj.argmin_late({1: float("inf")}) == 1                            # the only candidate: chosen, not an error


# ---- song-lock drift

def test_song_lock_refuses_non_contiguous_scenes_up_front():
    import h3_loop_scene as hs_
    ok = [{"start": 0.0, "end": 5.0}, {"start": 5.0, "end": 10.0}]
    hs_.check_contiguous(ok, "song_lock")
    gap = [{"start": 0.0, "end": 5.0}, {"start": 5.5, "end": 10.0}]
    with pytest.raises(ValueError, match="scene 1 starts at 5.5 s but scene 0 ends at 5.0 s"):
        hs_.check_contiguous(gap, "song_lock")
    hs_.check_contiguous(gap, "generated")                                    # dialogue runs may skip song time
    hs_.check_contiguous([{"start": 0, "end": 5.0}, {"start": 5.0000004, "end": 9}], "song_lock")   # within 1e-6


def test_the_scene_node_refuses_a_gap_before_anything_renders(tmp_path, monkeypatch):
    import h3_loop_scene as hs_
    monkeypatch.setattr(h3_loop, "_output_root", lambda: tmp_path)
    scenes = [{"start": 0.0, "end": 5.0, "references": ["a.png"]}, {"start": 6.0, "end": 10.0, "references": ["a.png"]}]
    state = {"run_name": "t", "scenes": scenes, "scene_index": 0, "scene_count": 2}
    with pytest.raises(ValueError, match="needs contiguous scenes"):
        hs_.LunaH3LoopScene.execute(state, "nope.wav")                       # before the lock file is even opened


def test_a_gap_in_a_saved_song_locked_run_stops_the_join(root):
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12, "gap": 5}])
    with pytest.raises(ValueError, match="not contiguous in song time"):
        join(root, 2, color_match="off")
    # A generated run does not care about song time.
    other = root.parent / "gen"
    build_run(other, [{"keep": 36}, {"keep": 36, "warm": 12, "gap": 5}], audio_mode="generated")
    res = join(other, 2, color_match="off")
    assert res["joins"][0]["drift"]["frames"] == 5.0


def test_the_join_record_carries_the_drift(root):
    build_run(root, [{"keep": 36}, {"keep": 36, "warm": 12}, {"keep": 36, "warm": 12}])
    res = join(root, 3, color_match="off", fallback="hard_cut", match_threshold=255)
    assert [j["drift"]["frames"] for j in res["joins"]] == [0.0, 0.0]
    assert json.loads((root / "join_02.json").read_text())["drift"] == {"frames": 0.0, "local_frames": 0.0}


def test_drift_stays_within_half_a_frame_however_many_scenes():
    """Beta's keep = round(end*24) - round(start*24) telescopes: the placed start of scene i is
    round(start_0*24) + sum(keep) = round(start_i*24), so the drift is the rounding of start_i alone (<= 0.5 frame)
    for any number of contiguous scenes. The per-join local drift can be larger, which is why it is not the test."""
    import h3_loop_scene as hs_
    beta = hs_.load_beta()
    rng = np.random.default_rng(3)
    worst, local_worst = 0.0, 0.0
    for _ in range(40):
        start = round(float(rng.uniform(0, 30)), 3)
        scenes, t0 = [], start
        for _i in range(25):                                       # 25 scenes, off the frame grid
            d = round(float(rng.uniform(2.1, 6.9)), 3)
            timing = beta.calculate_minimax_h3_timing(t0, t0 + d, warmup_frames=0, cooldown_frames=0,
                                                      source_start_seconds=t0, source_duration_seconds=10_000,
                                                      pad_warmup=False).to_dict() if d * 24 + 5 <= 362 else None
            scenes.append({"plan": {"start_seconds": t0}, "keep": timing["final_frame_count"]})
            t0 = round(t0 + d, 3)
        for i in range(25):
            rec = hj.song_drift(scenes, i)
            worst, local_worst = max(worst, abs(rec["frames"])), max(local_worst, abs(rec["local_frames"]))
    assert worst <= 0.5 + 1e-6
    assert local_worst > 0.5            # the local figure alone would have rejected valid contiguous lists


def test_the_master_slice_starts_on_the_frame_grid(root, tmp_path):
    lock = tmp_path / "ramp.wav"
    n = 130 * PER
    ramp = np.linspace(-0.9, 0.9, n).astype(np.float32)
    h3_loop.save_audio_wav({"waveform": stereo(ramp), "sample_rate": RATE}, lock)
    build_run(root, [{"keep": 24, "start_frac": 0.48}, {"keep": 24, "warm": 12}], lock=lock)
    scenes = hj.load_run(root, 2)
    assert abs(scenes[0]["plan"]["start_seconds"] * 24 - 24.48) < 1e-9
    wave, rate = hj.build_audio(root, scenes, {"total_frames": 48})
    assert rate == RATE and wave.shape[1] == 48 * PER
    # frame_round(1.02 s) = 24 (24.48 rounds down): the slice starts at frame 24 exactly, the picture's grid.
    assert np.array_equal(wave[0, :100], ramp[24 * PER:24 * PER + 100])


# ---- the emission of short scenes

def _run_emit(s, e, nin, nout, pending=None):
    out, blends = [], []
    pending = pending if pending is not None else [("p", k) for k in range(nin)]

    def blend(a, b, t):
        blends.append((a, b, round(t, 3)))
        return ("blend", a, b)
    tail = hj.emit_scene(lambda lo: iter(range(lo, 10_000)), s, e, nin, nout, pending, out.append, blend)
    return out, blends, tail


def test_emit_scene_normal_case():
    out, blends, tail = _run_emit(10, 40, 3, 4)
    assert [b[1] for b in blends] == [7, 8, 9] and [b[2] for b in blends] == [0.25, 0.5, 0.75]
    assert out[3:] == list(range(10, 36)) and tail == [36, 37, 38, 39]


def test_emit_scene_when_the_scene_is_shorter_than_its_blends():
    # e - s = 2 < nout = 4 (reachable with a time shift): the tail is F[e-4:e), read from before s, not from after it.
    out, blends, tail = _run_emit(15, 17, 3, 4)
    assert [b[1] for b in blends] == [12, 13, 14]                  # the partners of the previous tail
    assert out[3:] == []                                           # nothing between the blends
    assert tail == [13, 14, 15, 16]
    # and e - s smaller than nin + nout in general
    out, blends, tail = _run_emit(12, 15, 5, 5)
    assert tail == [10, 11, 12, 13, 14] and [b[1] for b in blends] == [7, 8, 9, 10, 11]


def test_emit_scene_no_blends():
    out, blends, tail = _run_emit(5, 9, 0, 0, pending=[])
    assert out == [5, 6, 7, 8] and blends == [] and tail is None
    with pytest.raises(ValueError, match="cannot plan"):
        _run_emit(9, 9, 0, 0, pending=[])


def test_emitted_frames_equal_the_joined_length_with_overlapping_blends():
    # Three scenes in a row through emit_scene: the totals match sum(e - s) however the windows overlap.
    plan = [(0, 30, 0, 6), (12, 40, 6, 3), (4, 26, 3, 0)]
    pending, emitted = None, 0
    for s, e, nin, nout in plan:
        out, _, pending = _run_emit(s, e, nin, nout, pending=pending if pending is not None else [])
        emitted += len(out)
    assert emitted == sum(e - s for s, e, _, _ in plan)

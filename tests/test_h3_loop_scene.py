"""Unit tests for Luna-H3-LoopScene / Luna-H3-LoopProfile and Luna-H3-LoopEnd's plan handling (S10), CPU only.

Needs ComfyUI on the path and Beta 2.0 (comfyui-vrgamedevgirl) in its custom_nodes: the timing is Beta's
own, so it is tested against Beta, not a stub. The song-lock window is compared with chain_driver.py's
ffmpeg `pad_audio` when the skill and ffmpeg are present.

Run from this folder: `pytest . --basetemp=D:/t/ComfyUI-LunaMiniMaxH3`.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("comfy_api.latest")
pytest.importorskip("comfy_execution.graph_utils")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import h3_loop  # noqa: E402
import h3_loop_scene as hs  # noqa: E402
from h3_loop import LunaH3LoopEnd, save_audio_wav  # noqa: E402
from h3_loop_scene import LunaH3LoopProfile, LunaH3LoopScene  # noqa: E402

try:
    BETA = hs.load_beta()
except Exception as exc:  # no Beta on this machine
    pytest.skip(f"Beta 2.0 not available: {exc}", allow_module_level=True)

SCENES = [
    {"name": "A", "prompt": "first", "start": 0.0, "end": 5.0, "references": ["{ref}"]},
    {"name": "B", "prompt": "second", "start": 5.0, "end": 10.0, "references": [{"previous_last_frame": True}],
     "overlap": 39},
]


# --------------------------------------------------------------------------- fixtures

@pytest.fixture
def run(tmp_path, monkeypatch):
    """A run folder, a 30 s 48 kHz lock WAV, one reference image; returns a state factory."""
    monkeypatch.setattr(h3_loop, "_output_root", lambda: tmp_path)
    (tmp_path / "input").mkdir()
    monkeypatch.setattr(h3_loop, "_input_root", lambda: tmp_path / "input")
    g = torch.Generator().manual_seed(7)
    lock = tmp_path / "lock.wav"
    save_audio_wav({"waveform": (torch.rand(1, 2, 48000 * 30, generator=g) * 2 - 1) * 0.5,
                    "sample_rate": 48000}, lock)
    from PIL import Image
    ref = tmp_path / "ref.png"
    Image.new("RGB", (32, 16), (200, 10, 10)).save(ref)
    scenes = json.loads(json.dumps(SCENES).replace("{ref}", ref.as_posix()))
    root = tmp_path / "h3_loops" / "t"
    root.mkdir(parents=True)

    def state(index=0, scenes_=None):
        sc = scenes_ or scenes
        return {"run_name": "t", "scenes": sc, "scene_index": index, "scene_count": len(sc), "start_scene": 0}

    return SimpleNamespace(root=root, lock=lock, ref=ref, state=state, scenes=scenes, tmp=tmp_path)


def _write_beta_scene(root, number, token_count, tail, pass1=None, ts=1000.0):
    """Fake Beta Save Latent files: the loaders' sidecar fields, empty tensors."""
    folder = root / "latents"
    folder.mkdir(exist_ok=True)
    for suffix, p1, stamp in ((".latent", False, ts), (".pass1.latent", True, ts - 5)):
        if p1 and not pass1:
            continue
        path = folder / f"scene_{number:03d}{suffix}"
        path.write_bytes(b"x")
        meta = {"scene_number": str(number), "token_count": str(token_count), "timestamp": str(stamp),
                "tail_padding_frames": str(tail), "video_shape": str([1, 24, token_count, 4, 4])}
        if p1:
            meta["pass"] = "1"
        path.with_name(path.name + ".json").write_text(json.dumps(meta), encoding="utf-8")


def _scene(run, index=0, state=None, mode="song_lock"):
    return LunaH3LoopScene.execute(state or run.state(index), run.lock.as_posix(), mode, "").result


OUT = ["state", "project_folder", "scene_number", "continuation", "pass1", "context_frames", "render_frames",
       "trim_start", "keep_frames", "visible_duration", "seed_1", "seed_2", "selflift_seed", "audio",
       "ref_1", "ref_2", "ref_3", "ref_4", "guide"]


def _named(result):
    return dict(zip(OUT, result))


# --------------------------------------------------------------------------- Beta timing

def test_beta_timing_is_the_installed_one():
    assert tuple(BETA.MASKED_CONTEXT_FRAMES) == (39, 90, 141, 192)
    first = hs.calculate(BETA, 0, 5, 300)
    assert first["timing"]["h3_frame_count"] == 124 and first["tail_padding_frames"] == 4
    second = hs.calculate(BETA, 5, 10, 300, {"token_count": 37, "tail_padding_frames": 4})
    assert second["context_plan"]["warmup_frames"] == 35 and second["timing"]["h3_frame_count"] == 158


def test_overlap_is_a_parameter():
    pred = {"token_count": 112, "tail_padding_frames": 4}
    warm = {o: hs.calculate(BETA, 20, 25, 300, pred, o)["context_plan"]["warmup_frames"] for o in hs.OVERLAPS}
    assert len(set(warm.values())) == 4 and warm[39] < warm[90] < warm[141] < warm[192]
    with pytest.raises(ValueError, match="tail padding"):
        hs.calculate(BETA, 20, 30, 300, {"token_count": 112, "tail_padding_frames": None})


def test_beta_check_catches_a_changed_grid():
    fake = SimpleNamespace(MASKED_CONTEXT_FRAMES=(39, 90, 141), calculate_minimax_h3_timing=None)
    with pytest.raises(RuntimeError, match="MASKED_CONTEXT_FRAMES"):
        hs._check_beta(fake)


# --------------------------------------------------------------------------- scene fields

@pytest.mark.parametrize("change, fragment", [
    ({"start": None}, "start' and 'end"),
    ({"end": 0.0}, "0 <= start < end"),
    ({"overlap": 40}, "overlap must be one of"),
    ({"join": "blend"}, "join must be"),
    ({"seeds": [1]}, "seeds must be two"),
    ({"seeds": [1, "x"]}, "seeds must be two"),
    ({"selflift_seed": -1}, "selflift_seed"),
    ({"preroll_frames": -2}, "preroll_frames"),
    ({"references": []}, "1 to 4"),
    ({"references": ["a"] * 5}, "1 to 4"),
    ({"references": [3]}, "image path"),
])
def test_scene_field_errors(change, fragment):
    scene = {"start": 0, "end": 5, "references": ["a.png"]}
    for k, v in change.items():
        if v is None:
            scene.pop(k)
        else:
            scene[k] = v
    with pytest.raises(ValueError, match=fragment):
        hs.scene_settings(scene, 3)


def test_previous_last_frame_not_on_the_first_scene_and_defaults():
    with pytest.raises(ValueError, match="no previous last frame"):
        hs.scene_settings({"start": 0, "end": 5, "references": [{"previous_last_frame": True}]}, 0)
    s = hs.scene_settings({"start": 0, "end": 5, "references": ["a.png"]}, 0)
    assert (s["overlap"], s["join"], s["seeds"], s["selflift_seed"], s["preroll_frames"]) == (
        39, "carried", [1111, 2222], 1111, 0)


def test_every_scene_needs_the_same_reference_count():
    ok = {"start": 0, "end": 5, "references": ["a.png"]}
    with pytest.raises(ValueError, match="same number of references"):
        hs.check_run_scenes([ok, dict(ok, references=["a.png", "b.png"])])


# --------------------------------------------------------------------------- song-lock window

def _chain_driver():
    path = Path.home() / ".claude" / "skills" / "music-video-from-audio" / "scripts" / "chain_driver.py"
    if not path.is_file() or not shutil.which("ffmpeg"):
        pytest.skip("chain_driver.py or ffmpeg not available")
    spec = importlib.util.spec_from_file_location("chain_driver_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_audio_window_matches_chain_driver_pad_audio(run):
    driver = _chain_driver()
    pred = {"token_count": 37, "tail_padding_frames": 4}
    # Scene starts 0.5 s into the song but the carried head wants ~1.46 s before it: leading silence.
    calc = hs.calculate(BETA, 0.5, 5.5, 30.0, pred, 39)
    t = calc["timing"]
    assert t["audio_leading_padding_seconds"] > 0.5
    wave, rate = hs.load_audio_file(run.lock)
    assert rate == 48000 and wave.shape == (2, 48000 * 30)
    ours = hs.audio_window(wave, rate, t)
    assert ours["sample_rate"] == 48000

    target = run.tmp / "driver_padded.wav"
    driver.pad_audio(run.lock.as_posix(), target.as_posix(), t)
    ref, ref_rate = hs.load_audio_file(target)
    assert ref_rate == 48000
    mine = ours["waveform"][0]
    assert abs(mine.shape[1] - ref.shape[1]) <= 1
    n = min(mine.shape[1], ref.shape[1])
    # ffmpeg writes 16-bit PCM, so the reference is quantised: half a step is 1.6e-5.
    assert torch.allclose(mine[:, :n], ref[:, :n], atol=5e-5)
    lead = round(t["audio_leading_padding_seconds"] * rate)
    assert float(mine[:, :lead].abs().max()) == 0.0 and float(mine[:, lead:lead + 100].abs().max()) > 0


def test_audio_window_without_leading_padding_is_a_plain_slice(run):
    t = hs.calculate(BETA, 12.0, 20.0, 30.0, {"token_count": 37, "tail_padding_frames": 4}, 39)["timing"]
    assert t["audio_leading_padding_seconds"] == 0
    wave, rate = hs.load_audio_file(run.lock)
    out = hs.audio_window(wave, rate, t)["waveform"][0]
    first = round(t["audio_trim_start_seconds"] * rate)
    assert out.shape[1] == round(t["audio_trim_duration_seconds"] * rate)
    assert torch.equal(out, wave[:, first:first + out.shape[1]])


def test_audio_window_keeps_the_sample_rate(run, tmp_path):
    save_audio_wav({"waveform": torch.rand(1, 1, 22050 * 12) - .5, "sample_rate": 22050}, tmp_path / "lo.wav")
    wave, rate = hs.load_audio_file(tmp_path / "lo.wav")
    assert rate == 22050 and wave.shape[0] == 1
    t = hs.calculate(BETA, 1.0, 6.0, 12.0)["timing"]
    assert hs.audio_window(wave, rate, t)["sample_rate"] == 22050


# --------------------------------------------------------------------------- Luna-H3-LoopScene node

def test_first_scene_is_fresh_and_fully_planned(run):
    o = _named(_scene(run, 0))
    t = hs.calculate(BETA, 0.0, 5.0, 30.0)["timing"]
    assert o["continuation"] is False and o["pass1"] is False and o["scene_number"] == 1
    assert o["project_folder"] == run.root.as_posix() and o["context_frames"] == 39
    assert (o["render_frames"], o["keep_frames"]) == (t["h3_frame_count"], t["final_frame_count"])
    assert o["trim_start"] == t["final_trim_start_seconds"] and o["visible_duration"] == t["final_trim_duration_seconds"]
    assert (o["seed_1"], o["seed_2"], o["selflift_seed"]) == (1111, 2222, 1111)
    plan = o["state"]["plan"]
    assert plan["token_count"] == BETA._frames_to_tokens(t["h3_frame_count"])
    assert plan["tail_padding_frames"] == hs.calculate(BETA, 0, 5, 30)["tail_padding_frames"] == 4
    assert o["audio"]["waveform"].shape[-1] == round(t["audio_trim_duration_seconds"] * 48000)
    name = o["ref_1"]
    assert name.startswith("luna_chain_refs/") and "\\" not in name and name.endswith(".png")
    assert (run.tmp / "input" / name).read_bytes() == run.ref.read_bytes()      # staged copy, same bytes
    assert o["ref_2"] == o["ref_3"] == o["ref_4"] == "" and o["state"]["plan"]["audio_mode"] == "song_lock"
    assert "plan" not in run.state(0)  # the input state is not changed


def test_carried_scene_reads_the_predecessor_sidecar_and_pass1(run):
    first = _named(_scene(run, 0))["state"]["plan"]
    _write_beta_scene(run.root, 1, first["token_count"], first["tail_padding_frames"], pass1=True)
    from PIL import Image
    Image.new("RGB", (8, 8), (1, 2, 3)).save(h3_loop_scene_last(run, 0))
    o = _named(_scene(run, 1))
    expect = hs.calculate(BETA, 5.0, 10.0, 30.0, {"token_count": first["token_count"],
                                                   "tail_padding_frames": first["tail_padding_frames"]}, 39)
    assert o["continuation"] is True and o["pass1"] is True and o["scene_number"] == 2
    assert o["render_frames"] == expect["timing"]["h3_frame_count"]
    assert o["state"]["plan"]["timing"]["final_trim_start_seconds"] == expect["timing"]["final_trim_start_seconds"]
    assert (run.tmp / "input" / o["ref_1"]).read_bytes() == h3_loop_scene_last(run, 0).read_bytes()  # the PNG, staged


def h3_loop_scene_last(run, index):
    return hs.last_frame_path(run.root, index)


def test_pass1_ignored_when_it_does_not_match(run):
    first = _named(_scene(run, 0))["state"]["plan"]
    _write_beta_scene(run.root, 1, first["token_count"], first["tail_padding_frames"], pass1=True)
    p1 = run.root / "latents" / "scene_001.pass1.latent.json"
    meta = json.loads(p1.read_text())
    meta["token_count"] = str(first["token_count"] + 5)
    p1.write_text(json.dumps(meta))
    from PIL import Image
    Image.new("RGB", (8, 8)).save(h3_loop_scene_last(run, 0))
    assert _named(_scene(run, 1))["pass1"] is False


def test_fresh_join_ignores_the_predecessor_and_uses_preroll(run):
    scenes = json.loads(json.dumps(run.scenes))
    scenes[1].update(join="fresh", preroll_frames=39, references=[run.ref.as_posix()])
    scenes[0]["references"] = [run.ref.as_posix()]
    o = _named(_scene(run, 1, run.state(1, scenes)))          # no sidecars exist: not read
    t = hs.calculate(BETA, 5.0, 10.0, 30.0, None, 39, 39)["timing"]
    assert o["continuation"] is False and o["pass1"] is False
    assert o["render_frames"] == t["h3_frame_count"] and o["trim_start"] == t["final_trim_start_seconds"]
    assert o["trim_start"] > 0


def test_overlap_per_scene_changes_the_head(run):
    first = _named(_scene(run, 0))["state"]["plan"]
    big = {"token_count": 112, "tail_padding_frames": 4}
    _write_beta_scene(run.root, 1, big["token_count"], big["tail_padding_frames"])
    from PIL import Image
    Image.new("RGB", (8, 8)).save(h3_loop_scene_last(run, 0))
    renders = {}
    for overlap in (39, 90):
        scenes = json.loads(json.dumps(run.scenes))
        scenes[1]["overlap"] = overlap
        o = _named(_scene(run, 1, run.state(1, scenes)))
        renders[overlap] = o
        assert o["context_frames"] == overlap
    assert renders[90]["trim_start"] > renders[39]["trim_start"]
    assert first["token_count"] > 0


def test_carried_scene_without_predecessor_files_names_the_gap(run):
    with pytest.raises(FileNotFoundError, match="scene 1 did not save it"):
        _scene(run, 1)


def test_scene_node_errors(run):
    with pytest.raises(FileNotFoundError, match="lock_audio"):
        LunaH3LoopScene.execute(run.state(0), "nope.wav").result
    with pytest.raises(ValueError, match="audio_mode must be"):
        LunaH3LoopScene.execute(run.state(0), run.lock.as_posix(), "both").result
    with pytest.raises(ValueError, match="Load Latent nodes hold"):
        LunaH3LoopScene.execute(run.state(0), run.lock.as_posix(), loader_folder="D:/elsewhere").result
    scenes = json.loads(json.dumps(run.scenes))
    scenes[0]["end"] = 99.0
    scenes = scenes[:1]                                      # one scene: the contiguity check has nothing to say
    with pytest.raises(ValueError, match="lock audio is"):
        _scene(run, 0, run.state(0, scenes))
    assert LunaH3LoopScene.execute(run.state(0), run.lock.as_posix(), loader_folder=run.root.as_posix()).result
    f = LunaH3LoopScene.fingerprint_inputs(state={})
    assert f != f


def test_scene_schema_builds():
    schema = LunaH3LoopScene.GET_SCHEMA()
    assert [o.id for o in schema.outputs] == OUT
    assert [i.id for i in schema.inputs] == ["state", "lock_audio", "audio_mode", "manual_cuts", "search_window",
                                            "time_lock", "loader_folder", "guide_mode", "guide_source",
                                            "guide_length", "guide_frame_idx", "guide_audio"]
    assert schema.outputs[-1].get_io_type() == "LUNA_H3_GUIDE"
    assert [o.get_io_type() for o in schema.outputs][-5:] == ["STRING"] * 4 + ["LUNA_H3_GUIDE"]


# --------------------------------------------------------------------------- Luna-H3-LoopProfile

ROOT_DA = Path("D:/Claude/DA-Music-Videos")
ROOT_RESEARCH = Path("D:/Claude/Luna-Research")


def _profile(scene_profile=None, default="normal", text=None):
    scene = {"start": 0, "end": 5, "references": ["a.png"]}
    if scene_profile:
        scene["profile"] = scene_profile
    state = {"run_name": "t", "scenes": [scene], "scene_index": 0, "scene_count": 1}
    res = LunaH3LoopProfile.execute(state, text or hs.DEFAULT_PROFILES_JSON, default).result
    names = ["profile_name", "unet_name", "low_lora_1_name", "low_lora_1_strength", "low_lora_2_name",
             "low_lora_2_strength", "high_lora_1_name", "high_lora_1_strength", "high_lora_2_name",
             "high_lora_2_strength", "steps", "transition_step", "use_high_model", "shift_video", "shift_audio"]
    return dict(zip(names, res))


def test_profile_defaults_normal_and_fast():
    n = _profile()
    assert n["profile_name"] == "normal" and "Pruned_v1.3_int8" in n["unet_name"]
    assert (n["low_lora_1_strength"], n["low_lora_2_strength"], n["high_lora_1_strength"],
            n["high_lora_2_strength"]) == (0.6, 0.0, 0.8, 0.0)
    assert (n["steps"], n["transition_step"], n["use_high_model"]) == (12, 9, True)
    assert (n["shift_video"], n["shift_audio"]) == (12.0, 3.0)
    f = _profile("fast")
    assert f["profile_name"] == "fast" and "v2_pruned_int8_convrot" in f["unet_name"]
    assert "dmad_minimax_h3_4step_full_critic" in f["low_lora_1_name"] and f["low_lora_1_strength"] == 0.8
    assert f["low_lora_2_strength"] == 0.0
    assert (f["steps"], f["transition_step"], f["use_high_model"]) == (8, 5, False)


def test_profile_defaults_match_their_source_files():
    chain = ROOT_RESEARCH / "long-video/2026-10-25-long-video/elira-shadow/chain.json"
    api = ROOT_RESEARCH / "long-video/2026-10-25-long-video/dojo-motion/renders/v2_std_dmad/scene_002_api.json"
    if not chain.is_file() or not api.is_file():
        pytest.skip("source recipes not on this machine")
    nodes = json.loads(chain.read_text(encoding="utf-8"))["settings"]["nodes"]
    n = hs.DEFAULT_PROFILES["normal"]
    assert [n["low_lora"][0]["strength"], n["low_lora"][1]["strength"], n["high_lora"][0]["strength"],
            n["high_lora"][1]["strength"]] == [nodes["95"]["strength_model"], nodes["136"]["strength_model"],
                                               nodes["131"]["strength_model"], nodes["132"]["strength_model"]]
    assert (n["steps"], n["transition_step"]) == (nodes["99"]["steps"], nodes["102"]["transition_step"])
    spec = (ROOT_DA / "SPEC.md").read_text(encoding="utf-8")
    assert "SelfLift schedule: 12 steps, 9 low + 3 high" in spec
    g = json.loads(api.read_text(encoding="utf-8"))
    f = hs.DEFAULT_PROFILES["fast"]
    assert f["unet_name"] == g["30"]["inputs"]["unet_name"]
    assert (f["low_lora"][0]["name"], f["low_lora"][0]["strength"]) == (
        g["95"]["inputs"]["lora_name"], g["95"]["inputs"]["strength_model"])
    assert f["low_lora"][1]["strength"] == g["136"]["inputs"]["strength_model"] == 0.0
    assert (f["steps"], f["transition_step"]) == (g["99"]["inputs"]["steps"], g["102"]["inputs"]["transition_step"])
    assert "high_res_model" not in g["102"]["inputs"] and f["use_high_model"] is False
    assert (f["shift_video"], f["shift_audio"]) == (g["203"]["inputs"]["shift_video"], g["203"]["inputs"]["shift_audio"])


def test_profile_errors():
    with pytest.raises(ValueError, match="profile 'slow'"):
        _profile("slow")
    base = json.loads(hs.DEFAULT_PROFILES_JSON)
    for change, fragment in (({"steps": 8, "transition_step": 8}, "transition_step < steps"),
                             ({"use_high_model": "yes"}, "use_high_model"),
                             ({"unet_name": ""}, "unet_name"),
                             ({"low_lora": [{"name": "a", "strength": 1}]}, "two"),
                             ({"shift_video": "x"}, "shift_video")):
        bad = json.loads(json.dumps(base))
        bad["normal"].update(change)
        with pytest.raises(ValueError, match=fragment):
            hs.parse_profiles(json.dumps(bad))
        assert fragment in LunaH3LoopProfile.validate_inputs(profiles=json.dumps(bad))
    assert "not valid JSON" in LunaH3LoopProfile.validate_inputs(profiles="{")
    assert LunaH3LoopProfile.validate_inputs(profiles=None) is True       # linked input: skipped
    assert LunaH3LoopProfile.validate_inputs(profiles=hs.DEFAULT_PROFILES_JSON) is True


def test_profile_schema_builds():
    schema = LunaH3LoopProfile.GET_SCHEMA()
    types = {o.id: o.get_io_type() for o in schema.outputs}
    assert types["unet_name"] == "*" and types["low_lora_1_name"] == "*" and types["use_high_model"] == "BOOLEAN"


# --------------------------------------------------------------------------- Luna-H3-LoopEnd with a plan

def _plan(**kw):
    plan = {"scene_number": 1, "project_folder": "x", "continuation": False, "pass1": False, "overlap": 39,
            "timing": {}, "audio_mode": "song_lock", "tail_padding_frames": 4, "token_count": 37, "render_frames": 20, "trim_frames": 5,
            "keep_frames": 10}
    plan.update(kw)
    return plan


def _end_state(run, plan, index=0, count=1):
    scenes = [{"start": i * 5, "end": i * 5 + 5, "references": ["a.png"]} for i in range(count)]
    return {"run_name": "t", "scenes": scenes, "scene_index": index, "scene_count": count, "plan": plan}


def _frames(n):
    g = torch.Generator().manual_seed(5)
    return torch.rand(n, 8, 8, 3, generator=g)


def test_end_trims_the_hidden_head_and_saves_the_last_frame(run, monkeypatch):
    _write_beta_scene(run.root, 1, 37, 4, pass1=True)
    images = _frames(20)
    out = LunaH3LoopEnd.execute(["1", 0], _end_state(run, _plan()), images=images, memory_policy="off")
    assert out.result[1] == 1
    manifest = json.loads((run.root / "scene_0000.manifest.json").read_text(encoding="utf-8"))
    assert manifest["info"]["frames"]["count"] == 10
    paths = {e["path"] for e in manifest["files"]}
    assert "scene_0000_last.png" in paths
    assert {"latents/scene_001.latent", "latents/scene_001.latent.json", "latents/scene_001.pass1.latent",
            "latents/scene_001.pass1.latent.json"} <= paths
    assert manifest["info"]["plan"]["keep_frames"] == 10
    from PIL import Image
    import numpy as np
    last = np.asarray(Image.open(run.root / "scene_0000_last.png"))
    expect = (images[14].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    assert np.array_equal(last, expect)                     # frame trim + keep - 1 = the last kept frame
    assert h3_loop.verify_scene(run.root, 0) == []


def test_end_refuses_a_render_of_the_wrong_length(run):
    _write_beta_scene(run.root, 1, 37, 4)
    with pytest.raises(ValueError, match="decoded to 19 frames but the plan rendered 20"):
        LunaH3LoopEnd.execute(["1", 0], _end_state(run, _plan()), images=_frames(19))
    assert not (run.root / "scene_0000.manifest.json").exists()


def test_end_checks_beta_saved_what_was_planned(run):
    with pytest.raises(FileNotFoundError, match="did not write"):
        LunaH3LoopEnd.execute(["1", 0], _end_state(run, _plan()), images=_frames(20))
    _write_beta_scene(run.root, 1, 40, 4)
    with pytest.raises(ValueError, match="40 tokens"):
        LunaH3LoopEnd.execute(["1", 0], _end_state(run, _plan()), images=_frames(20))
    _write_beta_scene(run.root, 1, 37, 0)
    with pytest.raises(ValueError, match="0 tail padding"):
        LunaH3LoopEnd.execute(["1", 0], _end_state(run, _plan()), images=_frames(20))


def test_end_hands_the_next_scene_a_state_without_the_plan(run, monkeypatch):
    _write_beta_scene(run.root, 1, 37, 4)
    prompt = {
        "1": {"class_type": h3_loop.START_ID, "inputs": {"scenes": "[]", "run_name": "t", "start_scene": 0}},
        "4": {"class_type": h3_loop.END_ID, "inputs": {"flow": ["1", 0], "state": ["1", 1]}},
    }
    from test_h3_loop import FakeDynPrompt
    monkeypatch.setattr(LunaH3LoopEnd, "hidden", SimpleNamespace(dynprompt=FakeDynPrompt(prompt), unique_id="4"),
                        raising=False)
    out = LunaH3LoopEnd.execute(["1", 0], _end_state(run, _plan(), index=0, count=2), images=_frames(20))
    start = next(n for n in out.expand.values() if n["class_type"] == h3_loop.START_ID)
    nxt = start["inputs"]["initial_state"]
    assert nxt["scene_index"] == 1 and "plan" not in nxt and nxt["scene_count"] == 2


def test_end_without_a_plan_behaves_as_in_s9(run):
    out = LunaH3LoopEnd.execute(["1", 0], {"run_name": "t", "scenes": [{"name": "x"}], "scene_index": 0,
                                           "scene_count": 1}, images=_frames(6))
    assert out.result[1] == 1
    m = json.loads((run.root / "scene_0000.manifest.json").read_text(encoding="utf-8"))
    assert m["info"]["frames"]["count"] == 6 and "plan" not in m["info"]
    assert not (run.root / "scene_0000_last.png").exists()


# --------------------------------------------------------------------------- the workflow builder's table

SKILL = Path.home() / ".claude" / "skills" / "music-video-from-audio"


def _builder():
    path = SKILL / "scripts" / "build_loop_graph.py"
    if not path.is_file():
        pytest.skip("music-video-from-audio skill not on this machine")
    spec = importlib.util.spec_from_file_location("build_loop_graph_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_builder_table_matches_the_pack():
    """build_loop_graph.py keeps its own copy of the four schemas (standard library only); it must not drift."""
    b = _builder()
    import h3_loop_guide
    for cls in (h3_loop.LunaH3LoopStart, h3_loop.LunaH3LoopEnd, LunaH3LoopScene, LunaH3LoopProfile,
                h3_loop_guide.LunaH3GuideApply, h3_loop_guide.LunaH3GuideClip):
        schema = cls.GET_SCHEMA()
        entry = b.LOOP_REG[schema.node_id]
        assert [i.id for i in schema.inputs] == list(entry["inputs"]), schema.node_id
        for i in schema.inputs:
            mine = entry["inputs"][i.id]
            assert mine["type"] == i.get_io_type(), (schema.node_id, i.id)
            assert mine["required"] == (not i.optional), (schema.node_id, i.id)
        assert [o.get_io_type() for o in schema.outputs] == entry["outputs"], schema.node_id
        assert entry["output_node"] == bool(schema.is_output_node), schema.node_id
    scene_ids = [o.id for o in LunaH3LoopScene.GET_SCHEMA().outputs]
    assert list(b.S) == scene_ids and [b.S[k] for k in scene_ids] == list(range(len(scene_ids)))
    prof_ids = [o.id for o in LunaH3LoopProfile.GET_SCHEMA().outputs]
    assert list(b.P) == prof_ids
    start_ids = [o.id for o in h3_loop.LunaH3LoopStart.GET_SCHEMA().outputs]
    assert start_ids[5] == "prompt" and start_ids[0] == "flow" and start_ids[1] == "state"
    end = h3_loop.LunaH3LoopEnd.GET_SCHEMA()
    assert list(next(i for i in end.inputs if i.id == "memory_policy").options) == list(h3_loop.MEMORY_POLICIES)
    assert list(LunaH3LoopScene.GET_SCHEMA().inputs[2].options) == list(hs.AUDIO_MODES)


def test_builder_profiles_file_equals_the_node_defaults():
    path = SKILL / "workflows" / "loop_profiles.json"
    if not path.is_file():
        pytest.skip("loop_profiles.json not on this machine")
    assert json.loads(path.read_text(encoding="utf-8")) == hs.DEFAULT_PROFILES


# --------------------------------------------------------------------------- audio modes (S10 follow-up)

def test_generated_mode_still_plans_from_the_lock_file_and_records_the_mode(run):
    a = _named(_scene(run, 0, mode="song_lock"))
    b = _named(_scene(run, 0, mode="generated"))
    assert b["state"]["plan"]["audio_mode"] == "generated" and a["state"]["plan"]["audio_mode"] == "song_lock"
    for k in ("render_frames", "trim_start", "keep_frames", "visible_duration", "seed_1", "ref_1"):
        assert a[k] == b[k]                                   # timing and references do not depend on the mode


def _wave(frames, rate=48000, ch=2):
    n = round(frames * rate / 24)
    return {"waveform": torch.arange(n, dtype=torch.float32).repeat(1, ch, 1), "sample_rate": rate}


def test_end_trims_generated_audio_with_the_frames(run):
    _write_beta_scene(run.root, 1, 37, 4)
    plan = _plan(audio_mode="generated")                      # render 20, trim 5, keep 10
    audio = _wave(20)
    LunaH3LoopEnd.execute(["1", 0], _end_state(run, plan), images=_frames(20), audio=audio)
    import wave as _w
    raw = (run.root / "scene_0000.wav").read_bytes()
    import numpy as np
    pcm = np.frombuffer(raw[raw.index(b"data") + 8:], dtype="<f4").reshape(-1, 2).T
    per = 48000 / 24
    assert pcm.shape[1] == round(10 * per)
    assert pcm[0, 0] == round(5 * per) and pcm[0, -1] == round(5 * per) + round(10 * per) - 1   # samples of frames 5..14
    manifest = json.loads((run.root / "scene_0000.manifest.json").read_text(encoding="utf-8"))
    assert manifest["info"]["audio"]["samples"] == round(10 * per)


def test_end_trims_whatever_audio_is_connected_in_either_mode(run):
    _write_beta_scene(run.root, 1, 37, 4)
    LunaH3LoopEnd.execute(["1", 0], _end_state(run, _plan(audio_mode="song_lock")), images=_frames(20), audio=_wave(20))
    m = json.loads((run.root / "scene_0000.manifest.json").read_text(encoding="utf-8"))
    assert m["info"]["audio"]["samples"] == round(10 * 48000 / 24)          # the kept frames, not the render
    for mode in ("song_lock", "generated"):
        with pytest.raises(ValueError, match="generated audio is"):
            LunaH3LoopEnd.execute(["1", 0], _end_state(run, _plan(audio_mode=mode)), images=_frames(20),
                                  audio=_wave(14))
    # Without a plan (S9 use) the audio is saved as it came.
    LunaH3LoopEnd.execute(["1", 0], {"run_name": "u", "scenes": [{}], "scene_index": 0, "scene_count": 1},
                          audio=_wave(20))
    m = json.loads((run.tmp / "h3_loops" / "u" / "scene_0000.manifest.json").read_text(encoding="utf-8"))
    assert m["info"]["audio"]["samples"] == round(20 * 48000 / 24)


def test_trim_audio_zero_fills_a_one_frame_shortfall():
    plan = _plan(audio_mode="generated", render_frames=20, trim_frames=10, keep_frames=10)
    out = h3_loop.trim_audio_to_plan(_wave(19), plan)
    assert out["waveform"].shape[-1] == round(10 * 48000 / 24) and float(out["waveform"][..., -10:].abs().max()) == 0


# --------------------------------------------------------------------------- staged references

def test_staging_is_idempotent_and_refuses_a_changed_copy(run):
    a = h3_loop.stage_reference(run.ref, run.root)
    b = h3_loop.stage_reference(run.ref, run.root)
    assert a == b and a == h3_loop.staged_name(run.root, run.ref.resolve())
    assert (run.tmp / "input" / a).is_file()
    with pytest.raises(FileNotFoundError):
        h3_loop.stage_reference(run.tmp / "missing.png", run.root)


def test_a_claimed_copy_is_replaced_an_unclaimed_one_is_not(run):
    name = h3_loop.stage_reference(run.ref, run.root)
    target = run.tmp / "input" / name
    target.write_bytes(b"left by a crashed run")                 # listed in staged_refs.json: ours
    assert h3_loop.stage_reference(run.ref, run.root) == name
    assert target.read_bytes() == run.ref.read_bytes()
    # The same for a rewritten last frame: new bytes at the same source path.
    from PIL import Image
    last = hs.last_frame_path(run.root, 0)
    Image.new("RGB", (4, 4), (9, 9, 9)).save(last)
    n1 = h3_loop.stage_reference(last, run.root)
    Image.new("RGB", (4, 4), (200, 9, 9)).save(last)
    assert h3_loop.stage_reference(last, run.root) == n1
    assert (run.tmp / "input" / n1).read_bytes() == last.read_bytes()
    # A file this run did not stage is refused.
    (run.root / h3_loop.STAGED_FILE).unlink()
    target.write_bytes(b"somebody else's")
    with pytest.raises(FileExistsError, match="not this run's"):
        h3_loop.stage_reference(run.ref, run.root)
    assert target.read_bytes() == b"somebody else's"


def test_last_run_scene_cleans_up_what_it_staged_and_only_that(run, monkeypatch):
    pre = run.tmp / "input" / "luna_chain_refs"                 # a folder that was already there
    pre.mkdir()
    keep = pre / "someone_elses.png"
    keep.write_bytes(b"x")
    name = h3_loop.stage_reference(run.ref, run.root)
    staged = run.tmp / "input" / name
    assert staged.is_file() and (run.root / h3_loop.STAGED_FILE).is_file()
    registry = json.loads((run.root / h3_loop.STAGED_FILE).read_text())
    assert len(registry["files"]) == 1 and len(registry["directories"]) == 1       # only the namespace folder is ours
    _write_beta_scene(run.root, 1, 37, 4)
    LunaH3LoopEnd.execute(["1", 0], _end_state(run, _plan()), images=_frames(20))   # the only scene: the last
    assert not staged.exists() and not staged.parent.exists()
    assert keep.is_file() and pre.is_dir()
    assert not (run.root / h3_loop.STAGED_FILE).exists()


def test_a_middle_scene_does_not_clean_up(run, monkeypatch):
    name = h3_loop.stage_reference(run.ref, run.root)
    _write_beta_scene(run.root, 1, 37, 4)
    prompt = {
        "1": {"class_type": h3_loop.START_ID, "inputs": {"scenes": "[]", "run_name": "t", "start_scene": 0}},
        "4": {"class_type": h3_loop.END_ID, "inputs": {"flow": ["1", 0], "state": ["1", 1]}},
    }
    from test_h3_loop import FakeDynPrompt
    monkeypatch.setattr(LunaH3LoopEnd, "hidden", SimpleNamespace(dynprompt=FakeDynPrompt(prompt), unique_id="4"),
                        raising=False)
    LunaH3LoopEnd.execute(["1", 0], _end_state(run, _plan(), index=0, count=2), images=_frames(20))
    assert (run.tmp / "input" / name).is_file()


# --------------------------------------------------------------------------- the builder, both audio modes

def _build(run, tmp_path, audio_mode, mutate=None, name=None, extra_args=()):
    import subprocess
    b = _builder()                                              # skips when the skill is absent
    chain = {
        "project_folder": (tmp_path / "proj").as_posix(), "master_audio": run.lock.as_posix(),
        "lock_audio": run.lock.as_posix(), "audio_mode": audio_mode, "engine": "selflift", "arm": "a",
        "size": [1920, 1088], "seeds": [1111, 2222],
        "settings": {"reference_size": "match", "reference_resize": {"all": {
            "mode": "max_mp", "max_mp": 2.09, "ratio": "1920:1088", "ratio_action": "pad", "snap": 32,
            "allow_upscale": False}}},
        "scenes": [{"start": 0.0, "end": 5.0, "prompt_file": "p.txt", "references": [run.ref.as_posix(), run.ref.as_posix()]},
                   {"start": 5.0, "end": 10.0, "prompt_file": "p.txt",
                    "references": [{"previous_last_frame": True}, run.ref.as_posix()]}]}
    if mutate:
        mutate(chain)
    (tmp_path / "p.txt").write_text("A prompt.", encoding="utf-8")
    with open(tmp_path / "chain.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(chain, fh)
    out = tmp_path / f"out_{audio_mode}"
    done = subprocess.run([sys.executable, "-I", str(Path(b.__file__)), "--chain", str(tmp_path / "chain.json"),
                           "--run-name", "t", "--out-dir", str(out), "--comfy-output", str(tmp_path),
                           *(["--name", name] if name else []), *extra_args],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    return json.loads((out / (f"{name}_api.json" if name else "long_video_loop_api.json")).read_text(encoding="utf-8"))


def test_builder_song_lock_mode_keeps_the_luna_loaders_and_the_song_lock(run, tmp_path):
    g = _build(run, tmp_path, "locked")
    assert g["302"]["inputs"]["audio_mode"] == "song_lock" and "152" not in g and "audio" not in g["304"]["inputs"]
    assert g["66"]["inputs"]["ref_audios.ref_audio_0"] == ["302", 13]
    assert g["14"]["inputs"]["image_paths"] == ["302", 14] and g["149"]["inputs"]["image_paths"] == ["302", 15]
    assert g["150"]["inputs"]["image_paths"] == "" and "ref_images.ref_image_2" not in g["66"]["inputs"]
    card = json.loads(g["14"]["inputs"]["asset_state"])
    assert card == {"items": [{"mode": "max_mp", "snap": 32, "max_mp": 2.09, "allow_upscale": False,
                               "ratio": "1920:1088", "ratio_action": "pad"}]}
    assert g["66"]["inputs"]["ref_images.ref_image_0"] == ["14", 7] and g["66"]["inputs"]["ref_image_size"] == "max"
    assert g["68"]["inputs"]["project_folder"] == (run.tmp / "h3_loops" / "t").as_posix()


def test_builder_generated_mode_follows_chain_driver(run, tmp_path):
    g = _build(run, tmp_path, "generated")
    assert g["302"]["inputs"]["audio_mode"] == "generated"
    assert not any(k.startswith("ref_audios.") for k in g["66"]["inputs"])
    assert not any(v == ["67", 0] for n in g.values() for v in n["inputs"].values())
    assert g["93"]["inputs"]["av_latent"] == ["66", 1] and g["71"]["inputs"]["on_false"] == ["66", 1]
    assert g["152"] == {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["106", 1], "audio_vae": ["33", 0]},
                        "_meta": g["152"]["_meta"]}
    assert g["304"]["inputs"]["audio"] == ["152", 0] and "108" not in g
    assert g["14"]["inputs"]["image_paths"] == ["302", 14]       # references are the same in both modes


# --------------------------------------------------------------------------- stale Beta files on overwrite_run

def _beta_files(root, numbers, pass1=True):
    for n in numbers:
        _write_beta_scene(root, n, 37, 4, pass1=pass1)


def test_overwrite_run_deletes_beta_latents_of_the_deleted_scenes_only(run):
    for i in range(3):
        h3_loop.save_scene(run.root, i, {}, audio=_wave(2))
    _beta_files(run.root, (1, 2, 3))
    other = run.tmp / "h3_loops" / "other" / "latents"            # a different run: never touched
    other.mkdir(parents=True)
    (other / "scene_002.latent").write_bytes(b"x")
    from h3_loop import LunaH3LoopStart
    LunaH3LoopStart.execute(json.dumps(run.scenes + [run.scenes[0]]), "t", 1, True)   # keep scene 0, drop 1 and 2
    left = sorted(p.name for p in (run.root / "latents").iterdir())
    assert left == ["scene_001.latent", "scene_001.latent.json", "scene_001.pass1.latent",
                    "scene_001.pass1.latent.json"]
    assert (other / "scene_002.latent").is_file()


def test_beta_files_without_a_manifest_also_block_a_fresh_start_unless_overwritten(run):
    _beta_files(run.root, (1,))                                   # a crash after Beta's save, before Luna-H3-LoopEnd
    from h3_loop import LunaH3LoopStart
    with pytest.raises(ValueError, match="already holds saved scene"):
        LunaH3LoopStart.execute(json.dumps(run.scenes), "t", 0)
    assert (run.root / "latents" / "scene_001.latent").is_file()
    LunaH3LoopStart.execute(json.dumps(run.scenes), "t", 0, True)
    assert list((run.root / "latents").iterdir()) == []


def test_resume_does_not_delete_the_predecessors_latents(run):
    h3_loop.save_scene(run.root, 0, {}, audio=_wave(2))
    _beta_files(run.root, (1,))
    from h3_loop import LunaH3LoopStart
    LunaH3LoopStart.execute(json.dumps(run.scenes), "t", 1)       # resume at scene 1: Beta scene 1 is the head
    assert (run.root / "latents" / "scene_001.latent").is_file()


# --------------------------------------------------------------------------- builder: merged settings

def test_builder_merges_chain_and_scene_settings(run, tmp_path):
    def mutate(chain):
        chain["settings"]["selflift_seed"] = 5
        chain["scenes"][1]["settings"] = {"selflift_seed": 9}
    g = _build(run, tmp_path, "locked", mutate)
    scenes = json.loads(g["301"]["inputs"]["scenes"])
    assert [sc.get("selflift_seed") for sc in scenes] == [5, 9]


def test_builder_refuses_reference_cards_that_differ_between_scenes(run, tmp_path):
    def mutate(chain):
        chain["scenes"][1]["settings"] = {"reference_size": "max", "reference_resize": {}}
    import subprocess
    with pytest.raises(AssertionError, match="differ between scenes"):
        _build(run, tmp_path, "locked", mutate)


def test_builder_names_its_files_with_name(run, tmp_path):
    _build(run, tmp_path, "locked", name="s10_x")
    assert (tmp_path / "out_locked" / "s10_x_api.json").is_file() and (tmp_path / "out_locked" / "s10_x.json").is_file()


def test_builder_join_defaults_equal_the_pack_defaults():
    import h3_loop_join
    b = _builder()
    assert b.JOIN_DEFAULTS == {k: v for k, v in h3_loop_join.DEFAULTS.items()}
    end = h3_loop.LunaH3LoopEnd.GET_SCHEMA()
    options = {i.id: list(i.options) for i in end.inputs if hasattr(i, "options")}
    for key, opts in (("cut_rule", h3_loop_join.CUT_RULES), ("breath_source", h3_loop_join.BREATH_SOURCES),
                      ("fallback", h3_loop_join.FALLBACKS), ("color_match", h3_loop_join.COLOR_MODES)):
        assert options[key] == list(opts) == b.LOOP_REG["Luna-H3-LoopEnd"]["inputs"][key]["options"]
    assert b.parse_join_options(["cut_rule=breath", "time_lock=false", "match_threshold=4", "search_window=9"]) == {
        "cut_rule": "breath", "time_lock": False, "match_threshold": 4.0, "search_window": 9}


# --------------------------------------------------------------------------- the time mapping, against Beta

def _fake_manifest(root, index, trim, keep):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"scene_{index:04d}.manifest.json").write_text(
        json.dumps({"info": {"plan": {"trim_frames": trim, "keep_frames": keep}}}), encoding="utf-8")


def test_overlap_mapping_agrees_with_betas_plan_for_every_overlap_and_length(tmp_path):
    """Chain scenes through Beta's own planner: for every overlap, scene length and predecessor, the previous
    scene's kept frame that head frame 0 shows is the same by Beta (start_frame - trim) and by keep - head, and
    both put head frame 0 on the same song time as that kept frame, to within a frame."""
    cases = 0
    for overlap in hs.OVERLAPS:
        for lengths in ((5.0, 5.0, 6.0), (7.5, 4.2, 8.0), (10.0, 3.0, 5.5), (4.0, 12.0, 6.0), (6.25, 6.25, 6.25)):
            starts = [0.0]
            for d in lengths:
                starts.append(starts[-1] + d)
            pred_info = None
            for i, d in enumerate(lengths):
                start, end = starts[i], starts[i + 1]
                try:
                    calc = hs.calculate(BETA, start, end, 120.0, pred_info, overlap)
                except ValueError:
                    break                                          # too long for 362 frames at this overlap
                t = calc["timing"]
                trim, keep = round(t["final_trim_start_seconds"] * 24), t["final_frame_count"]
                if i:
                    root = tmp_path / f"c{cases}"
                    _fake_manifest(root, i - 1, prev_trim, prev_keep)
                    m = hs.overlap_mapping(root, i, calc["context_plan"], t)
                    assert m["offset"] == prev_keep - trim == calc["context_plan"]["start_frame"] - prev_trim
                    # the same song time, from the two scenes' own clocks (frames)
                    head0 = start * 24 - trim                                  # song time of this scene's render frame 0
                    pred_frame = starts[i - 1] * 24 - prev_trim + calc["context_plan"]["start_frame"]
                    assert abs(head0 - pred_frame) <= 1.0
                    cases += 1
                prev_trim, prev_keep = trim, keep
                pred_info = {"token_count": BETA._frames_to_tokens(t["h3_frame_count"]),
                             "tail_padding_frames": calc["tail_padding_frames"]}
    assert cases >= 25


def test_overlap_mapping_refuses_a_plan_that_disagrees(tmp_path):
    _fake_manifest(tmp_path, 0, 0, 100)
    ctx = {"warmup_frames": 12, "start_frame": 50, "end_frame": 62}
    t = {"final_trim_start_seconds": 12 / 24}
    with pytest.raises(ValueError, match="disagrees with Beta's plan"):
        hs.overlap_mapping(tmp_path, 1, ctx, t)                   # Beta: 50 - 0; keep - head: 100 - 12 = 88
    assert hs.overlap_mapping(tmp_path, 1, dict(ctx, start_frame=88), t)["offset"] == 88
    with pytest.raises(ValueError, match="not Beta's warm-up"):
        hs.overlap_mapping(tmp_path, 1, dict(ctx, warmup_frames=13, start_frame=88), t)
    assert hs.overlap_mapping(tmp_path / "none", 1, ctx, t) is None    # no previous plan on disk (older run)


def test_loop_scene_plan_carries_the_overlap_mapping_and_song_times(run):
    first = _named(_scene(run, 0))["state"]["plan"]
    _write_beta_scene(run.root, 1, first["token_count"], first["tail_padding_frames"])
    # Luna-H3-LoopEnd would have saved scene 0's manifest with its plan; the second scene reads it.
    _fake_manifest(run.root, 0, first["trim_frames"], first["keep_frames"])
    from PIL import Image
    Image.new("RGB", (8, 8)).save(h3_loop_scene_last(run, 0))
    plan = _named(_scene(run, 1))["state"]["plan"]
    om = plan["overlap_map"]
    assert om["offset"] == first["keep_frames"] - plan["trim_frames"] == om["beta_start_frame"] - first["trim_frames"]
    assert plan["start_seconds"] == 5.0 and plan["end_seconds"] == 10.0 and plan["lock_audio"].endswith("lock.wav")
    assert plan["context_plan"]["warmup_frames"] == plan["trim_frames"]
    assert first["overlap_map"] is None                      # the first scene has no predecessor


def test_loop_scene_refuses_a_bad_manual_cut_before_rendering(run):
    first = _named(_scene(run, 0))["state"]["plan"]
    _write_beta_scene(run.root, 1, first["token_count"], first["tail_padding_frames"])
    _fake_manifest(run.root, 0, first["trim_frames"], first["keep_frames"])
    from PIL import Image
    Image.new("RGB", (8, 8)).save(h3_loop_scene_last(run, 0))
    keep0 = first["keep_frames"]
    ok = LunaH3LoopScene.execute(run.state(1), run.lock.as_posix(), "song_lock", f'{{"1": {keep0}}}')
    assert ok.result
    with pytest.raises(ValueError, match="falls outside the overlap"):
        LunaH3LoopScene.execute(run.state(1), run.lock.as_posix(), "song_lock", '{"1": 3}')
    with pytest.raises(ValueError, match="names join 5"):
        LunaH3LoopScene.execute(run.state(1), run.lock.as_posix(), "song_lock", '{"5": 3}')
    with pytest.raises(ValueError, match="join number 0"):
        LunaH3LoopScene.execute(run.state(1), run.lock.as_posix(), "song_lock", '{"0": 3}')
    # time_lock off: the range cannot be known yet, so the entry passes here.
    assert LunaH3LoopScene.execute(run.state(1), run.lock.as_posix(), "song_lock", '{"1": 3}', 0, False).result


def test_builder_scene_inputs_follow_loop_end(tmp_path):
    b = _builder()
    assert list(b.LOOP_REG["Luna-H3-LoopScene"]["inputs"]) == [i.id for i in LunaH3LoopScene.GET_SCHEMA().inputs]


# --------------------------------------------------------------------------- S12: the guide in the loop

def _tail_scene(run, n=30):
    """Scene 0 saved with n PNG frames (grey k/255) so scene 1 can take a tail guide from it."""
    import sys as _sys
    from unittest import mock as _mock
    frames_ = torch.stack([torch.full((8, 8, 3), k / 255.0) for k in range(n)])
    with _mock.patch.dict(_sys.modules, {"av": None}):
        h3_loop.save_scene(run.root, 0, {}, images=frames_)


def _guide_scene(run, **fields):
    scenes = json.loads(json.dumps(run.scenes))
    scenes[0]["references"] = [run.ref.as_posix()]
    scenes[1].update(join="fresh", references=[run.ref.as_posix()], **fields)
    return scenes


def test_guide_is_off_by_default_and_changes_nothing(run):
    o = _named(_scene(run, 0))
    assert o["guide"] == {"mode": "off", "source": "previous_tail", "keyframes": []}
    # the rest of the outputs are what they were before the guide existed
    assert o["render_frames"] == hs.calculate(BETA, 0.0, 5.0, 30.0)["timing"]["h3_frame_count"]


def test_a_tail_guide_for_a_fresh_scene_lands_at_the_first_visible_frame(run):
    _tail_scene(run)
    scenes = _guide_scene(run, preroll_frames=39)
    res = LunaH3LoopScene.execute(run.state(1, scenes), run.lock.as_posix(), "generated", "", 0, True, "",
                                  "add_guide", "previous_tail", 22, 4, False).result
    guide = _named(res)["guide"]
    plan = _named(res)["state"]["plan"]
    kf = guide["keyframes"][0]
    assert guide["mode"] == "add_guide" and plan["trim_frames"] == 39 and plan["continuation"] is False
    assert kf["index"] == 39 + 4 and kf["images"].shape == (22, 8, 8, 3)              # first visible frame + guide_frame_idx
    assert [round(float(f.mean()) * 255) for f in kf["images"]] == list(range(8, 30))   # the last 22 of scene 0's 30


def test_a_scene_field_overrides_the_node_defaults(run):
    _tail_scene(run)
    scenes = _guide_scene(run, guide_mode="add_guide_vlm", guide_length=5, guide_frame_idx=2)
    res = LunaH3LoopScene.execute(run.state(1, scenes), run.lock.as_posix(), "generated", "", 0, True, "",
                                  "add_guide", "previous_tail", 22, 0, False).result
    g = _named(res)["guide"]
    assert g["mode"] == "add_guide_vlm" and g["keyframes"][0]["images"].shape[0] == 5 and g["keyframes"][0]["index"] == 2
    off = _guide_scene(run, guide_mode="off")                                         # a scene can turn the default off
    res = LunaH3LoopScene.execute(run.state(1, off), run.lock.as_posix(), "generated", "", 0, True, "",
                                  "add_guide", "previous_tail", 22, 0, False).result
    assert _named(res)["guide"]["mode"] == "off"


def test_a_tail_guide_on_a_carried_scene_is_refused_before_scene_0_renders(run):
    scenes = json.loads(json.dumps(run.scenes))
    scenes[0]["references"] = [run.ref.as_posix()]
    args = (run.lock.as_posix(), "generated", "", 0, True, "", "add_guide", "previous_tail", 22, 0, False)
    # The node default alone: scene 0 and the carried scene 1 simply go without a tail guide.
    for index in (0,):
        assert _named(LunaH3LoopScene.execute(run.state(index, scenes), *args).result)["guide"]["mode"] == "off"
    # A scene that asks for the tail itself, on a carried join, is stopped before the first scene renders.
    scenes[1]["guide_mode"] = "add_guide"
    with pytest.raises(ValueError, match="carried scene is refused"):
        LunaH3LoopScene.execute(run.state(0, scenes), *args)


def test_a_guide_that_does_not_fit_stops_the_scene(run):
    _tail_scene(run)
    scenes = _guide_scene(run, guide_mode="add_guide", guide_frame_idx=500)
    with pytest.raises(ValueError, match="does not fit"):
        LunaH3LoopScene.execute(run.state(1, scenes), run.lock.as_posix(), "generated").result


def test_builder_wires_the_guide_nodes_and_leaves_node_66_alone(run, tmp_path):
    g = _build(run, tmp_path, "generated", extra_args=["--guide", "mode=add_guide_vlm", "--guide", "length=39",
                                                       "--guide", "frame_idx=2", "--guide", "audio=true"])
    sc = g["302"]["inputs"]
    assert (sc["guide_mode"], sc["guide_source"], sc["guide_length"], sc["guide_frame_idx"], sc["guide_audio"]) == (
        "add_guide_vlm", "previous_tail", 39, 2, True)
    assert g["306"]["inputs"] == {"clip": ["31", 0], "guide": ["302", 18], "width": ["64", 0], "height": ["65", 0]}
    assert g["307"]["inputs"] == {"positive": ["142", 0], "latent": ["66", 1], "guide": ["302", 18], "vae": ["32", 0],
                                  "audio_vae": ["33", 0]}
    assert g["66"]["class_type"] == "MiniMaxH3ReferenceToVideo" and g["66"]["inputs"]["clip"] == ["306", 0]
    assert g["102"]["inputs"]["positive"] == ["307", 0] and g["101"]["inputs"]["conditioning"] == ["142", 0]


def test_builder_default_is_guide_off(run, tmp_path):
    g = _build(run, tmp_path, "locked")
    sc = g["302"]["inputs"]
    assert sc["guide_mode"] == "off" and g["102"]["inputs"]["positive"] == ["307", 0]


def test_builder_carries_scene_guide_fields_and_appends_a_fresh_chain(run, tmp_path):
    def mutate(chain):
        chain["scenes"][1]["guide_mode"] = "add_guide"
        chain["scenes"][1]["guide_length"] = 5
        chain["scenes"][1]["guide_images"] = [{"path": run.ref.as_posix(), "frame": 3}]
        chain["scenes"][1]["guide_source"] = "images"
    extra = tmp_path / "more.json"
    more = {"project_folder": str(tmp_path), "master_audio": run.lock.as_posix(), "lock_audio": run.lock.as_posix(),
            "audio_mode": "generated", "engine": "selflift", "arm": "a", "size": [1920, 1088],
            "settings": {"reference_size": "match", "reference_resize": {"all": {
                "mode": "max_mp", "max_mp": 2.09, "ratio": "1920:1088", "ratio_action": "pad", "snap": 32,
                "allow_upscale": False}}},
            "scenes": [{"start": 0.0, "end": 6.0, "prompt_file": "p.txt", "references": [run.ref.as_posix(), run.ref.as_posix()]}]}
    with open(extra, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(more, fh)
    g = _build(run, tmp_path, "generated", mutate, extra_args=["--add-chain", str(extra)])
    scenes = json.loads(g["301"]["inputs"]["scenes"])
    assert len(scenes) == 3 and scenes[2]["join"] == "fresh" and "join" not in scenes[1]
    assert scenes[1]["guide_mode"] == "add_guide" and scenes[1]["guide_length"] == 5 and scenes[1]["guide_source"] == "images"
    assert scenes[1]["guide_images"] == [{"path": run.ref.as_posix(), "frame": 3}]

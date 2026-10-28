"""Unit tests for Luna-H3-LoopStart / Luna-H3-LoopEnd (S9), CPU only.

Needs ComfyUI on the path (comfy_api, comfy_execution.GraphBuilder, comfy.nested_tensor):
the loop expansion is tested against the real GraphBuilder, not a stub. Without ComfyUI the
whole module is skipped.

Run from this folder: `pytest . --basetemp=D:/t/ComfyUI-LunaMiniMaxH3`
(the pack's __init__.py makes pytest try to import the pack as a package when run from the root).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("comfy_api.latest")
pytest.importorskip("comfy_execution.graph_utils")
nested_tensor = pytest.importorskip("comfy.nested_tensor")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import h3_loop  # noqa: E402
from h3_loop import (  # noqa: E402
    END_ID, START_ID, LunaH3LoopEnd, LunaH3LoopStart,
    check_run_name, find_body, load_latent, parse_scenes, release_between_scenes,
    sha256_file, verify_scene,
)

SCENES = [{"name": "S1", "prompt": "a"}, {"name": "S2", "prompt": "b"}, {"name": "S3"}]
SCENES_JSON = json.dumps(SCENES)


# --------------------------------------------------------------------------- fakes

class FakeDynPrompt:
    """The four DynamicPrompt methods the loop code uses. Nodes are API-format dicts."""

    def __init__(self, prompt: dict):
        self.original = {k: dict(v) for k, v in prompt.items()}
        self.nodes = {k: dict(v) for k, v in prompt.items()}

    def add(self, graph: dict):
        for k, v in graph.items():
            self.nodes[k] = dict(v)

    def get_node(self, node_id):
        return self.nodes[node_id]

    def get_original_prompt(self):
        return self.original

    def get_display_node_id(self, node_id):
        return self.nodes[node_id].get("override_display_id", node_id)


def _body_prompt():
    """1 Start -> 2 Mid -> 4 End; 3 Side feeds Mid from outside; 5 Save hangs off Mid;
    6 consumes End; 7 hangs off Start but End never depends on it."""
    return {
        "1": {"class_type": START_ID, "inputs": {"scenes": SCENES_JSON, "run_name": "t", "start_scene": 0}},
        "2": {"class_type": "Mid", "inputs": {"a": ["1", 1], "b": ["3", 0]}},
        "3": {"class_type": "Side", "inputs": {}},
        "4": {"class_type": END_ID, "inputs": {"flow": ["1", 0], "state": ["1", 1], "images": ["2", 0],
                                               "fps": 24.0, "memory_policy": "off"}},
        "5": {"class_type": "SaveOut", "inputs": {"x": ["2", 0]}},
        "6": {"class_type": "Consumer", "inputs": {"c": ["4", 0]}},
        "7": {"class_type": "DanglingFromStart", "inputs": {"s": ["1", 2]}},
    }


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    monkeypatch.setattr(h3_loop, "_output_root", lambda: tmp_path)
    monkeypatch.setattr(h3_loop, "_output_node_types", lambda: {"SaveOut"})
    return tmp_path


def _images(n=4, h=8, w=8):
    g = torch.Generator().manual_seed(1)
    return torch.rand(n, h, w, 3, generator=g)


def _audio(samples=1600, sr=16000, ch=2):
    g = torch.Generator().manual_seed(2)
    return {"waveform": torch.rand(1, ch, samples, generator=g) * 2 - 1, "sample_rate": sr}


def _av_latent():
    g = torch.Generator().manual_seed(3)
    video = torch.randn(1, 24, 7, 4, 4, generator=g)
    audio = torch.randn(1, 32, 2, 10, generator=g)
    return {"samples": nested_tensor.NestedTensor((video, audio))}, video, audio


def _start(out_root, scenes=SCENES_JSON, run_name="t", start_scene=0, initial_state=None):
    kwargs = {} if initial_state is None else {"initial_state": initial_state}
    return LunaH3LoopStart.execute(scenes, run_name, start_scene, **kwargs).result


def _end(monkeypatch, dyn, end_id, flow, state, **kwargs):
    monkeypatch.setattr(LunaH3LoopEnd, "hidden",
                        SimpleNamespace(dynprompt=dyn, unique_id=end_id), raising=False)
    return LunaH3LoopEnd.execute(flow, state, **kwargs)


# --------------------------------------------------------------------------- scene JSON / run name

@pytest.mark.parametrize("text, fragment", [
    ("", "empty"),
    ("   ", "empty"),
    ("[{", "not valid JSON"),
    ('{"name": "x"}', "must be a JSON list"),
    ("[]", "empty list"),
    ('["x"]', "scene 0 must be a JSON object"),
    ('[{"prompt": 3}]', "'prompt' must be a string"),
])
def test_scene_json_errors(text, fragment):
    with pytest.raises(ValueError, match=fragment):
        parse_scenes(text)


def test_scene_json_ok():
    assert parse_scenes(SCENES_JSON) == SCENES


@pytest.mark.parametrize("name", ["", "../x", "a/b", "a\\b", ".hidden", "a..b", "x" * 81])
def test_bad_run_names(name):
    with pytest.raises(ValueError, match="run_name"):
        check_run_name(name)


def test_validate_inputs_reports_strings():
    assert LunaH3LoopStart.validate_inputs(SCENES_JSON, "t", 0) is True
    assert "not valid JSON" in LunaH3LoopStart.validate_inputs("[", "t", 0)
    assert "outside" in LunaH3LoopStart.validate_inputs(SCENES_JSON, "t", 3)
    assert "run_name" in LunaH3LoopStart.validate_inputs(SCENES_JSON, "../t", 0)


def test_start_outputs_and_never_cached(out_root):
    flow, state, idx, count, scene, prompt = _start(out_root)
    assert (idx, count, prompt) == (0, 3, "a")
    assert json.loads(scene) == SCENES[0]
    assert state["scene_index"] == 0 and flow["scene_index"] == 0
    assert _start(out_root, start_scene=0)[5] == "a"
    fp = LunaH3LoopStart.fingerprint_inputs(scenes=SCENES_JSON)
    assert fp != fp  # NaN
    assert _start(out_root, scenes='[{"name": "only"}]')[5] == ""  # no prompt field -> ""


def test_start_scene_out_of_range(out_root):
    with pytest.raises(ValueError, match="outside"):
        _start(out_root, start_scene=3)


# --------------------------------------------------------------------------- body detection / clone

def test_body_detection():
    dyn = FakeDynPrompt(_body_prompt())
    with mock.patch.object(h3_loop, "_output_node_types", lambda: {"SaveOut"}):
        body = find_body(["1", 0], dyn, "4")
    # Start, Mid, End and the Save hanging off Mid. Not Side (feeds in), Consumer (after End),
    # nor the branch off Start that End never depends on.
    assert set(body) == {"1", "2", "4", "5"}


def test_body_without_output_node_types_skips_the_save():
    dyn = FakeDynPrompt(_body_prompt())
    with mock.patch.object(h3_loop, "_output_node_types", lambda: set()):
        assert set(find_body(["1", 0], dyn, "4")) == {"1", "2", "4"}


def test_flow_must_come_from_start():
    dyn = FakeDynPrompt(_body_prompt())
    with pytest.raises(ValueError, match="directly from Luna-H3-LoopStart"):
        find_body(["2", 0], dyn, "4")
    with pytest.raises(ValueError, match="straight from"):
        find_body("not a link", dyn, "4")


def test_clone_repoints_inside_and_leaves_outside(out_root):
    dyn = FakeDynPrompt(_body_prompt())
    next_state = {"scene_index": 1, "marker": True}
    built = h3_loop.expand_body(["1", 0], next_state, dyn, "4", 2)
    g = built["graph"]
    by_class = {}
    for nid, node in g.items():
        by_class.setdefault(node["class_type"], []).append((nid, node))
    assert sorted(by_class) == sorted([START_ID, "Mid", END_ID, "SaveOut"])
    (start_id, start), (mid_id, mid), (end_id, end), (save_id, save) = (
        by_class[START_ID][0], by_class["Mid"][0], by_class[END_ID][0], by_class["SaveOut"][0])
    assert end_id.endswith("Recurse")
    assert start["inputs"]["initial_state"] == next_state
    assert mid["inputs"]["a"] == [start_id, 1]          # re-pointed at the cloned Start
    assert mid["inputs"]["b"] == ["3", 0]                # outside the body: untouched
    assert end["inputs"]["flow"] == [start_id, 0]
    assert end["inputs"]["state"] == [start_id, 1]
    assert end["inputs"]["images"] == [mid_id, 0]
    assert save["inputs"]["x"] == [mid_id, 0]
    assert mid["override_display_id"] == "2"
    # The expansion's results are the clone End's two outputs.
    assert built["outputs"] == ([end_id, 0], [end_id, 1])
    # The original prompt is not touched.
    assert "initial_state" not in dyn.nodes["1"]["inputs"]


# --------------------------------------------------------------------------- the loop, end to end

def _run_loop(monkeypatch, out_root, scenes=SCENES, run_name="t", start_scene=0):
    """Drive Start/End by hand over a model-free body, expanding each time like ComfyUI would."""
    prompt = _body_prompt()
    prompt["1"]["inputs"].update(scenes=json.dumps(scenes), run_name=run_name, start_scene=start_scene)
    dyn = FakeDynPrompt(prompt)
    start_id, end_id = "1", "4"
    indices, results = [], None
    for _ in range(len(scenes) + 2):  # bounded: a runaway loop fails the assertions instead of hanging
        inputs = dict(dyn.get_node(start_id)["inputs"])
        flow, state, idx, count, _scene, _prompt = LunaH3LoopStart.execute(
            inputs["scenes"], inputs["run_name"], inputs["start_scene"],
            **({"initial_state": inputs["initial_state"]} if "initial_state" in inputs else {}))
        indices.append(idx)
        out = _end(monkeypatch, dyn, end_id, [start_id, 0], state,
                   images=_images(), audio=_audio(), latent=_av_latent()[0], memory_policy="off")
        if out.expand is None:
            results = out.result
            break
        dyn.add(out.expand)
        end_id = next(k for k, v in out.expand.items() if v["class_type"] == END_ID)
        start_id = out.expand[end_id]["inputs"]["flow"][0]
    return indices, results


def test_three_scenes_save_three_and_one_manifest(monkeypatch, out_root):
    indices, results = _run_loop(monkeypatch, out_root)
    assert indices == [0, 1, 2]                      # the index advanced once per clone, then stopped
    root = out_root / "h3_loops" / "t"
    manifest_path, count, video, report = results
    assert count == 3 and video == "" and report == ""   # S9 bodies have no frames: nothing to join
    assert Path(manifest_path) == root / "run.manifest.json"
    run = json.loads((root / "run.manifest.json").read_text(encoding="utf-8"))
    assert run["scene_count"] == 3 and [s["scene_index"] for s in run["scenes"]] == [0, 1, 2]
    assert [s["scene"] for s in run["scenes"]] == SCENES
    for i in range(3):
        stem = f"scene_{i:04d}"
        for suffix in (".manifest.json", ".wav", ".latent.safetensors", ".latent.json"):
            assert (root / (stem + suffix)).is_file(), stem + suffix
        assert verify_scene(root, i) == []


def test_last_scene_alone_terminates_without_expanding(monkeypatch, out_root):
    _, state, *_ = _start(out_root, start_scene=0)
    state = dict(state, scene_index=2)
    # scenes 0..1 are not saved, but a plain End on the last scene does not check them for
    # saving; the run manifest does, so it must refuse to claim a complete run.
    dyn = FakeDynPrompt(_body_prompt())
    with pytest.raises(ValueError, match="not complete"):
        _end(monkeypatch, dyn, "4", ["1", 0], state, images=_images())


def test_end_outputs_are_none_before_the_last_scene(monkeypatch, out_root):
    _, state, *_ = _start(out_root)
    dyn = FakeDynPrompt(_body_prompt())
    with mock.patch.object(h3_loop, "_output_node_types", lambda: set()):
        out = _end(monkeypatch, dyn, "4", ["1", 0], state, images=_images())
    assert out.expand is not None and len(out.result) == 4
    assert not (out_root / "h3_loops" / "t" / "run.manifest.json").exists()


# --------------------------------------------------------------------------- what gets saved

def test_frames_mp4_when_av_present(out_root):
    pytest.importorskip("av")
    root = out_root / "r"
    info = h3_loop.save_scene(root, 0, {"name": "x"}, images=_images(), fps=24.0)
    manifest = json.loads(info["manifest"].read_text(encoding="utf-8"))
    assert manifest["info"]["frames"]["format"] == "mp4"
    assert manifest["info"]["frames"]["count"] == 4
    entry = manifest["files"][0]
    assert entry["path"] == "scene_0000.mp4" and entry["bytes"] > 0
    assert entry["sha256"] == sha256_file(root / "scene_0000.mp4")


def test_frames_png_without_av_or_with_odd_size(out_root):
    root = out_root / "r"
    with mock.patch.dict(sys.modules, {"av": None}):  # import av -> ImportError
        h3_loop.save_scene(root, 0, {}, images=_images(3))
    m = json.loads((root / "scene_0000.manifest.json").read_text(encoding="utf-8"))
    assert m["info"]["frames"]["format"] == "png"
    assert [e["path"] for e in m["files"]] == [f"scene_0000_frames/frame_{i:05d}.png" for i in range(3)]
    h3_loop.save_scene(root, 1, {}, images=_images(2, 7, 7))      # odd size, whatever av is
    m = json.loads((root / "scene_0001.manifest.json").read_text(encoding="utf-8"))
    assert m["info"]["frames"]["format"] == "png"


def test_audio_wav_is_float_and_exact(out_root):
    import numpy as np
    audio = _audio()
    path = out_root / "a.wav"
    info = h3_loop.save_audio_wav(audio, path)
    assert info == {"channels": 2, "samples": 1600, "sample_rate": 16000}
    raw = path.read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
    data_at = raw.index(b"data")
    pcm = np.frombuffer(raw[data_at + 8:], dtype="<f4").reshape(-1, 2).T
    np.testing.assert_array_equal(pcm, audio["waveform"][0].numpy())
    fmt_at = raw.index(b"fmt ")
    assert int.from_bytes(raw[fmt_at + 8:fmt_at + 10], "little") == 3  # IEEE float


def test_nested_latent_round_trip_both_streams(out_root):
    latent, video, audio = _av_latent()
    files = h3_loop.save_latent(latent, out_root, "scene_0000")
    sidecar = json.loads(files[1].read_text(encoding="utf-8"))
    streams = sidecar["entries"]["samples"]["streams"]
    assert sidecar["entries"]["samples"]["nested"] is True
    assert [s["shape"] for s in streams] == [list(video.shape), list(audio.shape)]
    assert all(s["dtype"] == "torch.float32" for s in streams)
    back = load_latent(files[0])
    assert back["samples"].is_nested
    v2, a2 = back["samples"].unbind()
    assert torch.equal(v2, video) and torch.equal(a2, audio)


def test_plain_latent_and_mask_round_trip(out_root):
    samples = torch.randn(1, 4, 3, 3, dtype=torch.bfloat16)
    mask = torch.ones(1, 1, 3, 3)
    files = h3_loop.save_latent({"samples": samples, "noise_mask": mask}, out_root, "s")
    back = load_latent(files[0])
    assert torch.equal(back["samples"], samples) and back["samples"].dtype == torch.bfloat16
    assert torch.equal(back["noise_mask"], mask)
    with pytest.raises(ValueError, match="no 'samples'"):
        h3_loop.save_latent({}, out_root, "bad")


def test_manifest_hashes_match_and_tampering_is_caught(out_root):
    root = out_root / "r"
    h3_loop.save_scene(root, 0, {"name": "x"}, audio=_audio(), latent=_av_latent()[0])
    m = json.loads((root / "scene_0000.manifest.json").read_text(encoding="utf-8"))
    assert {e["path"] for e in m["files"]} == {"scene_0000.wav", "scene_0000.latent.safetensors",
                                               "scene_0000.latent.json"}
    for e in m["files"]:
        assert sha256_file(root / e["path"]) == e["sha256"]
    assert verify_scene(root, 0) == []
    with open(root / "scene_0000.wav", "ab") as fh:
        fh.write(b"x")
    assert "does not match" in verify_scene(root, 0)[0]


# --------------------------------------------------------------------------- resume

def _save_prior(out_root, n):
    root = out_root / "h3_loops" / "t"
    for i in range(n):
        h3_loop.save_scene(root, i, SCENES[i], audio=_audio())
    return root


def test_resume_ok_when_earlier_scenes_are_intact(out_root):
    _save_prior(out_root, 2)
    assert _start(out_root, start_scene=2)[2] == 2


@pytest.mark.parametrize("damage", ["manifest", "file", "tamper"])
def test_resume_fails_on_missing_or_changed_file(out_root, damage):
    root = _save_prior(out_root, 2)
    if damage == "manifest":
        (root / "scene_0001.manifest.json").unlink()
    elif damage == "file":
        (root / "scene_0000.wav").unlink()
    else:
        (root / "scene_0001.wav").write_bytes(b"different")
    with pytest.raises(ValueError, match="cannot resume at scene 2"):
        _start(out_root, start_scene=2)


def test_resume_fails_with_nothing_saved(out_root):
    with pytest.raises(ValueError, match="scene 0: scene_0000.manifest.json is missing"):
        _start(out_root, start_scene=1)


def test_clone_start_does_not_re_verify(out_root):
    # A clone carries initial_state; it must not look at the disk or the widgets.
    _, state, *_ = _start(out_root)
    out = _start(out_root, scenes="[", run_name="..", start_scene=9, initial_state=dict(state, scene_index=1))
    assert out[2] == 1 and out[5] == "b"


# --------------------------------------------------------------------------- memory policy

def test_policy_off_does_nothing(out_root):
    mm = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"comfy.model_management": mm}):
        assert release_between_scenes("off", 0) == {"policy": "off"}
    assert mm.mock_calls == []


def test_policy_unload_and_fresh_call_the_model_manager():
    mm = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"comfy.model_management": mm}):
        with mock.patch.object(sys.modules["comfy"], "model_management", mm, create=True):
            release_between_scenes("unload_models", 0)
            mm.unload_all_models.assert_called_once()
            mm.soft_empty_cache.assert_called_once()
            mm.cleanup_models_gc.assert_not_called()
            release_between_scenes("fresh_scene", 1)
            mm.cleanup_models_gc.assert_called_once()
            assert mm.unload_all_models.call_count == 2


def test_bad_policy_rejected():
    with pytest.raises(ValueError, match="memory_policy"):
        release_between_scenes("sometimes", 0)
    assert "memory_policy" in LunaH3LoopEnd.validate_inputs(memory_policy="sometimes")
    assert LunaH3LoopEnd.validate_inputs(memory_policy="fresh_scene") is True


def test_end_with_policy_off_leaves_the_model_manager_alone(monkeypatch, out_root):
    mm = mock.MagicMock()
    _, state, *_ = _start(out_root)
    dyn = FakeDynPrompt(_body_prompt())
    with mock.patch.dict(sys.modules, {"comfy.model_management": mm}):
        _end(monkeypatch, dyn, "4", ["1", 0], state, images=_images(), memory_policy="off")
    assert mm.mock_calls == []


# --------------------------------------------------------------------------- schema

def test_schemas_build():
    start = LunaH3LoopStart.GET_SCHEMA()
    end = LunaH3LoopEnd.GET_SCHEMA()
    assert end.enable_expand and end.is_output_node
    flow_input = next(i for i in end.inputs if i.id == "flow")
    assert flow_input.rawLink is True
    assert [o.id for o in start.outputs] == ["flow", "state", "scene_index", "scene_count", "scene", "prompt"]
    assert [o.id for o in end.outputs] == ["manifest_path", "scene_count", "joined_video_path", "report_path"]
    info = LunaH3LoopEnd.GET_NODE_INFO_V1()
    assert info["input"]["required"]["flow"][1]["rawLink"] is True
    policy = info["input"]["optional"]["memory_policy"]
    assert policy[0] == "COMBO" and policy[1]["options"] == list(h3_loop.MEMORY_POLICIES)


# --------------------------------------------------------------------------- review fixes

def test_validate_inputs_skips_linked_inputs():
    # The executor hands None for an input wired to another node.
    assert LunaH3LoopStart.validate_inputs(None, None, None) is True
    assert LunaH3LoopStart.validate_inputs(None, "t", 99) is True            # no scene count known
    assert LunaH3LoopStart.validate_inputs(SCENES_JSON, None, 2) is True
    assert "not valid JSON" in LunaH3LoopStart.validate_inputs("[", None, None)
    assert "run_name" in LunaH3LoopStart.validate_inputs(None, "../x", None)
    assert LunaH3LoopStart.validate_inputs(SCENES_JSON, "t", None, overwrite_run=None) is True


def _branch_prompt():
    """Decode (8) feeds End and also CreateVideo (9) -> SaveVideo (10); 6 consumes End."""
    return {
        "1": {"class_type": START_ID, "inputs": {"scenes": SCENES_JSON, "run_name": "t", "start_scene": 0}},
        "8": {"class_type": "Decode", "inputs": {"s": ["1", 1]}},
        "9": {"class_type": "CreateVideo", "inputs": {"images": ["8", 0]}},
        "10": {"class_type": "SaveVideo", "inputs": {"video": ["9", 0]}},
        "4": {"class_type": END_ID, "inputs": {"flow": ["1", 0], "state": ["1", 1], "images": ["8", 0]}},
        "6": {"class_type": "Consumer", "inputs": {"c": ["4", 0]}},
        "11": {"class_type": "Preview", "inputs": {"x": ["6", 0]}},   # reaches an output, but only through End
    }


def test_output_branch_with_intermediates_is_in_the_body_every_scene(monkeypatch):
    monkeypatch.setattr(h3_loop, "_output_node_types", lambda: {"SaveVideo", "Preview"})
    dyn = FakeDynPrompt(_branch_prompt())
    assert set(find_body(["1", 0], dyn, "4")) == {"1", "8", "9", "10", "4"}   # not 6 or 11
    # Expand twice: the second pass runs on the first clone's ids.
    first = h3_loop.expand_body(["1", 0], {"scene_index": 1}, dyn, "4", 2)
    assert sorted(n["class_type"] for n in first["graph"].values()) == sorted(
        [START_ID, "Decode", "CreateVideo", "SaveVideo", END_ID])
    dyn.add(first["graph"])
    end1 = next(k for k, v in first["graph"].items() if v["class_type"] == END_ID)
    start1 = first["graph"][end1]["inputs"]["flow"][0]
    second = h3_loop.expand_body([start1, 0], {"scene_index": 2}, dyn, end1, 2)
    assert sorted(n["class_type"] for n in second["graph"].values()) == sorted(
        [START_ID, "Decode", "CreateVideo", "SaveVideo", END_ID])
    video = next(n for n in second["graph"].values() if n["class_type"] == "SaveVideo")
    create_id = next(k for k, n in second["graph"].items() if n["class_type"] == "CreateVideo")
    assert video["inputs"]["video"] == [create_id, 0]


def _saved_run(out_root, n=3):
    root = out_root / "h3_loops" / "t"
    for i in range(n):
        h3_loop.save_scene(root, i, SCENES[i], images=_images(2), audio=_audio())
    h3_loop.write_run_manifest(root, "t", n)
    return root


def test_fresh_start_refuses_existing_scenes_and_names_the_folder(out_root):
    root = _saved_run(out_root)
    with pytest.raises(ValueError, match="already holds saved scene") as exc:
        _start(out_root, start_scene=0)
    assert str(root) in str(exc.value)
    assert (root / "run.manifest.json").exists() and (root / "scene_0002.wav").exists()  # untouched


def test_overwrite_run_deletes_scenes_from_start_and_the_run_manifest(out_root):
    root = _saved_run(out_root)
    out = LunaH3LoopStart.execute(SCENES_JSON, "t", 1, True).result
    assert out[2] == 1
    assert not (root / "run.manifest.json").exists()
    assert (root / "scene_0000.manifest.json").exists() and (root / "scene_0000.wav").exists()
    leftovers = [p.name for p in root.iterdir() if p.name.startswith(("scene_0001", "scene_0002"))]
    assert leftovers == []


def test_overwrite_run_removes_png_frame_folders(out_root):
    root = out_root / "h3_loops" / "t"
    with mock.patch.dict(sys.modules, {"av": None}):
        h3_loop.save_scene(root, 0, SCENES[0], images=_images(2))
    assert (root / "scene_0000_frames").is_dir()
    LunaH3LoopStart.execute(SCENES_JSON, "t", 0, True)
    assert not (root / "scene_0000_frames").exists() and not (root / "scene_0000.manifest.json").exists()


def test_resume_removes_stale_run_manifest_but_keeps_earlier_scenes(out_root):
    root = _saved_run(out_root, 2)                      # scenes 0,1 + a run manifest
    assert (root / "run.manifest.json").exists()
    out = _start(out_root, start_scene=2)
    assert out[2] == 2
    assert not (root / "run.manifest.json").exists()
    assert verify_scene(root, 0) == [] and verify_scene(root, 1) == []


def test_no_part_files_and_a_failed_write_never_looks_complete(out_root):
    root = out_root / "r"
    h3_loop.save_scene(root, 0, {}, images=_images(2), audio=_audio(), latent=_av_latent()[0])
    assert [p.name for p in root.rglob("*.part")] == []

    import safetensors.torch as st

    def torn(tensors, path):
        Path(path).write_bytes(b"half")
        raise OSError("disk full")

    with mock.patch.object(st, "save_file", torn):
        with pytest.raises(OSError):
            h3_loop.save_scene(root, 1, {}, latent=_av_latent()[0])
    assert not (root / "scene_0001.latent.safetensors").exists()
    assert not (root / "scene_0001.manifest.json").exists()
    assert verify_scene(root, 1)  # reports scene 1 as missing


@pytest.mark.parametrize("name", ["CON", "con", "Prn", "AUX", "nul", "COM1", "com9", "LPT1", "lpt9",
                                  "CON.txt", "nul.run.1", "aux ", "run.", "run ", "COM1 .x"])
def test_windows_device_names_and_trailing_dots_rejected(name):
    with pytest.raises(ValueError, match="run_name"):
        check_run_name(name)


@pytest.mark.parametrize("name", ["COM0", "COM10", "CONSOLE", "console.1", "my run", "luna-1.2", "LPT"])
def test_similar_but_legal_names_pass(name):
    assert check_run_name(name) == name

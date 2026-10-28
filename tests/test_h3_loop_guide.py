"""Unit tests for the loop guide (S12), CPU only: tail cuts, keyframe rows against core Add Guide, the VLM items,
per-scene overrides, and off = unchanged.

Run from this folder: `pytest . --basetemp=D:/t/ComfyUI-LunaMiniMaxH3`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("comfy_api.latest")
nested_tensor = pytest.importorskip("comfy.nested_tensor")
pytest.importorskip("comfy.ldm.minimax.model")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import h3_loop  # noqa: E402
import h3_loop_guide as hg  # noqa: E402

H, W = 64, 96


class FakeVAE:
    audio_sample_rate = 32000

    def encode(self, x):
        if x.ndim == 4 and x.shape[-1] == 3:                       # video frames [n, H, W, 3]
            n, h, w, _ = x.shape
            return torch.full((1, 24, max(1, (n - 1) // 4 + 1), h // 16, w // 16), float(x.mean()))
        return torch.full((1, 32, 2, x.shape[1] // 1000 + 1), float(x.mean()))     # audio [1, L, C]


def positive(existing=None):
    return [[torch.zeros(1, 3, 5), ({"minimax_keyframes": existing} if existing else {})]]


def latent(T=7):
    video = torch.zeros(1, 24, T, H // 16, W // 16)
    audio = torch.zeros(1, 32, 2, round(22 / 24 * 40))
    return {"samples": nested_tensor.NestedTensor((video, audio))}


def frames(n, value=None):
    """n frames [n, H, W, 3]; frame k is the flat grey k/255 unless `value` says otherwise."""
    return torch.stack([torch.full((H, W, 3), (k if value is None else value) / 255.0) for k in range(n)])


def kf(index, n=5):
    return {"index": index, "images": frames(n), "audio": None}


def guide(mode="add_guide", *kfs):
    return {"mode": mode, "source": "previous_tail", "keyframes": list(kfs) or [kf(10)]}


# --------------------------------------------------------------------------- the tail

@pytest.fixture
def saved(tmp_path):
    root = tmp_path / "run"
    with mock.patch.dict(sys.modules, {"av": None}):               # PNG: exact pixel values
        h3_loop.save_scene(root, 0, {}, images=frames(50))
    return root


@pytest.mark.parametrize("n", [5, 22, 39])
def test_the_tail_is_cut_to_exactly_n_frames(saved, n):
    tail = hg.read_tail(saved, 0, n)
    assert tail.shape == (n, H, W, 3)
    assert [round(float(f.mean()) * 255) for f in tail] == list(range(50 - n, 50))        # the LAST n frames, in order


def test_a_tail_longer_than_the_scene_is_refused(saved):
    with pytest.raises(ValueError, match="has 50 frames; a 59-frame"):
        hg.read_tail(saved, 0, 59)


def test_the_valid_clip_lengths_follow_core():
    assert [hg.valid_clip_length(n) for n in (1, 4, 5, 21, 22, 38, 39, 60, 77)] == [1, 1, 5, 5, 22, 22, 39, 56, 73]


def test_a_clip_is_cut_to_the_longest_valid_length(tmp_path):
    import av
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=24)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        for k in range(45):
            for p in s.encode(av.VideoFrame.from_ndarray(frames(1, k * 5)[0].mul(255).byte().numpy(), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    assert hg.read_clip(str(path), 39).shape[0] == 39
    assert hg.read_clip(str(path), 22).shape[0] == 22
    assert hg.read_clip(str(path), 40).shape[0] == 39              # 40 is not a clip length: the largest below it


# --------------------------------------------------------------------------- the render frame

def test_frame_idx_counts_from_the_first_visible_frame():
    where = "scene"
    assert hg.render_index(0, 12, 100, 5, where) == 12
    assert hg.render_index(7, 12, 100, 5, where) == 19
    assert hg.render_index(-10, 12, 100, 5, where) == 90          # back from the end of the render, as core
    with pytest.raises(ValueError, match="hidden head"):
        hg.render_index(-95, 12, 100, 1, where)
    with pytest.raises(ValueError, match="does not fit"):
        hg.render_index(85, 12, 100, 5, where)
    assert hg.render_index(0, 0, 100, 22, where) == 0


# --------------------------------------------------------------------------- the keyframe rows, against core

core = None
try:
    import comfy_extras.nodes_minimax_h3 as core
except Exception:                                                   # pragma: no cover
    core = None


@pytest.mark.skipif(core is None, reason="core nodes_minimax_h3 not importable")
@pytest.mark.parametrize("n, idx", [(1, 0), (1, 9), (5, 0), (5, 10), (22, 0), (39, 0)])
def test_keyframe_rows_equal_core_add_guide(n, idx):
    T = 14 if n > 22 else 7                                         # 39 frames need a longer video
    lat, vae = latent(T), FakeVAE()
    imgs = frames(max(n, 1))
    mine = hg.apply_guide(positive(), lat, vae, None, guide("add_guide", {"index": idx, "images": imgs, "audio": None}))
    theirs = core.MiniMaxH3AddGuide.execute(positive(), lat, idx, vae=vae, image=imgs).result[0]
    a, b = mine[0][1]["minimax_keyframes"], theirs[0][1]["minimax_keyframes"]
    assert len(a) == len(b) == 1
    assert a[0]["resolved_frame_index"] == b[0]["resolved_frame_index"] == idx
    assert torch.equal(a[0]["latent"], b[0]["latent"]) and a[0].keys() == b[0].keys()


@pytest.mark.skipif(core is None, reason="core nodes_minimax_h3 not importable")
def test_keyframe_audio_equals_core_add_guide():
    lat, vae, avae = latent(7), FakeVAE(), FakeVAE()
    audio = {"waveform": torch.rand(1, 2, 32000), "sample_rate": 32000}
    g = guide("add_guide", {"index": 3, "images": frames(5), "audio": audio})
    mine = hg.apply_guide(positive(), lat, vae, avae, g)[0][1]["minimax_keyframes"][0]
    theirs = core.MiniMaxH3AddGuide.execute(positive(), lat, 3, vae=vae, audio_vae=avae, image=frames(5),
                                            audio=audio).result[0][0][1]["minimax_keyframes"][0]
    assert torch.equal(mine["audio_latent"], theirs["audio_latent"])
    assert mine["resolved_frame_index"] == theirs["resolved_frame_index"]


def test_several_keyframes_append_after_existing_ones_and_never_touch_the_input():
    existing = [{"resolved_frame_index": 0, "latent": torch.zeros(1)}]
    base = positive(existing)
    out = hg.apply_guide(base, latent(7), FakeVAE(), None, guide("add_guide", kf(2, 1), kf(10, 5)))
    rows = out[0][1]["minimax_keyframes"]
    assert [r["resolved_frame_index"] for r in rows] == [0, 2, 10]
    assert base[0][1]["minimax_keyframes"] == existing and len(existing) == 1


def test_a_guide_that_does_not_fit_is_refused():
    with pytest.raises(ValueError, match="does not fit"):
        hg.apply_guide(positive(), latent(7), FakeVAE(), None, guide("add_guide", kf(20, 5)))   # the video has 22 frames
    with pytest.raises(ValueError, match="needs the video vae"):
        hg.apply_guide(positive(), latent(7), None, None, guide("add_guide", kf(2, 1)))
    with pytest.raises(ValueError, match="AV latent"):
        hg.apply_guide(positive(), {"samples": torch.zeros(1, 4, 2, 2)}, FakeVAE(), None, guide("add_guide", kf(2, 1)))


def test_off_leaves_the_conditioning_exactly_as_it_was():
    base = positive([{"resolved_frame_index": 3, "latent": torch.zeros(1)}])
    for g in (None, {"mode": "off", "keyframes": []}, {"mode": "off", "keyframes": [kf(2)]},
              {"mode": "add_guide", "keyframes": []}):
        assert hg.apply_guide(base, latent(7), FakeVAE(), None, g) is base


# --------------------------------------------------------------------------- the VLM side

class FakeClip:
    def __init__(self):
        self.calls = []
        self.cloned = 0

    def clone(self):
        other = FakeClip()
        other.calls = self.calls
        other.cloned = self.cloned + 1
        return other

    def tokenize(self, text, return_word_ids=False, **kwargs):
        self.calls.append((text, kwargs))
        return {"tokens": text}


def labels(items):
    """The <Picture i> / <Video k> numbers the tokenizer would print, per item, in order."""
    count, out = {"image": 0, "video": 0, "audio": 0}, []
    for it in items:
        count[it["type"]] += 1
        out.append((it["type"], count[it["type"]]))
    return out


def ref_items():
    pic = lambda v: {"type": "image", "data": torch.full((1, 8, 8, 3), v)}
    return [pic(1.0), pic(2.0), pic(3.0), {"type": "audio"}]


def test_guide_pictures_go_after_the_reference_pictures_and_keep_their_numbers():
    refs = ref_items()
    out = hg.insert_guide_items(refs, [{"type": "image", "data": torch.zeros(1, 8, 8, 3)}])
    assert [it["type"] for it in out] == ["image", "image", "image", "image", "audio"]
    assert labels(out)[:3] == labels(refs)[:3] == [("image", 1), ("image", 2), ("image", 3)]     # refs: same numbers
    assert out[:3] == refs[:3] and out[-1] is refs[-1]
    assert labels(out)[3] == ("image", 4)


def test_a_guide_clip_goes_after_the_last_reference_video_or_before_standalone_audio():
    clip = {"type": "video", "data": torch.zeros(2, 8, 8, 3), "timestamps": [0.0, 0.5]}
    out = hg.insert_guide_items(ref_items(), [clip])
    assert [it["type"] for it in out] == ["image", "image", "image", "video", "audio"]
    with_video = [ref_items()[0], {"type": "audio"}, {"type": "video", "data": torch.zeros(2, 8, 8, 3)}, {"type": "audio"}]
    out = hg.insert_guide_items(with_video, [clip])
    assert [it["type"] for it in out] == ["image", "audio", "video", "video", "audio"]
    assert out[2] is with_video[2]                                  # the reference video is still Video 1
    assert hg.insert_guide_items([], [clip]) == [clip]


def test_vlm_items_use_the_core_layout_for_one_frame_and_for_a_clip():
    g = guide("add_guide_vlm", kf(10, 1), kf(20, 22))
    items = hg.vlm_items(g, 96, 64)
    assert items[0]["type"] == "image" and items[0]["data"].shape == (1, 64, 96, 3)
    assert items[1]["type"] == "video" and items[1]["data"].shape == (2, 64, 96, 3)         # 2 fps: frames 0 and 12
    assert items[1]["timestamps"] == [0.0, 0.5]
    # The picture is resized to the canvas like core's keyframe images (aspect-preserving cover crop).
    items = hg.vlm_items(guide("add_guide_vlm", {"index": 0, "images": frames(1)[:, :32, :32], "audio": None}), 96, 64)
    assert items[0]["data"].shape == (1, 64, 96, 3)


def test_the_vlm_clip_adds_the_guide_only_for_add_guide_vlm():
    clip = FakeClip()
    for mode in ("off", "add_guide"):
        assert hg.vlm_clip(clip, guide(mode), 96, 64) is clip                           # same object, nothing patched
    assert hg.vlm_clip(clip, None, 96, 64) is clip
    assert hg.vlm_clip(clip, {"mode": "add_guide_vlm", "keyframes": []}, 96, 64) is clip
    wrapped = hg.vlm_clip(clip, guide("add_guide_vlm", kf(10, 1)), 96, 64)
    assert wrapped is not clip and wrapped.cloned == 1
    refs = ref_items()
    wrapped.tokenize("a prompt", minimax_ref_items=refs)
    seen = clip.calls[-1][1]["minimax_ref_items"]
    assert [it["type"] for it in seen] == ["image", "image", "image", "image", "audio"]
    assert seen[:3] == refs[:3] and seen[3]["data"].shape == (1, 64, 96, 3)
    assert [it["type"] for it in refs] == ["image", "image", "image", "audio"]            # the caller's list is untouched
    # The original clip still tokenizes without a guide.
    clip.tokenize("plain", minimax_ref_items=refs)
    assert len(clip.calls[-1][1]["minimax_ref_items"]) == 4


def test_the_vlm_clip_without_references_uses_the_keyframe_image_path():
    clip = FakeClip()
    wrapped = hg.vlm_clip(clip, guide("add_guide_vlm", kf(10, 1)), 96, 64)
    wrapped.tokenize("p")
    imgs = clip.calls[-1][1]["images"]
    assert len(imgs) == 1 and imgs[0].shape == (1, 64, 96, 3)
    wrapped.tokenize("p", images=[torch.zeros(1, 8, 8, 3)])
    assert len(clip.calls[-1][1]["images"]) == 2


# --------------------------------------------------------------------------- scene fields

def test_scene_fields_override_the_node_defaults():
    defaults = {"guide_mode": "add_guide", "guide_source": "previous_tail", "guide_length": 22, "guide_frame_idx": 0}
    scene = {"guide_mode": "add_guide_vlm", "guide_length": 5, "guide_frame_idx": 6}
    g = hg.guide_fields(scene, defaults, 1)
    assert (g["mode"], g["source"], g["length"], g["frame_idx"], g["audio"]) == ("add_guide_vlm", "previous_tail", 5, 6, False)
    assert hg.guide_fields({}, defaults, 1)["length"] == 22
    assert hg.guide_fields({"guide_mode": "off"}, defaults, 1)["mode"] == "off"          # a scene can switch it off
    assert hg.guide_fields({}, {}, 0)["mode"] == "off"                                   # the built-in default is off


@pytest.mark.parametrize("scene, fragment", [
    ({"guide_mode": "maybe"}, "guide_mode must be one of"),
    ({"guide_mode": "add_guide", "guide_source": "web"}, "guide_source must be one of"),
    ({"guide_mode": "add_guide", "guide_length": 23}, "guide_length must be one of"),
    ({"guide_mode": "add_guide", "guide_frame_idx": 1.5}, "whole number"),
    ({"guide_mode": "add_guide", "guide_audio": "yes"}, "true or false"),
    ({"guide_mode": "add_guide", "guide_source": "clip"}, "needs guide_clip"),
    ({"guide_mode": "add_guide", "guide_source": "images"}, "needs guide_images"),
    ({"guide_mode": "add_guide", "guide_source": "images", "guide_images": [{"frame": 1}]}, "needs guide_images"),
])
def test_guide_field_errors(scene, fragment):
    with pytest.raises(ValueError, match=fragment):
        hg.guide_fields(scene, {}, 2)


def test_check_guides_refuses_what_the_loop_cannot_use_before_anything_renders(tmp_path):
    on = {"guide_mode": "add_guide"}
    ok = [{"start": 0, "end": 5}, {"start": 5, "end": 10, "join": "fresh"}]
    hg.check_guides(ok, on, "generated")                                                  # fresh scene: previous_tail is fine
    # Defaults alone: the first and the carried scenes simply go without a tail guide ...
    both = [{"start": 0, "end": 5}, {"start": 5, "end": 10}]
    hg.check_guides(both, on, "generated")
    assert [hg.guide_fields(sc, on, i)["mode"] for i, sc in enumerate(both)] == ["off", "off"]
    assert hg.guide_fields(ok[1], on, 1)["mode"] == "add_guide"
    # ... but a scene that asks for one itself and cannot have it is an error.
    with pytest.raises(ValueError, match="scene 0 guide: previous_tail needs a previous scene"):
        hg.check_guides([{"start": 0, "end": 5, "guide_mode": "add_guide"}], {}, "generated")
    with pytest.raises(ValueError, match="carried scene is refused"):
        hg.check_guides([{"start": 0, "end": 5}, {"start": 5, "end": 10, "guide_mode": "add_guide"}], {}, "generated")
    with pytest.raises(ValueError, match="carried scene is refused"):
        hg.check_guides(both[:1] + [{"start": 5, "end": 10, "guide_source": "previous_tail"}], on, "generated")
    hg.check_guides([{"start": 0, "end": 5}, {"start": 5, "end": 10}], {"guide_mode": "off"}, "generated")
    with pytest.raises(ValueError, match="guide_audio is for audio_mode generated"):
        hg.check_guides(ok, {**on, "guide_audio": True}, "song_lock")
    hg.check_guides(ok, {**on, "guide_audio": True}, "generated")
    with pytest.raises(FileNotFoundError, match="nope.mp4"):
        hg.check_guides([{"start": 0, "end": 5, "guide_mode": "add_guide", "guide_source": "clip",
                          "guide_clip": str(tmp_path / "nope.mp4")}], {}, "generated")
    # a carried scene may take a clip or key images: they do not repeat the head
    img = tmp_path / "k.png"
    from PIL import Image
    Image.new("RGB", (8, 8)).save(img)
    hg.check_guides([{"start": 0, "end": 5}, {"start": 5, "end": 10, "guide_mode": "add_guide_vlm", "guide_source": "images",
                                              "guide_images": [{"path": str(img), "frame": 4}]}], {}, "generated")


def test_build_guide_for_each_source(saved, tmp_path):
    from PIL import Image
    img = tmp_path / "k.png"
    Image.new("RGB", (W, H), (10, 20, 30)).save(img)
    g = hg.guide_fields({"guide_mode": "add_guide_vlm", "guide_length": 22, "guide_frame_idx": 3}, {}, 1)
    built = hg.build_guide(g, root=saved, index=1, trim=0, render_frames=124)
    assert built["mode"] == "add_guide_vlm" and built["keyframes"][0]["index"] == 3
    assert built["keyframes"][0]["images"].shape[0] == 22
    g = hg.guide_fields({"guide_mode": "add_guide", "guide_source": "images",
                         "guide_images": [{"path": str(img), "frame": 0}, {"path": str(img), "frame": -1}]}, {}, 1)
    built = hg.build_guide(g, root=saved, index=1, trim=12, render_frames=124)
    assert [k["index"] for k in built["keyframes"]] == [12, 123]
    assert built["keyframes"][0]["images"].shape == (1, H, W, 3)
    off = hg.build_guide(hg.guide_fields({}, {}, 1), root=saved, index=1, trim=12, render_frames=124)
    assert off == {"mode": "off", "source": "previous_tail", "keyframes": []}


def test_guide_audio_from_the_previous_scene_tail(tmp_path):
    root = tmp_path / "run"
    wave = torch.arange(30 * 2000, dtype=torch.float32).repeat(1, 2, 1)
    with mock.patch.dict(sys.modules, {"av": None}):
        h3_loop.save_scene(root, 0, {}, images=frames(30), audio={"waveform": wave[:, :, :30 * 2000], "sample_rate": 48000})
    g = hg.guide_fields({"guide_mode": "add_guide", "guide_length": 5, "guide_audio": True}, {}, 1)
    built = hg.build_guide(g, root=root, index=1, trim=0, render_frames=60)
    audio = built["keyframes"][0]["audio"]
    assert audio["sample_rate"] == 48000 and audio["waveform"].shape == (1, 2, 5 * 2000)
    assert float(audio["waveform"][0, 0, -1]) == 30 * 2000 - 1                     # the last 5 frames of the scene's audio


# --------------------------------------------------------------------------- the nodes

def test_nodes_build_and_pass_through_when_off():
    apply_schema, clip_schema = hg.LunaH3GuideApply.GET_SCHEMA(), hg.LunaH3GuideClip.GET_SCHEMA()
    assert [i.id for i in apply_schema.inputs] == ["positive", "latent", "guide", "vae", "audio_vae"]
    assert [i.id for i in clip_schema.inputs] == ["clip", "guide", "width", "height"]
    base, clip = positive(), FakeClip()
    off = {"mode": "off", "keyframes": []}
    assert hg.LunaH3GuideApply.execute(base, latent(7), off).result[0] is base
    assert hg.LunaH3GuideClip.execute(clip, off, 96, 64).result[0] is clip
    out = hg.LunaH3GuideApply.execute(base, latent(7), guide("add_guide", kf(2, 1)), vae=FakeVAE()).result[0]
    assert out[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 2

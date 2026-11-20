"""media_io.py: the frame counts made for H3 round UP onto the grid (luna_units), a cut to what a file holds rounds
down, no 362 cut; labels come from `who`; and the copy check.

media_io.py is identical in Luna-MinimaxH3 and Luna-Media-Tools (Luna-Media-Tools `tests/test_media_io_copy.py` is
the other side of the check). The copy check reads the sibling pack from env LUNA_PACKS_DIR (default D:\\Claude) and
skips when it is not there.
"""

import filecmp
import os
import sys
import unittest

import pytest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)
import luna_units  # noqa: E402
import media_io  # noqa: E402

PACKS_DIR = os.environ.get("LUNA_PACKS_DIR", r"D:\Claude")
OTHER = "Luna-Media-Tools"


class CopyTests(unittest.TestCase):
    def test_copy_is_identical_in_luna_media_tools(self):
        other = os.path.join(PACKS_DIR, OTHER, "media_io.py")
        if not os.path.isfile(other):
            self.skipTest(f"{OTHER} is not beside this pack")
        self.assertTrue(filecmp.cmp(os.path.join(PACK, "media_io.py"), other, shallow=False), f"{OTHER}/media_io.py differs")

    def test_no_node_name_in_a_message(self):
        # Every label comes from `who`; the header names both nodes, nothing else does.
        with open(os.path.join(PACK, "media_io.py"), encoding="utf-8") as fh:
            body = fh.read().split('"""', 2)[2]
        for name in ("Luna-H3-AssetLoader", "Luna-LoadVideo]", "[Luna-"):
            self.assertNotIn(name, body)


class FrameRuleTests(unittest.TestCase):
    def test_grid_frames_rounds_up_with_no_362_cut(self):
        self.assertEqual(media_io.grid_frames(120, "h3"), 124)
        self.assertEqual(media_io.grid_frames(40, "h3"), 56)           # up, not the nearest 39
        self.assertEqual(media_io.grid_frames(384, "h3"), 396)         # the old code cut this to 362
        self.assertEqual(media_io.grid_frames(120, "ltx"), 121)
        self.assertEqual(media_io.grid_frames(100, "ltx"), 105)
        self.assertEqual(media_io.grid_frames(120, "free"), 120)
        for n in range(1, 500):
            self.assertEqual(media_io.grid_frames(n, "h3"), luna_units.frames_up(n, "h3"))

    def test_grid_count_up_but_never_past_what_exists(self):
        self.assertEqual(media_io.grid_count(120, "h3", 200), 124)
        self.assertEqual(media_io.grid_count(130, "h3", 130), 124)     # 141 does not fit: what is there, cut down
        self.assertEqual(media_io.grid_count(500, "h3", 1000), 515)
        self.assertEqual(media_io.grid_count(10, "h3", 4), 0)

    def test_cut_to_what_exists_rounds_down(self):
        self.assertEqual(media_io.snap_frames_down(130), 124)
        self.assertEqual(media_io.snap_frames_down(4), 0)
        self.assertEqual(media_io.snap_frames_up(120), 124)

    def test_plan_load_warns_past_362(self):
        src = {"width": 640, "height": 360, "fps": 24.0, "duration": 20.0}
        p = media_io.plan_load(src, frames=384, grid="h3")
        self.assertEqual(p["frames"], 396)
        self.assertIn(luna_units.frames_warning(396, "h3"), p["notes"])

    def test_shot_step_at_the_shots_rate(self):
        self.assertAlmostEqual(media_io.song_shot_start(0.0, 2, 129 / 25, 22, 25.0), 129 / 25 - 22 / 25)
        self.assertAlmostEqual(media_io.song_shot_start(0.0, 2, 124 / 24, 22), 124 / 24 - 22 / 24)


def _write_clip(path, seconds, fps=24, size=16):
    import av
    import numpy as np
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width = s.height = size
        s.pix_fmt = "yuv420p"
        for k in range(int(seconds * fps)):
            img = np.full((size, size, 3), k % 255, dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def test_decode_video_keeps_past_362_and_warns(tmp_path):
    clip = tmp_path / "long.mp4"
    _write_clip(clip, 17)
    frames, info = media_io.decode_video(str(clip), 0.0, 16.0, who="Test-Node")
    assert info["frames"] == 396 == frames.shape[0]
    assert "396 frames is past the 362" in info["warning"]
    frames, info = media_io.decode_video(str(clip), 0.0, 5.0)
    assert info["frames"] == 124 and info["warning"] == ""


def test_labels_come_from_who(tmp_path):
    clip = tmp_path / "short.mp4"
    _write_clip(clip, 1)
    with pytest.raises(RuntimeError, match=r"^\[Test-Node\] No frames"):
        media_io.decode_clip(str(clip), 30.0, 24, 24.0, who="Test-Node")
    with pytest.raises(RuntimeError, match=r"^\[media_io\] No frames"):
        media_io.decode_clip(str(clip), 30.0, 24, 24.0)

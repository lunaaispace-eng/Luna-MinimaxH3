"""luna_units.py: the frame grid, the overlap list, the frame rate of each target, and the copy check.

Identical in Luna-Media-Tools and Luna-MinimaxH3 (`tests/test_luna_units.py`), like `luna_units.py` itself. The copy check
reads the sibling pack from env LUNA_PACKS_DIR (default D:\\Claude) and skips it when it is not there.
"""

import ast
import filecmp
import os
import sys
import unittest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)
import luna_units as lu  # noqa: E402

PACKS_DIR = os.environ.get("LUNA_PACKS_DIR", r"D:\Claude")
PACKS = ("Luna-Media-Tools", "Luna-MinimaxH3")


class FrameTests(unittest.TestCase):
    def test_up_onto_the_h3_grid(self):
        for n, want in ((0, 5), (1, 5), (5, 5), (6, 22), (120, 124), (124, 124), (125, 141), (362, 362), (363, 379)):
            self.assertEqual(lu.frames_up(n), want, n)

    def test_up_on_ltx_and_free(self):
        self.assertEqual(lu.frames_up(120, "ltx"), 121)
        self.assertEqual(lu.frames_up(121, "ltx"), 121)
        self.assertEqual(lu.frames_up(0, "ltx"), 1)
        self.assertEqual(lu.frames_up(7, "free"), 7)
        self.assertEqual(lu.frames_up(0, "free"), 1)

    def test_down_cuts_to_what_exists(self):
        for n, want in ((4, 0), (5, 5), (21, 5), (130, 124), (362, 362), (378, 362)):
            self.assertEqual(lu.frames_down(n), want, n)
        self.assertEqual(lu.frames_down(128, "ltx"), 121)
        self.assertEqual(lu.frames_down(0, "ltx"), 0)

    def test_seconds_become_whole_frames_then_go_up(self):
        self.assertEqual(lu.frames_for(5.0), 124)            # 120 frames -> 124
        self.assertEqual(lu.frames_for(124 / 24), 124)       # a grid count's own length gives it back
        self.assertEqual(lu.frames_for(243 / 24), 243)
        self.assertEqual(lu.frames_for(5.17), 124)           # 124.08 frames is 124 whole frames
        self.assertEqual(lu.frames_for(16), 396)             # 384 -> 396, not cut at 362
        self.assertEqual(lu.frames_for(5.0, 25, "ltx"), 129)
        self.assertEqual(lu.whole_frames(5.0625), 122)       # 121.5 frames, half up

    def test_past_362_warns_and_is_not_cut(self):
        self.assertEqual(lu.frames_warning(362), "")
        self.assertIn("362", lu.frames_warning(379))
        self.assertEqual(lu.frames_warning(400, "ltx"), "")
        f, warn = lu.h3_frames(380)
        self.assertEqual(f, 396)
        self.assertTrue(warn and warn.isascii())
        self.assertEqual(lu.h3_frames(120), (124, ""))

    def test_a_fraction_of_a_frame_counts_as_a_frame(self):
        self.assertEqual(lu.frames_up(5.1), 22)
        self.assertEqual(lu.h3_frames(124.9), (141, ""))
        self.assertEqual(lu.frames_up(124.00000000000001), 124)   # a float's last digit is no frame
        self.assertEqual(lu.frames_up(124 / 24 * 24), 124)
        self.assertEqual(lu.frames_down(124.9), 124)
        self.assertEqual(lu.frames_down(123.5), 107)                    # below 124: cut to 107
        self.assertEqual(lu.frames_down(124 / 24 * 24), 124)

    def test_exactly_half_a_frame_goes_up(self):
        # 0.9375 s x 24 = 22.5 frames: 23 whole frames, then up to 39 (Canvas's old Python round gave 22)
        self.assertEqual(lu.whole_frames(0.9375), 23)
        self.assertEqual(lu.frames_for(0.9375), 39)

    def test_on_grid(self):
        self.assertTrue(lu.on_grid(124))
        self.assertTrue(lu.on_grid(5))
        self.assertFalse(lu.on_grid(125))
        self.assertFalse(lu.on_grid(0))
        self.assertFalse(lu.on_grid(124.9))
        self.assertTrue(lu.on_grid(124.0))
        self.assertTrue(lu.on_grid(121, "ltx"))
        self.assertTrue(lu.on_grid(7, "free"))

    def test_an_unknown_grid_raises_in_ascii(self):
        with self.assertRaises(ValueError):
            lu.frames_up(10, "nope")
        try:
            lu.first_frames("全")
        except ValueError as e:
            self.assertTrue(str(e).isascii())


class OverlapTests(unittest.TestCase):
    def test_values_on_the_list_pass_without_a_word(self):
        for o in lu.OVERLAPS:
            self.assertEqual(lu.snap_overlap(o), (o, ""))

    def test_any_other_value_snaps_to_the_nearest_with_a_warning(self):
        for v, want in ((30, 22), (31, 39), (60, 39), (100, 90), (500, 192), (-5, 0), (11, 0), (5, 0)):
            got, warn = lu.snap_overlap(v)
            self.assertEqual(got, want, v)
            self.assertTrue(warn.isascii() and str(want) in warn, v)

    def test_fractions_are_measured_as_they_are(self):
        self.assertEqual(lu.snap_overlap(30.9)[0], 39)     # nearer 39 than 22
        for v in (22.9, -0.9, 38.5):
            got, warn = lu.snap_overlap(v)
            self.assertTrue(warn, v)                       # off the list: always a warning
        self.assertEqual(lu.snap_overlap(39.0), (39, ""))
        for bad in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                lu.snap_overlap(bad)


class TargetTests(unittest.TestCase):
    def test_grid_and_rate_per_target(self):
        self.assertEqual(lu.grid_for("h3"), "h3")
        self.assertEqual(lu.grid_for(" LTX "), "ltx")
        self.assertEqual(lu.grid_for("general"), "free")
        self.assertEqual(lu.grid_for(None), "free")
        self.assertEqual(lu.fps_for("h3", 30), 24.0)
        self.assertEqual(lu.fps_for("ltx"), 25.0)
        self.assertEqual(lu.fps_for("general", 30), 30.0)
        self.assertEqual(lu.fps_for("general", 29.97), 29.97)
        for bad in (None, 0, -1, "abc", "inf", float("nan"), 10 ** 400):
            self.assertEqual(lu.fps_for("general", bad), 24.0, bad)


class CopyTests(unittest.TestCase):
    def test_module_imports_only_the_standard_library(self):
        with open(os.path.join(PACK, "luna_units.py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertLessEqual(names, {"__future__", "math"})

    def test_copies_are_identical_in_every_pack(self):
        checked = 0
        for pack in PACKS:
            other = os.path.join(PACKS_DIR, pack)
            if not os.path.isdir(other) or os.path.normcase(os.path.abspath(other)) == os.path.normcase(PACK):
                continue
            for rel in ("luna_units.py", os.path.join("tests", "test_luna_units.py")):
                path = os.path.join(other, rel)
                self.assertTrue(os.path.isfile(path), f"{pack} has no {rel}")
                self.assertTrue(filecmp.cmp(os.path.join(PACK, rel), path, shallow=False), f"{pack}/{rel} differs")
            checked += 1
        if not checked:
            self.skipTest("no sibling pack beside this one")


if __name__ == "__main__":
    unittest.main()

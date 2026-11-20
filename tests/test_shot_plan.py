"""Luna-H3-ShotList's script rules (shot_plan): parsing, chaining starts with an overlap,
grid snapping, JSON, and the errors that name the wrong line.

    E:\\ComfyUI-Easy-Install\\python_embeded\\python.exe -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shot_plan import parse_time, plan_scenes, scene_line, snap  # noqa: E402

SCRIPT = """# a test song
overlap: 22
0:12.00 | 5.17 | the singer walks into the light
        | 124f | close-up on the guitar  # the red one
0:30.5  | 3s   | wide shot, the crowd
"""


class SnapTests(unittest.TestCase):
    """Build 2: a scene's count rounds UP onto the grid (luna_units.frames_up), as Luna-H3-Canvas does."""

    def test_h3(self):
        self.assertEqual(snap(120, "h3"), 124)
        self.assertEqual(snap(110, "h3"), 124)          # was 107 (nearest)
        self.assertEqual(snap(107, "h3"), 107)
        self.assertEqual(snap(1, "h3"), 5)

    def test_ltx_and_free(self):
        self.assertEqual(snap(100, "ltx"), 105)         # was 97 (nearest)
        self.assertEqual(snap(97, "ltx"), 97)
        self.assertEqual(snap(77, "free"), 77)


class ParseTests(unittest.TestCase):
    def test_times(self):
        self.assertEqual(parse_time("1:02.5"), 62.5)
        self.assertEqual(parse_time("0:01:02"), 62.0)
        self.assertEqual(parse_time("12s"), 12.0)
        with self.assertRaises(ValueError):
            parse_time("1::2")

    def test_script(self):
        p = plan_scenes(SCRIPT)
        s1, s2, s3 = p["scenes"]
        self.assertEqual((s1["start"], s1["frames"]), (12.0, 124))
        # scene 2 starts 22 frames before scene 1 ends: 12 + 124/24 - 22/24 = 16.25
        self.assertAlmostEqual(s2["start"], 16.25)
        self.assertTrue(s2["auto_start"])
        self.assertEqual(s2["overlap"], 22)
        self.assertEqual(s2["prompt"], "close-up on the guitar  # the red one")
        self.assertEqual((s3["start"], s3["frames"], s3["overlap"]), (30.5, 73, 0))   # 3 s = 72 -> 73
        self.assertEqual(s1["line"], 2)

    def test_overlap_argument_and_first_start(self):
        p = plan_scenes("start: 1:00\n| 5 | a\n| 5 | b", overlap_frames=0)
        self.assertEqual(p["scenes"][0]["start"], 60.0)
        self.assertAlmostEqual(p["scenes"][1]["start"], 60 + 124 / 24)

    def test_json(self):
        p = plan_scenes('{"overlap": 39, "scenes": [{"start": 2, "frames": 124, "prompt": "x"}, {"seconds": 1}]}')
        self.assertEqual(p["scenes"][0]["frames"], 124)
        self.assertAlmostEqual(p["scenes"][1]["start"], 2 + 124 / 24 - 39 / 24)
        self.assertIsNone(p["scenes"][0]["line"])

    def test_json_overlap_off_the_list_snaps(self):
        # 5 frames is nearest 0 on the shared list: back to back, with a warning.
        p = plan_scenes('{"overlap": 5, "scenes": [{"start": 2, "frames": 39, "prompt": "x"}, {"seconds": 1}]}')
        self.assertEqual(p["overlap"], 0)
        self.assertAlmostEqual(p["scenes"][1]["start"], 2 + 39 / 24)
        self.assertTrue(any("Overlap 5" in w for w in p["warnings"]))

    def test_ltx_runs_at_25(self):
        p = plan_scenes("0 | 4 | a", grid="ltx")
        self.assertEqual((p["fps"], p["scenes"][0]["frames"]), (25, 105))   # 100 frames -> up onto 8k+1

    def test_errors_name_the_line(self):
        with self.assertRaisesRegex(ValueError, "line 2"):
            plan_scenes("0 | 5 | a\nnonsense")
        with self.assertRaisesRegex(ValueError, "line 1: length"):
            plan_scenes("0 | five | a")
        with self.assertRaisesRegex(ValueError, "empty"):
            plan_scenes("  ")
        with self.assertRaisesRegex(ValueError, "would not move on"):
            plan_scenes("0 | 5f | a\n | 5f | b", overlap_frames=22)

    def test_line_writer(self):
        self.assertEqual(scene_line("0:12.00", "124f", "a"), "0:12.00  | 124f   | a")


class SharedRuleTests(unittest.TestCase):
    """Build 2 acceptance checks: frame counts up onto the grid, seconds half up, overlap on the shared list
    (SPEC 1 and 2, 2026-11-20)."""

    def frames(self, length, grid="h3"):
        return plan_scenes(f"0 | {length} | a", grid=grid)["scenes"][0]["frames"]

    def test_110f_is_124(self):
        self.assertEqual(self.frames("110f"), 124)

    def test_5s_is_124(self):
        self.assertEqual(self.frames("5s"), 124)

    def test_0_9375s_is_39_as_canvas(self):
        self.assertEqual(self.frames("0.9375s"), 39)       # 22.5 frames -> 23 (half up) -> 39

    def test_past_362_warns_and_keeps(self):
        p = plan_scenes("0 | 16 | a")
        s = p["scenes"][0]
        self.assertEqual(s["frames"], 396)
        self.assertTrue(any("past the 362 frames" in w for w in s["warnings"]))

    def test_overlap_30_snaps_to_22_with_a_warning(self):
        p = plan_scenes("0 | 124f | a\n | 124f | b", overlap_frames=30)
        self.assertEqual((p["overlap"], p["scenes"][1]["overlap"]), (22, 22))
        self.assertEqual(p["scenes"][0]["overlap"], 0)
        self.assertEqual(len(p["warnings"]), 1)
        self.assertIn("30", p["warnings"][0])
        self.assertIn("22 is used", p["warnings"][0])
        p["warnings"][0].encode("ascii")

    def test_overlap_192_is_silent(self):
        p = plan_scenes("0 | 243f | a\n | 243f | b", overlap_frames=192)
        self.assertEqual((p["overlap"], p["scenes"][1]["overlap"], p["warnings"]), (192, 192, []))
        self.assertEqual(p["scenes"][1]["warnings"], [])

    def test_script_overlap_line_snaps(self):
        p = plan_scenes("overlap: 50\n0 | 124f | a\n | 124f | b", overlap_frames=22)
        self.assertEqual(p["scenes"][1]["overlap"], 39)
        self.assertIn("Overlap 50", p["warnings"][0])

    def test_pinned_start_snaps_its_overlap_and_moves(self):
        p = plan_scenes("0 | 124f | first\n4 | 124f | second", grid="h3")
        s2 = p["scenes"][1]
        self.assertEqual(s2["overlap"], 22)                # 28 frames written -> 22 on the list
        self.assertAlmostEqual(s2["start"], 4.25)          # 124/24 - 22/24
        self.assertEqual(len(s2["warnings"]), 1)
        self.assertIn("4.000", s2["warnings"][0])
        self.assertIn("4.250", s2["warnings"][0])
        s2["warnings"][0].encode("ascii")

    def test_pinned_start_on_the_list_is_kept(self):
        # 124/24 - 39/24 = 3.5417 s: 39 frames shared, already on the list.
        p = plan_scenes(f"0 | 124f | a\n{(124 - 39) / 24} | 124f | b")
        self.assertEqual((p["scenes"][1]["overlap"], p["scenes"][1]["warnings"]), (39, []))
        self.assertAlmostEqual(p["scenes"][1]["start"], 85 / 24)

    def test_pinned_start_after_the_end_is_a_gap(self):
        p = plan_scenes("0 | 124f | a\n9 | 124f | b")
        self.assertEqual((p["scenes"][1]["overlap"], p["scenes"][1]["start"], p["scenes"][1]["warnings"]),
                         (0, 9.0, []))

    def test_pinned_start_whose_snapped_overlap_eats_the_scene_raises(self):
        # 39 frames, then a start 1 frame after it: 38 shared -> 39 = the whole scene before.
        with self.assertRaisesRegex(ValueError, "as long as that scene"):
            plan_scenes(f"0 | 39f | a\n{1 / 24} | 124f | b")


class ShotListNodeTests(unittest.TestCase):
    """The node's sockets (order pinned) and its info line with the warnings."""

    @classmethod
    def setUpClass(cls):
        # ComfyUI's `server` may be importable without a running PromptServer: hide it while shot_list imports,
        # so the module takes its no-routes branch.
        import importlib.util
        from types import ModuleType
        saved = sys.modules.get("server")
        sys.modules["server"] = ModuleType("server")
        try:
            spec = importlib.util.spec_from_file_location(
                "shot_list_under_test", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                     "shot_list.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if saved is None:
                sys.modules.pop("server", None)
            else:
                sys.modules["server"] = saved
        cls.node = module.LunaShotList

    def test_schema_order_is_pinned(self):
        inputs = self.node.INPUT_TYPES()
        self.assertEqual(list(inputs), ["required"])
        self.assertEqual(list(inputs["required"]), ["script", "scene", "grid", "overlap"])
        opts = inputs["required"]["overlap"][1]
        self.assertEqual((opts["default"], opts["min"], opts["max"]), (22, 0, 361))   # saved values still load
        self.assertEqual(self.node.RETURN_NAMES,
                         ("scene", "scenes", "frames", "seconds", "song_start", "prompt", "overlap", "info"))
        self.assertEqual(self.node.RETURN_TYPES, ("INT", "INT", "INT", "FLOAT", "FLOAT", "STRING", "INT", "STRING"))

    def test_info_carries_the_overlap_warning_and_scene_1_overlap_is_0(self):
        script = "0 | 124f | a\n | 124f | b"
        r1 = self.node().pick(script, 1, "H3 (24 fps, 17k+5)", 30)["result"]
        r2 = self.node().pick(script, 2, "H3 (24 fps, 17k+5)", 30)["result"]
        self.assertEqual((r1[6], r2[6]), (0, 22))
        self.assertIn("WARNING", r2[7])
        self.assertIn("22 is used", r2[7])
        r2[7].encode("ascii")

    def test_info_names_the_moved_start(self):
        r = self.node().pick("0 | 124f | first\n4 | 124f | second", 2, "H3 (24 fps, 17k+5)", 22)["result"]
        self.assertEqual(r[6], 22)
        self.assertAlmostEqual(r[4], 4.25)
        self.assertIn("4.000", r[7])
        self.assertIn("4.250", r[7])

    def test_192_is_silent(self):
        r = self.node().pick("0 | 243f | a\n | 243f | b", 2, "H3 (24 fps, 17k+5)", 192)["result"]
        self.assertEqual(r[6], 192)
        self.assertNotIn("WARNING", r[7])


if __name__ == "__main__":
    unittest.main()

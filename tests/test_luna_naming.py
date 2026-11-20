"""luna_naming.py: the shared tag table, the @ map, the @ replace and its warnings, and the copy check.

Identical in Luna-LLM-Tools, Luna-Media-Tools and Luna-MinimaxH3 (`tests/test_luna_naming.py`), like `luna_naming.py`
itself. The copy check reads the sibling packs from env LUNA_PACKS_DIR (default D:\\Claude) and skips a pack that is not
there.
"""

import ast
import filecmp
import os
import sys
import unittest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)
import luna_naming as ln  # noqa: E402

PACKS_DIR = os.environ.get("LUNA_PACKS_DIR", r"D:\Claude")
PACKS = ("Luna-LLM-Tools", "Luna-Media-Tools", "Luna-MinimaxH3")


class RowTests(unittest.TestCase):
    def test_each_row_renders_its_own_tags(self):
        self.assertEqual(ln.tag("minimax", "image", 1), "<Picture 1>")
        self.assertEqual(ln.tag("minimax", "video", 2), "<Video 2>")
        self.assertEqual(ln.tag("minimax", "audio", 3), "<Audio 3>")
        self.assertEqual(ln.tag("qwen", "image", 2), "<image2>")
        self.assertEqual(ln.tag("cloud", "image", 1), "Image 1")
        self.assertEqual(ln.tag("grok", "image", 1), "<IMAGE_0>")   # Grok counts from 0

    def test_rows_without_clips_or_sounds_give_none(self):
        for row in ("qwen", "cloud", "grok"):
            self.assertIsNone(ln.tag(row, "video", 1))
            self.assertIsNone(ln.tag(row, "audio", 1))
        self.assertIsNone(ln.tag("nope", "image", 1))
        self.assertIsNone(ln.tag("minimax", "image", 0))

    def test_targets_and_labels_map_to_rows(self):
        self.assertEqual(ln.row_for("h3"), "minimax")
        self.assertEqual(ln.row_for("MiniMax Image"), "minimax")
        self.assertEqual(ln.row_for(" Qwen "), "qwen")
        self.assertEqual(ln.row_for("General"), "cloud")
        self.assertEqual(ln.row_for("grok"), "grok")
        self.assertIsNone(ln.row_for("ltx"))
        self.assertIsNone(ln.row_for(None))


class AssignTests(unittest.TestCase):
    def test_pictures_not_sent_are_skipped(self):
        # cards 1 and 3 are keyframes on the shot; 2 and 4 are references -> positions 1 and 2
        a = ln.assign("minimax", ["key", "ref", "key", "ref"])
        self.assertEqual(a["image"], {1: 1, 3: 2})
        self.assertEqual(a["order"], [1, 3])

    def test_edit_base_goes_first_where_the_row_edits_the_first_picture(self):
        roles = ["ref", "style", "edit"]
        for row in ("minimax", "qwen", "grok"):
            a = ln.assign(row, roles)
            self.assertEqual(a["order"], [2, 0, 1], row)
            self.assertEqual(a["image"], {2: 1, 0: 2, 1: 3}, row)
        self.assertEqual(ln.assign("cloud", roles)["order"], [0, 1, 2])

    def test_sound_order_is_minimax_order(self):
        # the same case as Luna-MinimaxH3 tests/test_asset_timeline.py (h3_tags)
        clips = [{"sent": True, "sound": True}, {"sent": True, "sound": False}, {"sent": False, "sound": True}]
        a = ln.assign("minimax", [], clips, [False, True, True])
        self.assertEqual(a["video"], {0: 1, 1: 2})
        self.assertEqual(a["video_audio"], {0: 1})           # clip 2 has none; clip 3 sits on the shot
        self.assertEqual(a["audio"], {1: 2, 2: 3})           # the locked song (card 1) is no reference

    def test_empty_slots_are_skipped(self):
        a = ln.assign("minimax", ["ref", None, "ref"], [], [])
        self.assertEqual(a["image"], {0: 1, 2: 2})


class MapTests(unittest.TestCase):
    def setUp(self):
        self.naming = ln.make("minimax", ["ref", "edit", "key"],
                              [{"sent": True, "sound": True}], [False, True])

    def test_map_covers_every_card(self):
        m = self.naming["map"]
        self.assertEqual(m["@image2"], ("image", 1))         # the edit base first
        self.assertEqual(m["@image1"], ("image", 2))
        self.assertIsNone(m["@image3"])                      # a keyframe: on the shot, not sent
        self.assertEqual(m["@video1"], ("video", 1))
        self.assertIsNone(m["@audio1"])                      # the song lock
        self.assertEqual(m["@audio2"], ("audio", 2))         # after the clip's own sound
        self.assertEqual(self.naming["counts"], {"image": 3, "video": 1, "audio": 2})

    def test_names_line(self):
        line = ln.names_line(self.naming)
        self.assertTrue(line.startswith("@ names: "))
        self.assertIn("@image1 = <Picture 2>", line)
        self.assertIn("@image2 = <Picture 1>", line)
        self.assertIn("@image3 = not sent", line)
        self.assertIn("@audio2 = <Audio 2>", line)
        self.assertIn("@video1 = not taken by this model", ln.names_line(self.naming, row="qwen"))
        self.assertEqual(ln.names_line(None), "")

    def test_remake_switches_categories_off(self):
        off = ln.remake(self.naming, pictures=False)
        self.assertIsNone(off["map"]["@image1"])
        self.assertIsNone(off["map"]["@image2"])
        self.assertEqual(off["map"]["@video1"], ("video", 1))
        # the clip's sound off: the standalone sound moves up to <Audio 1>
        self.assertEqual(ln.remake(self.naming, clip_sound=False)["map"]["@audio2"], ("audio", 1))
        self.assertIsNone(ln.remake(self.naming, clips=False)["map"]["@video1"])
        self.assertIsNone(ln.remake(self.naming, sounds=False)["map"]["@audio2"])
        self.assertEqual(ln.remake(self.naming), self.naming)   # nothing off: the same naming

    def test_from_refs(self):
        self.assertIs(ln.from_refs({"naming": self.naming}), self.naming)
        self.assertIsNone(ln.from_refs({"images": []}))
        self.assertIsNone(ln.from_refs(None))
        self.assertIsNone(ln.from_refs({"naming": {"row": "nope", "map": {}}}))


class ReplaceTests(unittest.TestCase):
    def setUp(self):
        self.naming = ln.make("minimax", ["ref", "edit"], [], [])

    def test_user_sentence_is_rewritten_with_the_tags(self):
        text = "Change the clothes of the woman from @image1 with the ones from the woman in @image2."
        out, warnings = ln.replace(text, self.naming)
        self.assertEqual(out, "Change the clothes of the woman from <Picture 2> with the ones from the woman in <Picture 1>.")
        self.assertEqual(warnings, [])

    def test_the_consumer_row_renders_the_same_positions(self):
        self.assertEqual(ln.replace("@image2 and @image1", self.naming, row="qwen")[0], "<image1> and <image2>")
        self.assertEqual(ln.replace("@image2 and @image1", self.naming, row="grok")[0], "<IMAGE_0> and <IMAGE_1>")

    def test_no_at_name_changes_nothing(self):
        text = "A woman in a red coat, <Picture 1> as the face."
        self.assertEqual(ln.replace(text, self.naming), (text, []))
        self.assertEqual(ln.replace("", None), ("", []))

    def test_missing_item_warns_and_becomes_words(self):
        out, warnings = ln.replace("Put @image5 behind her.", self.naming)
        self.assertEqual(out, "Put the picture behind her.")
        self.assertEqual(warnings, ["@image5 names no picture: 2 picture(s) here."])

    def test_item_not_sent_warns(self):
        naming = ln.make("minimax", ["key", "ref"])
        out, warnings = ln.replace("Start on @image1.", naming)
        self.assertEqual(out, "Start on the picture.")
        self.assertIn("not sent", warnings[0])

    def test_kind_the_model_does_not_take_warns(self):
        out, warnings = ln.replace("Move like @video1.", ln.make("qwen", ["ref"], [{"sent": True}]))
        self.assertEqual(out, "Move like the clip.")
        self.assertEqual(warnings, ["@video1: this model takes no clips."])

    def test_case_possessive_and_email_like_text(self):
        out, _ = ln.replace("@Image2's coat; mail a@image1.com", self.naming)
        self.assertEqual(out, "<Picture 1>'s coat; mail a@image1.com")

    def test_a_longer_token_is_no_name_and_still_never_reaches_the_model(self):
        out, warnings = ln.replace("@image1st, @image2_extra and @image.", self.naming)
        self.assertEqual(out, "the picture, the picture and the picture.")
        self.assertEqual(len(warnings), 3)
        self.assertIn("is not an @ name", warnings[0])

    def test_adjacent_names_are_two_names(self):
        out, warnings = ln.replace("@image1@image2@image3", ln.make("minimax", ["ref"]))
        self.assertEqual(out, "<Picture 1>the picturethe picture")
        self.assertEqual(len(warnings), 2)

    def test_unicode_spellings_never_throw(self):
        # "İ" (U+0130) and "ı" (U+0131) are no ASCII kind: no name, text kept, no error
        text = "@İmage1 and @ımage1"
        self.assertEqual(ln.replace(text, self.naming), (text, []))
        # a non-ASCII digit after the kind is no name: words and a warning, no error
        out, warnings = ln.replace("@image² and @image١", self.naming)
        self.assertEqual((out, len(warnings)), ("the picture and the picture", 2))
        self.assertIsNone(ln.read_names_line("@ names: @İmage1 = <Picture 1>."))

    def test_one_warning_per_name(self):
        _, warnings = ln.replace("@image9 and @image9", self.naming)
        self.assertEqual(len(warnings), 1)

    def test_without_naming_the_number_is_the_place_sent(self):
        self.assertEqual(ln.replace("@image1 into @image2", None, row="grok", counts={"image": 2}),
                         ("<IMAGE_0> into <IMAGE_1>", []))
        self.assertEqual(ln.replace("@image1", None, counts={"image": 1})[0], "Image 1")   # default row: cloud
        out, warnings = ln.replace("@image3", None, row="cloud", counts={"image": 2})
        self.assertEqual(out, "the picture")
        self.assertEqual(warnings, ["@image3 names no picture: 2 picture(s) here."])
        out, warnings = ln.replace("@image1", None, row="cloud", counts={"image": 0})
        self.assertEqual((out, len(warnings)), ("the picture", 1))       # no picture attached at all

    def test_without_naming_or_counts_nothing_is_invented(self):
        out, warnings = ln.replace("@image999 and @image1", None)
        self.assertEqual(out, "the picture and the picture")
        self.assertEqual(len(warnings), 2)
        # a kind the counts leave out counts 0
        out, warnings = ln.replace("@audio1", None, row="minimax", counts={"image": 1})
        self.assertEqual((out, len(warnings)), ("the sound", 1))

    def test_mismatch(self):
        self.assertEqual(ln.mismatch("minimax", "minimax"), "")
        self.assertEqual(ln.mismatch(None, "minimax"), "")
        self.assertIn("<image1>", ln.mismatch("qwen", "minimax"))
        self.assertIn("<Picture 1>", ln.mismatch("qwen", "minimax"))
        # every message is ASCII: a node prints it, and a console that is not UTF-8 must not stop the run
        self.assertTrue(ln.mismatch("qwen", "minimax").isascii())
        self.assertTrue(all(w.isascii() for w in ln.replace("@image9 @video1 @audio1 @image1st", None)[1]))


class NamesLineTests(unittest.TestCase):
    def setUp(self):
        self.naming = ln.make("minimax", ["ref", "edit", "key"], [{"sent": True, "sound": True}], [False, True])

    def test_round_trip_in_every_row(self):
        for row in ln.ROWS:
            read = ln.read_names_line(ln.names_line(self.naming, row))
            self.assertEqual(read["counts"], self.naming["counts"], row)
            for key, entry in self.naming["map"].items():
                taken = entry is not None and ln.tag(row, entry[0], entry[1]) is not None
                self.assertEqual(read["map"][key], entry if taken else None, (row, key))
            self.assertEqual(read["row"], row)

    def test_read_inside_a_brief_and_after_the_refs_out_switches(self):
        line = ln.names_line(ln.remake(self.naming, clip_sound=False))
        brief = f"Reference images, in order: <Picture 1> = 1:1; <Picture 2> = 16:9. {line} On the shot (124 frames)."
        read = ln.read_names_line(brief)
        self.assertEqual(read["map"]["@audio2"], ("audio", 1))
        self.assertEqual(read["map"]["@image2"], ("image", 1))
        self.assertIsNone(ln.read_names_line("Reference images, in order: <Picture 1> = 1:1."))
        self.assertIsNone(ln.read_names_line(None))

    def test_pick_prefers_the_line_the_model_is_shown(self):
        refs = {"naming": self.naming}
        shown = ln.names_line(ln.remake(self.naming, pictures=False))
        self.assertIsNone(ln.pick(shown, refs)["map"]["@image2"])
        self.assertIs(ln.pick("no line here", refs), self.naming)
        self.assertIsNone(ln.pick("no line here", None))

    def test_a_line_with_nothing_sent_shows_no_row(self):
        naming = ln.make("minimax", ["ref"])
        line = ln.names_line(ln.remake(naming, pictures=False))
        self.assertEqual(line, "@ names: @image1 = not sent.")
        self.assertIsNone(ln.read_names_line(line)["row"])
        picked = ln.pick(line, {"naming": naming})
        self.assertEqual(picked["row"], "minimax")               # from refs: no false mismatch
        self.assertIsNone(picked["map"]["@image1"])              # the line's map still wins
        self.assertEqual(ln.mismatch("minimax", picked["row"]), "")
        self.assertIsNone(ln.pick(line, None)["row"])

    def test_put_names_line_renders_in_the_model_row(self):
        brief = "Reference images: <Picture 1>. " + ln.names_line(self.naming) + " End."
        out = ln.put_names_line(brief, ln.read_names_line(brief), "qwen")
        self.assertIn("@image1 = <image2>; @image2 = <image1>", out)
        self.assertTrue(out.startswith("Reference images: <Picture 1>. @ names: "))
        self.assertTrue(out.endswith(" End."))
        self.assertEqual(ln.put_names_line("Image 1 (edit base, 64x64)", self.naming, "cloud").splitlines()[1],
                         ln.names_line(self.naming, "cloud"))
        self.assertEqual(ln.put_names_line("same", None), "same")


class CopyTests(unittest.TestCase):
    def test_module_imports_only_the_standard_library(self):
        with open(os.path.join(PACK, "luna_naming.py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertLessEqual(names, {"__future__", "re"})

    def test_copies_are_identical_in_every_pack(self):
        checked = 0
        for pack in PACKS:
            other = os.path.join(PACKS_DIR, pack)
            if not os.path.isdir(other) or os.path.normcase(os.path.abspath(other)) == os.path.normcase(PACK):
                continue
            for rel in ("luna_naming.py", os.path.join("tests", "test_luna_naming.py")):
                path = os.path.join(other, rel)
                self.assertTrue(os.path.isfile(path), f"{pack} has no {rel}")
                self.assertTrue(filecmp.cmp(os.path.join(PACK, rel), path, shallow=False), f"{pack}/{rel} differs")
            checked += 1
        if not checked:
            self.skipTest("no sibling pack beside this one")


if __name__ == "__main__":
    unittest.main()

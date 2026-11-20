"""Asset Loader / Refs Out / Reference Loader / Canvas wiring (plans/asset-loader-connect.md):
the loader's new outputs, no continuation, an empty loader that runs, the canvas size passed on,
refs["song"] and refs["items"], and Refs Out's switches, rebuilt brief and `audio` output; Build 2
(SPEC § 1, § 2): one frame count and shot rate per run, the overlap row, the vocal stem, the refs contract and
Refs Out's `refs` output.

    E:\\ComfyUI-Easy-Install\\python_embeded\\python.exe -m unittest discover -s tests -v
Loads the whole pack through ComfyUI's modules, like test_registration.py.
"""

import json
import os
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_registration  # noqa: E402

PACKNAME = "Luna_MinimaxH3_under_test"


def _module(name):
    if f"{PACKNAME}.{name}" not in sys.modules:
        test_registration.load_pack()
    return sys.modules[f"{PACKNAME}.{name}"]


AL = _module("asset_loader")
RO = _module("h3_refs_out")
RL = _module("h3_reference_loader")
Loader = AL.LunaAssetLoader
RefsOut = RO.LunaH3RefsOut

IMAGE_NAMES = tuple(f"image_{i}" for i in range(1, 10))
SONG = {"waveform": torch.zeros(1, 2, 100), "sample_rate": 48000}
# Luna-H3-RefsOut's output slots: 0-17 the H3 sockets, then asset_brief, audio, refs.
BRIEF, AUDIO, REFS = 18, 19, 20


def fake_load(path, st=None):
    w, h = {"a.png": (640, 480), "b.png": (480, 640)}.get(path, (320, 320))
    return torch.zeros(1, h, w, 3), (w, h), None


def clip_stub(sound):
    return [{"batch": torch.zeros(5, 8, 8, 3), "sound": sound, "info": {"name": "c.mp4", "fps": 24.0},
             "line": "c.mp4 from 0:00.00, 5 frames (0.21 s) at 24 fps, 8x8"}]


def probe_stub(clips):
    return [{"video": True, "fps": 24.0, "width": 8, "height": 8, "duration": 10.0, "audio": True} for _ in clips]


class LoaderOutputTests(unittest.TestCase):
    def test_return_names_and_types(self):
        self.assertEqual(Loader.RETURN_NAMES[:10], ("images", "width", "height", "asset_brief", "audio", "refs",
                                                    "timeline", "video", "frames", "overlap"))
        self.assertEqual(Loader.RETURN_NAMES[10:19], IMAGE_NAMES)   # all nine, always
        self.assertEqual(Loader.RETURN_NAMES[19:], ("fps", "vocals"))   # appended after image_9
        self.assertEqual(Loader.RETURN_TYPES[7:10], ("IMAGE", "INT", "INT"))
        self.assertEqual(len(Loader.RETURN_TYPES), len(Loader.RETURN_NAMES))
        self.assertEqual(len(Loader.OUTPUT_TOOLTIPS), len(Loader.RETURN_NAMES))
        self.assertIn("Luna-LLMPrompt", Loader.OUTPUT_TOOLTIPS[7])
        self.assertIn("Luna-LLMPrompt", Loader.OUTPUT_TOOLTIPS[8])

    def test_no_continuation_left(self):
        self.assertFalse(hasattr(Loader, "_continuation"))
        self.assertFalse(hasattr(AL, "_output_newest"))
        self.assertNotIn("previous_frames", Loader.INPUT_TYPES()["optional"])

    def test_old_media_state_with_a_continuation_key_is_read_and_ignored(self):
        media = AL._parse_media('{"continuation": {"on": true, "from": "folder"}, "shot": {"frames": 39}}')
        self.assertNotIn("continuation", media)
        self.assertEqual(media["shot_frames"], 39)
        self.assertIsNotNone(Loader.IS_CHANGED("", 2, media_state='{"continuation": {"on": true}}'))

    def test_empty_loader_runs(self):
        out = Loader().load("", 2)["result"]
        self.assertEqual(len(out), len(Loader.RETURN_NAMES))
        self.assertIsNone(out[0])                    # images
        self.assertEqual(out[1:3], (0, 0))
        self.assertEqual(out[3], "Nothing loaded.")
        self.assertIsNone(out[4])                    # audio
        self.assertEqual(out[5]["images"], [])
        self.assertIsNone(out[7])                    # video
        self.assertEqual(out[8], 124)                # frames: the card default
        self.assertEqual(out[9], 22)                 # overlap: the node's setting, 22 by default, song or none
        self.assertTrue(all(x is None for x in out[10:19]))       # image_1..9
        self.assertEqual(out[19], 24.0)                             # fps: the shot's rate, no clip
        self.assertIsNone(out[20])                                  # vocals

    def test_empty_loader_hands_on_the_canvas_and_the_seconds(self):
        out = Loader().load("", 2, seconds=5.0, width=832, height=480)["result"]
        self.assertEqual((out[1], out[2], out[8]), (832, 480, 124))

    def test_frames_is_on_the_h3_grid(self):
        # One count: seconds x 24 in whole frames, rounded UP onto 17k+5 (luna_units.frames_for), because it feeds
        # H3's length. The canvas's seconds (frames / 24) is already on it.
        with patch.object(AL, "_load", fake_load):
            self.assertEqual(Loader().load("a.png", 1)["result"][8], 124)
            self.assertEqual(Loader().load("a.png", 1, seconds=243 / 24)["result"][8], 243)
            self.assertEqual(Loader().load("a.png", 1, seconds=5.0)["result"][8], 124)

    def test_width_height_picture_1_unless_the_canvas_is_wired(self):
        with patch.object(AL, "_load", fake_load):
            r = Loader().load("a.png\nb.png", 2)["result"]
            self.assertEqual(r[1:3], (640, 480))
            r = Loader().load("a.png\nb.png", 2, width=1344, height=768)["result"]
            self.assertEqual(r[1:3], (1344, 768))
            r = Loader().load("a.png\nb.png", 2, width=1344)["result"]       # only one wired: not the canvas
            self.assertEqual(r[1:3], (640, 480))
            r = Loader().load("a.png\nb.png", 2, width=0, height=0)["result"]
            self.assertEqual(r[1:3], (640, 480))

    def test_video_is_clip_1_whatever_its_role(self):
        media = '{"video": [{"file": "c.mp4", "role": "guide"}]}'
        with patch.object(AL, "_load", fake_load), patch.object(Loader, "_clips", staticmethod(
                lambda *a, **k: clip_stub(None))), patch.object(Loader, "_probe", staticmethod(probe_stub)):
            r = Loader().load("a.png", 1, media_state=media)["result"]
        self.assertEqual(tuple(r[7].shape), (5, 8, 8, 3))

    def test_no_clip_means_no_video(self):
        with patch.object(AL, "_load", fake_load):
            self.assertIsNone(Loader().load("a.png", 1)["result"][7])

    def test_refs_carry_song_and_items(self):
        media = '{"audio": [{"file": "s.mp3"}], "video": [{"file": "c.mp4", "role": "ref"}]}'
        song_line = "Song: s.mp3 0:00.00-0:00.02 (0.02 s)."
        with patch.object(AL, "_load", fake_load), \
                patch.object(Loader, "_clips", staticmethod(lambda *a, **k: clip_stub(SONG))), \
                patch.object(Loader, "_probe", staticmethod(probe_stub)), \
                patch.object(Loader, "_song", staticmethod(lambda *a, **k: (SONG, {"name": "s.mp3", "start": 0.0, "seconds": 0.02}, song_line))), \
                patch.object(Loader, "_sounds", staticmethod(lambda *a, **k: {})):
            r = Loader().load("a.png\nb.png", 2, media_state=media)["result"]
        refs = r[5]
        self.assertIs(refs["song"], SONG)
        items = refs["items"]
        self.assertEqual(len(items["images"]), 2)
        self.assertNotIn("<Picture", items["images"][0])             # no tags
        self.assertEqual(len(items["videos"]), 1)
        self.assertNotIn("<Video", items["videos"][0])
        self.assertEqual(items["video_audios"], [True])
        self.assertEqual(items["audios"], [])
        self.assertEqual(items["song"], song_line)
        self.assertIn("different aspect ratios", items["picture_warning"])
        self.assertFalse(any("different aspect ratios" in x for x in items["extra"]))
        self.assertNotIn("different aspect ratios", RO.build_brief(refs, pictures=False))   # goes with the pictures
        self.assertIn("different aspect ratios", RO.build_brief(refs))
        self.assertIn("<Picture 1>", refs["brief"])                    # brief as before
        self.assertIn("<Video 1>", refs["brief"])

    def test_pictures_only_has_no_song(self):
        with patch.object(AL, "_load", fake_load):
            refs = Loader().load("a.png", 1)["result"][5]
        self.assertIsNone(refs["song"])
        self.assertEqual(refs["items"]["song"], "")
        self.assertEqual(refs["items"]["video_audios"], [])


def refs_with(pictures=2, clip_sounds=(True,), sounds=2, song=True):
    return {
        "images": [f"img{i}" for i in range(pictures)],
        "videos": [f"vid{k}" for k in range(len(clip_sounds))],
        "video_audios": [f"snd{k}" if on else None for k, on in enumerate(clip_sounds)],
        "audios": [f"aud{j}" for j in range(sounds)],
        "brief": "THE LOADER BRIEF",
        "song": SONG if song else None,
        "items": {"images": [f"pic desc {i}" for i in range(pictures)],
                  "videos": [f"vid desc {k}" for k in range(len(clip_sounds))],
                  "video_audios": list(clip_sounds),
                  "audios": [f"aud desc {j}" for j in range(sounds)],
                  "song": "Song: x." if song else "",
                  "extra": ["On the shot (124 frames, 5.17 s): frame 0: card 1."]},
    }


class RefsOutTests(unittest.TestCase):
    def test_outputs_and_inputs(self):
        self.assertEqual(RefsOut.RETURN_NAMES[BRIEF:], ("asset_brief", "audio", "refs"))   # refs appended last
        self.assertEqual(RefsOut.RETURN_TYPES[BRIEF:], ("STRING", "AUDIO", "LUNA_H3_REFS"))
        self.assertEqual(len(RefsOut.RETURN_NAMES), 9 + 3 + 3 + 3 + 3)
        self.assertEqual(len(RefsOut.OUTPUT_TOOLTIPS), len(RefsOut.RETURN_NAMES))
        req = RefsOut.INPUT_TYPES()["required"]
        self.assertEqual(list(req), ["refs", "pictures", "clips", "clip_sound", "sounds", "song"])
        for k in ("pictures", "clips", "clip_sound", "sounds", "song"):
            self.assertEqual(req[k][0], "BOOLEAN")
            self.assertIs(req[k][1]["default"], True)
            self.assertTrue(req[k][1]["tooltip"])

    def test_all_on_delivers_everything(self):
        out = RefsOut().unpack(refs_with())
        self.assertEqual(out[0], "img0")
        self.assertIsNone(out[2])
        self.assertEqual(out[9], "vid0")
        self.assertEqual(out[12], "snd0")
        self.assertEqual(out[15], "aud0")
        self.assertIs(out[AUDIO], SONG)

    def test_each_switch_silences_its_sockets(self):
        n = RefsOut()
        out = n.unpack(refs_with(), pictures=False)
        self.assertTrue(all(x is None for x in out[0:9]))
        self.assertEqual(out[9], "vid0")
        out = n.unpack(refs_with(), clips=False)
        self.assertTrue(all(x is None for x in out[9:15]))     # clip sound goes with its clip
        self.assertEqual(out[15], "aud0")
        out = n.unpack(refs_with(), clip_sound=False)
        self.assertEqual(out[9], "vid0")
        self.assertTrue(all(x is None for x in out[12:15]))
        out = n.unpack(refs_with(), sounds=False)
        self.assertTrue(all(x is None for x in out[15:18]))
        out = n.unpack(refs_with(), song=False)
        self.assertIsNone(out[AUDIO])

    def test_brief_numbering_all_on(self):
        brief = RefsOut().unpack(refs_with())[BRIEF]
        order = [brief.index(t) for t in ("<Picture 1>", "<Picture 2>", "<Audio 1> = the soundtrack of <Video 1>",
                                           "<Video 1> =", "<Audio 2> = aud desc 0", "<Audio 3> = aud desc 1",
                                           "Song: x.", "On the shot")]
        self.assertEqual(order, sorted(order))
        self.assertNotIn("THE LOADER BRIEF", brief)

    def test_clip_sound_off_recounts_the_standalone_sounds(self):
        brief = RefsOut().unpack(refs_with(), clip_sound=False)[BRIEF]
        self.assertNotIn("soundtrack", brief)
        self.assertIn("<Audio 1> = aud desc 0", brief)
        self.assertIn("<Audio 2> = aud desc 1", brief)

    def test_two_clips_one_silent(self):
        brief = RefsOut().unpack(refs_with(clip_sounds=(False, True), sounds=1))[BRIEF]
        self.assertNotIn("soundtrack of <Video 1>", brief)
        self.assertLess(brief.index("<Audio 1> = the soundtrack of <Video 2>"), brief.index("<Video 2> ="))
        self.assertIn("<Audio 2> = aud desc 0", brief)

    def test_switched_off_categories_leave_the_brief(self):
        n = RefsOut()
        self.assertNotIn("<Picture", n.unpack(refs_with(), pictures=False)[BRIEF])
        b = n.unpack(refs_with(), clips=False)[BRIEF]
        self.assertNotIn("<Video", b)
        self.assertNotIn("soundtrack", b)
        self.assertIn("<Audio 1> = aud desc 0", b)
        self.assertNotIn("Song:", n.unpack(refs_with(), song=False)[BRIEF])
        self.assertNotIn("<Audio", n.unpack(refs_with(), sounds=False, clip_sound=False)[BRIEF])

    def test_sounds_without_picture_or_clip_warn(self):
        warn = "MiniMax H3 does not accept reference sounds without a reference picture or clip"
        b = RefsOut().unpack(refs_with(), pictures=False, clips=False)[BRIEF]
        self.assertIn(warn, b)
        self.assertNotIn(warn, RefsOut().unpack(refs_with())[BRIEF])
        self.assertNotIn(warn, RefsOut().unpack(refs_with(), pictures=False, clips=False, sounds=False)[BRIEF])

    def test_old_refs_without_items(self):
        old = {"images": ["i"], "videos": ["v"], "video_audios": ["a"], "audios": ["x"], "brief": "OLD BRIEF"}
        out = RefsOut().unpack(old)
        self.assertEqual(out[BRIEF], "OLD BRIEF")
        self.assertIsNone(out[AUDIO])                       # no song in old refs
        b = RefsOut().unpack(old, clip_sound=False)[BRIEF]
        self.assertNotEqual(b, "OLD BRIEF")
        self.assertIn("<Audio 1>.", b)                   # the standalone sound counts from 1
        self.assertNotIn("soundtrack", b)

    def test_not_a_dict(self):
        out = RefsOut().unpack(None)
        self.assertEqual(len(out), len(RefsOut.RETURN_NAMES))


LN = _module("luna_naming")

# Acceptance case A: picture cards [key, ref, ref]; clip V1 a reference with its sound kept; sound card 1 the
# song, locked; sound card 2 a reference sound.
CASE_A = json.dumps({"pictures": [{"role": "key", "at": 0}, {}, {}],
                     "video": [{"file": "c.mp4", "role": "ref"}],
                     "audio": [{"file": "s.mp3", "role": "lock"}, {"file": "b.wav", "role": "ref"}]})
SONG_LINE = "Song: s.mp3 0:00.00-0:00.02 (0.02 s)."
SOUND_LINE = "b.wav 0:00.00-0:01.00 (1.00 s)"
NAMES_A = ("@ names: @image1 = not sent; @image2 = <Picture 1>; @image3 = <Picture 2>; @video1 = <Video 1>; "
           "@audio1 = not sent; @audio2 = <Audio 2>.")


def load_case(media=CASE_A, paths="a.png\nb.png\nc.png"):
    with patch.object(AL, "_load", fake_load), \
            patch.object(Loader, "_clips", staticmethod(lambda *a, **k: clip_stub(SONG))), \
            patch.object(Loader, "_probe", staticmethod(probe_stub)), \
            patch.object(Loader, "_song", staticmethod(lambda *a, **k: (SONG, {"name": "s.mp3", "start": 0.0, "seconds": 0.02}, SONG_LINE))), \
            patch.object(Loader, "_sounds", staticmethod(lambda cards, *a: {1: (SONG, SOUND_LINE)})):
        return Loader().load(paths, len(paths.split("\n")), media_state=media)["result"]


class NamingTests(unittest.TestCase):
    """The @ names (SPEC § 2, Naming): refs["naming"], the loader's names line, Refs Out's recounted one."""

    def test_a_map(self):
        refs = load_case()[5]
        self.assertEqual(refs["naming"]["row"], "minimax")
        self.assertEqual(refs["naming"]["map"], {"@image1": None, "@image2": ("image", 1), "@image3": ("image", 2),
                                                 "@video1": ("video", 1), "@audio1": None, "@audio2": ("audio", 2)})
        # The lists say exactly what refs carries: two pictures, one clip with its sound, one standalone sound.
        self.assertEqual((len(refs["images"]), len(refs["videos"]), len(refs["audios"])), (2, 1, 1))
        self.assertIsNotNone(refs["video_audios"][0])

    def test_a_map_agrees_with_the_brief_tags(self):
        out = load_case()
        refs, brief = out[5], out[3]
        for key, entry in refs["naming"]["map"].items():
            if entry is not None:
                self.assertIn(f"{LN.tag('minimax', *entry)} = ", brief, key)
        self.assertIn("<Picture 1> = 480x640", brief)          # card 2 (b.png) is <Picture 1>
        self.assertIn("<Audio 2> = " + SOUND_LINE, brief)

    def test_a_brief_has_the_names_line_after_the_references(self):
        brief = load_case()[3]
        self.assertIn(NAMES_A, brief)
        self.assertIn("@image2 = <Picture 1>", brief)
        self.assertIn("@audio2 = <Audio 2>", brief)
        at = brief.index(NAMES_A)
        self.assertLess(brief.index("<Audio 2> = " + SOUND_LINE), at)
        self.assertLess(at, brief.index("On the shot"))
        self.assertLess(at, brief.index(SONG_LINE))
        self.assertEqual(load_case()[5]["brief"], brief)
        self.assertNotIn("@", " ".join(load_case()[5]["items"]["extra"]))   # Refs Out writes its own line

    def test_b_refs_out_recounts_the_names(self):
        refs = load_case()[5]
        b_sound = RO.build_brief(refs, clip_sound=False)
        self.assertIn("@audio2 = <Audio 1>", b_sound)
        self.assertIn("<Audio 1> = " + SOUND_LINE, b_sound)      # the inventory says the same
        b_pics = RO.build_brief(refs, pictures=False)
        self.assertIn("@image2 = not sent", b_pics)
        self.assertNotIn("<Picture", b_pics)
        self.assertIn(NAMES_A, RO.build_brief(refs))            # all on: the loader's own line
        # B2: a writer reading the line back gets the switched map.
        self.assertEqual(LN.read_names_line(b_sound)["map"]["@audio2"], ("audio", 1))
        self.assertIsNone(LN.read_names_line(b_pics)["map"]["@image2"])
        self.assertEqual(RefsOut().unpack(refs, clip_sound=False)[BRIEF], b_sound)

    def test_b3_a_qwen_payload_is_written_in_minimax_tags(self):
        refs = {"images": ["i0", "i1"],
                "items": {"images": ["1:1", "16:9"], "videos": [], "video_audios": [], "audios": [], "song": "",
                          "extra": []},
                "naming": LN.make("qwen", ["ref", "edit"])}
        brief = RO.build_brief(refs)
        self.assertIn("Reference images, in order: <Picture 1> = 1:1; <Picture 2> = 16:9.", brief)
        self.assertIn("@ names: @image1 = <Picture 2>; @image2 = <Picture 1>.", brief)

    def test_c_refs_without_naming_give_todays_brief(self):
        refs = load_case()[5]
        old = {k: v for k, v in refs.items() if k != "naming"}
        for switches in ({}, {"pictures": False}, {"clip_sound": False}, {"clips": False}, {"sounds": False},
                         {"song": False}):
            with_names = RO.build_brief(refs, **switches)
            without = RO.build_brief(old, **switches)
            self.assertNotIn("@ names", without)
            self.assertEqual(without, with_names.replace(" " + LN.names_line(
                LN.remake(refs["naming"], **{k: v for k, v in switches.items() if k != "song"}), "minimax"), ""))

    def test_c_old_refs_without_items_keep_their_brief_and_get_the_line(self):
        refs = load_case()[5]
        old = {k: v for k, v in refs.items() if k != "items"}
        self.assertEqual(RO.build_brief(old), refs["brief"])                 # its line is already the minimax one
        b = RO.build_brief(old, clip_sound=False)
        self.assertIn("@audio2 = <Audio 1>", b)
        self.assertEqual(b.count("@ names:"), 1)

    def test_d_nothing_loaded(self):
        refs = Loader().load("", 2)["result"][5]
        naming = refs["naming"]
        self.assertEqual(naming["map"], {})
        self.assertEqual(naming["counts"], {"image": 0, "video": 0, "audio": 0})
        self.assertIs(LN.from_refs(refs), naming)
        self.assertEqual(RO.build_brief(refs), "")
        self.assertEqual(RefsOut().unpack(refs)[BRIEF], "")

    def test_keyframe_only_names_every_card_not_sent(self):
        media = json.dumps({"pictures": [{"role": "key", "at": 0}]})
        out = load_case(media, "a.png")
        self.assertEqual(out[5]["naming"]["map"], {"@image1": None})
        self.assertIn("@ names: @image1 = not sent.", out[3])


class ReferenceLoaderRefsTests(unittest.TestCase):
    INFO = {"seconds": 5.17, "frames": 124, "width": 768, "height": 432, "source_size": (768, 432),
            "source_fps": 24.0, "start": 0.0, "name": "c.mp4"}

    def test_items_and_song(self):
        a = {"seconds": 3.0, "sample_rate": 48000, "channels": 2, "start": 0.0, "short": False,
             "asked": 3.0, "name": "a.wav"}
        items = RL._build_items([(640, 480), (480, 640)], [self.INFO], [None], [a])
        self.assertEqual(len(items["images"]), 2)
        self.assertEqual(items["video_audios"], [False])
        self.assertEqual(len(items["videos"]), 1)
        self.assertEqual(len(items["audios"]), 1)
        self.assertEqual(items["song"], "")
        self.assertIn("different aspect ratios", items["picture_warning"])
        self.assertNotIn("<", "".join(items["images"] + items["videos"] + items["audios"]))

    def test_refs_out_works_on_it(self):
        items = RL._build_items([(640, 480)], [self.INFO], ["snd"], [])
        refs = {"images": ["i"], "videos": ["v"], "video_audios": ["snd"], "audios": [], "song": None,
                "items": items, "brief": "b"}
        out = RefsOut().unpack(refs)
        self.assertIn("<Audio 1> = the soundtrack of <Video 1>.", out[BRIEF])
        self.assertIsNone(out[AUDIO])


class CanvasRegisteredTests(unittest.TestCase):
    def test_frames_replaces_length_in_the_registered_node(self):
        pack = sys.modules[PACKNAME]
        cls = pack.NODE_CLASS_MAPPINGS["Luna-H3-Canvas"]
        self.assertEqual(list(cls.RETURN_NAMES[:3]), ["width", "height", "frames"])
        self.assertEqual(cls.RETURN_TYPES[2], "INT")
        self.assertEqual(len(cls.RETURN_NAMES), 8)
        # V3 gives an id-less output the id "_{slot}_{type}_", so the rename moves no link.
        schema = cls.GET_SCHEMA()
        self.assertEqual(schema.outputs[2].id, "_2_INT_")
        self.assertEqual(schema.inputs[-1].id, "image")
        self.assertTrue(schema.inputs[-1].optional)



class LoaderChainTests(unittest.TestCase):
    """2026-11-03: the song start, the segment and the overlap are set on the Asset Loader's card."""

    CARD = {"file": "song.wav", "start": 0.0, "duration": 5.0, "fill": "silence", "position": "shot",
            "song_start": 87.44, "shot": 2, "overlap": 22}

    def window(self, chain, card=None, song_start=None):
        seen = {}

        def fake_window(path, start, want, fill, describe, **k):
            seen["start"], seen["want"] = start, want
            return SONG, {"name": "song.wav", "start": start, "seconds": want, "total": 300.0, "filled": 0.0,
                          "taken": want}

        with patch.object(AL, "_media_file", lambda f: "song.wav"), patch.object(AL, "load_audio_window", fake_window):
            _, info, _ = Loader._song([dict(card or self.CARD)], 124 / 24, None, song_start, chain)
        return seen, info

    def test_clip_moves_the_cards_shot_on(self):
        for clip in (0, 1, 2):
            seen, info = self.window({"clip": clip})
            self.assertEqual(info["shot"], 2 + clip)
            self.assertAlmostEqual(seen["start"], 87.44 + (1 + clip) * (124 - 22) / 24)
            self.assertAlmostEqual(seen["want"], 124 / 24)

    def test_chain_needs_shot_mode_and_no_song_start(self):
        with self.assertRaisesRegex(RuntimeError, "By shot number"):
            self.window({"clip": 0}, card=dict(self.CARD, position="free"))
        with self.assertRaisesRegex(RuntimeError, "By shot number"):
            self.window({"clip": 0}, song_start=10.0)
        with self.assertRaisesRegex(RuntimeError, "no song"):
            Loader._song([], 5.0, None, None, {"clip": 0})


# --- Build 2: one frame count, one shot rate, the overlap row, the vocal stem, the refs contract -------------------

MIO = _module("media_io")
LU = _module("luna_units")
SR = 48000


def song_wave(sign=1.0):
    """60 s of stereo whose sample k holds k (sign -1: -k), so a window's first sample says where it starts."""
    ramp = torch.arange(60 * SR, dtype=torch.float64) * sign
    return torch.stack([ramp, ramp]), SR


class FakeMedia:
    """The real Asset Loader path (_probe, _clips, _song, _vocals, _sounds, media_io's windows) on fake files:
    probe_media, decode_clip, clip_soundtrack and media_io._decode_audio are the stubbed boundaries."""

    def __init__(self, src_fps=30.0, sound=None):
        self.src_fps, self.sound, self.decodes = src_fps, sound, []

    def decode(self, path, start, frames, fps, size, fit, nth, precision, grid, who=None):
        n = -(-int(frames) // int(nth))
        self.decodes.append({"start": start, "frames": frames, "fps": fps, "nth": nth})
        return torch.zeros(n, 8, 8, 3), {
            "name": os.path.basename(path), "frames": n, "asked": int(frames), "short": False,
            "seconds": frames / fps, "fps": fps / nth, "source_fps": self.src_fps, "width": 8, "height": 8,
            "source_size": (8, 8), "start": start, "nth": nth}

    def probe(self, path):
        return {"name": os.path.basename(path), "video": True, "audio": self.sound is not None, "fps": self.src_fps,
                "width": 1920, "height": 1080, "duration": 60.0}

    def audio(self, path):
        return song_wave(-1.0 if "vocals" in path else 1.0)

    def run(self, media, paths="", **kw):
        with patch.object(AL, "_media_file", lambda f: f), patch.object(AL, "probe_media", self.probe), \
                patch.object(AL, "decode_clip", self.decode), \
                patch.object(AL, "clip_soundtrack", lambda *a: self.sound), \
                patch.object(MIO, "_decode_audio", self.audio), patch.object(AL, "_load", fake_load):
            result = Loader().load(paths, max(1, len(paths.split("\n")) if paths else 1),
                                   media_state=json.dumps(media) if isinstance(media, dict) else media, **kw)["result"]
        # Every value read from the declared outputs, by socket name.
        self.assert_declared(result)
        return dict(zip(Loader.RETURN_NAMES, result))

    @staticmethod
    def assert_declared(result):
        assert len(result) == len(Loader.RETURN_NAMES) == len(Loader.RETURN_TYPES) == 21


def target(name, **extra):
    return {"settings": {"target": name, **extra}}


def clip(**card):
    return {"video": [{"file": "c.mp4", **card}]}


class FrameCountTests(unittest.TestCase):
    """SPEC § 1: one count per run, rounded UP onto the target's grid; 362 is a warning, not a cut."""

    def test_h3_5_seconds_is_124_frames_at_24(self):
        out = FakeMedia().run(target("h3"), "a.png", seconds=5.0)
        self.assertEqual((out["frames"], out["fps"]), (124, 24.0))
        self.assertEqual(out["timeline"]["frames"], 124)
        self.assertEqual(out["timeline"]["fps"], 24.0)

    def test_h3_16_seconds_is_396_with_the_warning_and_nothing_cut(self):
        out = FakeMedia().run(target("h3"), "a.png", seconds=16)
        self.assertEqual(out["frames"], 396)
        self.assertIn("396 frames is past the 362 frames H3 was trained on", out["asset_brief"])
        self.assertIn("396 frames is past", out["refs"]["brief"])
        self.assertTrue(any("396 frames is past" in x for x in out["refs"]["items"]["extra"]))

    def test_ltx_5_seconds_is_129_frames_at_25(self):
        out = FakeMedia().run(target("ltx"), "a.png", seconds=5.0)
        self.assertEqual((out["frames"], out["fps"]), (129, 25.0))

    def test_general_takes_clip_1s_delivered_rate(self):
        out = FakeMedia(30.0).run({**target("general"), **clip()}, seconds=5.0)
        self.assertEqual((out["fps"], out["frames"]), (30.0, 150))
        out = FakeMedia(30.0).run({**target("general"), **clip(nth=2, count="free")}, seconds=5.0)
        self.assertEqual((out["fps"], out["frames"]), (15.0, 75))

    def test_h3_clip_at_its_30_fps_source_warns_and_keeps_124(self):
        out = FakeMedia(30.0).run({**target("h3"), **clip(fps="source")}, seconds=5.0)
        self.assertEqual((out["frames"], out["fps"]), (124, 30.0))
        self.assertIn("Clip V1 (c.mp4) is delivered at 30 fps, not the shot's 24 fps", out["asset_brief"])
        out = FakeMedia(30.0).run({**target("h3"), **clip(fps="target")}, seconds=5.0)
        self.assertEqual((out["frames"], out["fps"]), (124, 24.0))
        self.assertNotIn("is delivered at", out["asset_brief"])

    def test_a_clip_cards_own_count_past_362_warns(self):
        # The card's own length (380 -> 396 on the H3 grid), not the shot's: warned in the console and the brief,
        # and kept by Refs Out (items["extra"]). Nothing is cut.
        import contextlib
        import io
        fm = FakeMedia(24.0)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            out = fm.run({**target("h3"), **clip(frames=380)}, seconds=5.0)
        self.assertEqual(fm.decodes[0]["frames"], 396)
        self.assertIn("Clip V1 (c.mp4): 396 frames is past the 362 frames H3 was trained on", out["asset_brief"])
        self.assertTrue(any("Clip V1 (c.mp4): 396 frames is past" in x for x in out["refs"]["items"]["extra"]))
        self.assertIn("Clip V1 (c.mp4): 396 frames is past", RO.build_brief(out["refs"], clips=False))
        self.assertIn("Clip V1 (c.mp4): 396 frames is past", printed.getvalue())
        out = FakeMedia(24.0).run({**target("h3"), **clip(frames=124)}, seconds=5.0)
        self.assertNotIn("frames is past", out["asset_brief"])

    def test_unwired_takes_the_stored_length_up_onto_the_grid(self):
        out = FakeMedia().run({"shot": {"frames": 380}}, "a.png")
        self.assertEqual(out["frames"], 396)
        self.assertIn("396 frames is past", out["asset_brief"])
        self.assertEqual(FakeMedia().run({"shot": {"frames": 120}, **target("ltx")}, "a.png")["frames"], 121)

    def test_the_brief_and_the_placements_use_the_same_count(self):
        # A keyframe at frame 123 fits a 124-frame shot (5.0 s); the old ungridded 120 refused it.
        media = {"pictures": [{"role": "key", "at": 123}]}
        out = FakeMedia().run(media, "a.png", seconds=5.0)
        self.assertIn("On the shot (124 frames, 5.17 s)", out["asset_brief"])
        with self.assertRaisesRegex(RuntimeError, "fits from frame 0 to 123"):
            FakeMedia().run({"pictures": [{"role": "key", "at": 124}]}, "a.png", seconds=5.0)


class SongWindowTests(unittest.TestCase):
    """SPEC § 2: the song and the vocal stem follow the shot length, from the same start."""

    SONG = {"file": "s.wav", "start": 10.0, "duration": 30.0, "fill": "silence"}

    def test_free_song_is_exactly_the_shot_from_its_start(self):
        out = FakeMedia().run({"audio": [self.SONG]}, "a.png", seconds=5.0)
        wave = out["audio"]["waveform"]
        self.assertEqual(wave.shape[-1], round(124 / 24 * SR))       # 124/24 s to the sample, not the card's 30 s
        self.assertEqual(float(wave[0, 0, 0]), 10.0 * SR)            # from 10 s
        self.assertIsNone(out["vocals"])
        self.assertIsNone(out["refs"]["vocals"])
        self.assertEqual(out["refs"]["items"]["vocals"], "")

    def test_the_vocal_stem_has_the_same_start_and_length(self):
        out = FakeMedia().run({"audio": [dict(self.SONG, vocals="s_vocals.wav")]}, "a.png", seconds=5.0)
        song, voc = out["audio"]["waveform"], out["vocals"]["waveform"]
        self.assertEqual(voc.shape, song.shape)
        self.assertEqual(float(voc[0, 0, 0]), -10.0 * SR)            # the stem file, from the same 10 s
        self.assertIs(out["refs"]["vocals"], out["vocals"])
        line = out["refs"]["items"]["vocals"]
        self.assertIn("for the lip-sync lock", line)
        self.assertNotIn("<Audio", line)                              # no number baked in (inventory renders it)
        self.assertNotIn("<Audio", out["refs"]["items"]["song"])
        self.assertIn("for the saved video", out["refs"]["items"]["song"])
        self.assertLess(out["asset_brief"].index("Song:"), out["asset_brief"].index("Vocal stem:"))
        lock = [it for it in out["timeline"]["items"] if it["kind"] == "audio"][0]
        self.assertIs(lock["vocals"], out["vocals"])                 # the lock carries the stem

    def test_reference_sounds_are_at_most_the_shot(self):
        media = {"audio": [self.SONG, {"file": "r.wav", "role": "ref", "duration": 0},
                           {"file": "q.wav", "role": "ref", "duration": 2.0}]}
        out = FakeMedia().run(media, "a.png", seconds=5.0)
        lengths = [a["waveform"].shape[-1] for a in out["refs"]["audios"]]
        self.assertEqual(lengths, [round(124 / 24 * SR), 2 * SR])


class OverlapTests(unittest.TestCase):
    """SPEC § 2: one overlap setting on the node; old saves take sound card 1's; off the list snaps with a warning."""

    def test_old_save_takes_sound_card_1s_overlap(self):
        media = {"audio": [{"file": "s.wav", "overlap": 39}]}
        self.assertEqual(AL._parse_media(json.dumps(media))["settings"]["overlap"], 39)
        self.assertEqual(FakeMedia().run(media, "a.png", seconds=5.0)["overlap"], 39)

    def test_the_setting_without_a_song(self):
        self.assertEqual(FakeMedia().run({"settings": {"overlap": 90}}, "a.png")["overlap"], 90)
        self.assertEqual(Loader().load("", 2, media_state='{"settings": {"overlap": 90}}')["result"][9], 90)

    def test_off_the_list_snaps_with_a_warning(self):
        out = FakeMedia().run({"settings": {"overlap": 50}}, "a.png")
        self.assertEqual(out["overlap"], 39)
        self.assertIn("Overlap 50 is not on the list", out["asset_brief"])
        self.assertIn("Overlap 50 is not on the list", RO.build_brief(out["refs"], pictures=False))

    def test_the_migrated_overlap_outlives_the_song(self):
        # The text the frontend writes once the song is removed from an old save (tests/test_asset_frames.mjs,
        # AFTER_REMOVAL), and the old save itself: both 39.
        self.assertEqual(FakeMedia().run('{"settings":{"overlap":39}}', "a.png")["overlap"], 39)
        self.assertEqual(FakeMedia().run('{"audio":[{"file":"song.wav","overlap":39}],"settings":{}}', "a.png",
                                         seconds=5.0)["overlap"], 39)

    def test_the_setting_wins_over_the_card(self):
        media = {"audio": [{"file": "s.wav", "overlap": 39}], "settings": {"overlap": 141}}
        self.assertEqual(FakeMedia().run(media, "a.png")["overlap"], 141)


class ChainShiftTests(unittest.TestCase):
    """SPEC § 2: clip cards by shot number move with the chain's clip like the song; a free card stays."""

    def test_clip_1_moves_song_and_shot_clip_cards_by_one_step(self):
        step = 124 / 24 - 22 / 24
        media = {"audio": [{"file": "s.wav", "position": "shot", "song_start": 10.0, "shot": 1}],
                 "video": [{"file": "a.mp4", "start": 2.0, "position": "shot", "shot": 1},
                           {"file": "b.mp4", "start": 2.0}],
                 "settings": {"overlap": 22}}
        fm = FakeMedia(24.0)
        out = fm.run(media, "a.png", seconds=124 / 24, chain={"clip": 1})
        self.assertAlmostEqual(out["timeline"]["frames"] / 24, 124 / 24)
        song_start = float(out["audio"]["waveform"][0, 0, 0]) / SR
        self.assertAlmostEqual(song_start, 10.0 + step, places=4)
        self.assertAlmostEqual(fm.decodes[0]["start"], 2.0 + step)   # by shot number: the same step
        self.assertAlmostEqual(fm.decodes[1]["start"], 2.0)          # free: stays

    def test_clip_card_by_shot_follows_the_wired_shot_without_a_chain(self):
        starts = Loader._clip_starts([{"file": "a", "start": 1.0, "position": "shot", "shot": 1}], 124 / 24, shot=3)
        self.assertAlmostEqual(starts[0], 1.0 + 2 * (124 - 22) / 24)


class RefsContractTests(unittest.TestCase):
    """SPEC § 2, refs: the keys, roles, keyframes and the one brief text."""

    KEYS = {"images", "roles", "videos", "video_audios", "audios", "song", "vocals", "keyframes", "brief", "items",
            "naming", "width", "height"}

    def test_keys_roles_and_keyframes(self):
        media = {"pictures": [{}, {"role": "key", "at": 0}, {}]}
        refs = FakeMedia().run(media, "a.png\nb.png\nc.png", seconds=5.0)["refs"]
        self.assertEqual(set(refs), self.KEYS)
        self.assertEqual(refs["roles"], ["reference"] * len(refs["images"]))
        self.assertEqual(len(refs["images"]), 2)
        self.assertEqual([(k["card"], k["at"]) for k in refs["keyframes"]], [(2, 0)])
        self.assertEqual(tuple(refs["keyframes"][0]["image"].shape), (1, 640, 480, 3))
        self.assertEqual(set(Loader().load("", 2)["result"][5]), self.KEYS)       # the empty loader too

    def test_the_loader_brief_is_the_inventory(self):
        refs = FakeMedia().run({"audio": [{"file": "s.wav", "vocals": "s_vocals.wav"}], **clip()},
                               "a.png\nb.png", seconds=5.0)["refs"]
        self.assertEqual(refs["brief"], LN.inventory(refs["items"], refs["naming"]))
        self.assertEqual(RO.build_brief(refs), refs["brief"])

    def test_a_clip_kept_with_no_track_says_so_in_both_briefs(self):
        out = FakeMedia(24.0, sound=None).run(clip(sound="keep"), "a.png", seconds=5.0)
        self.assertIn("it has no sound", out["asset_brief"])
        self.assertIn("it has no sound", out["refs"]["items"]["videos"][0])
        self.assertIn("it has no sound", RefsOut().unpack(out["refs"])[BRIEF])

    def test_fps_and_vocals_are_sockets_19_and_20(self):
        result = Loader().load("", 2, seconds=5.0, media_state='{"settings": {"target": "ltx"}}')["result"]
        self.assertEqual((result[19], result[20]), (25.0, None))


class RefsOutRefsTests(unittest.TestCase):
    """Luna-H3-RefsOut's `refs` output: the bundle filtered by the switches, its brief = asset_brief."""

    def loaded(self):
        media = {"audio": [{"file": "s.wav", "vocals": "s_vocals.wav"}, {"file": "r.wav", "role": "ref"}],
                 **clip(), "pictures": [{}, {"role": "key", "at": 0}]}
        return FakeMedia(24.0, sound=SONG).run(media, "a.png\nb.png\nc.png", seconds=5.0)["refs"]

    def test_all_on_keeps_every_item(self):
        refs = self.loaded()
        out = RefsOut().unpack(refs)
        r = out[REFS]
        for key in ("images", "roles", "videos", "video_audios", "audios"):
            self.assertEqual(len(r[key]), len(refs[key]), key)
        self.assertIs(r["song"], refs["song"])
        self.assertIs(r["vocals"], refs["vocals"])
        self.assertEqual(r["keyframes"], refs["keyframes"])
        self.assertEqual(r["brief"], out[BRIEF])
        self.assertEqual(r["brief"], refs["brief"])

    def test_every_switch_set_gives_the_brief_back_from_its_own_items(self):
        import itertools
        refs = self.loaded()
        for flags in itertools.product((True, False), repeat=5):
            sw = dict(zip(("pictures", "clips", "clip_sound", "sounds", "song"), flags))
            out = RefsOut().unpack(refs, **sw)
            r = out[REFS]
            self.assertEqual(r["brief"], out[BRIEF], sw)
            self.assertEqual(LN.inventory(r["items"], r["naming"]), out[BRIEF], sw)
            self.assertEqual(set(r), set(refs), sw)

    def test_pictures_off(self):
        out = RefsOut().unpack(self.loaded(), pictures=False)
        r = out[REFS]
        self.assertEqual((r["images"], r["roles"]), ([], []))
        self.assertNotIn("<Picture", r["brief"])
        self.assertEqual(r["brief"], out[BRIEF])

    def test_song_off_takes_the_vocal_stem_too(self):
        out = RefsOut().unpack(self.loaded(), song=False)
        self.assertIsNone(out[REFS]["song"])
        self.assertIsNone(out[REFS]["vocals"])
        self.assertNotIn("Vocal stem:", out[BRIEF])

    def test_clip_sound_off_keeps_the_pairing(self):
        r = RefsOut().unpack(self.loaded(), clip_sound=False)[REFS]
        self.assertEqual(len(r["videos"]), 1)
        self.assertEqual(r["video_audios"], [None])

    def test_old_refs_without_items_or_naming(self):
        old = {"images": ["i"], "videos": [], "video_audios": [], "audios": [], "brief": "OLD BRIEF"}
        r = RefsOut().unpack(old)[REFS]
        self.assertEqual(r["brief"], "OLD BRIEF")
        self.assertNotIn("naming", r)


class SchemaTests(unittest.TestCase):
    """Saved workflows keep working: every old slot where it was, new ones last."""

    def test_loader_outputs_and_inputs(self):
        # SPEC § 2: 10 fixed, then image_1..image_9 always, then fps and vocals; no saved slot moves.
        self.assertEqual(Loader.RETURN_NAMES, ("images", "width", "height", "asset_brief", "audio", "refs", "timeline",
                                               "video", "frames", "overlap") + IMAGE_NAMES + ("fps", "vocals"))
        self.assertEqual(Loader.RETURN_TYPES, ("IMAGE", "INT", "INT", "STRING", "AUDIO", "LUNA_H3_REFS",
                                               "LUNA_TIMELINE", "IMAGE", "INT", "INT") + ("IMAGE",) * 9
                         + ("FLOAT", "AUDIO"))
        self.assertEqual(len(Loader.OUTPUT_TOOLTIPS), 21)
        t = Loader.INPUT_TYPES()
        self.assertEqual(list(t["required"]), ["image_paths", "image_count"])
        self.assertEqual(list(t["optional"]), ["asset_state", "media_state", "seconds", "shot", "width", "height",
                                               "song_start", "chain"])
        tips = dict(zip(Loader.RETURN_NAMES, Loader.OUTPUT_TOOLTIPS))
        self.assertIn("Luna-SaveVideo's frame_rate", tips["fps"])
        self.assertIn("Luna-LLMPrompt's fps", tips["fps"])
        self.assertIn("lip-sync lock", tips["vocals"])

    def test_refs_out_outputs(self):
        self.assertEqual(RefsOut.RETURN_NAMES[:BRIEF], tuple(f"ref_image_{i}" for i in range(9))
                         + tuple(f"ref_video_{i}" for i in range(3)) + tuple(f"ref_video_audio_{i}" for i in range(3))
                         + tuple(f"ref_audio_{i}" for i in range(3)))
        self.assertEqual(RefsOut.RETURN_NAMES[-1], "refs")
        self.assertIn("Luna-LLMPrompt", RefsOut.OUTPUT_TOOLTIPS[-1])

    def test_tooltips_name_the_rule(self):
        tips = dict(zip(Loader.RETURN_NAMES, Loader.OUTPUT_TOOLTIPS))
        self.assertIn("UP", tips["frames"])
        self.assertIn("362", tips["frames"])
        self.assertIn("MiniMaxH3ReferenceToVideo", tips["frames"])
        self.assertIn("Overlap row", tips["overlap"])
        self.assertIn("keyframes", tips["refs"])
        self.assertIn("362", Loader.INPUT_TYPES()["optional"]["seconds"][1]["tooltip"])

    def test_messages_are_ascii(self):
        out = FakeMedia(30.0).run({**clip(fps="source"), "audio": [{"file": "s.wav", "vocals": "s_vocals.wav"}],
                                   "settings": {"overlap": 50}}, "a.png", seconds=16)
        out["asset_brief"].encode("ascii")


if __name__ == "__main__":
    unittest.main()

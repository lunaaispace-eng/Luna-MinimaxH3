"""Asset Loader / Refs Out / Reference Loader / Canvas wiring (plans/asset-loader-connect.md):
the loader's new outputs, no continuation, an empty loader that runs, the canvas size passed on,
refs["song"] and refs["items"], and Refs Out's switches, rebuilt brief and `audio` output.

    E:\\ComfyUI-Easy-Install\\python_embeded\\python.exe -m unittest discover -s tests -v
Loads the whole pack through ComfyUI's modules, like test_registration.py.
"""

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


def fake_load(path, st=None):
    w, h = {"a.png": (640, 480), "b.png": (480, 640)}.get(path, (320, 320))
    return torch.zeros(1, h, w, 3), (w, h), None


def clip_stub(sound):
    return [{"batch": torch.zeros(5, 8, 8, 3), "sound": sound, "info": {"name": "c.mp4"},
             "line": "c.mp4 from 0:00.00, 5 frames (0.21 s) at 24 fps, 8x8"}]


class LoaderOutputTests(unittest.TestCase):
    def test_return_names_and_types(self):
        self.assertEqual(Loader.RETURN_NAMES[:9], ("images", "width", "height", "asset_brief", "audio", "refs",
                                                   "timeline", "video", "frames"))
        self.assertEqual(Loader.RETURN_NAMES[9:], IMAGE_NAMES)      # image_N strictly last
        self.assertEqual(Loader.RETURN_TYPES[7:9], ("IMAGE", "INT"))
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
        self.assertTrue(all(x is None for x in out[9:]))

    def test_empty_loader_hands_on_the_canvas_and_the_seconds(self):
        out = Loader().load("", 2, seconds=5.0, width=832, height=480)["result"]
        self.assertEqual((out[1], out[2], out[8]), (832, 480, 124))

    def test_frames_is_on_the_h3_grid(self):
        # shot_frames is round(seconds x 24); the `frames` output snaps it to the nearest
        # 17k+5 because it feeds H3's length. The canvas's seconds (frames / 24) is already on it.
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
                lambda *a, **k: clip_stub(None))):
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
        self.assertEqual(RefsOut.RETURN_NAMES[-2:], ("asset_brief", "audio"))
        self.assertEqual(RefsOut.RETURN_TYPES[-1], "AUDIO")
        self.assertEqual(len(RefsOut.RETURN_NAMES), 9 + 3 + 3 + 3 + 2)
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
        self.assertIs(out[-1], SONG)

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
        self.assertIsNone(out[-1])

    def test_brief_numbering_all_on(self):
        brief = RefsOut().unpack(refs_with())[-2]
        order = [brief.index(t) for t in ("<Picture 1>", "<Picture 2>", "<Audio 1> = the soundtrack of <Video 1>",
                                           "<Video 1> =", "<Audio 2> = aud desc 0", "<Audio 3> = aud desc 1",
                                           "Song: x.", "On the shot")]
        self.assertEqual(order, sorted(order))
        self.assertNotIn("THE LOADER BRIEF", brief)

    def test_clip_sound_off_recounts_the_standalone_sounds(self):
        brief = RefsOut().unpack(refs_with(), clip_sound=False)[-2]
        self.assertNotIn("soundtrack", brief)
        self.assertIn("<Audio 1> = aud desc 0", brief)
        self.assertIn("<Audio 2> = aud desc 1", brief)

    def test_two_clips_one_silent(self):
        brief = RefsOut().unpack(refs_with(clip_sounds=(False, True), sounds=1))[-2]
        self.assertNotIn("soundtrack of <Video 1>", brief)
        self.assertLess(brief.index("<Audio 1> = the soundtrack of <Video 2>"), brief.index("<Video 2> ="))
        self.assertIn("<Audio 2> = aud desc 0", brief)

    def test_switched_off_categories_leave_the_brief(self):
        n = RefsOut()
        self.assertNotIn("<Picture", n.unpack(refs_with(), pictures=False)[-2])
        b = n.unpack(refs_with(), clips=False)[-2]
        self.assertNotIn("<Video", b)
        self.assertNotIn("soundtrack", b)
        self.assertIn("<Audio 1> = aud desc 0", b)
        self.assertNotIn("Song:", n.unpack(refs_with(), song=False)[-2])
        self.assertNotIn("<Audio", n.unpack(refs_with(), sounds=False, clip_sound=False)[-2])

    def test_sounds_without_picture_or_clip_warn(self):
        warn = "MiniMax H3 does not accept reference sounds without a reference picture or clip"
        b = RefsOut().unpack(refs_with(), pictures=False, clips=False)[-2]
        self.assertIn(warn, b)
        self.assertNotIn(warn, RefsOut().unpack(refs_with())[-2])
        self.assertNotIn(warn, RefsOut().unpack(refs_with(), pictures=False, clips=False, sounds=False)[-2])

    def test_old_refs_without_items(self):
        old = {"images": ["i"], "videos": ["v"], "video_audios": ["a"], "audios": ["x"], "brief": "OLD BRIEF"}
        out = RefsOut().unpack(old)
        self.assertEqual(out[-2], "OLD BRIEF")
        self.assertIsNone(out[-1])                       # no song in old refs
        b = RefsOut().unpack(old, clip_sound=False)[-2]
        self.assertNotEqual(b, "OLD BRIEF")
        self.assertIn("<Audio 1>.", b)                   # the standalone sound counts from 1
        self.assertNotIn("soundtrack", b)

    def test_not_a_dict(self):
        out = RefsOut().unpack(None)
        self.assertEqual(len(out), len(RefsOut.RETURN_NAMES))


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
        self.assertIn("<Audio 1> = the soundtrack of <Video 1>.", out[-2])
        self.assertIsNone(out[-1])


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


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the chain nodes (h3_chain.py), on CPU tensors.

Stubs `comfy.nested_tensor` and `comfy.utils` when ComfyUI is not on the path, like test_h3_latent.py.
The output folder is a temporary directory; the VHS Meta Batch Manager is a small fake.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


class _StubNestedTensor:
    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


def _stub_reshape_mask(mask, output_shape):
    import torch
    if tuple(mask.shape[2:]) == tuple(output_shape[2:]):
        return mask
    return torch.ones(output_shape, dtype=mask.dtype, device=mask.device)


def _install_comfy_stubs():
    try:
        import comfy.nested_tensor  # noqa: F401
        import comfy.utils  # noqa: F401
        return
    except ImportError:
        pass
    comfy = ModuleType("comfy")
    nested = ModuleType("comfy.nested_tensor")
    utils = ModuleType("comfy.utils")
    nested.NestedTensor = _StubNestedTensor
    utils.reshape_mask = _stub_reshape_mask
    comfy.nested_tensor = nested
    comfy.utils = utils
    sys.modules.update({"comfy": comfy, "comfy.nested_tensor": nested, "comfy.utils": utils})


_install_comfy_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import h3_chain  # noqa: E402

FRAMES = 141            # 17 * 8 + 5
TOKENS = 42             # 2 + 5 * 8
AUDIO = 235             # 141 / 24 * 40
LOW = (6, 8)            # low grid
FULL = (12, 16)


class FakeManager:
    def __init__(self, frames=FRAMES):
        self.frames_per_batch = frames
        self.inputs = {}
        self.unique_id = "5"   # as left by step 0 of the same run
        self.total_frames = 0
        self.has_closed_inputs = False

    def reset(self):
        self.unique_id = None


def prompt_for(step):
    return {"10": {"inputs": {"meta_batch": ["5", 0]}}, "5": {"inputs": {"requeue": step}}}


def av(fill, hw=FULL, audio_fill=None):
    video = torch.full((1, 24, TOKENS, *hw), float(fill))
    audio = torch.full((1, 32, 2, AUDIO), float(fill if audio_fill is None else audio_fill))
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def locked_song(latent):
    video, audio = latent["samples"].unbind()
    out = dict(latent)
    out["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (torch.ones((1, 1, *video.shape[2:])), torch.zeros((1, 1, *audio.shape[2:]))))
    return out


def cont(chain, *args, **kwargs):
    return h3_chain.LunaH3ChainContinue().continue_clip(chain, *args, overlap=chain["_test_overlap"], **kwargs)


def rem(chain, *args, **kwargs):
    return h3_chain.LunaH3ChainRemember().remember(chain, *args, overlap=chain["_test_overlap"], **kwargs)


def join(chain, *args, **kwargs):
    return h3_chain.LunaH3ChainJoin().join(chain, *args, overlap=chain["_test_overlap"], **kwargs)


class ChainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = h3_chain._output_dir
        h3_chain._output_dir = lambda: self.tmp.name

    def tearDown(self):
        h3_chain._output_dir = self._old
        self.tmp.cleanup()

    def chain(self, step, clips=3, overlap="22", locked=True, manager=None):
        manager = manager or FakeManager()
        out = h3_chain.LunaH3Chain().run(manager, clips, 100, locked, prompt=prompt_for(step),
                                         unique_id="10", first_prompt="a", later_prompt="b")
        out[0]["_test_overlap"] = int(overlap)   # what Luna-H3-AssetLoader's overlap output would give
        return out, manager

    # ------------------------------------------------------------------ Luna-H3-Chain
    def test_steps(self):
        (out, manager) = self.chain(2)
        chain, seed, clip, frames, seconds, prompt, first = out
        self.assertEqual((chain["clip"], seed, clip, frames), (2, 102, 3, FRAMES))
        self.assertNotIn("overlap", chain, "the overlap comes from the Asset Loader, not the chain")
        self.assertAlmostEqual(seconds, FRAMES / 24)
        self.assertEqual((prompt, first), ("b", False))
        self.assertTrue(manager.has_closed_inputs)

    def test_22_needs_the_song_locked(self):
        (c, _) = self.chain(0, locked=False)
        with self.assertRaisesRegex(ValueError, "36.67"):
            h3_chain._overlap(c[0], 22, "test")
        self.assertEqual(h3_chain._overlap(c[0], 39, "test"), 39)

    def test_frames_off_grid_or_overlap_too_long(self):
        with self.assertRaisesRegex(ValueError, "17k\\+5"):
            self.chain(0, manager=FakeManager(140))
        (c, _) = self.chain(0, manager=FakeManager(141))
        with self.assertRaisesRegex(ValueError, "shorter"):
            h3_chain._overlap(c[0], 141, "test")

    def test_overlap_missing_or_not_set_on_the_loader(self):
        (c0, _) = self.chain(0)
        with self.assertRaisesRegex(ValueError, "connect Luna-H3-AssetLoader overlap"):
            h3_chain.LunaH3ChainContinue().continue_clip(c0[0], locked_song(av(0.0)), "t", False)
        with self.assertRaisesRegex(ValueError, "sound card"):
            h3_chain.LunaH3ChainContinue().continue_clip(c0[0], locked_song(av(0.0)), "t", False, overlap=0)

    def test_lost_batch_state(self):
        manager = FakeManager()
        manager.unique_id = "other"
        with self.assertRaisesRegex(ValueError, "lost"):
            self.chain(1, manager=manager)

    def test_past_the_clip_count(self):
        with self.assertRaisesRegex(ValueError, "past 3 clips"):
            self.chain(3)

    # ------------------------------------------------------------------ Remember / Continue
    def remember(self, chain, fill, low=True):
        low_latent = av(fill + 0.5, hw=LOW) if low else None
        return rem(chain, av(fill, audio_fill=fill + 0.25), "t", low_latent)[0]

    def test_head_copied_locked_and_song_mask_kept(self):
        (c0, _), (c1, _) = self.chain(0), self.chain(1)
        self.remember(c0[0], 7.0)
        latent, low_head, _ = cont(c1[0], locked_song(av(0.0)), "t", False)
        video, audio = latent["samples"].unbind()
        vmask, amask = latent["noise_mask"].unbind()
        v_n = h3_chain.video_steps(22)
        self.assertEqual(v_n, 7)
        self.assertTrue(torch.all(video[:, :, :v_n] == 7.0))
        self.assertTrue(torch.all(video[:, :, v_n:] == 0.0))
        self.assertTrue(torch.all(vmask[:, :, :v_n] == 0) and torch.all(vmask[:, :, v_n:] == 1))
        self.assertTrue(torch.all(audio == 0.0), "song locked: the clip before's sound is not copied")
        self.assertTrue(torch.all(amask == 0), "the song stays locked over the whole clip")
        self.assertEqual(tuple(low_head["samples"].shape), (1, 24, v_n, *LOW))
        self.assertTrue(torch.all(low_head["samples"] == 7.5))

    def test_sound_copied_when_not_locked(self):
        (c0, _), (c1, _) = self.chain(0, overlap="39", locked=False), self.chain(1, overlap="39", locked=False)
        self.remember(c0[0], 3.0)
        latent, _, _ = cont(c1[0], av(0.0), "t", False)
        audio = latent["samples"].unbind()[1]
        amask = latent["noise_mask"].unbind()[1]
        a_n = h3_chain.audio_steps(39)
        self.assertEqual(a_n, 65)
        self.assertTrue(torch.all(audio[..., :a_n] == 3.25) and torch.all(audio[..., a_n:] == 0))
        self.assertTrue(torch.all(amask[..., :a_n] == 0) and torch.all(amask[..., a_n:] == 1))

    def test_clip_one_unchanged_and_no_low_head(self):
        (c0, _) = self.chain(0)
        latent = locked_song(av(1.0))
        out, low_head, _ = cont(c0[0], latent, "t", True)
        self.assertIs(out, latent)
        self.assertIsNone(low_head)

    def test_keep_look_puts_clip_one_into_slot_zero(self):
        (c0, _), (c1, _), (c2, _) = self.chain(0), self.chain(1), self.chain(2)
        self.remember(c0[0], 1.0)
        self.remember(c1[0], 2.0)
        latent, low_head, _ = cont(c2[0], locked_song(av(0.0)), "t", True)
        video = latent["samples"].unbind()[0]
        self.assertTrue(torch.all(video[:, :, :1] == 1.0) and torch.all(video[:, :, 1:7] == 2.0))
        self.assertTrue(torch.all(low_head["samples"][:, :, :1] == 1.5) and torch.all(low_head["samples"][:, :, 1:] == 2.5))

    def test_missing_ending_and_size_change(self):
        (c1, _) = self.chain(1)
        with self.assertRaisesRegex(ValueError, "no saved ending"):
            cont(c1[0], locked_song(av(0.0)), "t", False)
        (c0, _) = self.chain(0)
        self.remember(c0[0], 1.0)
        with self.assertRaisesRegex(ValueError, "differ in size"):
            cont(c1[0], locked_song(av(0.0, hw=LOW)), "t", False)

    def test_clip_one_clears_old_endings(self):
        (c0, _), (c1, _) = self.chain(0), self.chain(1)
        self.remember(c0[0], 1.0)
        path = self.remember(c1[0], 2.0)[0]
        self.assertTrue(os.path.isfile(path))
        self.remember(c0[0], 1.0)
        self.assertFalse(os.path.isfile(path))

    def test_song_locked_but_latent_not_locked(self):
        (c0, _) = self.chain(0)
        with self.assertRaisesRegex(ValueError, "not"):
            cont(c0[0], av(0.0), "t", False)

    def test_join_refuses_another_frame_count(self):
        (c0, _) = self.chain(0)
        saved = self.remember(c0[0], 0.0)
        images = torch.zeros((FRAMES * 2 - 1, 4, 4, 3))
        with self.assertRaisesRegex(ValueError, "Interpolation"):
            join(c0[0], images, {"waveform": torch.zeros(1, 1, 100), "sample_rate": 48000}, saved)

    def test_join_fills_a_short_sound_without_silence(self):
        (c0, _) = self.chain(0, overlap="39", locked=False)
        saved = self.remember(c0[0], 0.0)
        sr = 32000
        wave = torch.ones(1, 2, round(FRAMES / 24 * sr) - 267)
        _, audio, _ = join(c0[0], torch.zeros((FRAMES, 4, 4, 3)),
                                                   {"waveform": wave, "sample_rate": sr}, saved)
        self.assertEqual(audio["waveform"].shape[-1], round(FRAMES / 24 * sr))
        self.assertTrue(torch.all(audio["waveform"] == 1.0))

    def test_whole_clip_saved_only_when_asked(self):
        (c0, _), (c1, _) = self.chain(0), self.chain(1)
        rem_node = h3_chain.LunaH3ChainRemember()
        path = rem_node.remember(c0[0], av(3.0, audio_fill=3.25), "t", av(3.5, hw=LOW), save_whole_clip=True, overlap=22)[0][0]
        folder = os.path.dirname(path)
        rem_node.remember(c1[0], av(4.0, audio_fill=4.25), "t", av(4.5, hw=LOW), overlap=22)
        import safetensors.torch
        whole = safetensors.torch.load_file(os.path.join(folder, "whole_0000.latent"))
        video, audio = h3_chain._streams(av(3.0, audio_fill=3.25), "test")
        self.assertEqual(tuple(whole["video"].shape), tuple(video.shape))
        self.assertEqual(tuple(whole["audio"].shape), tuple(audio.shape))
        self.assertTrue(torch.all(whole["low_video"] == 3.5))
        self.assertFalse(os.path.exists(os.path.join(folder, "whole_0001.latent")))

    def test_bad_name(self):
        with self.assertRaisesRegex(ValueError, "letters"):
            h3_chain.ending_path("../x", 0)

    # ------------------------------------------------------------------ Join
    def test_join_cuts_head_and_gathers_sound(self):
        sr = 48000
        pieces = []
        for step in range(3):
            (c, _) = self.chain(step)
            saved = self.remember(c[0], float(step))
            images = torch.full((FRAMES, 4, 4, 3), float(step))
            wave = torch.arange(round(FRAMES / 24 * sr), dtype=torch.float32).reshape(1, 1, -1) + step * 1e6
            out_images, out_audio, clip_audio = join(
                c[0], images, {"waveform": wave, "sample_rate": sr}, saved)
            cut = 0 if step == 0 else 22
            self.assertEqual(out_images.shape[0], FRAMES - cut)
            # clip_audio is this clip's own window on every clip, the last one included.
            self.assertEqual(clip_audio["waveform"].shape[-1], round(out_images.shape[0] / 24 * sr))
            self.assertEqual(float(clip_audio["waveform"][0, 0, 0]), step * 1e6 + round(cut / 24 * sr))
            pieces.append(out_images.shape[0])
        self.assertEqual(out_audio["waveform"].shape[-1], round(sum(pieces) / 24 * sr) + 0 * len(pieces))
        first_of_clip_2 = out_audio["waveform"][0, 0, round(FRAMES / 24 * sr)]
        self.assertEqual(float(first_of_clip_2), 1e6 + round(22 / 24 * sr))


if __name__ == "__main__":
    unittest.main()

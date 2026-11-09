"""Synthetic checkpoints only: never load installed weights or initialize CUDA."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h3_harness  # noqa: E402,F401 -- selects CPU before any ComfyUI import
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import luna_upscaler as up  # noqa: E402


NAME = "minimax_h3_latent_upscaler_3d_fp16.safetensors"


def small_model(**kwargs):
    config = dict(channels=32, in_blocks=2, out_blocks=2, temporal_every=2,
                  temporal_kernel=3, dropout=0.0)
    config.update(kwargs)
    return up.LatentResizer3D(**config).eval()


class UpscalerTests(unittest.TestCase):
    def setUp(self):
        up.MODEL_CACHE.clear()

    def tearDown(self):
        up.MODEL_CACHE.clear()

    def test_model_list_starts_with_none_and_merges_folders(self):
        def names(folder):
            return ([NAME, "nested/" + NAME, "unrelated.safetensors"]
                    if folder == "latent_upscale_models" else [NAME, NAME.replace("fp16", "bf16")])
        with patch.object(up.folder_paths, "get_filename_list", side_effect=names):
            models = up.list_upscaler_models()
        self.assertEqual(models[0], "none")
        self.assertEqual(len(models), 4)
        self.assertIn(NAME, models)
        self.assertIn("nested/" + NAME, models)
        self.assertNotIn("unrelated.safetensors", models)

    def test_model_list_with_missing_folders(self):
        with patch.object(up.folder_paths, "get_filename_list", side_effect=KeyError):
            self.assertEqual(up.list_upscaler_models(), ["none"])

    def test_build_from_synthetic_state_dict_and_keep_time(self):
        for temporal_every in (0, 1, 2, 3):
            with self.subTest(temporal_every=temporal_every):
                state = small_model(in_blocks=4, temporal_every=temporal_every).state_dict()
                restored = up._model_from_state_dict(state)
                with torch.inference_mode():
                    result = restored(torch.randn(2, 24, 3, 2, 3), scale=2,
                                      target_size=(3, 4, 6))
                self.assertEqual(result.shape, (2, 24, 3, 4, 6))
                self.assertTrue(torch.isfinite(result).all())
                self.assertFalse(restored.training)

    def test_scale_only_keeps_time_and_chunked_video_keeps_time(self):
        model = small_model(in_blocks=1, out_blocks=1)
        with torch.inference_mode():
            result = model(torch.randn(1, 24, 3, 2, 2), scale=2)
            chunked = model(torch.randn(1, 24, 35, 2, 2), scale=2,
                            target_size=(35, 3, 5))
        self.assertEqual(result.shape, (1, 24, 3, 4, 4))
        self.assertEqual(chunked.shape, (1, 24, 35, 3, 5))
        self.assertTrue(torch.isfinite(chunked).all())

    def test_prefixed_and_wrapped_state_dict(self):
        state = small_model(temporal_every=0).state_dict()
        wrapped = {"model": {"upscaler." + k: v for k, v in state.items()}}
        wrapped["model"]["unrelated.weight"] = torch.zeros(1)
        model = up._model_from_state_dict(wrapped)
        self.assertEqual(set(model.state_dict()), set(state))

    def test_incompatible_weights_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "3D H3"):
            up._model_from_state_dict({"conv_in.weight": torch.zeros(32, 24, 3, 3)})

    def test_loads_once_on_cpu_and_returns_exact_requested_grid(self):
        state = small_model().state_dict()
        z = torch.randn(1, 24, 3, 2, 3, dtype=torch.float64)
        original = z.clone()
        with patch.object(up.folder_paths, "get_full_path", return_value="synthetic.safetensors"), \
                patch.object(up.comfy.utils, "load_torch_file", return_value=state) as load, \
                patch.object(up.mm, "get_torch_device", return_value=torch.device("cpu")), \
                patch.object(up.mm, "soft_empty_cache") as empty:
            for hw in ((5, 7), (3, 4)):
                result = up.learned_latent_lift(z, hw, NAME)
                self.assertEqual(result.shape, (1, 24, 3, *hw))
                self.assertEqual(result.device, z.device)
                self.assertEqual(result.dtype, z.dtype)
                self.assertTrue(torch.isfinite(result).all())
                self.assertFalse(result.requires_grad)
            load.assert_called_once_with("synthetic.safetensors", safe_load=True,
                                         device=torch.device("cpu"))
            self.assertEqual(empty.call_count, 2)
        self.assertTrue(torch.equal(z, original))
        self.assertEqual(next(up.MODEL_CACHE[NAME].parameters()).device.type, "cpu")

    def test_vae_normalization_and_final_resize(self):
        class Echo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(()))

            def forward(self, x, **kwargs):
                seen.append(x.clone())
                return x

        seen = []
        up.MODEL_CACHE[NAME] = Echo()
        z = torch.randn(1, 24, 3, 2, 3)
        with patch.object(up.mm, "get_torch_device", return_value=torch.device("cpu")), \
                patch.object(up.mm, "soft_empty_cache"):
            result = up.learned_latent_lift(z, (5, 7), NAME)
        mean = torch.tensor(up.LATENTS_MEAN).view(1, 24, 1, 1, 1)
        std = torch.tensor(up.LATENTS_STD).view(1, 24, 1, 1, 1)
        torch.testing.assert_close(seen[0], (z - mean) / std)
        torch.testing.assert_close(result, F.interpolate(z, (3, 5, 7), mode="trilinear", align_corners=False))

    def test_cleanup_even_when_inference_fails(self):
        model = small_model()
        up.MODEL_CACHE[NAME] = model
        with patch.object(up.mm, "get_torch_device", return_value=torch.device("cpu")), \
                patch.object(up.mm, "soft_empty_cache") as empty, \
                patch.object(model, "forward", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                up.learned_latent_lift(torch.randn(1, 24, 2, 2, 2), (4, 4), NAME)
        empty.assert_called_once()
        self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_invalid_input_and_missing_model(self):
        with self.assertRaisesRegex(ValueError, "24"):
            up.learned_latent_lift(torch.zeros(1, 4, 2, 2, 2), (4, 4), NAME)
        with self.assertRaisesRegex(ValueError, "positive"):
            up.learned_latent_lift(torch.zeros(1, 24, 2, 2, 2), (0, 4), NAME)
        with patch.object(up.folder_paths, "get_full_path", return_value=None):
            with self.assertRaises(FileNotFoundError):
                up.learned_latent_lift(torch.zeros(1, 24, 2, 2, 2), (4, 4), NAME)


class TemporalPadTests(unittest.TestCase):
    """The chunk path's halos mirror the clip instead of repeating frame 0 (selflift-Avatar issue #5)."""

    def clip(self, frames):
        return torch.arange(frames, dtype=torch.float32).view(1, 1, frames, 1, 1).expand(1, 2, frames, 2, 2).clone()

    def test_head_is_mirrored_not_a_hold_of_frame_0(self):
        padded = up.temporal_pad(self.clip(72), 5)
        self.assertEqual(padded[0, 0, :7, 0, 0].tolist(), [5, 4, 3, 2, 1, 0, 1])
        self.assertEqual(padded[0, 0, -6:, 0, 0].tolist(), [71, 70, 69, 68, 67, 66])

    def test_frame_0_appears_once_in_the_head(self):
        head = up.temporal_pad(self.clip(72), 5)[0, 0, :10, 0, 0]
        self.assertEqual(int((head == 0).sum()), 1)

    def test_clip_shorter_than_the_overlap_falls_back_to_replicate(self):
        padded = up.temporal_pad(self.clip(4), 5)
        self.assertEqual(padded.shape[2], 14)
        self.assertEqual(padded[0, 0, :5, 0, 0].tolist(), [0, 0, 0, 0, 0])

    def test_no_overlap_is_unchanged(self):
        x = self.clip(10)
        self.assertTrue(torch.equal(up.temporal_pad(x, 0), x))

    def test_chunk_path_uses_the_mirror(self):
        calls = []
        real = up.temporal_pad

        def spy(x, overlap):
            calls.append((x.shape[2], overlap))
            return real(x, overlap)

        with patch.object(up, "temporal_pad", spy):
            model = small_model(in_blocks=1, out_blocks=1)
            with torch.no_grad():
                model(torch.randn(1, 24, 40, 2, 2), scale=2)
        self.assertTrue(calls and calls[0][0] == 40)


class TemporalWindowTests(unittest.TestCase):
    """Each chunk reads `overlap` frames of context on both sides of the frames it writes."""

    def test_every_window_has_context_on_both_sides(self):
        T, chunk, overlap = 72, 32, 5
        windows = list(up.temporal_windows(T, chunk, overlap))
        self.assertEqual([w[:2] for w in windows], [(0, 32), (32, 64), (64, 72)])
        for seg_start, seg_end, out_start, out_end, lo, hi in windows:
            # padded index p is clip frame p - overlap
            self.assertEqual(out_start - (lo - overlap), overlap)
            self.assertEqual((hi - overlap) - out_end, overlap)
        self.assertEqual(windows[-1][5], T + 2 * overlap)  # the tail reads its mirrored halo

    def test_written_frames_cover_the_clip(self):
        covered = set()
        for _, _, out_start, out_end, _, _ in up.temporal_windows(72, 32, 5):
            covered.update(range(out_start, out_end))
        self.assertEqual(covered, set(range(72)))

    def test_chunked_equals_one_pass_for_a_local_operator(self):
        """With an operator that sees one frame either side, chunking must give the same frames as one pass."""
        def local_seg(x, scale, size):
            kernel = torch.full((x.shape[1], 1, 3, 1, 1), 1.0 / 3, dtype=x.dtype)
            x = F.conv3d(x, kernel, padding=(1, 0, 0), groups=x.shape[1])
            return F.interpolate(x, size=size, mode="nearest")

        model = small_model(in_blocks=1, out_blocks=1)
        overlap = model.in_blocks[1].dwconv.weight.shape[2]
        x = torch.randn(1, 24, 72, 2, 2)
        with patch.object(model, "_forward_seg", side_effect=local_seg), torch.no_grad():
            chunked = model(x, scale=2, target_size=(72, 4, 4))
            whole = local_seg(up.temporal_pad(x, overlap), 1.0, (72 + 2 * overlap, 4, 4))[:, :, overlap:-overlap]
        torch.testing.assert_close(chunked, whole)


if __name__ == "__main__":
    unittest.main()

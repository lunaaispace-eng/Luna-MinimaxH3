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


if __name__ == "__main__":
    unittest.main()

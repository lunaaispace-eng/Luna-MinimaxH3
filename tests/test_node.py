import gc
import sys
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h3_harness import FakeVAE, av_latent, empty_cond, sampler, tiny_h3_model  # noqa: E402

import torch  # noqa: E402

import comfy.model_management  # noqa: E402
import comfy.model_sampling  # noqa: E402
import comfy.samplers  # noqa: E402
import luna_sampler  # noqa: E402
from luna_sampler import LunaH3Sampler  # noqa: E402

DMAD = "MiniMaxH3/minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors"

INPUT_ORDER = [
    "model", "positive", "negative", "vae", "latent_image",
    "lora_low", "strength_low", "shift_low", "audio_shift_low", "sampler_low", "cfg_low",
    "lora_high", "strength_high", "shift_high", "audio_shift_high", "sampler_high", "cfg_high",
    "scheduler", "total_steps", "transition", "highres_denoise", "noise_seed",
    "lowres_scale", "upscaler_model", "rho", "w_min", "w_max",
    "low_res_head", "sigmas_low", "sigmas_high", "high_res_model", "positive_low", "negative_low",
]


def streams(latent):
    return latent["samples"].unbind()


class NodeTests(unittest.TestCase):
    def schema(self):
        return LunaH3Sampler.define_schema()

    def test_schema_names_order_defaults_and_tooltips(self):
        schema = self.schema()
        self.assertEqual(schema.node_id, "Luna-H3-Sampler")
        self.assertEqual(schema.display_name, "Luna-H3-Sampler")
        self.assertEqual(schema.category, "Luna-MinimaxH3")
        self.assertIn("one trajectory", schema.description)
        self.assertIn("finishes in pass 2", schema.description)
        self.assertIn("locked song", schema.description)
        self.assertIn("low_res_head", schema.description)

        self.assertEqual([item.id for item in schema.inputs], INPUT_ORDER)
        for item in schema.inputs:
            self.assertIsInstance(item.tooltip, str)
            self.assertTrue(item.tooltip.strip(), item.id)
        self.assertEqual([item.id for item in schema.outputs], ["latent", "low_res_latent"])
        for item in schema.outputs:
            self.assertTrue(item.tooltip.strip(), item.id)

        by_id = {item.id: item for item in schema.inputs}
        self.assertEqual(by_id["lora_low"].options[0], "none")
        self.assertEqual(by_id["lora_low"].default, "none")
        self.assertEqual(by_id["lora_high"].default, "none")
        self.assertEqual(by_id["strength_low"].default, 0.8)
        self.assertEqual((by_id["strength_low"].min, by_id["strength_low"].max), (0.0, 2.0))
        self.assertEqual(by_id["strength_high"].default, 0.8)
        self.assertEqual(by_id["shift_low"].default, 12.0)
        self.assertEqual(by_id["audio_shift_low"].default, 3.0)
        self.assertEqual(by_id["shift_high"].default, 12.0)
        self.assertEqual(by_id["audio_shift_high"].default, 3.0)
        self.assertEqual(by_id["cfg_low"].default, 1.0)
        self.assertEqual(by_id["cfg_high"].default, 1.0)
        self.assertEqual(by_id["scheduler"].options, list(comfy.samplers.SCHEDULER_NAMES))
        self.assertEqual(by_id["scheduler"].default, "simple")
        self.assertEqual(by_id["total_steps"].default, 12)
        self.assertEqual((by_id["total_steps"].min, by_id["total_steps"].max), (1, 200))
        self.assertEqual(by_id["transition"].default, 8)
        self.assertEqual((by_id["transition"].min, by_id["transition"].max), (1, 199))
        self.assertEqual(by_id["highres_denoise"].default, 1.0)
        self.assertEqual((by_id["highres_denoise"].min, by_id["highres_denoise"].max), (0.01, 1.0))
        self.assertEqual(by_id["lowres_scale"].default, 0.5)
        self.assertEqual((by_id["lowres_scale"].min, by_id["lowres_scale"].max), (0.25, 1.0))
        self.assertEqual(by_id["rho"].default, 0.0)
        self.assertEqual(by_id["w_min"].default, 0.5)
        self.assertEqual(by_id["w_max"].default, 1.0)
        self.assertTrue(by_id["low_res_head"].optional)
        self.assertTrue(by_id["low_res_head"].lazy)
        self.assertTrue(by_id["sigmas_low"].optional)
        self.assertTrue(by_id["sigmas_high"].optional)
        self.assertFalse(by_id["sigmas_low"].lazy)
        self.assertTrue(by_id["high_res_model"].optional)
        self.assertFalse(by_id["high_res_model"].lazy)
        self.assertTrue(by_id["positive_low"].optional)
        self.assertTrue(by_id["negative_low"].optional)

    def test_low_res_head_is_lazy_only_when_connected(self):
        self.assertEqual(LunaH3Sampler.check_lazy_status(low_res_head=None), ["low_res_head"])
        self.assertEqual(LunaH3Sampler.check_lazy_status(), [])
        self.assertEqual(LunaH3Sampler.check_lazy_status(low_res_head={"samples": None}), [])

    def test_upscaler_widget_default_with_and_without_model(self):
        name = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
        for names, expected in ((["none"], "none"), (["none", name], name)):
            with self.subTest(names=names), patch("luna_sampler.list_upscaler_models", return_value=names):
                widget = next(item for item in self.schema().inputs if item.id == "upscaler_model")
                self.assertEqual(widget.options, names)
                self.assertEqual(widget.default, expected)
                self.assertTrue(widget.tooltip)

    def test_upscaler_none_passes_none_and_name_passes_callable(self):
        name = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
        seen = []
        z = torch.zeros(1, 24, 2, 2, 2)

        def sample(*args, **kwargs):
            lifter = kwargs["latent_lifter"]
            seen.append(lifter)
            if lifter is not None:
                self.assertIs(lifter(z, (4, 4)), z)
            return args[4], args[4]

        with patch("luna_sampler.two_pass_sample", sample), \
                patch("luna_sampler.learned_latent_lift", return_value=z) as learned:
            plain = LunaH3Sampler.execute(**self.kwargs(upscaler_model="none"))
            learned.assert_not_called()
            selected = LunaH3Sampler.execute(**self.kwargs(upscaler_model=name))
            learned.assert_called_once_with(z, (4, 4), name)
        self.assertIsNone(seen[0])
        self.assertTrue(callable(seen[1]))
        self.assertIn("upscaler: none (nearest)", plain.ui.as_dict()["text"][0])
        self.assertIn("upscaler: " + name, selected.ui.as_dict()["text"][0])

    def test_learned_upscaler_runs_between_real_cpu_passes(self):
        import luna_upscaler

        name = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
        upscaler = luna_upscaler.LatentResizer3D(
            channels=32, in_blocks=1, out_blocks=1, temporal_kernel=3).eval().requires_grad_(False)
        source = av_latent(audio_mask=torch.zeros(1, 1, 2, 8))
        with patch.dict(luna_upscaler.MODEL_CACHE, {name: upscaler}, clear=True), \
                patch("luna_sampler.learned_latent_lift", wraps=luna_upscaler.learned_latent_lift) as learned:
            result = LunaH3Sampler.execute(**self.kwargs(latent_image=source, upscaler_model=name))
        learned.assert_called_once()
        self.assertEqual(learned.call_args.args[1:], ((4, 4), name))
        video, audio = streams(result[0])
        low_video, low_audio = streams(result[1])
        self.assertEqual(video.shape, streams(source)[0].shape)
        self.assertEqual(low_video.shape[-2:], (2, 2))
        self.assertTrue(torch.isfinite(video).all())
        self.assertTrue(torch.equal(audio, streams(source)[1]))
        self.assertTrue(torch.equal(low_audio, audio))

    def kwargs(self, model=None, **overrides):
        args = dict(
            model=model if model is not None else tiny_h3_model(),
            positive=empty_cond(),
            negative=empty_cond(),
            vae=FakeVAE(),
            latent_image=av_latent(),
            lora_low="none",
            strength_low=0.8,
            shift_low=12.0,
            audio_shift_low=3.0,
            sampler_low=sampler("euler"),
            cfg_low=1.0,
            lora_high="none",
            strength_high=0.8,
            shift_high=12.0,
            audio_shift_high=3.0,
            sampler_high=sampler("euler"),
            cfg_high=1.0,
            scheduler="simple",
            total_steps=4,
            transition=2,
            highres_denoise=1.0,
            noise_seed=0,
            lowres_scale=0.5,
            rho=0.0,
            w_min=0.5,
            w_max=1.0,
        )
        args.update(overrides)
        return args

    def test_execute_returns_two_latents_and_readout(self):
        model = tiny_h3_model()
        source = av_latent()
        with patch("luna_sampler.comfy.utils.load_torch_file") as load_file, \
                patch("luna_sampler.comfy.sd.load_lora_for_models") as load_lora:
            result = LunaH3Sampler.execute(**self.kwargs(model=model, latent_image=source, shift_high=5.0, audio_shift_high=2.0))
        load_file.assert_not_called()
        load_lora.assert_not_called()
        self.assertEqual(streams(result[0])[0].shape, streams(source)[0].shape)
        self.assertEqual(streams(result[0])[1].shape, streams(source)[1].shape)
        self.assertEqual(tuple(streams(result[1])[0].shape[-2:]), (2, 2))
        self.assertTrue(torch.isfinite(streams(result[0])[0]).all())
        self.assertTrue(torch.isfinite(streams(result[0])[1]).all())
        text = result.ui.as_dict()["text"][0]
        self.assertIn("pass 1: steps 1-2 (low) / pass 2: steps 3-4 (high)", text)
        self.assertIn("pass 1: none × 0.80, shift 12.0 / 3.0, sigmas", text)
        self.assertIn("pass 2: none × 0.80, shift 5.0 / 2.0, sigmas", text)
        self.assertIn("hand-over sigma", text)
        self.assertNotIn("no recipe", text)
        self.assertEqual(model.get_model_object("model_sampling").shift, 12.0)
        self.assertEqual(model.get_model_object("model_sampling").audio_shift, 3.0)

    def test_per_arm_shift_lands_on_that_arm(self):
        seen = {}
        real = luna_sampler.two_pass_sample

        def spy(*args, **kwargs):
            seen["low"] = args[0]
            seen["high"] = kwargs["high_res_model"]
            seen["denoise"] = kwargs["highres_denoise"]
            return real(*args, **kwargs)

        with patch("luna_sampler.two_pass_sample", spy):
            LunaH3Sampler.execute(**self.kwargs(shift_low=7.0, audio_shift_low=4.0, shift_high=5.0, audio_shift_high=2.0))

        for arm, shift, audio in ((seen["low"], 7.0, 4.0), (seen["high"], 5.0, 2.0)):
            sampling = arm.get_model_object("model_sampling")
            self.assertIsInstance(sampling, comfy.model_sampling.CONST)
            self.assertEqual(sampling.shift, shift)
            self.assertEqual(sampling.audio_shift, audio)
            options = arm.model_options["transformer_options"]
            self.assertEqual(options["minimax_h3_sigma_shift_video"], shift)
            self.assertEqual(options["minimax_h3_sigma_shift_audio"], audio)
        self.assertIsNone(seen["denoise"])
        self.assertIsNot(seen["low"], seen["high"])

    def test_high_res_model_builds_pass_2_and_empty_uses_model(self):
        real = luna_sampler.two_pass_sample

        def run(**overrides):
            seen = {}

            def spy(*args, **kwargs):
                seen["low"] = args[0]
                seen["high"] = kwargs["high_res_model"]
                return real(*args, **kwargs)

            with patch("luna_sampler.two_pass_sample", spy):
                result = LunaH3Sampler.execute(**self.kwargs(**overrides))
            return seen, result.ui.as_dict()["text"][0]

        base, second = tiny_h3_model(), tiny_h3_model()
        seen, text = run(model=base, high_res_model=second, shift_high=5.0, audio_shift_high=2.0)
        self.assertTrue(seen["low"].is_clone(base))
        self.assertTrue(seen["high"].is_clone(second))
        self.assertFalse(seen["high"].is_clone(base))
        sampling = seen["high"].get_model_object("model_sampling")
        self.assertEqual((sampling.shift, sampling.audio_shift), (5.0, 2.0))
        self.assertIn("pass 2 model: high_res_model", text)
        self.assertEqual(second.get_model_object("model_sampling").shift, 12.0)

        seen, text = run(model=base)
        self.assertTrue(seen["high"].is_clone(base))
        self.assertIn("pass 2 model: model", text)

    def test_named_lora_loads_and_none_does_not(self):
        calls = []

        def fake_load(path, safe_load=True):
            calls.append(("file", path, safe_load))
            return {"lora": True}

        def fake_apply(model, clip, lora, strength, strength_clip):
            calls.append(("apply", clip, strength, strength_clip))
            return model, None

        def fake_sample(*args, **kwargs):
            return args[4], args[4]

        with patch("luna_sampler.folder_paths.get_full_path_or_raise", return_value="D:/fake/lora.safetensors"), \
                patch("luna_sampler.comfy.utils.load_torch_file", fake_load), \
                patch("luna_sampler.comfy.sd.load_lora_for_models", fake_apply), \
                patch("luna_sampler.two_pass_sample", fake_sample):
            result = LunaH3Sampler.execute(**self.kwargs(
                lora_low="none", lora_high=DMAD, total_steps=12, transition=8,
                sampler_high=sampler("er_sde"), audio_shift_high=2.0))
            warned = LunaH3Sampler.execute(**self.kwargs(
                lora_low="none", lora_high=DMAD, total_steps=12, transition=8, sampler_high=sampler("er_sde")))
        self.assertEqual(calls, [
            ("file", "D:/fake/lora.safetensors", True), ("apply", None, 0.8, 0),
            ("file", "D:/fake/lora.safetensors", True), ("apply", None, 0.8, 0),
        ])
        self.assertNotIn("no recipe", result.ui.as_dict()["text"][0])
        self.assertIn(f"pass 2 {DMAD} recipe: audio_shift 2.0, set 3.0", warned.ui.as_dict()["text"][0])

    def test_highres_denoise_one_is_none_and_overrides_pass_through(self):
        seen = {}

        def fake_sample(*args, **kwargs):
            seen["denoise"] = kwargs["highres_denoise"]
            seen["sigmas_low"] = kwargs["sigmas_low"]
            seen["sigmas_high"] = kwargs["sigmas_high"]
            return args[4], args[4]

        custom = torch.tensor([1.0, 0.5, 0.0])
        with patch("luna_sampler.two_pass_sample", fake_sample):
            LunaH3Sampler.execute(**self.kwargs(highres_denoise=1.0, sigmas_low=custom, sigmas_high=custom))
        self.assertIsNone(seen["denoise"])
        self.assertTrue(torch.equal(seen["sigmas_low"], custom))
        self.assertTrue(torch.equal(seen["sigmas_high"], custom))

        with patch("luna_sampler.two_pass_sample", fake_sample):
            text = LunaH3Sampler.execute(**self.kwargs(highres_denoise=0.5)).ui.as_dict()["text"][0]
        self.assertEqual(seen["denoise"], 0.5)
        self.assertIn("highres_denoise 0.50", text)

    def test_impossible_transition_raises(self):
        with patch("luna_sampler.two_pass_sample") as sample:
            with self.assertRaisesRegex(ValueError, "transition 4 is not in \\[1, 3\\]"):
                LunaH3Sampler.execute(**self.kwargs(total_steps=4, transition=4))
            with self.assertRaisesRegex(ValueError, "pass 1 strength 2.5 is outside \\[0, 2\\]"):
                LunaH3Sampler.execute(**self.kwargs(strength_low=2.5))
        sample.assert_not_called()

    def test_cancelled_arm_hands_its_loaded_entry_back_to_the_input_model(self):
        """A cancelled run frees the arm's clones in one collection; ComfyUI's entry must not be orphaned."""
        base = tiny_h3_model()
        gc.collect()
        gc.disable()
        try:
            arm = luna_sampler._arm_model(base, "none", 0.0, 12.0, 3.0, {})
            entry = comfy.model_management.LoadedModel(arm)
            entry.real_model = weakref.ref(base.model)
            run = {"arm": arm}
            run["self"] = run  # the reference cycle a dead run leaves behind
            del arm, run
        finally:
            gc.enable()
        gc.collect()
        self.assertFalse(entry.is_dead())
        self.assertIs(entry.model, base)


if __name__ == "__main__":
    unittest.main()

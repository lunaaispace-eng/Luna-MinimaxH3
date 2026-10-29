"""Integration smoke tests: real ComfyUI H3 model and samplers, entirely on CPU."""

import unittest

from h3_harness import FakeVAE, av_latent, empty_cond, sampler, sigmas, tiny_h3_model

import torch
import comfy.model_base
import comfy.model_management
import comfy.model_sampling
import comfy.sample
import comfy.samplers


class HarnessTests(unittest.TestCase):
    def run_sampler(self, name, latent):
        model = tiny_h3_model()
        samples = latent["samples"]
        noise = comfy.sample.prepare_noise(samples, seed=17)
        with torch.inference_mode():
            output = comfy.samplers.sample(
                model, noise, empty_cond(), empty_cond(), cfg=1.0,
                device=model.load_device, sampler=sampler(name), sigmas=sigmas(4),
                latent_image=samples, denoise_mask=latent.get("noise_mask"),
                disable_pbar=True, seed=17,
            )
        self.assertTrue(output.is_nested)
        self.assertEqual(len(output.unbind()), 2)
        for actual, original in zip(output.unbind(), samples.unbind()):
            self.assertEqual(actual.shape, original.shape)
            self.assertEqual(actual.device.type, "cpu")
            self.assertTrue(torch.isfinite(actual).all().item())
        return output

    def test_real_samplers_preserve_nested_shapes(self):
        for name in ("euler", "res_multistep", "er_sde"):
            with self.subTest(sampler=name):
                latent = av_latent()
                output = self.run_sampler(name, latent)
                self.assertFalse(torch.equal(output.unbind()[0], latent["samples"].unbind()[0]))
                self.assertFalse(torch.equal(output.unbind()[1], latent["samples"].unbind()[1]))

    def test_zero_audio_mask_preserves_audio_exactly(self):
        for name in ("euler", "res_multistep", "er_sde"):
            with self.subTest(sampler=name):
                latent = av_latent(audio_mask=torch.zeros((1, 1, 2, 8)))
                original = [stream.clone() for stream in latent["samples"].unbind()]
                output = self.run_sampler(name, latent)
                # ComfyUI keeps mask-0 audio only up to float32 roundoff (MEASURED 2026-10-28:
                # euler 7e-9, res_multistep 7e-9, er_sde exact). Luna-H3-Sampler restores it bit for bit.
                audio = output.unbind()[1]
                error = (audio - original[1]).abs().max().item()
                self.assertLess(error, 1e-6, f"{name}: masked audio max absolute error {error}")
                self.assertFalse(torch.equal(output.unbind()[0], original[0]))
                for actual, before in zip(latent["samples"].unbind(), original):
                    self.assertTrue(torch.equal(actual, before))

    def test_model_is_real_cpu_h3_with_deterministic_weights_and_shifts(self):
        rng = torch.random.get_rng_state().clone()
        a = tiny_h3_model(shift=8.0, audio_shift=2.0, seed=7)
        b = tiny_h3_model(shift=8.0, audio_shift=2.0, seed=7)
        c = tiny_h3_model(seed=8)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        self.assertIsInstance(a.model, comfy.model_base.MiniMaxH3)
        self.assertIsInstance(a.model.model_sampling, comfy.model_sampling.ModelSamplingAV)
        self.assertTrue(comfy.model_management.cpu_mode())
        self.assertEqual(a.load_device, torch.device("cpu"))
        self.assertEqual(a.offload_device, torch.device("cpu"))
        self.assertEqual(a.model.model_sampling.shift, 8.0)
        self.assertEqual(a.model.model_sampling.audio_shift, 2.0)
        self.assertEqual(a.model.diffusion_model.sigma_shift_video, 8.0)
        self.assertEqual(a.model.diffusion_model.sigma_shift_audio, 2.0)
        for key, value in a.model.state_dict().items():
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue(torch.isfinite(value).all().item(), key)
            self.assertTrue(torch.equal(value, b.model.state_dict()[key]), key)
        self.assertFalse(torch.equal(a.model.diffusion_model.video_patch_proj.weight,
                                     c.model.diffusion_model.video_patch_proj.weight))

    def test_latent_audio_duration_masks_and_seed(self):
        for t, audio_t in ((1, 2), (2, 8), (7, 37)):
            with self.subTest(t=t):
                latent = av_latent(t=t)
                self.assertEqual(latent["samples"].unbind()[0].shape, (1, 24, t, 4, 4))
                self.assertEqual(latent["samples"].unbind()[1].shape, (1, 32, 2, audio_t))
                self.assertNotIn("noise_mask", latent)
        a = av_latent(audio_t=11, video_mask=torch.zeros((1, 1, 2, 4, 4)), seed=2)
        b = av_latent(audio_t=11, seed=2)
        self.assertTrue(torch.equal(a["samples"].unbind()[1], b["samples"].unbind()[1]))
        self.assertTrue(a["noise_mask"].is_nested)
        self.assertTrue(torch.all(a["noise_mask"].unbind()[0] == 0).item())
        self.assertTrue(torch.all(a["noise_mask"].unbind()[1] == 1).item())
        audio_only = av_latent(audio_mask=torch.zeros((1, 1, 2, 8)))
        self.assertTrue(torch.all(audio_only["noise_mask"].unbind()[0] == 1).item())

    def test_fake_vae_round_trip_shapes(self):
        vae = FakeVAE()
        for t, frames in ((1, 1), (2, 5), (7, 22)):
            with self.subTest(t=t):
                video = av_latent(t=t, h=2, w=3)["samples"].unbind()[0]
                pixels = vae.decode(video)
                self.assertEqual(pixels.shape, (1, frames, 32, 48, 3))
                self.assertTrue(torch.equal(pixels, vae.decode(video)))
                encoded = vae.encode(pixels)
                self.assertEqual(encoded.shape, video.shape)
                self.assertTrue(torch.equal(encoded, vae.encode(pixels)))
                self.assertEqual(vae.encode(pixels[0]).shape, video.shape)

    def test_sampler_and_sigma_helpers(self):
        self.assertIsInstance(sampler("euler"), comfy.samplers.KSAMPLER)
        schedule = sigmas(4, start=0.8)
        self.assertEqual(schedule.shape, (5,))
        self.assertAlmostEqual(schedule[0].item(), 0.8)
        self.assertEqual(schedule[-1].item(), 0.0)
        self.assertTrue(torch.all(schedule[:-1] > schedule[1:]).item())
        self.assertEqual(schedule.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()

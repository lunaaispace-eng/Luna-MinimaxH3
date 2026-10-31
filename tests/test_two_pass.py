import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h3_harness import FakeVAE, av_latent, empty_cond, sampler, tiny_h3_model  # noqa: E402  (sets CPU first)

import torch  # noqa: E402

import comfy.nested_tensor  # noqa: E402
import comfy.samplers  # noqa: E402
import luna_two_pass  # noqa: E402
from luna_two_pass import arm_sigmas, audio_handover, two_pass_sample  # noqa: E402


def nearest_lift(z0_low, vae, out_hw, rho, w_min, w_max, latent_format, latent_lifter=None, mask=None):
    return torch.nn.functional.interpolate(z0_low, size=(z0_low.shape[2],) + tuple(out_hw), mode="nearest")


def streams(latent):
    return latent["samples"].unbind()


class TwoPassTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_h3_model()
        self.vae = FakeVAE()

    def run_two_pass(self, latent=None, low="euler", high="euler", total=4, transition=2, scale=0.5, **kw):
        latent = latent if latent is not None else av_latent()
        return two_pass_sample(self.model, empty_cond(), empty_cond(), self.vae, latent, sampler(low), sampler(high),
                               "simple", total, transition, 0, 1.0, 1.0, scale, 0.0, 0.5, 1.0, lift=nearest_lift, **kw)

    def spy_sigmas(self, **kw):
        seen = []
        real = luna_two_pass._sample

        def spy(model, noise, latent, masks, nested, positive, negative, cfg, smp, sigmas, seed, on_x0=None):
            seen.append((model, sigmas.clone(), latent.unbind()[0].clone()))
            return real(model, noise, latent, masks, nested, positive, negative, cfg, smp, sigmas, seed, on_x0)

        with patch.object(luna_two_pass, "_sample", spy):
            result = self.run_two_pass(**kw)
        return seen, result

    def test_any_sampler_both_passes(self):
        source = streams(av_latent())
        for low, high in (("euler", "euler"), ("res_multistep", "er_sde"), ("er_sde", "res_multistep")):
            with self.subTest(low=low, high=high):
                latent, low_latent = self.run_two_pass(low=low, high=high)
                self.assertEqual(streams(latent)[0].shape, source[0].shape)
                self.assertEqual(streams(latent)[1].shape, source[1].shape)
                self.assertTrue(torch.isfinite(streams(latent)[0]).all())
                self.assertTrue(torch.isfinite(streams(latent)[1]).all())

    def test_one_schedule_split_at_transition(self):
        seen, _ = self.spy_sigmas(total=8, transition=5)
        full = comfy.samplers.calculate_sigmas(self.model.get_model_object("model_sampling"), "simple", 8)
        self.assertTrue(torch.equal(seen[0][1], full[:6]))
        self.assertTrue(torch.equal(seen[1][1], full[5:]))

    def test_each_arm_on_its_own_shift(self):
        high = tiny_h3_model(shift=5.0, audio_shift=2.0)
        seen, (latent, low_latent) = self.spy_sigmas(total=8, transition=5, high_res_model=high)
        own = comfy.samplers.calculate_sigmas(high.get_model_object("model_sampling"), "simple", 8)
        self.assertTrue(torch.equal(seen[1][1], own[5:]))
        self.assertTrue(torch.isfinite(streams(latent)[1]).all())

    def test_highres_denoise_sets_the_start(self):
        seen, _ = self.spy_sigmas(total=8, transition=5, highres_denoise=0.5)
        expected = arm_sigmas(self.model, "simple", 8, 5, low=False, denoise=0.5)
        self.assertEqual(seen[1][1].numel(), 4)
        self.assertTrue(torch.equal(seen[1][1], expected))

    def test_highres_denoise_one_is_the_transition_slice(self):
        a = arm_sigmas(self.model, "simple", 8, 5, low=False, denoise=1.0)
        b = arm_sigmas(self.model, "simple", 8, 5, low=False)
        self.assertTrue(torch.equal(a, b))
        lower = arm_sigmas(self.model, "simple", 8, 5, low=False, denoise=0.5)
        self.assertLess(float(lower[0]), float(b[0]))

    def test_soft_video_mask_stays_soft_in_pass_1(self):
        mask = torch.full((1, 1, 2, 4, 4), 0.25)
        _, low_latent = self.run_two_pass(av_latent(video_mask=mask))
        self.assertTrue(torch.allclose(low_latent["noise_mask"].unbind()[0], torch.full((1, 24, 2, 2, 2), 0.25)))

    def test_bad_overrides_refused(self):
        for bad in (torch.tensor([1.0, 1.0]), torch.tensor([1.0, float("nan")]), torch.tensor([[1.0, 0.5]]),
                    torch.tensor([0.5, 0.8])):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "sigmas_high"):
                    self.run_two_pass(sigmas_high=bad)

    def test_audio_noise_divided_by_noise_scale(self):
        def pass2_audio_noise(scale):
            high = tiny_h3_model()
            high.get_model_object("model_sampling").noise_scale = scale
            seen = []
            real = luna_two_pass._sample

            def spy(model, noise, *args, **kwargs):
                seen.append(noise.unbind()[1].clone())
                return real(model, noise, *args, **kwargs)

            with patch.object(luna_two_pass, "_sample", spy):
                self.run_two_pass(high_res_model=high)
            return seen[1]

        self.assertTrue(torch.allclose(pass2_audio_noise(2.0) * 2.0, pass2_audio_noise(1.0), atol=1e-6))

    def test_transition_must_be_part_way(self):
        for t in (0, 4):
            with self.assertRaisesRegex(ValueError, "transition"):
                self.run_two_pass(total=4, transition=t)

    def test_audio_handover_same_arm_is_identity(self):
        g = torch.Generator().manual_seed(1)
        x0, eps = torch.randn(1, 32, 2, 8, generator=g), torch.randn(1, 32, 2, 8, generator=g)
        sigma, scale = 0.4, 4.0
        state = (1 - sigma) * x0 + sigma * eps
        x0_vae, eps_out = audio_handover(state, x0, sigma, sigma, scale, scale)
        rebuilt = (1 - sigma) * (x0_vae * scale) + sigma * eps_out
        self.assertTrue(torch.allclose(rebuilt, state, atol=1e-6))

    def test_low_res_latent_carries_final_audio(self):
        latent, low_latent = self.run_two_pass()
        self.assertTrue(torch.equal(streams(latent)[1], streams(low_latent)[1]))

    def test_locked_song_kept_in_both_outputs(self):
        source = av_latent(audio_mask=torch.zeros((1, 1, 2, 8)))
        song = streams(source)[1].clone()
        latent, low_latent = self.run_two_pass(source)
        self.assertTrue(torch.equal(streams(latent)[1], song))
        self.assertTrue(torch.equal(streams(low_latent)[1], song))

    def test_low_res_latent_grid_and_mask(self):
        latent, low_latent = self.run_two_pass(av_latent(video_mask=torch.ones((1, 1, 2, 4, 4))))
        h = max(2, round(4 * 0.5 / 2) * 2)
        self.assertEqual(tuple(streams(low_latent)[0].shape[-2:]), (h, h))
        self.assertEqual(tuple(low_latent["noise_mask"].unbind()[0].shape[-2:]), (h, h))

    def test_kept_video_exact_with_locked_song(self):
        mask = torch.ones((1, 1, 2, 4, 4))
        mask[:, :, 0] = 0
        source = av_latent(video_mask=mask, audio_mask=torch.zeros((1, 1, 2, 8)))
        video, song = (s.clone() for s in streams(source))
        latent, _ = self.run_two_pass(source)
        self.assertTrue(torch.equal(streams(latent)[0][:, :, 0], video[:, :, 0]))
        self.assertFalse(torch.equal(streams(latent)[0][:, :, 1], video[:, :, 1]))
        self.assertTrue(torch.equal(streams(latent)[1], song))

    def run_with_keyframes(self, **kw):
        """Conditioning as MiniMaxH3ImageToVideo builds it; returns the keyframe latent each pass saw."""
        seen = []
        real = luna_two_pass._sample

        def spy(model, noise, latent, masks, nested, positive, negative, cfg, smp, sigmas, seed, on_x0=None):
            seen.append((positive[0][1]["minimax_keyframes"][0]["latent"].clone(),
                         negative[0][1]["minimax_keyframes"][0]["latent"].clone()))
            return real(model, noise, latent, masks, nested, empty_cond(), empty_cond(), cfg, smp, sigmas, seed, on_x0)

        def cond(value, grid):
            kf = {"resolved_frame_index": 0, "latent": torch.full((1, 24, 1, grid, grid), value)}
            return [[torch.zeros((1, 4, 8)), {"minimax_keyframes": [kf]}]]

        with patch.object(luna_two_pass, "_sample", spy):
            two_pass_sample(self.model, cond(1.0, 4), cond(2.0, 4), self.vae, av_latent(), sampler("euler"),
                            sampler("euler"), "simple", 4, 2, 0, 1.0, 1.0, 0.5, 0.0, 0.5, 1.0, lift=nearest_lift,
                            **{k: cond(v, 2) for k, v in kw.items()})
        return seen

    def test_low_conditioning_reaches_pass_1_only(self):
        seen = self.run_with_keyframes(positive_low=3.0, negative_low=4.0)
        self.assertTrue(torch.equal(seen[0][0], torch.full((1, 24, 1, 2, 2), 3.0)))
        self.assertTrue(torch.equal(seen[0][1], torch.full((1, 24, 1, 2, 2), 4.0)))
        self.assertTrue(torch.equal(seen[1][0], torch.full((1, 24, 1, 4, 4), 1.0)))
        self.assertTrue(torch.equal(seen[1][1], torch.full((1, 24, 1, 4, 4), 2.0)))

    def test_without_low_conditioning_pass_1_shrinks_the_keyframes(self):
        seen = self.run_with_keyframes()
        self.assertTrue(torch.allclose(seen[0][0], torch.full((1, 24, 1, 2, 2), 1.0)))
        self.assertEqual(tuple(seen[1][0].shape[-2:]), (4, 4))

    def test_high_model_defaults_to_low(self):
        a, _ = self.run_two_pass()
        b, _ = self.run_two_pass(high_res_model=self.model)
        self.assertTrue(torch.equal(streams(a)[0], streams(b)[0]))
        self.assertTrue(torch.equal(streams(a)[1], streams(b)[1]))

    def test_sigma_overrides(self):
        with self.assertRaisesRegex(ValueError, "sigmas_low"):
            self.run_two_pass(sigmas_low=torch.tensor([1.0]))
        with self.assertRaisesRegex(ValueError, "sigmas_high"):
            self.run_two_pass(sigmas_high=torch.tensor([0.5]))
        with self.assertRaisesRegex(ValueError, "above sigma 0"):
            self.run_two_pass(sigmas_low=torch.tensor([1.0, 0.0]))

    def test_lowres_scale_one(self):
        latent, low_latent = self.run_two_pass(scale=1.0)
        self.assertEqual(streams(low_latent)[0].shape, streams(latent)[0].shape)

    def low_head_setup(self, protect=True, grid=2):
        mask = torch.ones((1, 1, 2, 4, 4))
        if protect:
            mask[:, :, 0] = 0
        source = av_latent(video_mask=mask)
        head_video = torch.full((1, 24, 1, grid, grid), 0.25)
        head = {"samples": comfy.nested_tensor.NestedTensor((head_video, streams(source)[1]))}
        return source, head, head_video

    def test_low_head_lands_in_low_prefix(self):
        source, head, head_video = self.low_head_setup()
        seen, _ = self.spy_sigmas(latent=source, low_res_head=head)
        self.assertTrue(torch.equal(seen[0][2][:, :, :1], head_video))

    def test_low_head_refused_when_wrong(self):
        source, head, _ = self.low_head_setup(grid=4)
        with self.assertRaisesRegex(ValueError, "low_res_head"):
            self.run_two_pass(source, low_res_head=head)
        source, head, _ = self.low_head_setup(protect=False)
        with self.assertRaisesRegex(ValueError, "protected"):
            self.run_two_pass(source, low_res_head=head)


if __name__ == "__main__":
    unittest.main()

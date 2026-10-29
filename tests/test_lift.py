import os
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

_COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", "E:/ComfyUI-Easy-Install/ComfyUI")).resolve()
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_COMFY_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import comfy.latent_formats  # noqa: E402

from luna_lift import lift_video  # noqa: E402


class _TinyFakeVAE:
    """Self-contained deterministic fake H3 VAE for unit testing `lift_video`."""

    spatial_ratio = 16
    temporal_ratio = 4

    def __init__(self, pixel_bias: float = 0.35):
        self.pixel_bias = float(pixel_bias)
        self.decode_calls = 0
        self.encode_calls = 0

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1
        b, c, t, h, w = z.shape
        frames = 1 + max(0, t - 1) * self.temporal_ratio
        h_px = h * self.spatial_ratio
        w_px = w * self.spatial_ratio
        folded = z.reshape(b, 3, c // 3, t, h, w).mean(dim=2)
        up = F.interpolate(folded.float(), size=(frames, h_px, w_px), mode="trilinear", align_corners=False)
        return up.movedim(1, -1).to(dtype=z.dtype)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        b, frames, h_px, w_px, _ = pixels.shape
        t = 1 if frames <= 1 else 1 + (frames - 1) // self.temporal_ratio
        h = h_px // self.spatial_ratio
        w = w_px // self.spatial_ratio
        chw = pixels.movedim(-1, 1).float()
        pooled = F.interpolate(chw, size=(t, h, w), mode="area")
        base = pooled.repeat(1, 8, 1, 1, 1)
        # Add deterministic spatial curvature so z_pix differs non-uniformly from z_lat.
        gy = torch.linspace(-1.0, 1.0, h, device=pixels.device, dtype=torch.float32).view(1, 1, 1, h, 1)
        gx = torch.linspace(-1.0, 1.0, w, device=pixels.device, dtype=torch.float32).view(1, 1, 1, 1, w)
        pattern = self.pixel_bias * (gy * gy + 0.5 * gx)
        return (base + pattern).to(dtype=pixels.dtype)


class _AffineLatentFormat:
    """LatentFormat with non-trivial shift and scale to verify blend space."""

    def __init__(self, scale: float = 2.5, shift: float = -0.4):
        self.scale = float(scale)
        self.shift = float(shift)

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.shift) * self.scale

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent / self.scale) + self.shift


class LiftVideoTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.latent_format = comfy.latent_formats.MiniMaxH3Video()
        self.z0_low = torch.randn(1, 24, 2, 4, 6, dtype=torch.float32)
        self.out_hw = (8, 12)

    def test_output_shape_dtype_and_device(self):
        vae = _TinyFakeVAE()
        z_low = self.z0_low.to(dtype=torch.bfloat16)
        out = lift_video(z_low, vae, self.out_hw, rho=0.25, w_min=0.3, w_max=0.7, latent_format=self.latent_format)
        self.assertEqual(tuple(out.shape), (1, 24, 2, 8, 12))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.device, z_low.device)
        self.assertTrue(torch.isfinite(out).all())

    def test_identity_when_out_hw_matches_input(self):
        vae = _TinyFakeVAE()
        lifter_calls = 0

        def counting_lifter(z, hw):
            nonlocal lifter_calls
            lifter_calls += 1
            return z

        out = lift_video(
            self.z0_low,
            vae,
            (4, 6),
            rho=0.5,
            w_min=0.2,
            w_max=0.8,
            latent_format=self.latent_format,
            latent_lifter=counting_lifter,
        )
        self.assertTrue(torch.equal(out, self.z0_low))
        self.assertEqual(vae.decode_calls, 0)
        self.assertEqual(vae.encode_calls, 0)
        self.assertEqual(lifter_calls, 0)

    def test_rho_zero_equals_latent_route_nearest_and_custom_lifter(self):
        vae = _TinyFakeVAE()
        out_nearest = lift_video(
            self.z0_low,
            vae,
            self.out_hw,
            rho=0.0,
            w_min=1.0,
            w_max=1.0,
            latent_format=self.latent_format,
            latent_lifter=None,
        )
        expected_nearest = F.interpolate(self.z0_low, size=(2, 8, 12), mode="nearest")
        self.assertTrue(torch.allclose(out_nearest, expected_nearest, atol=1e-6))
        self.assertEqual(vae.decode_calls, 0)
        self.assertEqual(vae.encode_calls, 0)

        def custom_lifter(z: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
            return F.interpolate(z, size=(z.shape[2], hw[0], hw[1]), mode="trilinear", align_corners=False) * 1.25 + 0.1

        out_custom = lift_video(
            self.z0_low,
            vae,
            self.out_hw,
            rho=0.0,
            w_min=1.0,
            w_max=1.0,
            latent_format=self.latent_format,
            latent_lifter=custom_lifter,
        )
        expected_custom = custom_lifter(self.z0_low, self.out_hw)
        self.assertTrue(torch.allclose(out_custom, expected_custom, atol=1e-6))
        self.assertEqual(vae.decode_calls, 0)
        self.assertEqual(vae.encode_calls, 0)

    def test_pure_pixel_route_and_w_max_zero_skip(self):
        vae = _TinyFakeVAE()
        lifter_called = False

        def forbidden_lifter(z: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
            nonlocal lifter_called
            lifter_called = True
            return F.interpolate(z, size=(z.shape[2], hw[0], hw[1]), mode="nearest")

        # Pure pixel route: rho=1, w_min=w_max=1 gives weight 1 on z^pix everywhere and skips latent_lifter.
        out_pix = lift_video(
            self.z0_low,
            vae,
            self.out_hw,
            rho=1.0,
            w_min=1.0,
            w_max=1.0,
            latent_format=self.latent_format,
            latent_lifter=forbidden_lifter,
        )
        self.assertFalse(lifter_called)
        self.assertEqual(vae.decode_calls, 1)
        self.assertEqual(vae.encode_calls, 1)

        # Manual pixel route: decode -> per-frame 2D bilinear resize -> encode.
        px_low = vae.decode(self.z0_low)
        b_px, f_px, h_px, w_px, c_px = px_low.shape
        frames_2d = px_low.reshape(b_px * f_px, h_px, w_px, c_px).movedim(-1, 1)
        resized_2d = F.interpolate(frames_2d, size=(8 * 16, 12 * 16), mode="bilinear", align_corners=False)
        px_high = resized_2d.movedim(1, -1).reshape(b_px, f_px, 8 * 16, 12 * 16, c_px)
        expected_pix = vae.encode(px_high)
        self.assertTrue(torch.allclose(out_pix, expected_pix, atol=1e-6))
        self.assertFalse(torch.allclose(out_pix, F.interpolate(self.z0_low, size=(2, 8, 12), mode="nearest")))

        # When w_max == 0 (e.g. rho=1, w_min=w_max=0), Eq. (8) gives W=0 everywhere and skips the pixel route.
        vae_skip = _TinyFakeVAE()
        out_w0 = lift_video(
            self.z0_low,
            vae_skip,
            self.out_hw,
            rho=1.0,
            w_min=0.0,
            w_max=0.0,
            latent_format=self.latent_format,
        )
        expected_lat = F.interpolate(self.z0_low, size=(2, 8, 12), mode="nearest")
        self.assertTrue(torch.allclose(out_w0, expected_lat, atol=1e-6))
        self.assertEqual(vae_skip.decode_calls, 0)
        self.assertEqual(vae_skip.encode_calls, 0)

    def test_intermediate_blend_matches_paper_equations_and_latent_format(self):
        vae = _TinyFakeVAE(pixel_bias=0.5)
        fmt = _AffineLatentFormat(scale=2.0, shift=-0.25)
        rho, w_min, w_max = 0.25, 0.3, 0.7

        out = lift_video(
            self.z0_low,
            vae,
            self.out_hw,
            rho=rho,
            w_min=w_min,
            w_max=w_max,
            latent_format=fmt,
        )

        # Manual computation of Eqs. (4), (5), (7), (8), (9) in flow-matching space.
        z_in = fmt.process_in(self.z0_low)
        z_lat = F.interpolate(z_in, size=(2, 8, 12), mode="nearest")
        px_low = vae.decode(self.z0_low)
        b_px, f_px, h_px, w_px, c_px = px_low.shape
        px_high = (
            F.interpolate(
                px_low.reshape(b_px * f_px, h_px, w_px, c_px).movedim(-1, 1),
                size=(8 * 16, 12 * 16),
                mode="bilinear",
                align_corners=False,
            )
            .movedim(1, -1)
            .reshape(b_px, f_px, 8 * 16, 12 * 16, c_px)
        )
        z_pix = fmt.process_in(vae.encode(px_high))
        delta = z_pix - z_lat
        s = delta.abs().mean(dim=1, keepdim=True)
        q = torch.quantile(s.reshape(-1), 1.0 - rho)
        m = s >= q
        s_min = s[m].min()
        s_max = s[m].max()
        w = torch.where(m, w_min + (w_max - w_min) * ((s - s_min) / (s_max - s_min + 1e-8)), torch.zeros_like(s))
        expected = fmt.process_out((1.0 - w) * z_lat + w * z_pix)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_masked_out_regions_do_not_drive_weights_and_are_not_restored(self):
        class _PointwisePixelVAE(_TinyFakeVAE):
            """Pixel route depends only on each token's own spatial neighborhood (nearest) + offset."""

            def decode(self, z: torch.Tensor) -> torch.Tensor:
                self.decode_calls += 1
                b, c, t, h, w = z.shape
                up = F.interpolate(
                    z[:, :3].float(),
                    size=(t, h * self.spatial_ratio, w * self.spatial_ratio),
                    mode="nearest",
                )
                return up.movedim(1, -1).to(dtype=z.dtype)

            def encode(self, pixels: torch.Tensor) -> torch.Tensor:
                self.encode_calls += 1
                b, frames, h_px, w_px, _ = pixels.shape
                h = h_px // self.spatial_ratio
                w = w_px // self.spatial_ratio
                chw = pixels.movedim(-1, 1).float()
                pooled = F.interpolate(chw, size=(frames, h, w), mode="nearest")
                base = pooled.repeat(1, 8, 1, 1, 1)
                gy = torch.linspace(0.1, 1.0, h, device=pixels.device).view(1, 1, 1, h, 1)
                gx = torch.linspace(0.1, 1.0, w, device=pixels.device).view(1, 1, 1, 1, w)
                return (base + (gy * gx) * (base.abs().mean(dim=1, keepdim=True) + 0.5)).to(dtype=pixels.dtype)

        vae = _PointwisePixelVAE()
        mask = torch.ones((1, 1, 2, 8, 12), dtype=torch.float32)
        mask[:, :, 0, :, :] = 0.0  # Frame 0 is kept (masked out); frame 1 is generated.

        z_a = self.z0_low.clone()
        z_b = self.z0_low.clone()
        # Inject huge outliers into frame 0 (the kept region).
        z_b[:, :, 0, :, :] = z_b[:, :, 0, :, :] * 500.0 + 1000.0

        out_a = lift_video(
            z_a, vae, self.out_hw, rho=0.4, w_min=0.2, w_max=0.9, latent_format=self.latent_format, mask=mask
        )
        out_b = lift_video(
            z_b, vae, self.out_hw, rho=0.4, w_min=0.2, w_max=0.9, latent_format=self.latent_format, mask=mask
        )
        # Generated region (frame 1) must be completely unaffected by outliers in kept region (frame 0).
        self.assertTrue(torch.allclose(out_a[:, :, 1], out_b[:, :, 1], atol=1e-6))

        # Without the mask, the frame 0 outliers steal the top-rho quantile and change frame 1 weights.
        out_unmasked_a = lift_video(
            z_a, vae, self.out_hw, rho=0.4, w_min=0.2, w_max=0.9, latent_format=self.latent_format, mask=None
        )
        out_unmasked_b = lift_video(
            z_b, vae, self.out_hw, rho=0.4, w_min=0.2, w_max=0.9, latent_format=self.latent_format, mask=None
        )
        self.assertFalse(torch.allclose(out_unmasked_a[:, :, 1], out_unmasked_b[:, :, 1], atol=1e-4))

        # Kept region (frame 0) gets weight 0 (remains z_lat) and is not restored from any external source by lift_video.
        z_lat_b = F.interpolate(z_b, size=(2, 8, 12), mode="nearest")
        self.assertTrue(torch.allclose(out_b[:, :, 0], z_lat_b[:, :, 0], atol=1e-6))

    def test_invalid_weights_raise_value_error(self):
        vae = _TinyFakeVAE()
        bad_cases = [
            (-0.01, 0.3, 0.7),
            (1.01, 0.3, 0.7),
            (0.2, -0.1, 0.7),
            (0.2, 0.3, 1.1),
            (0.2, 0.8, 0.3),  # w_min > w_max
            (float("nan"), 0.3, 0.7),
            (0.2, float("nan"), 0.7),
            (0.2, 0.3, float("inf")),
            (True, 0.3, 0.7),
            (0.2, False, 0.7),
        ]
        for rho, w_min, w_max in bad_cases:
            with self.subTest(rho=rho, w_min=w_min, w_max=w_max):
                with self.assertRaises(ValueError):
                    lift_video(
                        self.z0_low,
                        vae,
                        self.out_hw,
                        rho=rho,
                        w_min=w_min,
                        w_max=w_max,
                        latent_format=self.latent_format,
                    )


if __name__ == "__main__":
    unittest.main()

"""Artifact-Aware Consistency Lift (SelfLift-zero) for MiniMax H3 video latents.

Implemented from the SelfLift paper:
    Tingyan Wen et al., "SelfLift: Accelerating Few-Step Diffusion via
    Self-Recovering Resolution Transition", arXiv:2609.02036, Sec. 3.2-3.3,
    Eqs. (4)-(9), and Algorithm 1 (Appendix A).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

# Numerical stability constant epsilon in Eq. (8) (paper leaves the value unspecified).
_EPS = 1e-8


def _process_in(z: torch.Tensor, latent_format) -> torch.Tensor:
    if latent_format is None or not hasattr(latent_format, "process_in"):
        return z
    return latent_format.process_in(z)


def _process_out(z: torch.Tensor, latent_format) -> torch.Tensor:
    if latent_format is None or not hasattr(latent_format, "process_out"):
        return z
    return latent_format.process_out(z)


def _latent_route(
    z_low: torch.Tensor,
    out_hw: tuple[int, int],
    latent_lifter: Callable[[torch.Tensor, tuple[int, int]], torch.Tensor] | None,
) -> torch.Tensor:
    """Direct latent lifting route z^lat = U^lat(z0_low) (Eq. 4; or learned G_phi in Eq. 11).

    When `latent_lifter` is None, uses nearest-neighbor spatial interpolation
    across (H, W) while preserving temporal length T (paper default, Sec. 3.2
    and Appendix B).
    """
    b, c, t, _, _ = z_low.shape
    h_out, w_out = out_hw
    if latent_lifter is not None:
        z_lat = latent_lifter(z_low, (h_out, w_out))
        if not isinstance(z_lat, torch.Tensor) or tuple(z_lat.shape) != (b, c, t, h_out, w_out):
            got = tuple(z_lat.shape) if isinstance(z_lat, torch.Tensor) else type(z_lat).__name__
            raise ValueError(
                f"lift_video: latent_lifter must return Tensor of shape {(b, c, t, h_out, w_out)}, got {got}"
            )
        return z_lat
    return F.interpolate(z_low.float(), size=(t, h_out, w_out), mode="nearest").to(dtype=z_low.dtype)


def _pixel_route_vae(
    z0_low_vae: torch.Tensor,
    vae,
    out_hw: tuple[int, int],
) -> torch.Tensor:
    """Pixel-VAE re-encoding route Y = U^pix(D_L(z0_low)), z^pix = E_H(Y) in VAE space (Eq. 5).

    Decodes the low-resolution VAE-space latent, resizes each decoded frame
    spatially to the target pixel grid using 2D bilinear interpolation
    (align_corners=False), and re-encodes at the target resolution.
    """
    b, c, t, h_low, w_low = z0_low_vae.shape
    h_out, w_out = out_hw

    pixels_low = vae.decode(z0_low_vae)
    if not isinstance(pixels_low, torch.Tensor) or pixels_low.ndim not in (4, 5):
        got = tuple(pixels_low.shape) if isinstance(pixels_low, torch.Tensor) else type(pixels_low).__name__
        raise ValueError(f"lift_video: vae.decode must return a 4D or 5D pixel tensor, got {got}")

    if pixels_low.ndim == 5:
        b_px, f_px, h_px, w_px, c_px = pixels_low.shape
        h_high_px = (h_px // h_low) * h_out if h_px % h_low == 0 else round(h_px * h_out / h_low)
        w_high_px = (w_px // w_low) * w_out if w_px % w_low == 0 else round(w_px * w_out / w_low)
        frames_2d = pixels_low.reshape(b_px * f_px, h_px, w_px, c_px).movedim(-1, 1).float()
        resized_2d = F.interpolate(frames_2d, size=(h_high_px, w_high_px), mode="bilinear", align_corners=False)
        pixels_high = resized_2d.movedim(1, -1).reshape(b_px, f_px, h_high_px, w_high_px, c_px).to(
            dtype=pixels_low.dtype
        )
    else:
        f_px, h_px, w_px, c_px = pixels_low.shape
        h_high_px = (h_px // h_low) * h_out if h_px % h_low == 0 else round(h_px * h_out / h_low)
        w_high_px = (w_px // w_low) * w_out if w_px % w_low == 0 else round(w_px * w_out / w_low)
        frames_2d = pixels_low.movedim(-1, 1).float()
        resized_2d = F.interpolate(frames_2d, size=(h_high_px, w_high_px), mode="bilinear", align_corners=False)
        pixels_high = resized_2d.movedim(1, -1).to(dtype=pixels_low.dtype)

    z_pix_vae = vae.encode(pixels_high)
    if not isinstance(z_pix_vae, torch.Tensor) or tuple(z_pix_vae.shape) != (b, c, t, h_out, w_out):
        got = tuple(z_pix_vae.shape) if isinstance(z_pix_vae, torch.Tensor) else type(z_pix_vae).__name__
        raise ValueError(
            f"lift_video: vae.encode must return Tensor of shape {(b, c, t, h_out, w_out)}, got {got}"
        )
    return z_pix_vae


def _prepare_active_mask(
    mask: torch.Tensor,
    target_shape: tuple[int, int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Return a boolean mask [B, 1, T, H, W] where True = generated (mask > 0), False = kept (mask == 0)."""
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"lift_video: mask must be a torch.Tensor or None, got {type(mask).__name__}")
    b, _, t, h_out, w_out = target_shape
    m = mask.to(device=device)
    if m.ndim == 3:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.ndim == 4:
        m = m.unsqueeze(1)
    elif m.ndim != 5:
        raise ValueError(f"lift_video: mask must have 3, 4, or 5 dims, got shape {tuple(mask.shape)}")

    if tuple(m.shape[2:]) != (t, h_out, w_out):
        m = F.interpolate(m.float(), size=(t, h_out, w_out), mode="nearest")

    if m.shape[0] == 1 and b > 1:
        m = m.expand(b, -1, -1, -1, -1)
    elif m.shape[0] != b:
        raise ValueError(f"lift_video: mask batch size {m.shape[0]} does not match latent batch size {b}")

    return (m > 0).any(dim=1, keepdim=True)


def lift_video(
    z0_low: torch.Tensor,
    vae,
    out_hw: tuple[int, int],
    rho: float,
    w_min: float,
    w_max: float,
    latent_format,
    latent_lifter: Callable[[torch.Tensor, tuple[int, int]], torch.Tensor] | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lift a low-resolution clean video latent to `out_hw` via Artifact-Aware Consistency Lift.

    Implements SelfLift-zero from arXiv:2609.02036, Sec. 3.2-3.3, Eqs. (4)-(9):
      - Eq. (4) Latent route: `z^lat = U^lat(z0_low)` using `latent_lifter` if
        given, or nearest-neighbor spatial upsampling when `None`.
      - Eq. (5) Pixel route: `Y = U^pix(D_L(z0_low))`, `z^pix = E_H(Y)` via
        `vae.decode`, per-frame bilinear resize, and `vae.encode`.
      - Eq. (7) Consistency residual and artifact-risk map:
        `Delta = z^pix - z^lat`, `s_i = (1 / C) ||Delta_i||_1`,
        `M_i = 1[s_i >= Q_{1-rho}({s_j})]`.
      - Eq. (8) Spatially adaptive correction weights:
        `W_i = M_i * (w_min + (w_max - w_min) * (s_i - s_min) / (s_max - s_min + eps))`,
        with `s_min` and `s_max` taken over the selected region `{i : M_i = 1}`.
      - Eq. (9) Artifact-Aware Consistency Lift blend:
        `z_hat = z^lat + W * Delta = (1 - W) * z^lat + W * z^pix`.

    Args:
        z0_low: VAE-space video latent tensor of shape `[B, 24, T, h, w]`.
        vae: ComfyUI VAE (or compatible object with `decode` and `encode`).
        out_hw: Target spatial latent dimensions `(H, W)`.
        rho: Top-fraction of inconsistent locations to correct (`0 <= rho <= 1`).
        w_min: Minimum blending weight toward the pixel route (`0 <= w_min <= w_max <= 1`).
        w_max: Maximum blending weight toward the pixel route (`0 <= w_min <= w_max <= 1`).
        latent_format: ComfyUI LatentFormat (`process_in` / `process_out`) defining the
            flow-matching latent space in which Eqs. (4) and (7)-(9) operate.
        latent_lifter: Optional callable `(z, (H, W)) -> z_up` replacing nearest upsampling.
        mask: Optional full-grid video noise mask (`1` = generated, `0` = kept). Kept
            regions (`0`) are excluded from the quantile and min/max statistics and get
            weight `0`; the caller restores kept regions afterwards.

    Returns:
        Lifted VAE-space video latent of shape `[B, 24, T, H, W]`.
    """
    for name, val in (("rho", rho), ("w_min", w_min), ("w_max", w_max)):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"lift_video: {name} must be a float in [0, 1], got {val!r}")
    if not (0.0 <= rho <= 1.0) or not (0.0 <= w_min <= w_max <= 1.0):
        raise ValueError(
            f"lift_video: require 0 <= rho <= 1 and 0 <= w_min <= w_max <= 1, "
            f"got rho={rho}, w_min={w_min}, w_max={w_max}"
        )
    if not isinstance(z0_low, torch.Tensor) or z0_low.ndim != 5:
        got = tuple(z0_low.shape) if isinstance(z0_low, torch.Tensor) else type(z0_low).__name__
        raise ValueError(f"lift_video: z0_low must be a 5D tensor [B, C, T, h, w], got {got}")
    if not isinstance(out_hw, (tuple, list)) or len(out_hw) != 2:
        raise ValueError(f"lift_video: out_hw must be a 2-tuple (H, W), got {out_hw}")
    h_out, w_out = int(out_hw[0]), int(out_hw[1])
    if h_out < 1 or w_out < 1:
        raise ValueError(f"lift_video: out_hw dimensions must be positive, got {(h_out, w_out)}")

    b, c, t, h_low, w_low = z0_low.shape
    if (h_out, w_out) == (h_low, w_low):
        return z0_low

    orig_device = z0_low.device
    orig_dtype = z0_low.dtype
    target_shape = (b, c, t, h_out, w_out)
    active_mask = _prepare_active_mask(mask, target_shape, orig_device) if mask is not None else None

    # Skip the pixel route when it gets no weight (rho == 0 or w_max == 0)
    # or when all tokens are masked out (kept).
    if rho == 0.0 or w_max == 0.0 or (active_mask is not None and not bool(torch.any(active_mask))):
        z_in = _process_in(z0_low, latent_format)
        z_lat = _latent_route(z_in, (h_out, w_out), latent_lifter)
        return _process_out(z_lat, latent_format).to(device=orig_device, dtype=orig_dtype)

    # Skip the latent route when it gets no weight anywhere (rho == 1 and w_min == 1 on all tokens).
    if rho == 1.0 and w_min == 1.0 and (active_mask is None or bool(torch.all(active_mask))):
        z_pix_vae = _pixel_route_vae(z0_low, vae, (h_out, w_out)).to(device=orig_device, dtype=orig_dtype)
        z_pix = _process_in(z_pix_vae, latent_format)
        return _process_out(z_pix, latent_format).to(device=orig_device, dtype=orig_dtype)

    # Paired lifts in flow-matching latent space (Eqs. 4-5).
    z_in = _process_in(z0_low, latent_format)
    z_lat = _latent_route(z_in, (h_out, w_out), latent_lifter).to(device=orig_device, dtype=torch.float32)
    z_pix_vae = _pixel_route_vae(z0_low, vae, (h_out, w_out)).to(device=orig_device, dtype=orig_dtype)
    z_pix = _process_in(z_pix_vae, latent_format).to(device=orig_device, dtype=torch.float32)

    # Eq. (7): consistency residual Delta = z^pix - z^lat and local L1 proxy s_i = (1/C) ||Delta_i||_1.
    delta = z_pix - z_lat
    s = delta.abs().mean(dim=1, keepdim=True)  # [B, 1, T, H, W]

    weights = torch.zeros_like(s)
    for idx in range(b):
        s_b = s[idx]
        act_b = active_mask[idx] if active_mask is not None else None
        pool = s_b[act_b] if act_b is not None else s_b.reshape(-1)
        if pool.numel() == 0:
            continue

        if rho >= 1.0:
            m_b = torch.ones_like(s_b, dtype=torch.bool) if act_b is None else act_b.clone()
        else:
            q = torch.quantile(pool, 1.0 - float(rho))
            m_b = (s_b >= q) if act_b is None else (act_b & (s_b >= q))

        if not bool(torch.any(m_b)):
            continue

        # Eq. (8): normalize selected scores into [w_min, w_max] over the selected region.
        sel = s_b[m_b]
        s_min = sel.min()
        s_max = sel.max()
        w_b = float(w_min) + (float(w_max) - float(w_min)) * ((s_b - s_min) / (s_max - s_min + _EPS))
        weights[idx] = torch.where(m_b, w_b, weights[idx])

    # Eq. (9): z_hat = z^lat + W * Delta = (1 - W) * z^lat + W * z^pix.
    z_hat = (1.0 - weights) * z_lat + weights * z_pix
    out = _process_out(z_hat.to(dtype=z_in.dtype), latent_format)
    return out.to(device=orig_device, dtype=orig_dtype)

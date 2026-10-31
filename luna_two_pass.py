"""SelfLift two-resolution sampling for MiniMax H3, on one schedule split at the transition.

`total_steps` is one schedule. The first `transition` steps run at low resolution (pass 1), the rest at high
resolution (pass 2). Each arm computes that schedule on its own model chain (its LoRA and shift node), so each pass
runs on sigmas built for its own shift. Both passes run inside ComfyUI's own sampler, so any SAMPLER works.

The hand-over between the passes is exact flow arithmetic, not a sampler step:
- video: pass 1's last clean estimate is lifted to the high grid and re-noised with fresh noise to pass 2's first sigma;
- audio: keeps its own trajectory. Its clean estimate and its noise are recovered from pass 1's final state, then
  re-combined at pass 2's first sigma, in each arm's own audio scale. The audio is never cut short: it finishes in
  pass 2, next to the full-size video. A locked song (audio mask 0) is restored bit for bit.
  A pass 1 that stopped at FINISHED_SIGMA or below has no noise left to recover (dividing by that sigma magnifies the
  sampler's own error), so pass 2 draws fresh noise for video and audio from one seed, as a native second pass does.
"""

import logging

import torch

import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils

FINISHED_SIGMA = 0.01


def _split(samples):
    if getattr(samples, "is_nested", False):
        return list(samples.unbind()), True
    return [samples], False


def _pack(streams, nested):
    return comfy.nested_tensor.NestedTensor(streams) if nested else streams[0]


def _check_sigmas(sigmas, name):
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError(f"Luna-H3-Sampler: {name} needs at least 2 values (one step), as a 1-D tensor")
    if not torch.isfinite(sigmas).all() or not bool((sigmas[1:] < sigmas[:-1]).all()):
        raise ValueError(f"Luna-H3-Sampler: {name} must be finite and strictly descending")
    if float(sigmas[0]) > 1.0 or float(sigmas[-1]) < 0.0:
        raise ValueError(f"Luna-H3-Sampler: {name} must lie in [0, 1] (rectified flow)")


def _model_sampling(model, name):
    ms = model.get_model_object("model_sampling")
    if not isinstance(ms, comfy.model_sampling.CONST):
        raise ValueError(f"Luna-H3-Sampler: {name} must be a rectified-flow model (MiniMax H3)")
    return ms


def arm_sigmas(model, scheduler, total_steps, transition, low, denoise=None):
    """The arm's share of one `total_steps` schedule, computed on the arm's own model_sampling (its shift).

    low=True: sigmas[0 .. transition]. low=False: sigmas[transition .. end], or, with `denoise`, the last
    `total_steps - transition` steps of the whole schedule stretched to `int(total_steps / denoise)` steps, so 1.0 is
    exactly the transition slice and lower values start pass 2 at a lower sigma (it changes the lifted video less).
    """
    ms = model.get_model_object("model_sampling")
    if low:
        return comfy.samplers.calculate_sigmas(ms, scheduler, total_steps).cpu()[: transition + 1]
    steps = total_steps - transition
    if denoise is None:
        return comfy.samplers.calculate_sigmas(ms, scheduler, total_steps).cpu()[transition:]
    if not 0.0 < denoise <= 1.0:
        raise ValueError("Luna-H3-Sampler: highres_denoise must be in (0, 1]")
    return comfy.samplers.calculate_sigmas(ms, scheduler, int(total_steps / denoise)).cpu()[-(steps + 1):]


def audio_handover(state, x0, sigma_from, sigma_to, scale_from, scale_to):
    """Move the audio from pass 1's final state to pass 2's first sigma; returns (x0 in VAE units, noise).

    `state` and `x0` are in pass 1's model space (audio carried at `scale_from`). Flow: x = (1 - s) * x0 + s * eps.
    Pass 2's sampler rebuilds `(1 - sigma_to) * process_in(x0_vae) + sigma_to * eps` with `process_in` scaling the
    audio by `scale_to`, which is exactly the audio's own trajectory at pass 2's noise level.
    """
    sigma_from = float(sigma_from)
    eps = (state - (1.0 - sigma_from) * x0) / sigma_from
    return x0 / scale_from, eps


def _full_masks(noise_mask, streams):
    """One full-shape float mask per stream (1 = generate, 0 = keep), or None when the latent has no mask."""
    if noise_mask is None:
        return None
    parts = list(noise_mask.unbind()) if getattr(noise_mask, "is_nested", False) else [noise_mask]
    masks = []
    for i, stream in enumerate(streams):
        if i < len(parts) and parts[i] is not None:
            masks.append(comfy.utils.reshape_mask(parts[i].float(), stream.shape).to(stream.device))
        else:
            masks.append(torch.ones(stream.shape, device=stream.device))
    return masks


def _resize_video(x, size, mode):
    return torch.nn.functional.interpolate(x.float(), size=size, mode=mode).to(x.dtype)


def _keyframes_at(conditioning, h, w):
    """Keyframe latents share the generation grid on H3; pass 1 needs them at the low grid."""
    out = []
    for cond, extra in conditioning:
        keyframes = extra.get("minimax_keyframes")
        if keyframes:
            extra = dict(extra)
            resized = []
            for kf in keyframes:
                kf = dict(kf)
                lat = kf.get("latent")
                if lat is not None and lat.ndim == 5 and tuple(lat.shape[-2:]) != (h, w):
                    kf["latent"] = _resize_video(lat, (lat.shape[2], h, w), "trilinear")
                resized.append(kf)
            extra["minimax_keyframes"] = resized
        out.append([cond, extra])
    return out


def _restore_exact(outputs, anchors, masks):
    """Exact keep: where a mask is 0 the anchor comes back bit for bit (soft masks are the sampler's business)."""
    return [torch.where(m.to(o.device) == 0, a.to(o), o) for o, a, m in zip(outputs, anchors, masks)]


def _write_low_head(low_video, low_mask, head_latent):
    head = _split(head_latent["samples"])[0][0]
    b, c, t, h, w = low_video.shape
    if head.ndim != 5 or head.shape[:2] != (b, c) or tuple(head.shape[-2:]) != (h, w) or not 0 < head.shape[2] <= t:
        raise ValueError("Luna-H3-Sampler: low_res_head must match the low grid, batch and channels "
                         "and be no longer than the video")
    k = head.shape[2]
    if low_mask is None or not torch.all(low_mask[:, :, :k] == 0):
        raise ValueError("Luna-H3-Sampler: low_res_head needs a protected video prefix (mask 0) in latent_image")
    low_video = low_video.clone()
    low_video[:, :, :k] = head.to(low_video)
    return low_video


def _sample(model, noise, latent, masks, nested, positive, negative, cfg, sampler, sigmas, seed, on_x0=None):
    preview = None
    try:
        import latent_preview
        preview = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)
    except Exception:  # previews are optional (tests, headless runs)
        pass

    def callback(step, x0, x, total):
        if on_x0 is not None:
            on_x0(x0)
        if preview is not None:
            return preview(step, x0, x, total)

    mask = _pack(masks, nested) if masks is not None else None
    return comfy.samplers.sample(model, noise, positive, negative, cfg, model.load_device, sampler, sigmas,
                                 model.model_options, latent_image=latent, denoise_mask=mask, callback=callback,
                                 disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)


def two_pass_sample(low_res_model, positive, negative, vae, latent_image, sampler_low, sampler_high, scheduler,
                    total_steps, transition, noise_seed, cfg_low, cfg_high, lowres_scale, rho, w_min, w_max,
                    latent_lifter=None, high_res_model=None, low_res_head=None, highres_denoise=None,
                    sigmas_low=None, sigmas_high=None, lift=None, positive_low=None, negative_low=None):
    """Return (latent, low_res_latent), both ComfyUI LATENT dicts in VAE space.

    `sigmas_low` / `sigmas_high` override the computed schedules. `lift` defaults to `luna_lift.lift_video`.
    `positive_low` / `negative_low` replace `positive` / `negative` in pass 1 only: conditioning built at pass 1's size.
    """
    if not 1 <= transition < total_steps:
        raise ValueError("Luna-H3-Sampler: transition must be at least 1 and below total_steps")
    if not 0.25 <= lowres_scale <= 1.0:
        raise ValueError("Luna-H3-Sampler: lowres_scale must be between 0.25 and 1")
    if not 0.0 <= rho <= 1.0 or not 0.0 <= w_min <= w_max <= 1.0:
        raise ValueError("Luna-H3-Sampler: need 0 <= rho <= 1 and 0 <= w_min <= w_max <= 1")
    high_model = high_res_model if high_res_model is not None else low_res_model
    ms_low = _model_sampling(low_res_model, "low_res_model")
    ms_high = _model_sampling(high_model, "high_res_model")
    if sigmas_low is None:
        sigmas_low = arm_sigmas(low_res_model, scheduler, total_steps, transition, low=True)
    if sigmas_high is None:
        sigmas_high = arm_sigmas(high_model, scheduler, total_steps, transition, low=False, denoise=highres_denoise)
    _check_sigmas(sigmas_low, "sigmas_low")
    _check_sigmas(sigmas_high, "sigmas_high")
    if not 0.0 < float(sigmas_low[-1]) < 1.0:
        raise ValueError("Luna-H3-Sampler: pass 1 must stop above sigma 0 and below 1 (the transition is part-way)")
    noise_scale_high = float(getattr(ms_high, "noise_scale", 1.0))
    if noise_scale_high == 0.0:
        raise ValueError("Luna-H3-Sampler: high_res_model has noise_scale 0")
    if lift is None:
        from .luna_lift import lift_video as lift

    samples = comfy.sample.fix_empty_latent_channels(
        low_res_model, latent_image["samples"], latent_image.get("downscale_ratio_spacial", None),
        latent_image.get("downscale_ratio_temporal", None))
    streams, nested = _split(samples)
    if not nested or len(streams) < 2 or streams[0].ndim != 5:
        raise ValueError("Luna-H3-Sampler: latent_image must be an H3 audio+video latent")
    device = comfy.model_management.intermediate_device()
    streams = [s.to(device) for s in streams]
    video, audio = streams[0], streams[1:]
    _, _, T, H, W = video.shape
    h = max(2, round(H * lowres_scale / 2) * 2)
    w = max(2, round(W * lowres_scale / 2) * 2)
    masks = _full_masks(latent_image.get("noise_mask"), streams)
    batch_index = latent_image.get("batch_index", None)
    latent_format = low_res_model.get_model_object("latent_format")
    scale_low = float(getattr(ms_low, "audio_scale", 1.0))
    scale_high = float(getattr(ms_high, "audio_scale", 1.0))
    logging.info("[Luna-H3-Sampler] low=%s high=%s steps=%d+%d sigma %.4f -> %.4f",
                 (T, h, w), (T, H, W), sigmas_low.numel() - 1, sigmas_high.numel() - 1,
                 float(sigmas_low[-1]), float(sigmas_high[0]))

    # Pass 1: the first `transition` steps at the low grid.
    low_video = _resize_video(video, (T, h, w), "trilinear")
    low_masks = None
    if masks is not None:
        low_masks = [_resize_video(masks[0], (T, h, w), "nearest")] + masks[1:]
    if low_res_head is not None:
        low_video = _write_low_head(low_video, low_masks[0] if low_masks else None, low_res_head)
    low_in = [low_video] + audio
    low_latent = _pack(low_in, nested)
    last_x0 = {}
    positive_1 = positive if positive_low is None else positive_low
    negative_1 = negative if negative_low is None else negative_low
    state_vae = _sample(low_res_model, comfy.sample.prepare_noise(low_latent, noise_seed, batch_index), low_latent,
                        low_masks, nested, _keyframes_at(positive_1, h, w), _keyframes_at(negative_1, h, w), cfg_low,
                        sampler_low, sigmas_low, noise_seed, on_x0=lambda x0: last_x0.__setitem__("x0", x0))
    if "x0" not in last_x0:
        raise RuntimeError("Luna-H3-Sampler: pass 1 gave no clean estimate (the sampler reported no step)")
    x0_streams = [s.to(device).float() for s in _split(last_x0["x0"])[0]]
    sigma_k = float(sigmas_low[-1])
    sigma_s = float(sigmas_high[0])

    # Video: pass 1's clean estimate, kept regions exact, lifted to the high grid.
    z0_low = latent_format.process_out(x0_streams[0]).to(video.dtype)
    if low_masks is not None:
        z0_low = torch.where(low_masks[0] == 0, low_video, z0_low)
    lifted = lift(z0_low, vae, (H, W), rho, w_min, w_max, latent_format, latent_lifter=latent_lifter,
                  mask=masks[0] if masks is not None else None).to(video)
    if masks is not None:
        lifted = torch.where(masks[0] == 0, video, lifted)

    # Audio: its own trajectory, moved from pass 1's sigma to pass 2's. The sampler returned
    # process_out(state / (1 - sigma_k)); undo that to get pass 1's model-space state.
    state_streams = [s.to(device).float() for s in _split(state_vae)[0]]
    audio_x0, audio_eps = [], []
    for state_a, x0_a in zip(state_streams[1:], x0_streams[1:]):
        x0_vae, eps = audio_handover(state_a * scale_low * (1.0 - sigma_k), x0_a, sigma_k, sigma_s,
                                     scale_low, scale_high)
        audio_x0.append(x0_vae.to(audio[0].dtype))
        audio_eps.append(eps / noise_scale_high)  # the sampler multiplies its noise by noise_scale
    if masks is not None:  # a locked song stays the song
        audio_x0 = [torch.where(m == 0, a, x) for x, a, m in zip(audio_x0, audio, masks[1:])]

    # Pass 2: the rest of the schedule at the high grid.
    high_masks = masks if masks is not None else None
    high_in = [lifted] + audio_x0
    if sigma_k <= FINISHED_SIGMA:
        high_noise = comfy.sample.prepare_noise(_pack(high_in, nested), (noise_seed + 1) % (1 << 64), batch_index)
    else:
        video_noise = comfy.sample.prepare_noise(lifted, (noise_seed + 1) % (1 << 64), batch_index)
        high_noise = _pack([video_noise] + [e.to(video_noise) for e in audio_eps], nested)
    out = _sample(high_model, high_noise, _pack(high_in, nested), high_masks, nested, positive, negative, cfg_high,
                  sampler_high, sigmas_high, (noise_seed + 1) % (1 << 64))
    dtype = comfy.model_management.intermediate_dtype()
    out_streams = [s.to(device=device, dtype=dtype) for s in _split(out)[0]]
    logging.info("[Luna-H3-Sampler] check: video mask %s; lifted std %.4f; pass 2 changed the video by %.4f "
                 "(mean abs); sigmas_high %s",
                 "none" if masks is None else "mean %.3f min %.3f" % (float(masks[0].mean()), float(masks[0].min())),
                 float(lifted.float().std()), float((out_streams[0].float() - lifted.float()).abs().mean()),
                 [round(float(s), 4) for s in sigmas_high])
    if masks is not None:  # after the last cast, so a kept value is not rounded again
        out_streams = _restore_exact(out_streams, [video] + audio, masks)
    latent = latent_image.copy()
    latent["samples"] = _pack(out_streams, nested)

    # low_res_latent: pass 1's clean video on the low grid (the continuation head) with the final audio.
    low_res_latent = latent_image.copy()
    low_res_latent["samples"] = _pack([z0_low.to(dtype=dtype)] + out_streams[1:], nested)
    if low_masks is not None:
        low_res_latent["noise_mask"] = _pack(low_masks, nested)
    return latent, low_res_latent

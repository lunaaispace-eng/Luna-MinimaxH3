"""CPU fixtures backed by ComfyUI's real MiniMax H3 and sampler code.

Import before other ComfyUI modules: this harness selects ComfyUI's CPU mode.
FakeVAE is a deterministic shape double, not a learned codec.
"""

from __future__ import annotations

import os
import sys

# Core imports must not create bytecode inside the read-only ComfyUI installation.
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("COMFYUI_ROOT", "E:/ComfyUI-Easy-Install/ComfyUI"))

import comfy.cli_args

comfy.cli_args.args.cpu = True

import torch
import torch.nn.functional as F

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.samplers
import comfy.supported_models


def tiny_h3_model(
    shift: float = 12.0, audio_shift: float = 3.0, seed: int = 0
) -> comfy.model_patcher.ModelPatcher:
    """One real DiT/refiner block, float32 weights, CPU only.

    H3 audio is stereo (`[B,32,2,T]`, PackedLayout / unpack_audio); the fixture uses it natively.
    """
    if shift <= 0 or audio_shift <= 0:
        raise ValueError("H3 shifts must be positive")
    if not comfy.model_management.cpu_mode():
        raise RuntimeError("Import h3_harness before initializing ComfyUI on a GPU")
    config = comfy.supported_models.MiniMaxH3({
        "image_model": "minimax_h3",
        "hidden_size": 6,
        "num_layers": 1,
        "token_refiner_num_layers": 1,
        "num_attention_heads": 1,
        "attention_head_dim": 6,
        "ffn_hidden_size": 6,
        "text_dim": 8,
        "timestep_input_dim": 2,
        "time_embed_hidden_size": 6,
        "time_embed_dim": 6,
        # Three rotary axes, one frequency each, require six head dimensions.
        "rope_inv_freq_len": 1,
        "sigma_shift_video": shift,
        "sigma_shift_audio": audio_shift,
        "dtype": torch.float32,
    })
    config.sampling_settings.update(shift=shift, audio_shift=audio_shift)
    cpu = torch.device("cpu")
    model = comfy.model_base.MiniMaxH3(config, device=cpu)
    # ComfyUI deliberately leaves checkpoint parameters uninitialized. Initialize
    # every parameter and the rotary buffer using a private CPU RNG instead.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.diffusion_model.named_parameters():
            if parameter.ndim == 1:
                parameter.fill_(1.0 if name.endswith("weight") else 0.0)
            else:
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
        model.diffusion_model.rope.inv_freq.fill_(1.0)
    model.model_sampling.set_parameters(shift=shift, audio_shift=audio_shift)
    model.eval()
    patcher = comfy.model_patcher.ModelPatcher(model, load_device=cpu, offload_device=cpu)

    return patcher


def av_latent(
    t: int = 2, h: int = 4, w: int = 4, audio_t: int | None = None,
    video_mask=None, audio_mask=None, seed: int = 0,
) -> dict:
    """A nested LATENT, with separate sampler-compatible stream masks if supplied."""
    if min(t, h, w) < 1:
        raise ValueError("Latent dimensions must be positive")
    if audio_t is None:
        # MiniMaxH3AV.fix_empty_latent's video-to-audio duration convention.
        frames_per_token = (1, 4, 4, 4, 4)
        audio_t = round(sum(frames_per_token[i % 5] for i in range(t)) * (5.0 / 3.0))
    if audio_t < 1:
        raise ValueError("audio_t must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    video = torch.randn((1, 24, t, h, w), generator=generator, device="cpu")
    audio = torch.randn((1, 32, 2, audio_t), generator=generator, device="cpu")
    result = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}
    if video_mask is not None or audio_mask is not None:
        masks = []
        for mask, shape in ((video_mask, (1, 1, t, h, w)),
                            (audio_mask, (1, 1, 2, audio_t))):
            # Single-channel masks broadcast across latent channels in prepare_mask.
            masks.append(torch.ones(shape) if mask is None else
                         torch.as_tensor(mask, dtype=torch.float32, device="cpu").clone())
        result["noise_mask"] = comfy.nested_tensor.NestedTensor(masks)
    return result


class FakeVAE:
    """H3 video VAE shape conventions, with simple deterministic resampling.

    decode: [B,24,T,H,W] -> [B,frames,16H,16W,3] in [0,1].
    encode accepts that channels-last batch or a flattened [frames,H,W,3].
    Round trips preserve canonical H3 lengths (T=1 or T=5k+2), not values.
    """

    def decode(self, z):
        if z.ndim != 5 or z.shape[1] != 24:
            raise ValueError("Expected [B,24,T,H,W] video latents")
        t = z.shape[2]
        if t != 1 and (t < 2 or (t - 2) % 5):
            raise ValueError("H3 VAE latent time must be 1 or 5k+2")
        frames = max(1, (t - 2) // 5 * 17 + 5)
        pixels = F.interpolate(z[:, :3].float(),
                               size=(frames, z.shape[3] * 16, z.shape[4] * 16),
                               mode="nearest")
        return pixels.sigmoid().movedim(1, -1)

    def encode(self, pixels):
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(0)
        if pixels.ndim != 5 or pixels.shape[-1] != 3:
            raise ValueError("Expected [B,frames,H,W,3] or [frames,H,W,3]")
        frames, h, w = pixels.shape[1:4]
        if min(h, w) < 16 or h % 16 or w % 16:
            raise ValueError("Pixel height and width must be multiples of 16")
        t = max(1, (frames - 5) // 17 * 5 + 2) if frames > 1 else 1
        rgb = F.interpolate(pixels.movedim(-1, 1).float(), size=(t, h // 16, w // 16),
                            mode="nearest")
        return rgb.repeat(1, 8, 1, 1, 1)


def sampler(name: str) -> comfy.samplers.KSAMPLER:
    return comfy.samplers.sampler_object(name)


def sigmas(n: int, start: float = 1.0) -> torch.Tensor:
    if n < 1 or not 0 < start <= 1:
        raise ValueError("Need at least one step and 0 < start <= 1")
    return torch.linspace(start, 0.0, n + 1, dtype=torch.float32, device="cpu")


def empty_cond() -> list:
    """One zero text token; call separately for positive and negative conditioning."""
    return [[torch.zeros((1, 1, 8), dtype=torch.float32, device="cpu"), {}]]

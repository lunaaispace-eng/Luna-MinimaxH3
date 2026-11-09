"""Luna-H3-Chain, -ChainContinue, -ChainRemember, -ChainJoin — many H3 clips in one queue, each continuing the one before.

One Luna-MetaBatch (Luna-Media-Tools; VHS's Meta Batch Manager works the same) runs the graph once per clip. Clip 1 is a
normal generation. Every later clip copies the last `overlap` frames of the clip before into its own start and locks them
(noise mask 0), so the picture carries on where that clip ended. Join cuts the copied frames off again and gathers the
sound, and Luna-SaveVideo on the same meta_batch writes one video, closed on the last clip with the whole soundtrack.

The song settings live on Luna-H3-AssetLoader's sound card (2026-11-03, Peter: "it needs to be the other way around, set it
in asset loader"): Luna-H3-Chain -> the loader's `chain` input gives only the clip number; the card's Start, Shot and
overlap stay the user's. The loader's `overlap` output feeds ChainContinue, ChainRemember and ChainJoin.

Overlap: 22, 39, 90, 141 or 192 frames, all 5 + 17k, so the copied run is 2 + 5k whole video latents starting on a
one-frame latent. 39 + 51k frames also land on whole 40 Hz audio steps (39 frames = 65); 22 frames are 36.67, so 22 is
allowed only with the song locked, when the previous clip's sound is not copied. A mask already on the latent (a
locked song) is kept outside the head.

Endings are saved under output/luna_h3_chain/<name>/clip_NNNN.latent: the video and audio tail of `latent`, and the
video tail of Luna-H3-Sampler's `low_res_latent` when it is connected, for the next clip's `low_res_head`.

Adapted from TrentNodes `nodes/h3_video_chain.py`, `h3_chain_batch.py` and `h3_keystone_graft.py`
(MIT, Copyright (c) 2025 Trent; see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import logging
import os
import re
import shutil

import torch

import comfy.nested_tensor
import comfy.utils

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:   # imported outside ComfyUI (tests)
    class ExecutionBlocker:
        def __init__(self, message):
            self.message = message

LOG = "[Luna-H3-Chain]"
FPS = 24
AUDIO_HZ = 40
OVERLAPS = ("22", "39", "90", "141", "192")
LOCKED_ONLY = 22   # not a whole number of audio steps
FOLDER = "luna_h3_chain"
CHAIN = "LUNA_H3_CHAIN"
SAVED = "LUNA_H3_CHAIN_SAVED"


def video_steps(frames: int) -> int:
    """Video latents of a 17k+5 frame run: 1, 4, 4, 4, 4 frames each, so 5k+2."""
    return 2 + 5 * ((int(frames) - 5) // 17)


def audio_steps(frames: int) -> int:
    return round(int(frames) / FPS * AUDIO_HZ)


def on_grid(frames: int) -> bool:
    return int(frames) >= 5 and (int(frames) - 5) % 17 == 0


def _output_dir() -> str:
    import folder_paths
    return folder_paths.get_output_directory()


def _clean_name(name: str) -> str:
    text = (name or "").strip()
    if not text or len(text) > 64 or not re.fullmatch(r"[A-Za-z0-9._ -]+", text) or text.strip(".") == "":
        raise ValueError(f"{LOG} name '{name}': use 1-64 letters, digits, spaces, dots, dashes or underscores.")
    return text


def ending_path(name: str, clip: int) -> str:
    return os.path.join(_output_dir(), FOLDER, _clean_name(name), "clip_%04d.latent" % int(clip))


def _streams(latent, who):
    samples = latent["samples"]
    if not getattr(samples, "is_nested", False) or len(samples.unbind()) < 2:
        raise ValueError(f"{LOG} {who}: this is not a MiniMax H3 audio+video latent.")
    video, audio = samples.unbind()[:2]
    return video, audio


def _masks(latent, video, audio):
    """Full-length masks (1 = generate, 0 = keep) for both streams, keeping any mask already on the latent."""
    mask = latent.get("noise_mask")
    parts = list(mask.unbind()) if getattr(mask, "is_nested", False) else ([mask] if mask is not None else [])
    out = []
    for i, stream in enumerate((video, audio)):
        shape = (stream.shape[0], 1) + tuple(stream.shape[2:])
        if i < len(parts) and parts[i] is not None:
            out.append(comfy.utils.reshape_mask(parts[i].float(), shape).to(stream.device).clone())
        else:
            out.append(torch.ones(shape, device=stream.device))
    return out


def _overlap(chain, overlap, who):
    """The overlap from Luna-H3-AssetLoader's sound card, checked against this chain."""
    if overlap is None:
        raise ValueError(f"{LOG} {who}: connect Luna-H3-AssetLoader overlap to overlap.")
    n = int(overlap)
    if str(n) not in OVERLAPS:
        raise ValueError(f"{LOG} {who}: the overlap is {n} frames; set the sound card's overlap on Luna-H3-AssetLoader to "
                         f"one of {', '.join(OVERLAPS)} (a song must be loaded on it).")
    if n >= chain["frames"]:
        raise ValueError(f"{LOG} {who}: the overlap ({n} frames) must be shorter than a clip ({chain['frames']} frames).")
    if n == LOCKED_ONLY and not chain["song_locked"]:
        raise ValueError(f"{LOG} {who}: 22 frames are 36.67 audio steps, so the previous clip's sound cannot be copied "
                         f"exactly. Lock the song (song_locked on) or set the overlap to 39, 90, 141 or 192.")
    return n


def _advance(manager, clips, prompt, unique_id):
    """This run's clip index, from the batch manager's requeue count (TrentNodes H3ChainStep)."""
    uid = str(prompt[str(unique_id)]["inputs"]["meta_batch"][0])
    step = int(prompt[uid]["inputs"].get("requeue", 0))
    if not 0 <= step < clips:
        raise ValueError(f"{LOG} the run went past {clips} clips; queue a fresh run.")
    if getattr(manager, "inputs", None):
        raise ValueError(f"{LOG} the batch manager drives the clips here; disconnect other batch loaders from it.")
    if step == 0:
        # The manager resets its id after finishing, and its cached output can outlive the run.
        manager.reset()
        manager.unique_id = uid
    elif manager.unique_id != uid:
        raise ValueError(f"{LOG} the batch state was lost; queue a fresh run.")
    manager.total_frames = clips * manager.frames_per_batch
    manager.has_closed_inputs = step == clips - 1
    return step


class LunaH3Chain:
    DESCRIPTION = (
        "Makes one video from several H3 clips in one queue. Connect a Luna-MetaBatch (its frames_per_batch is "
        "every clip's length, on the 17k+5 grid), set how many clips, and queue once. Every clip after the first "
        "starts with the last frames of the clip before, locked, so the picture carries on.\n\n"
        "The song start, the first segment and the overlap are set on Luna-H3-AssetLoader's sound card (Position: By "
        "shot number). Wire chain -> Luna-H3-AssetLoader chain, Luna-H3-ChainContinue, Luna-H3-ChainRemember and "
        "Luna-H3-ChainJoin; the loader's overlap -> the same three; seed -> the sampler's noise_seed; prompt -> the "
        "prompt writer."
    )
    CATEGORY = "Luna-MinimaxH3"
    FUNCTION = "run"
    RETURN_TYPES = (CHAIN, "INT", "INT", "INT", "FLOAT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("chain", "seed", "clip", "frames", "seconds", "prompt", "is_first_clip")
    OUTPUT_TOOLTIPS = (
        "Connect to Luna-H3-AssetLoader chain, Luna-H3-ChainContinue, Luna-H3-ChainRemember and Luna-H3-ChainJoin.",
        "Seed of this clip: seed + clip index. Connect to Luna-H3-Sampler noise_seed.",
        "This clip's number in the chain, from 1.",
        "Every clip's length in frames, from the batch manager.",
        "Every clip's length in seconds, frames / 24. Connect to Luna-H3-AssetLoader seconds or Luna-H3-Canvas.",
        "first_prompt on clip 1, later_prompt on every other clip.",
        "True on clip 1 only.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "meta_batch": ("VHS_BatchManager", {"forceInput": True,
                    "tooltip": "Luna-MetaBatch.meta_batch (or VHS's Meta Batch Manager). It reruns the graph once per "
                               "clip; its frames_per_batch is every clip's length. Wire the same meta_batch into the "
                               "Luna-SaveVideo that takes Luna-H3-ChainJoin's outputs."}),
                "clips": ("INT", {"default": 3, "min": 1, "max": 1000,
                    "tooltip": "How many clips this run makes."}),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Seed of clip 1; each later clip adds 1."}),
                "song_locked": ("BOOLEAN", {"default": True,
                    "tooltip": "On: the song is locked in every clip's latent (audio mask 0), so the previous clip's sound "
                               "is not copied; the overlap may be 22. Off: H3 makes the sound, the head copies the "
                               "previous clip's sound too, and the overlap must be 39 or more."}),
            },
            "optional": {
                "first_prompt": ("STRING", {"forceInput": True, "tooltip": "Prompt for clip 1."}),
                "later_prompt": ("STRING", {"forceInput": True,
                    "tooltip": "Prompt for every later clip. Not connected: first_prompt for all clips."}),
                "clips_already_made": ("INT", {"default": 0, "min": 0, "max": 1000,
                    "tooltip": "0 starts a new video. To add clips to the last run, set how many it made; this run "
                               "continues from its last clip. Keep every other setting the same."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, meta_batch, clips, seed, song_locked, prompt=None, unique_id=None,
            first_prompt=None, later_prompt=None, clips_already_made=0):
        frames = int(meta_batch.frames_per_batch)
        if not on_grid(frames):
            raise ValueError(f"{LOG} the batch manager's frames_per_batch is {frames}; H3 clips are 17k+5 frames "
                             f"(124, 141, 209, 243, 328, 362 ...).")
        step = _advance(meta_batch, int(clips), prompt, unique_id)
        clip = int(clips_already_made) + step
        chain = {"clip": clip, "first_clip": int(clips_already_made), "song_locked": bool(song_locked),
                 "frames": frames, "manager": meta_batch}
        first = clip == 0
        text = first_prompt if first or later_prompt is None else later_prompt
        logging.info("%s clip %d of %d", LOG, clip + 1, int(clips_already_made) + int(clips))
        return (chain, (int(seed) + clip) % (1 << 64), clip + 1, frames, frames / FPS,
                ExecutionBlocker(None) if text is None else text, first)


class LunaH3ChainContinue:
    DESCRIPTION = (
        "On clip 1 this hands the latent on unchanged. On every later clip it copies the saved end of the clip before "
        "into the start of this latent and locks it (noise mask 0), so the video carries on without a jump. A mask "
        "already on the latent, such as a locked song, stays as it is outside the head.\n\n"
        "Wire the clip's latent before sampling (after the song lock) into latent, then latent -> Luna-H3-Sampler "
        "latent_image and low_res_head -> Luna-H3-Sampler low_res_head."
    )
    CATEGORY = "Luna-MinimaxH3"
    FUNCTION = "continue_clip"
    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "low_res_head", "start_image")
    OUTPUT_TOOLTIPS = (
        "Latent for Luna-H3-Sampler latent_image. Clip 1: unchanged. Later clips: starts with the end of the clip before, locked.",
        "The clip before's pass 1 tail, for Luna-H3-Sampler low_res_head. Empty on clip 1, or when no low_res_latent was saved.",
        "first_image on clip 1, the first copied frame on later clips. Connect to first_frame of MiniMax H3 Image to Video.",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chain": (CHAIN, {"tooltip": "From Luna-H3-Chain."}),
                "overlap": ("INT", {"forceInput": True,
                    "tooltip": "Frames each clip shares with the one before: Luna-H3-AssetLoader overlap (its sound card)."}),
                "latent": ("LATENT", {"tooltip": "This clip's H3 audio+video latent before sampling, after the song lock."}),
                "name": ("STRING", {"default": "chain",
                    "tooltip": "Same name as on Luna-H3-ChainRemember. The endings sit in output/luna_h3_chain/<name>/."}),
                "keep_look_of_clip_1": ("BOOLEAN", {"default": True,
                    "tooltip": "Puts one frame of clip 1 into the hidden first frame of every later clip, so colour and "
                               "faces drift less over a long chain. Join cuts that frame away, so it never shows."}),
            },
            "optional": {
                "first_image": ("IMAGE", {"tooltip": "Start image of clip 1."}),
                "vae": ("VAE", {"tooltip": "Video VAE. Needed only for start_image on later clips."}),
            },
        }

    def continue_clip(self, chain, latent, name, keep_look_of_clip_1, first_image=None, vae=None, overlap=None):
        who = "Luna-H3-ChainContinue"
        n = _overlap(chain, overlap, who)
        if chain["song_locked"]:
            video, audio = _streams(latent, who)
            if not torch.all(_masks(latent, video, audio)[1] == 0):
                raise ValueError(f"{LOG} {who}: Luna-H3-Chain says the song is locked, but this latent's sound is not "
                                 f"(its audio mask is not 0). Lock the song, or turn song_locked off and set the overlap "
                                 f"to 39 or more.")
        if chain["clip"] == 0:
            return (latent, None, ExecutionBlocker(None) if first_image is None else first_image)

        path = ending_path(name, chain["clip"] - 1)
        if not os.path.isfile(path):
            raise ValueError(f"{LOG} {who}: no saved ending at {path}. Use the same name on Luna-H3-ChainRemember "
                             f"and queue a fresh run.")
        import safetensors.torch
        data = safetensors.torch.load_file(path, device="cpu")
        video, audio = _streams(latent, who)
        v_n, a_n = video_steps(n), audio_steps(n)
        src_video = data["video"]
        if src_video.shape[2] < v_n or v_n >= video.shape[2]:
            raise ValueError(f"{LOG} {who}: an overlap of {n} frames does not fit this clip or the saved ending.")
        if tuple(src_video.shape[:2]) + tuple(src_video.shape[3:]) != tuple(video.shape[:2]) + tuple(video.shape[3:]):
            raise ValueError(f"{LOG} {who}: the saved ending {tuple(src_video.shape)} and this latent "
                             f"{tuple(video.shape)} differ in size. Keep every clip at one size.")
        video_mask, audio_mask = _masks(latent, video, audio)
        video = video.clone()
        video[:, :, :v_n] = src_video[:, :, -v_n:].to(video)
        video_mask[:, :, :v_n] = 0.0
        if not chain["song_locked"]:
            if a_n >= audio.shape[-1] or data["audio"].shape[-1] < a_n:
                raise ValueError(f"{LOG} {who}: an overlap of {n} frames does not fit this clip's sound.")
            audio = audio.clone()
            audio[..., :a_n] = data["audio"][..., -a_n:].to(audio)
            audio_mask[..., :a_n] = 0.0

        low_head = None
        if "low_video" in data:
            low_head = {"samples": data["low_video"][:, :, -v_n:].clone()}

        if keep_look_of_clip_1:
            first = safetensors.torch.load_file(ending_path(name, 0), device="cpu")
            # Each saved tail starts on a one-frame latent (overlap 5 + 17k -> 2 + 5k latents), so index 0 is one frame.
            video[:, :, :1] = first["video"][:, :, -v_n:][:, :, :1].to(video)
            if low_head is not None and "low_video" in first:
                low_head["samples"][:, :, :1] = first["low_video"][:, :, -v_n:][:, :, :1]

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        if vae is None:
            start = ExecutionBlocker(f"{LOG} {who}: connect the video VAE to get start_image on later clips.")
        else:
            # The first two latents decode to five frames; the first one is the start image.
            start = vae.decode(video[:, :, :2])
            start = start.reshape(-1, *start.shape[-3:])[:1]
        logging.info("%s clip %d continues from %s (%d frames%s)", LOG, chain["clip"] + 1, path, n,
                     ", low-res head" if low_head is not None else "")
        return (out, low_head, start)


class LunaH3ChainRemember:
    DESCRIPTION = (
        "Saves the end of this clip so the next clip can continue from it: the video and audio tail of latent, and "
        "the video tail of low_res_latent when it is connected. Clip 1 clears endings left from an earlier run of the "
        "same name.\n\nWire Luna-H3-Sampler latent -> latent and low_res_latent -> low_res_latent, saved -> "
        "Luna-H3-ChainJoin saved."
    )
    CATEGORY = "Luna-MinimaxH3"
    FUNCTION = "remember"
    OUTPUT_NODE = True
    RETURN_TYPES = (SAVED,)
    RETURN_NAMES = ("saved",)
    OUTPUT_TOOLTIPS = ("Connect to Luna-H3-ChainJoin saved, so the next clip starts only after this ending is on disk.",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chain": (CHAIN, {"tooltip": "From Luna-H3-Chain."}),
                "overlap": ("INT", {"forceInput": True,
                    "tooltip": "Frames each clip shares with the one before: Luna-H3-AssetLoader overlap (its sound card)."}),
                "latent": ("LATENT", {"tooltip": "The sampled latent of this clip: Luna-H3-Sampler latent."}),
                "name": ("STRING", {"default": "chain", "tooltip": "Same name as on Luna-H3-ChainContinue."}),
            },
            "optional": {
                "low_res_latent": ("LATENT", {
                    "tooltip": "Luna-H3-Sampler low_res_latent. Saved, it becomes the next clip's low_res_head."}),
                "save_whole_clip": ("BOOLEAN", {"default": False,
                    "tooltip": "Also save this clip's whole latent (video, audio, low-res video) as whole_NNNN.latent "
                               "beside the endings, about 60 MB per 8 s clip at 2 MP, to decode or rejoin it later."}),
            },
        }

    def remember(self, chain, latent, name, low_res_latent=None, save_whole_clip=False, overlap=None):
        video, audio = _streams(latent, "Luna-H3-ChainRemember")
        n = _overlap(chain, overlap, "Luna-H3-ChainRemember")
        path = ending_path(name, chain["clip"])
        folder = os.path.dirname(path)
        if chain["clip"] == chain["first_clip"] == 0:
            shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)
        tensors = {
            "video": video[:, :, -video_steps(n):].detach().cpu().contiguous(),
            "audio": audio[..., -audio_steps(n):].detach().cpu().contiguous(),
        }
        if low_res_latent is not None:
            low = low_res_latent["samples"]
            low = low.unbind()[0] if getattr(low, "is_nested", False) else low
            tensors["low_video"] = low[:, :, -video_steps(n):].detach().cpu().contiguous()
        import safetensors.torch
        safetensors.torch.save_file(tensors, path)
        if save_whole_clip:
            whole = {"video": video.detach().cpu().contiguous(), "audio": audio.detach().cpu().contiguous()}
            if low_res_latent is not None:
                whole["low_video"] = low.detach().cpu().contiguous()
            safetensors.torch.save_file(whole, os.path.join(folder, "whole_%04d.latent" % chain["clip"]))
        return ([path],)


class LunaH3ChainJoin:
    DESCRIPTION = (
        "Cuts the copied frames off every clip after the first and keeps each clip's sound for the end, so the batch "
        "manager's Luna-SaveVideo gets one continuous video and one soundtrack.\n\nWire the decoded frames into "
        "images, the clip's sound (the song window from Luna-H3-AssetLoader audio, or the decoded audio) into audio, "
        "Luna-H3-ChainRemember saved into saved; images and audio -> Luna-SaveVideo with meta_batch. To keep every "
        "clip on disk, so a stopped run loses nothing: images and clip_audio -> a second Luna-SaveVideo without "
        "meta_batch."
    )
    CATEGORY = "Luna-MinimaxH3"
    FUNCTION = "join"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "AUDIO", "AUDIO")
    RETURN_NAMES = ("images", "audio", "clip_audio")
    OUTPUT_TOOLTIPS = (
        "This clip without the copied frames. Connect to Luna-SaveVideo images (the one with meta_batch).",
        "The whole soundtrack on the last clip. Connect to the same Luna-SaveVideo audio.",
        "This clip's own sound, cut to images, on every clip. Connect to a per-clip Luna-SaveVideo (no meta_batch).",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chain": (CHAIN, {"tooltip": "From Luna-H3-Chain."}),
                "overlap": ("INT", {"forceInput": True,
                    "tooltip": "Frames each clip shares with the one before: Luna-H3-AssetLoader overlap (its sound card)."}),
                "images": ("IMAGE", {"tooltip": "Decoded frames of this clip."}),
                "audio": ("AUDIO", {"tooltip": "Sound of this clip: the song window or the decoded audio."}),
                "saved": (SAVED, {"tooltip": "From Luna-H3-ChainRemember. The next clip starts only after the ending is saved."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def join(self, chain, images, audio, saved, overlap=None):
        clip, manager = chain["clip"], chain["manager"]
        if images.shape[0] != chain["frames"]:
            raise ValueError(f"{LOG} Luna-H3-ChainJoin: the clip has {images.shape[0]} frames, the chain makes "
                             f"{chain['frames']} at 24 fps. Frame generation (Interpolation) changes the count: turn it "
                             f"off for a chain.")
        n = _overlap(chain, overlap, "Luna-H3-ChainJoin")
        n = 0 if clip == 0 else n
        images = images[n:]

        # Cut and pad the sound to the picture's length, so small offsets cannot add up down the chain. H3's own sound
        # can be a few milliseconds short of the picture; that gap is filled with the clip's last sound mirrored, not
        # silence, so the join does not click.
        sr = int(audio["sample_rate"])
        start = round(n / FPS * sr)
        length = round(images.shape[0] / FPS * sr)
        wave = audio["waveform"][..., start:start + length].float()
        short = length - wave.shape[-1]
        if 0 < short < wave.shape[-1] and wave.ndim == 3:
            wave = torch.nn.functional.pad(wave, (0, short), mode="reflect")
        else:
            wave = torch.nn.functional.pad(wave, (0, max(0, short)))
        wave = wave.detach().to(device="cpu", dtype=torch.float32).contiguous()

        import safetensors.torch
        folder = os.path.dirname(saved[-1])
        safetensors.torch.save_file({"waveform": wave, "sample_rate": torch.tensor(sr)},
                                    os.path.join(folder, "sound_%04d.safetensors" % clip))
        clip_audio = {"waveform": wave, "sample_rate": sr}
        out = clip_audio
        if manager.has_closed_inputs:
            # The meta-batch save ignores the sound until the last clip, then muxes the whole track.
            chunks = []
            for k in range(chain["first_clip"], clip + 1):
                part_path = os.path.join(folder, "sound_%04d.safetensors" % k)
                if not os.path.isfile(part_path):
                    raise ValueError(f"{LOG} Luna-H3-ChainJoin: the sound of clip {k + 1} is missing ({part_path}). "
                                     f"Queue a fresh run.")
                part = safetensors.torch.load_file(part_path)
                if int(part["sample_rate"]) != sr or part["waveform"].shape[:-1] != wave.shape[:-1]:
                    raise ValueError(f"{LOG} Luna-H3-ChainJoin: the sample rate or the channels changed between clips.")
                chunks.append(part["waveform"])
            out = {"waveform": torch.cat(chunks, dim=-1), "sample_rate": sr}
        logging.info("%s clip %d joined after %s", LOG, clip + 1, ", ".join(saved))
        return (images, out, clip_audio)


NODE_CLASS_MAPPINGS = {
    "Luna-H3-Chain": LunaH3Chain,
    "Luna-H3-ChainContinue": LunaH3ChainContinue,
    "Luna-H3-ChainRemember": LunaH3ChainRemember,
    "Luna-H3-ChainJoin": LunaH3ChainJoin,
}
NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}

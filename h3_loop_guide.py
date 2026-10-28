"""Luna-H3 guide: keyframe latents (and, optionally, the same frames to the VLM) for a loop scene.

S12 of plans/S9-S14-loop-nodes.md. Default off. Three nodes around the SelfLift graph:

- Luna-H3-LoopScene builds the scene's guide (`LUNA_H3_GUIDE`): which frames, at which render frame, with which audio.
- **Luna-H3-GuideApply** adds the guide to the positive conditioning as `minimax_keyframes` rows: clean
  latent rows aligned to target time, the same thing core `MiniMaxH3AddGuide` does (adapted from it, see
  THIRD_PARTY_NOTICES.md). The VLM never sees them.
- **Luna-H3-GuideClip** (add_guide_vlm only) hands the same frames to the VLM, the FL2VA pairing of keyframe
  latents + keyframe images (core `MiniMaxH3ImageToVideo`). Node 66 (`MiniMaxH3ReferenceToVideo`) tokenizes its own
  prompt and references inside `execute`, and the tokenizer takes `minimax_ref_items` OR `images`, so the least
  invasive way in is a clone of the CLIP whose `tokenize` appends the guide frames to `minimax_ref_items`: node 66
  stays the core node, and the references' own items, order and latents are untouched. Guide pictures go in after
  the last reference picture (reference ordinals do not move), a guide clip as a 2 fps video item after the last
  reference video, exactly the way core presents a reference video to the VLM.

Scene fields (override the Luna-H3-LoopScene defaults per scene): guide_mode (off | add_guide | add_guide_vlm),
guide_source (previous_tail | clip | images), guide_length (5 | 22 | 39), guide_frame_idx, guide_audio,
guide_clip (a video file), guide_images ([{"path": ..., "frame": ...}]).

`guide_frame_idx` counts from the scene's FIRST VISIBLE frame (negative: back from the end of the render, as core
Add Guide). The carried head is the first `trim_frames` frames of the render and is already pinned by the masked
latent, so a guide index never lands inside it.

Licence: GPL-3.0. `make_keyframe` and the VLM item layout are adapted from ComfyUI core
(comfy_extras/nodes_minimax_h3.py: MiniMaxH3AddGuide, MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo),
GPL-3.0.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path
from typing import Any

import torch
from comfy_api.latest import io

try:
    from .h3_loop import LOG, LoopState, _scene_stem  # noqa: F401
except ImportError:  # flat import in the unit tests
    from h3_loop import LOG, LoopState, _scene_stem  # noqa: F401

logger = logging.getLogger("Luna.H3Guide")

MODES = ("off", "add_guide", "add_guide_vlm")
SOURCES = ("previous_tail", "clip", "images")
LENGTHS = (5, 22, 39)
FPS = 24
DEFAULTS = {"guide_mode": "off", "guide_source": "previous_tail", "guide_length": 22, "guide_frame_idx": 0,
            "guide_audio": False}

GuideType = io.Custom("LUNA_H3_GUIDE")


def _join():
    try:
        from . import h3_loop_join
    except ImportError:
        import h3_loop_join
    return h3_loop_join


def _scene_mod():
    try:
        from . import h3_loop_scene
    except ImportError:
        import h3_loop_scene
    return h3_loop_scene


# --------------------------------------------------------------------------- settings

def guide_fields(scene: dict, defaults: dict, index: int) -> dict:
    """The scene's effective guide settings (scene fields over the node defaults), validated."""
    where = f"{LOG} scene {index} guide"
    d = {**DEFAULTS, **{k: v for k, v in defaults.items() if v is not None}}
    mode = scene.get("guide_mode", d["guide_mode"])
    if mode not in MODES:
        raise ValueError(f"{where}: guide_mode must be one of {MODES}, got {mode!r}.")
    source = scene.get("guide_source", d["guide_source"])
    if source not in SOURCES:
        raise ValueError(f"{where}: guide_source must be one of {SOURCES}, got {source!r}.")
    length = scene.get("guide_length", d["guide_length"])
    if length not in LENGTHS:
        raise ValueError(f"{where}: guide_length must be one of {LENGTHS} (the clip lengths H3 accepts), got {length!r}.")
    idx = scene.get("guide_frame_idx", d["guide_frame_idx"])
    if type(idx) is not int or not -9999 <= idx <= 9999:
        raise ValueError(f"{where}: guide_frame_idx must be a whole number.")
    audio = scene.get("guide_audio", d["guide_audio"])
    if not isinstance(audio, bool):
        raise ValueError(f"{where}: guide_audio must be true or false.")
    # A guide that comes only from the node defaults and wants the previous tail applies where there is a usable
    # tail: a fresh scene. The first scene and carried scenes (which start from the tail as a pinned latent) go
    # without. A scene that asks for it in its own fields and cannot have it is an error (check_guides).
    carried = index > 0 and scene.get("join", "carried") != "fresh"
    if (mode != "off" and source == "previous_tail" and "guide_mode" not in scene and "guide_source" not in scene
            and (index == 0 or carried)):
        mode = "off"
    out = {"mode": mode, "source": source, "length": int(length), "frame_idx": idx, "audio": audio,
           "clip": scene.get("guide_clip"), "images": scene.get("guide_images")}
    if mode == "off":
        return out
    if source == "clip" and (not isinstance(out["clip"], str) or not out["clip"].strip()):
        raise ValueError(f"{where}: guide_source clip needs guide_clip, the path of a video file.")
    if source == "images":
        imgs = out["images"]
        if (not isinstance(imgs, list) or not imgs
                or any(not isinstance(x, dict) or not isinstance(x.get("path"), str) or type(x.get("frame", 0)) is not int
                       for x in imgs)):
            raise ValueError(f"{where}: guide_source images needs guide_images: [{{\"path\": ..., \"frame\": 0}}, ...].")
    return out


def check_guides(scenes: list[dict], defaults: dict, audio_mode: str) -> None:
    """Every scene's guide, before anything renders: fields valid, files present, and no guide the loop cannot use."""
    for i, scene in enumerate(scenes):
        g = guide_fields(scene, defaults, i)
        if g["mode"] == "off":
            continue
        carried = i > 0 and scene.get("join", "carried") != "fresh"
        if g["source"] == "previous_tail":
            if i == 0:
                raise ValueError(f"{LOG} scene 0 guide: previous_tail needs a previous scene.")
            if carried:
                raise ValueError(
                    f"{LOG} scene {i} guide: previous_tail on a carried scene is refused. A carried scene already "
                    "starts from the previous scene's last frames as a pinned latent (the hard head); a second copy "
                    "as a keyframe would anchor the same picture twice. Use join: fresh for a scene that should "
                    "start from the previous tail through a guide, or another guide_source.")
        if g["audio"] and audio_mode == "song_lock":
            raise ValueError(f"{LOG} scene {i} guide: guide_audio is for audio_mode generated; a song-locked run's "
                             "audio is the locked song.")
        paths = [g["clip"]] if g["source"] == "clip" else [x["path"] for x in g["images"]] if g["source"] == "images" else []
        for p in paths:
            if not Path(p).is_file():
                raise FileNotFoundError(f"{LOG} scene {i} guide: file not found: {p}")


def render_index(frame_idx: int, trim: int, render_frames: int, count: int, where: str) -> int:
    """Render-frame index of a guide placed `frame_idx` frames from the first visible frame (negative: from the end)."""
    idx = trim + frame_idx if frame_idx >= 0 else render_frames + frame_idx
    if idx < trim:
        raise ValueError(f"{where}: guide_frame_idx {frame_idx} lands in the hidden head (render frames 0..{trim - 1}), "
                         "which the carried latent already pins.")
    if idx + count > render_frames:
        raise ValueError(f"{where}: a {count}-frame guide at guide_frame_idx {frame_idx} does not fit in the "
                         f"{render_frames} rendered frames.")
    return idx


# --------------------------------------------------------------------------- the frames

def _to_tensor(frames: list) -> torch.Tensor:
    import numpy as np
    return torch.from_numpy(np.stack(frames).astype("float32") / 255.0)


def read_tail(root: Path, scene_index: int, n: int) -> torch.Tensor:
    """The previous scene's last `n` delivered frames [n, H, W, 3] in 0..1: cut to exactly n, because core Add Guide
    keeps the FIRST 17k+5 frames of a longer batch."""
    j = _join()
    manifest = json.loads((Path(root) / f"{_scene_stem(scene_index)}.manifest.json").read_text(encoding="utf-8"))
    info = manifest["info"].get("frames")
    if not info:
        raise ValueError(f"{LOG} scene {scene_index} has no saved frames to take a guide from.")
    if int(info["count"]) < n:
        raise ValueError(f"{LOG} scene {scene_index} has {info['count']} frames; a {n}-frame tail guide needs {n}.")
    tail: deque = deque(maxlen=n)
    for frame in j.iter_frames(j.frame_source(Path(root), _scene_stem(scene_index), info)):
        tail.append(frame)
    return _to_tensor(list(tail))


def read_clip(path: str, length: int) -> torch.Tensor:
    """The first `length` frames of a video file, or the longest valid clip (5, 22, 39...) that the file has."""
    import av
    frames = []
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            frames.append(f.to_ndarray(format="rgb24"))
            if len(frames) >= length:
                break
    if not frames:
        raise ValueError(f"{LOG} {path} has no video frames.")
    n = len(frames)
    if n >= 5:
        while n % 17 != 5:
            n -= 1
    else:
        n = 1
    return _to_tensor(frames[:n])


def read_image(path: str) -> torch.Tensor:
    from PIL import Image
    import numpy as np
    with Image.open(path) as im:
        return torch.from_numpy(np.asarray(im.convert("RGB"), dtype="float32") / 255.0)[None]


def audio_slice(wave: torch.Tensor, rate: int, frames: int, tail: bool) -> dict:
    """The audio of `frames` video frames: the last ones of a scene (tail) or the first ones of a clip."""
    n = round(frames * rate / FPS)
    seg = wave[:, -n:] if tail else wave[:, :n]
    return {"waveform": seg.unsqueeze(0), "sample_rate": int(rate)}


def build_guide(g: dict, *, root: Path, index: int, trim: int, render_frames: int) -> dict:
    """The guide for one scene: {"mode", "source", "keyframes": [{"index": render frame, "images", "audio"}]}.
    Raises if the frames cannot be found or do not fit."""
    where = f"{LOG} scene {index} guide"
    if g["mode"] == "off":
        return {"mode": "off", "source": g["source"], "keyframes": []}
    keyframes = []
    if g["source"] == "previous_tail":
        frames = read_tail(root, index - 1, g["length"])
        audio = None
        if g["audio"]:
            wav = Path(root) / f"{_scene_stem(index - 1)}.wav"
            if not wav.is_file():
                raise FileNotFoundError(f"{where}: guide_audio needs scene {index - 1}'s saved audio ({wav.name}).")
            wave, rate = _scene_mod().load_audio_file(wav)
            audio = audio_slice(wave, rate, g["length"], tail=True)
        keyframes.append({"index": render_index(g["frame_idx"], trim, render_frames, frames.shape[0], where),
                          "images": frames, "audio": audio})
    elif g["source"] == "clip":
        frames = read_clip(g["clip"], g["length"])
        audio = None
        if g["audio"]:
            wave, rate = _scene_mod().load_audio_file(Path(g["clip"]))
            audio = audio_slice(wave, rate, frames.shape[0], tail=False)
        keyframes.append({"index": render_index(g["frame_idx"], trim, render_frames, frames.shape[0], where),
                          "images": frames, "audio": audio})
    else:
        for item in g["images"]:
            keyframes.append({"index": render_index(int(item.get("frame", 0)), trim, render_frames, 1, where),
                              "images": read_image(item["path"]), "audio": None})
    return {"mode": g["mode"], "source": g["source"], "keyframes": keyframes}


# --------------------------------------------------------------------------- the keyframe (adapted from core)

def _resize(image: torch.Tensor, width: int, height: int, crop: str) -> torch.Tensor:
    import comfy.utils
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def valid_clip_length(n: int) -> int:
    """Core Add Guide: batches under 5 frames use one image, longer ones are cut to the largest 17k + 5."""
    if n < 5:
        return 1
    while n % 17 != 5:
        n -= 1
    return n


def make_keyframe(latent: dict, vae, audio_vae, resolved_index: int, images: torch.Tensor, audio: dict | None) -> dict:
    """One `minimax_keyframes` entry, as core MiniMaxH3AddGuide builds it (adapted)."""
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
    samples = latent["samples"]
    if (not getattr(samples, "is_nested", False) or len(samples.tensors) != 2 or samples.tensors[0].ndim != 5
            or samples.tensors[0].shape[1] != 24):
        raise ValueError(f"{LOG} Luna-H3-GuideApply expects a MiniMax H3 AV latent.")
    video = samples.tensors[0]
    height, width = video.shape[3] * 16, video.shape[4] * 16
    frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(video.shape[2]))
    n = valid_clip_length(images.shape[0])
    if resolved_index < 0 or resolved_index + n > frame_count:
        raise ValueError(f"{LOG} a {n}-frame guide at render frame {resolved_index} does not fit in the video's "
                         f"{frame_count} frames.")
    keyframe: dict[str, Any] = {"resolved_frame_index": resolved_index}
    if vae is None:
        raise ValueError(f"{LOG} Luna-H3-GuideApply needs the video vae.")
    keyframe["latent"] = vae.encode(_resize(images[:n], width, height, "center"))
    if audio is not None:
        if audio_vae is None:
            raise ValueError(f"{LOG} guide audio needs the audio_vae input.")
        import comfy.audio
        waveform, rate = audio["waveform"], audio["sample_rate"]
        vae_rate = getattr(audio_vae, "audio_sample_rate", 32000)
        if rate != vae_rate:
            waveform = comfy.audio.resample(waveform, rate, vae_rate)
        z = audio_vae.encode(waveform[:1].movedim(1, -1))
        max_rt = math.floor(samples.tensors[1].shape[-1] - FRAME_RESCALE * resolved_index)
        if max_rt < 1:
            raise ValueError(f"{LOG} guide frame {resolved_index} is past the end of the video's audio track.")
        if z.shape[-1] > max_rt:
            z = z[..., :max_rt].clone()
        keyframe["audio_latent"] = z
    return keyframe


def apply_guide(positive, latent: dict, vae, audio_vae, guide: dict | None):
    """Conditioning with the guide's keyframes added; unchanged (the same object) when the guide is off."""
    if not guide or guide.get("mode", "off") == "off" or not guide.get("keyframes"):
        return positive
    import node_helpers
    keyframes = list(positive[0][1].get("minimax_keyframes", []))
    for kf in guide["keyframes"]:
        keyframes.append(make_keyframe(latent, vae, audio_vae, int(kf["index"]), kf["images"], kf.get("audio")))
    return node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})


# --------------------------------------------------------------------------- the VLM side

def vlm_items(guide: dict, width: int, height: int) -> list[dict]:
    """The guide frames as tokenizer items: one frame an image item, a clip a 2 fps video item (core's layout)."""
    items = []
    for kf in guide.get("keyframes", []):
        images = kf["images"]
        n = valid_clip_length(images.shape[0])
        frames = _resize(images[:n], width, height, "center")
        if n == 1:
            items.append({"type": "image", "data": frames[:1]})
        else:
            idx = list(range(0, n, FPS // 2))
            items.append({"type": "video", "data": frames[idx], "timestamps": [i / 2.0 for i in range(len(idx))]})
    return items


def insert_guide_items(ref_items: list[dict], guide_items: list[dict]) -> list[dict]:
    """Guide pictures after the last reference picture, guide videos after the last reference video (before any
    standalone audio): the references keep their position and their <Picture i> / <Video k> numbers."""
    out = list(ref_items)
    pictures = [g for g in guide_items if g["type"] == "image"]
    videos = [g for g in guide_items if g["type"] == "video"]
    last = max((i for i, it in enumerate(out) if it["type"] == "image"), default=-1)
    out[last + 1:last + 1] = pictures
    last_video = max((i for i, it in enumerate(out) if it["type"] == "video"), default=-1)
    if last_video >= 0:
        out[last_video + 1:last_video + 1] = videos
    else:
        first_audio = next((i for i, it in enumerate(out) if it["type"] == "audio"), len(out))
        out[first_audio:first_audio] = videos
    return out


def vlm_clip(clip, guide: dict | None, width: int, height: int):
    """`clip` itself unless the guide is add_guide_vlm; else a clone whose tokenize also presents the guide frames."""
    if not guide or guide.get("mode") != "add_guide_vlm" or not guide.get("keyframes"):
        return clip
    items = vlm_items(guide, width, height)
    out = clip.clone()
    inner = out.tokenize

    def tokenize(text, return_word_ids=False, **kwargs):
        ref = kwargs.get("minimax_ref_items")
        if ref:
            kwargs["minimax_ref_items"] = insert_guide_items(ref, items)
        else:       # no references: the keyframe-image path (FL2VA), pictures only
            pics = list(kwargs.get("images", []))
            for it in items:
                pics += [it["data"][k:k + 1] for k in range(it["data"].shape[0])] if it["type"] == "video" else [it["data"]]
            kwargs["images"] = pics
        return inner(text, return_word_ids=return_word_ids, **kwargs)

    out.tokenize = tokenize
    return out


# --------------------------------------------------------------------------- nodes

class LunaH3GuideApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Luna-H3-GuideApply",
            display_name="Luna-H3-GuideApply",
            category="Luna-MinimaxH3",
            description=(
                "Adds Luna-H3-LoopScene's guide to the positive conditioning as clean keyframe rows aligned to target "
                "time, as core Add Guide for MiniMax H3 does (the VLM does not see them). Off: the conditioning "
                "comes through unchanged. Use after the conditioning and before the samplers; give it the "
                "empty AV latent of the reference node for the frame count."),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Latent.Input("latent", tooltip="An H3 AV latent of the scene's length (node 66's)."),
                GuideType.Input("guide", tooltip="From Luna-H3-LoopScene."),
                io.Vae.Input("vae", optional=True, tooltip="The video VAE (needed when the guide is on)."),
                io.Vae.Input("audio_vae", optional=True, tooltip="The audio VAE (needed for guide audio)."),
            ],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, positive, latent, guide, vae=None, audio_vae=None):
        return io.NodeOutput(apply_guide(positive, latent, vae, audio_vae, guide))


class LunaH3GuideClip(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Luna-H3-GuideClip",
            display_name="Luna-H3-GuideClip",
            category="Luna-MinimaxH3",
            description=(
                "For guide_mode add_guide_vlm: a CLIP whose tokenizer also shows the guide frames to the VLM (the "
                "FL2VA pairing: keyframe latents plus keyframe images). Put it between the text encoder and the "
                "reference node; the references keep their own items and numbers. In any other mode the CLIP "
                "comes through unchanged."),
            inputs=[
                io.Clip.Input("clip"),
                GuideType.Input("guide", tooltip="From Luna-H3-LoopScene."),
                io.Int.Input("width", default=1920, min=32, max=16384, step=32,
                             tooltip="The generation canvas the guide frames are fitted to."),
                io.Int.Input("height", default=1088, min=32, max=16384, step=32),
            ],
            outputs=[io.Clip.Output(display_name="clip")],
        )

    @classmethod
    def execute(cls, clip, guide, width, height):
        return io.NodeOutput(vlm_clip(clip, guide, int(width), int(height)))

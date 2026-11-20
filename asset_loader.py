"""Luna-H3-AssetLoader — one ordered set of reference images, many consumers.

Loads up to 9 images and hands them out three ways at once:

  image_1 … image_N   each at its ORIGINAL size, in card order, for the native
                      MiniMax H3 sockets (ref_image_0 … ref_image_8, or slots
                      1 and 2 as first_frame / last_frame in FL2VA)
  images              one conformed IMAGE batch, for prompt writers that only
                      ever flatten it anyway (Luna Agent Chat, LLM Prompt)
  width / height      image 1's dimensions, to drive the generation nodes
  asset_brief         "<Picture 1> = 1920x1056 (20:11) …" for the prompt writer

Why originals and a batch, rather than one or the other: an IMAGE batch cannot
hold mixed sizes, and H3's REF2VA path deliberately keeps each reference at its
own aspect ratio — conforming them there would destroy information the model
wants. The numbered outputs stay untouched; only the batch is conformed.

Ordering is the whole point. Card order here is the same order H3 numbers its
references, so slot N is <Picture N> everywhere downstream, and the brief states
that mapping explicitly rather than leaving a prompt writer to guess.

Per-image resize is modelled on ComfyUI-Pixaroma
(https://github.com/pixaroma/ComfyUI-Pixaroma), MIT, Copyright (c) 2026 pixaroma.

The `asset_state` schema comes from that pack's `RESIZE_DEFAULTS` in
`nodes/_resize_helpers.py`: the key names and their defaults — `mode`
(off | max_mp | longest_side | scale_factor), `ratio_action`, `crop_anchor`,
`pad_color`, `snap`, `allow_upscale` — are theirs, as is the idea of
keeping the whole thing as a JSON state blob rather than a row of widgets. Credit
where it is due: that design is why nine images can each carry their own settings
without the node growing nine sets of controls.

The code here was written against that schema rather than copied from it, and
differs where this node's job differs: one state per image instead of one per
node, `allow_upscale` defaulting to False because MiniMax H3 never upscales a
reference, `snap` documented against H3's CANVAS_MULTIPLE of 32, and the order
fixed as aspect -> size -> snap.

Sound cards (v3, 2026-10-16). A sound card holds a song and the window of it the
shot uses: start, and what to do when the window runs off the end (silence or
loop). The window is stored in `media_state` (JSON) and comes out on `audio`,
exactly as long as the shot (frames / the shot's rate), so the song always matches
the shot length. Sound card 1 may carry a vocal stem (`vocals`), cut to the same
window. Window maths: media_io.

One count per run (Luna-MinimaxH3 SPEC § 1 and § 2): the shot's rate is
luna_units.fps_for(target, clip 1's delivered rate); `frames` is the wired seconds
(or the stored shot length) rounded UP onto the target's grid, past 362 on H3 a
warning in the brief, never a cut. The brief, the timeline, the placements, the
song, the vocal stem and the clip cards by shot number all use that count. The
overlap is one node setting (`settings.overlap`; an old save takes sound card 1's).

Outputs (SPEC § 2): the ten fixed outputs at slots 0-9, image_1..image_9 at 10-18 whatever
the number of picture cards, then `fps` (19) and `vocals` (20). The frontend sends each
link's own slot number with the prompt and ComfyUI reads that index of the returned tuple,
so the node always has all 21 and image_count only labels the image sockets past it.
Workflows saved before this order are migrated in the frontend before the graph is
configured (web/luna_asset_loader.js, migrateWorkflowJSON): the older fixed outputs go in
before image_1 as before, then the slots a save lacks are appended, so no saved link moves.

  video   the decoded frames of clip card V1 (whatever its role), for prompt writers
  frames  the shot length in frames, for the generation node's length and a prompt
          writer's frame count

Width and height: when the canvas size is wired (width AND height), those are
passed on, so the loader and the canvas can never disagree; picture 1's size only
when nothing is wired. An empty loader is not an error: it hands on empty outputs
and a brief that says "Nothing loaded.", so a graph can run before cards are added.

No continuation (removed 2026-10-29). Motion context is the previous render's
latent, copied into the new render's first frames and locked; it lives in the loop.
The old card decoded the previous video's pixels and placed them as a guide, which
is a different thing. An old media_state that still has a "continuation" key is read
and the key ignored.
"""

from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
import node_helpers

from . import luna_units
from .asset_images import apply_state, conform, to_rgb
from .asset_timeline import (ROW, check_placements, h3_tags, naming_cards, normalize_sound_role, shot_frames,
                             target_fps, target_grid, thinning_error)
from .luna_naming import inventory, make as make_naming
from .media_io import (FPS, clip_bytes, clip_soundtrack, decode_clip, filmstrip_jpeg, grid_frames, strip_annotation,
                       load_audio, load_audio_window, probe_media, song_shot_count, song_shot_start)

WHO = "Luna-H3-AssetLoader"
DEFAULT_OVERLAP = 22   # the node's overlap row starts here (and so did sound card 1's)

MAX_IMAGES = 9  # MiniMax H3's own reference ceiling
MAX_SOUNDS = 3  # MiniMax H3 takes three standalone sounds
MAX_VIDEOS = 3  # MiniMax H3 takes three reference clips
MEMORY_CAP = 6 * 1024 ** 3   # refuse a clip bigger than this BEFORE decoding it

# Aspect ratios a reader is likely to have a name for. Reported alongside the
# exact ratio, never instead of it: 832x1216 is exactly 13:19, and calling it
# "2:3" without saying so is how a wrong frame shape ends up in a prompt.
_COMMON_AR = [(1, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4),
              (9, 16), (16, 9), (1, 2), (2, 1), (21, 9)]


def _describe_ar(w: int, h: int) -> str:
    ratio = w / h
    a, b = min(_COMMON_AR, key=lambda c: abs(c[0] / c[1] - ratio))
    nearest = f"{a}:{b}"
    orient = "portrait" if ratio < 0.99 else ("landscape" if ratio > 1.01 else "square")

    g = math.gcd(w, h) or 1
    ew, eh = w // g, h // g
    if (ew, eh) == (a, b):
        return f"{w}x{h} ({nearest}, {orient})"
    if max(ew, eh) <= 32:
        # A small exact ratio is worth naming: 832x1216 really is 13:19, and
        # calling it 2:3 unqualified is how a wrong frame shape reaches a prompt.
        return f"{w}x{h} (exactly {ew}:{eh}, nearest standard {nearest}, {orient})"
    # 1672x941 reduces to 1672:941, which tells nobody anything. Give the decimal.
    return f"{w}x{h} ({ratio:.2f}:1, nearest standard {nearest}, {orient})"


def _split_paths(raw: str) -> list[str]:
    """One entry per line. The card UI will write this same field later."""
    out: list[str] = []
    for line in (raw or "").replace("\r", "\n").split("\n"):
        line = line.strip().strip('"')
        if line:
            out.append(line)
    return out


def _load(path: str, st: dict | None = None) -> tuple[torch.Tensor, tuple[int, int], str | None]:
    """One image -> ([1, H, W, C] float 0-1, original (w, h), note or None).

    The ORIGINAL size is returned alongside because the brief reports what you
    supplied, not what the loader produced — otherwise a resize would quietly
    rewrite the record of what your source actually was. The note says when
    transparent areas were filled (with the card's pad colour, black by default).
    """
    resolved = folder_paths.get_annotated_filepath(path)
    if not resolved or not os.path.isfile(resolved):
        raise RuntimeError(
            f"[Luna-H3-AssetLoader] Image not found: {path}. Put it in ComfyUI's "
            f"input folder and use the file name, one per line."
        )
    img = node_helpers.pillow(Image.open, resolved)
    img, note = to_rgb(ImageOps.exif_transpose(img), (st or {}).get("pad_color") or "#000000")
    original = img.size
    if st:
        img = apply_state(img, st)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...], original, note


# Per-image resize state. The key names follow ComfyUI-Pixaroma's LoadImageMini
# state (MIT, compatible with this pack) so the two feel the same to use; the
# implementation below is our own and covers only what H3 work needs.
#
# mode defaults to "off" — nothing is resized unless asked. That default is the
# whole point: H3's REF2VA path keeps each reference's own aspect ratio, so
# conforming references by default would destroy information the model wants.
_DEFAULT_ITEM = {
    "mode": "off",           # off | max_mp | longest_side | scale_factor
    "max_mp": 2.0,
    "longest_side": 1024,
    "scale_factor": 1.0,
    "ratio": "",             # "" keeps the source aspect; else "16:9", "2:3", …
    "ratio_action": "crop",  # crop | pad
    "crop_anchor": "center",
    "pad_color": "#000000",
    "snap": 0,               # 0 = off; 32 matches H3's CANVAS_MULTIPLE
    "allow_upscale": False,  # H3 never upscales a reference; the loader is the
                             # only place it can happen, so make it deliberate
}


def _overlap_setting(data: dict, settings_raw: dict) -> tuple[int, str]:
    """The node's overlap in frames, and a warning or "": `settings.overlap` when set, else sound card 1's overlap
    (saves made before the node-wide row), else 22; a value off luna_units.OVERLAPS snaps to the nearest."""
    raw = settings_raw.get("overlap")
    if raw is None:
        first = next((c for c in (data.get("audio") or []) if isinstance(c, dict) and c.get("file")), None)
        raw = first.get("overlap") if first and first.get("overlap") is not None else DEFAULT_OVERLAP
    try:
        return luna_units.snap_overlap(float(raw))
    except (TypeError, ValueError):
        return DEFAULT_OVERLAP, f"Overlap {ascii(raw)} is not a number of frames; {DEFAULT_OVERLAP} is used."


def _parse_media(raw: str) -> dict:
    """media_state JSON -> {"audio": [card, ...]}. A card: file, start, duration
    (seconds, kept for old saves; the window is the shot length), fill ("silence" | "loop"); sound card 1 also
    `vocals` (the vocal stem's file, "" = none)."""
    try:
        data = json.loads(raw or "{}")
    except (TypeError, ValueError):
        raise RuntimeError("[Luna-H3-AssetLoader] media_state is not valid JSON. "
                           "Clear the sound cards and add them again.")
    cards = []
    for c in (data.get("audio") or [])[:MAX_SOUNDS]:
        if not isinstance(c, dict) or not c.get("file"):
            continue
        # Sound 1 is the song and drives the shot unless told otherwise; later
        # sounds are references for MiniMax H3 unless placed on the shot.
        role = normalize_sound_role(c.get("role"), len(cards))
        cards.append({
            "role": role,
            "at": max(0, int(c.get("at") or 0)),
            "file": str(c["file"]),
            "start": max(0.0, float(c.get("start") or 0.0)),
            "duration": max(0.0, float(c.get("duration") or 0.0)),
            "fill": "loop" if c.get("fill") == "loop" else "silence",
            # Song position: "free" = the window is where start says; "shot" = the
            # window follows the shot number from song_start, `overlap` frames shared.
            "position": "shot" if c.get("position") == "shot" else "free",
            "song_start": max(0.0, float(c.get("song_start") or 0.0)),
            "shot": max(1, int(c.get("shot") or 1)),
            # Kept for old saves only: the node's overlap is `settings.overlap` (_overlap_setting).
            "overlap": max(0, int(c.get("overlap") if c.get("overlap") is not None else DEFAULT_OVERLAP)),
            # The vocal stem rides sound card 1 only: the same window as the song.
            "vocals": str(c.get("vocals") or "") if not cards else "",
        })
    clips = []
    for c in (data.get("video") or [])[:MAX_VIDEOS]:
        if not isinstance(c, dict) or not c.get("file"):
            continue
        fps = c.get("fps", "target")
        clips.append({
            "role": "guide" if c.get("role") == "guide" else "ref",
            "at": max(0, int(c.get("at") or 0)),
            "file": str(c["file"]),
            "start": max(0.0, float(c.get("start") or 0.0)),
            "frames": max(1, int(c.get("frames") or 124)),
            # "target" = the target model's rate (24 for MiniMax H3), "source" = as filmed
            "fps": fps if fps in ("target", "source") else max(1.0, float(fps)),
            "count": "free" if c.get("count") == "free" else "target",
            "size": c.get("size") if c.get("size") in ("canvas", "edge", "source") else "canvas",
            "fit": c.get("fit") if c.get("fit") in ("crop", "pad", "stretch") else "crop",
            "nth": max(1, int(c.get("nth") or 1)),
            "sound": "drop" if c.get("sound") == "drop" else "keep",
            "precision": "fp32" if c.get("precision") == "fp32" else "fp16",
            # Like the song card: "free" = the clip opens at `start`; "shot" = `start` is shot 1's and shot n opens
            # (n - 1) x (shot length - overlap) later, n being `shot` (+ the chain's clip).
            "position": "shot" if c.get("position") == "shot" else "free",
            "shot": max(1, int(c.get("shot") or 1)),
        })
    # Picture roles, in card order (index-aligned with image_paths). Missing = reference.
    pics = []
    for p in (data.get("pictures") or [])[:MAX_IMAGES]:
        p = p if isinstance(p, dict) else {}
        pics.append({"role": "key" if p.get("role") == "key" else "ref", "at": max(0, int(p.get("at") or 0))})
    shot = data.get("shot") if isinstance(data.get("shot"), dict) else {}
    # Node-wide settings from the ⚙ panel.
    st = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    overlap, overlap_warning = _overlap_setting(data, st)
    settings = {
        "target": st.get("target") if st.get("target") in ("h3", "ltx", "general") else "h3",
        "memory_gb": min(64.0, max(1.0, float(st.get("memory_gb") or MEMORY_CAP / 1024 ** 3))),
        "overlap": overlap,
    }
    return {"audio": cards, "video": clips, "pictures": pics,
            "shot_frames": max(1, int(shot.get("frames") or 124)), "settings": settings,
            "warnings": [overlap_warning] if overlap_warning else []}


def _clock(t: float) -> str:
    m, s = divmod(max(0.0, t), 60.0)
    return f"{int(m)}:{s:05.2f}"


def _card_fps(c: dict, info: dict, target: str) -> float:
    """The rate a clip card decodes at, before every-Nth: "target" the target's (General: the file's own),
    "source" the file's own, else the number set on the card."""
    if c["fps"] == "target":
        return target_fps(target, info.get("fps"))
    if c["fps"] == "source":
        return float(info.get("fps") or FPS)
    return float(c["fps"])


def _delivered_fps(c: dict, info: dict, target: str) -> float:
    """The rate of the frames a clip card delivers: its decode rate after every-Nth (media_io.decode_clip's fps)."""
    return _card_fps(c, info, target) / max(1, int(c["nth"]))


def _parse_state(raw: str, count: int) -> list[dict]:
    """JSON -> one settled state dict per image.

    Accepts {"all": {...}} to apply one state to every image, {"items": [...]}
    for per-image control, or both, where items win. Anything malformed falls
    back to defaults rather than failing the run — a bad state string should not
    cost a render.
    """
    base = dict(_DEFAULT_ITEM)
    items: list[dict] = []
    try:
        parsed = json.loads(raw) if raw and raw.strip() else {}
        if isinstance(parsed, dict):
            shared = parsed.get("all")
            if isinstance(shared, dict):
                base.update({k: v for k, v in shared.items() if k in _DEFAULT_ITEM})
            listed = parsed.get("items")
            if isinstance(listed, list):
                items = [x if isinstance(x, dict) else {} for x in listed]
    except Exception:
        pass

    out = []
    for i in range(count):
        st = dict(base)
        if i < len(items):
            st.update({k: v for k, v in items[i].items() if k in _DEFAULT_ITEM})
        out.append(st)
    return out


class LunaAssetLoader:
    DESCRIPTION = (
        "Load up to 9 ordered reference images and a song through one node. Numbered "
        "outputs keep each image at its original size for the MiniMax H3 reference sockets; "
        "the `images` batch feeds prompt writers. The reference cards, in card order, are <Picture 1>, "
        "<Picture 2>, and so on (a keyframe card is not counted). In the prompt, point at a card by its @ name, "
        "its card number: @image2, @video1, @audio2. Each card shows its @ name and the tag H3 receives, and "
        "asset_brief states that mapping in its @ names line so nothing has to guess it.\n\n"
        "Upload a sound file and it becomes a sound card. Select it to see the whole song: "
        "drag the window to choose the part the shot uses, or type its Start. The window is always the "
        "shot length: wire `seconds` (for example from Luna-H3-Canvas) and the shot follows it; unwired, "
        "the shot length set on the node is used. `audio` delivers exactly that window; sound card 1 may "
        "also carry a vocal stem, cut to the same window.\n\n"
        "`frames` is the shot length rounded UP onto the Target's grid (H3 17k+5, LTX 8k+1); past 362 "
        "frames on H3 the brief warns and nothing is cut. The Overlap row on the node sets how many frames "
        "each shot shares with the one before, for the song, the vocal stem and the clip cards by shot number.\n\n"
        "`video` is the frames of the first clip card and `frames` the shot length in frames, for "
        "Luna-LLMPrompt. Wire the canvas width and height and they pass straight through; unwired, "
        "picture 1's size is given. An empty loader runs and delivers nothing."
    )
    CATEGORY = "Luna-MinimaxH3"
    FUNCTION = "load"

    # ComfyUI maps outputs to the returned tuple by INDEX, and the frontend sends each link's own slot number, so
    # every output keeps its index for good: the ten fixed outputs (audio/refs/timeline v3, video/frames v4,
    # overlap v5, moved up for old saves by the frontend migration), image_1..image_9 always all nine, then the
    # outputs appended after them (TAIL_OUTPUTS: fps, vocals).
    # Appended after image_9 (SPEC § 2): every run makes them (_run), every node has their sockets.
    TAIL_OUTPUTS = {
        "fps": ("FLOAT", "The frame rate of the clips, as FLOAT: clip card V1's delivered rate (its own fps choice, "
                         "after every-Nth), the rate of the frames on `video`; with no clip, the shot's rate (H3 24, "
                         "LTX 25, General 24). With every clip on 'Target' fps, frames / fps is the shot length. For "
                         "Luna-SaveVideo's frame_rate and Luna-LLMPrompt's fps."),
        "vocals": ("AUDIO", "Sound card 1's vocal stem, cut to exactly the song's window (same start, length and "
                            "fill): the stem for H3's lip-sync lock (MiniMaxH3AddGuide's audio, or the loop's lock "
                            "audio), while `audio` stays the full mix. Empty when the card has no vocal stem."),
    }
    RETURN_TYPES = (("IMAGE", "INT", "INT", "STRING", "AUDIO", "LUNA_H3_REFS", "LUNA_TIMELINE", "IMAGE", "INT", "INT")
                    + ("IMAGE",) * MAX_IMAGES + tuple(t for t, _ in TAIL_OUTPUTS.values()))
    RETURN_NAMES = (("images", "width", "height", "asset_brief", "audio", "refs", "timeline", "video",
                     "frames", "overlap") + tuple(f"image_{i + 1}" for i in range(MAX_IMAGES)) + tuple(TAIL_OUTPUTS))
    # ComfyUI shows these when you hover an output socket, and the ⓘ panel lists
    # them too — one definition, both places, so they cannot disagree.
    OUTPUT_TOOLTIPS = (
        "Every loaded picture as ONE IMAGE batch at image 1's size, each fitted whole with "
        "black bars (never stretched), because a batch cannot hold mixed dimensions. For prompt writers "
        "(Luna Agent Chat, LLM Prompt) which flatten it to individual JPEGs anyway. Empty when only "
        "clips or sounds are loaded.",
        "The width to generate at. With the width AND height inputs wired (the canvas size) it is "
        "that width, so the loader and the canvas never disagree; unwired it is image 1's width AFTER "
        "its resize; with no pictures either, 0.",
        "The height to generate at. Same rule as width: the wired canvas height, else image 1's height "
        "AFTER its resize, else 0.",
        "A one-line inventory — '<Picture 1> = 1920x1056 (20:11, landscape); ...' — with the tag H3 "
        "receives for each reference; a warning when picture cards 1 and 2 differ in aspect ratio, which "
        "FL2VA silently fixes by stretching the first frame and cropping the last; the @ names line "
        "('@ names: @image1 = not sent; @image2 = <Picture 1>; ...': each card's @imageN, @videoN, @audioN "
        "and the tag it becomes, 'not sent' when it sits on the shot); then the song and the vocal stem with "
        "their jobs, what is on the shot, and the warnings (past 362 frames, a clip at another rate than the "
        "shot, an overlap off the list). luna_naming.inventory writes it, the same text Luna-H3-RefsOut writes. "
        "For Luna-LLMPrompt's or Luna-LLMPromptAPI's asset_brief, so the writer knows which reference is which "
        "and what each @ name means. With Luna-H3-RefsOut in the graph, wire its asset_brief instead.",
        "The song window from sound card 1, the full mix: exactly the shot length (frames / the shot's frame "
        "rate), from the card's Start or its shot number. For Luna-SaveVideo's audio, so the finished video has "
        "the same sound, and for a ref_audio socket of MiniMaxH3ReferenceToVideo (H3's reference sound).",
        "Every reference on one wire, in the order H3 numbers them (keyframes, guide clips and the locked "
        "song are not in it): `images` with their `roles` (one word each, 'reference'), `videos`, "
        "`video_audios`, `audios`, the song window (`song`) and its vocal stem (`vocals`, or None), "
        "`keyframes` (the key pictures: card, frame, picture), the brief, `items` and `naming`, the @ names "
        "map (each card's @imageN, @videoN, @audioN to the tag H3 receives), `width` and `height`. "
        "For Luna-H3-RefsOut's refs, which splits it into the ref_image, ref_video, ref_video_audio and "
        "ref_audio sockets and rebuilds the @ names line for its switches; or Luna-LLMPrompt's or "
        "Luna-LLMPromptAPI's refs.",
        "What sits on the shot timeline and where, in frames at the shot's rate (its `fps`): keyframe "
        "pictures, guide clips and guide or locked sounds (the song's lock also carries the vocal stem).",
        "The frames of clip card V1 (whatever its role), decoded as the card says. For Luna-LLMPrompt's "
        "video input. Empty when no clip is loaded.",
        "The shot length in frames, one count for the whole node: the seconds input at the shot's rate "
        "(H3 24, LTX 25, General clip 1's) when wired, else the length set on the node, rounded UP onto the "
        "Target's grid (H3 17k+5: 5.0 s is 124; LTX 8k+1). Past 362 on H3 asset_brief warns and nothing is cut. "
        "For MiniMaxH3ReferenceToVideo's length and Luna-LLMPrompt's frames.",
        "The node's overlap, in frames (the Overlap row: 22, 39, 90, 141, 192; a saved value off the list snaps "
        "to the nearest with a warning): how many frames each shot shares with the one before, for the song, "
        "the vocal stem and the clip cards by shot number, with or without a song. For Luna-H3-ChainContinue's, "
        "Luna-H3-ChainRemember's and Luna-H3-ChainJoin's overlap.",
    ) + tuple(
        f"Picture {i + 1} at its ORIGINAL size (only the batch is conformed), "
        f"in card order. Wire to the MiniMax H3 ref_image_{i} socket — or in FL2VA, "
        f"{'first_frame' if i == 0 else 'last_frame' if i == 1 else f'reference {i + 1}'}. "
        f"Empty when fewer than {i + 1} images are loaded."
        for i in range(MAX_IMAGES)
    ) + tuple(tip for _, tip in TAIL_OUTPUTS.values())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_paths": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "One image file name per line, from ComfyUI's input folder. Order is everything: line 1 is picture card 1 (@image1), line 2 card 2 (@image2), and so on; the reference cards among them are <Picture 1>, <Picture 2> … in that order (a keyframe card is not counted). In FL2VA the first two lines are the first and last frame.",
                }),
                "image_count": ("INT", {
                    "default": 2, "min": 1, "max": MAX_IMAGES,
                    "tooltip": "How many picture cards the node expects. All nine image sockets (image_1 to image_9) are always there, so no saved wire ever moves; the sockets past this number are labelled (spare). Set it with the Image slots stepper at the top left of the node (click the number to type one); it also follows the cards: uploading, removing or typing a file name sets it. Sockets beyond the number of loaded images stay empty.",
                }),
            },
            "optional": {
                "asset_state": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Per-image resize, as JSON. Empty = every image passed through untouched, which is what REF2VA wants. {\"all\": {...}} applies one state to every image; {\"items\": [{...}, ...]} sets them individually and wins over \"all\". Keys: mode (off|max_mp|longest_side|scale_factor), max_mp, longest_side, scale_factor, ratio (\"16:9\"), ratio_action (crop|pad), crop_anchor, pad_color, snap (32 for H3), allow_upscale. Example — conform the first two frames for FL2VA and leave the references alone: {\"items\":[{\"ratio\":\"16:9\",\"snap\":32},{\"ratio\":\"16:9\",\"snap\":32}]}",
                }),
                # Added LAST on purpose: saved workflows keep their widget values by
                # position, and a widget appended at the end shifts nothing.
                "media_state": ("STRING", {
                    "default": "",
                    "tooltip": "The cards and the node settings, as JSON, written by the card UI: {\"audio\": [{\"file\": \"song.mp3\", \"vocals\": \"song_vocals.wav\", \"start\": 42.5, \"fill\": \"silence\"}], \"settings\": {\"overlap\": 22}}. start is seconds; the window is the shot length; vocals (sound card 1 only) is the vocal stem, cut to the same window; fill is what happens when the window runs past the end (silence or loop); settings.overlap is the node's Overlap row (an old save without it takes sound card 1's overlap).",
                }),
                "seconds": ("FLOAT", {
                    "forceInput": True,
                    "tooltip": "The shot length in seconds, from Luna-H3-Canvas's seconds. Wired, it becomes `frames` at the shot's rate, rounded UP onto the Target's grid (5.0 s is 124 on H3; past 362 a warning, no cut), and the song window, the vocal stem and the clip cards by shot number follow it. Unwired, the shot length set on the node is used.",
                }),
                "shot": ("INT", {
                    "forceInput": True,
                    "tooltip": "Which shot of the song this run makes, counting from 1. Only used when the sound card's Position is 'By shot number': shot n starts where shot n-1 started plus the shot length minus the overlap. Unwired, the shot number on the sound card is used.",
                }),
                "width": ("INT", {
                    "forceInput": True,
                    "tooltip": "The generation width, for example from the canvas node. Video cards set to 'Match canvas' are cut to width x height. Unwired, they keep their shape at a 768 short edge.",
                }),
                "height": ("INT", {
                    "forceInput": True,
                    "tooltip": "The generation height. See width.",
                }),
                # Added LAST, like every input since media_state: nothing saved shifts.
                "song_start": ("FLOAT", {
                    "forceInput": True,
                    "tooltip": "Where in the song this shot starts, in seconds — from Luna-H3-ShotList, for scenes of different lengths. Wired, it wins over the sound card's Start and over 'By shot number'; the window's length still comes from seconds (or the card).",
                }),
                "chain": ("LUNA_H3_CHAIN", {
                    "tooltip": "From Luna-H3-Chain: which clip of the chain this run makes. The sound card must be on 'By shot number'; its Start and Shot and the node's Overlap row stay yours, and clip k plays segment Shot + k - 1. Clip cards on 'By shot number' move the same way; a free clip card stays where it is. Wire this loader's overlap output to Luna-H3-ChainContinue, -ChainRemember and -ChainJoin.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, image_paths, image_count, asset_state="", media_state="", **kwargs):
        # Hash the file contents, not just the names: re-uploading a changed
        # picture under the same name must re-run the graph. Songs are big, so
        # they are keyed on size and modification time instead.
        m = hashlib.sha256()
        m.update(f"{image_count}|{asset_state}|{media_state}".encode("utf-8"))
        for p in _split_paths(image_paths):
            resolved = folder_paths.get_annotated_filepath(p)
            m.update(p.encode("utf-8"))
            if resolved and os.path.isfile(resolved):
                with open(resolved, "rb") as fh:
                    m.update(fh.read())
        try:
            media = _parse_media(media_state)
            files = [c["file"] for c in media["audio"] + media["video"]]
            files += [c["vocals"] for c in media["audio"] if c.get("vocals")]
        except RuntimeError:
            files = []
        for name in files:
            resolved = folder_paths.get_annotated_filepath(name)
            if resolved and os.path.isfile(resolved):
                s = os.stat(resolved)
                m.update(f"{name}|{s.st_size}|{s.st_mtime_ns}".encode("utf-8"))
        return m.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image_paths, image_count, **kwargs):
        paths = _split_paths(image_paths)
        if len(paths) > MAX_IMAGES:
            return (f"MiniMax H3 takes at most {MAX_IMAGES} reference images, "
                    f"but {len(paths)} lines are listed.")
        return True

    def load(self, image_paths, image_count, asset_state="", media_state="", seconds=None, shot=None,
             width=None, height=None, song_start=None, chain=None):
        ui, outs = self._run(image_paths, image_count, asset_state, media_state, seconds, shot, width, height,
                             song_start, chain)
        return {"ui": ui, "result": tuple(outs[name] for name in self.RETURN_NAMES)}

    def _run(self, image_paths, image_count, asset_state="", media_state="", seconds=None, shot=None,
             width=None, height=None, song_start=None, chain=None):
        """One run -> (ui, {output name: value}). Every output is made here, fps and vocals included,
        so `load` only picks the declared sockets out by name."""
        paths = _split_paths(image_paths)
        media = _parse_media(media_state)
        cards, clips, settings = media["audio"], media["video"], media["settings"]
        target, overlap = settings["target"], settings["overlap"]
        grid = target_grid(target)
        # One shot rate per run: H3 24, LTX 25, General clip 1's delivered rate (24 with no clip).
        probes = self._probe(clips)
        rate = luna_units.fps_for(target, _delivered_fps(clips[0], probes[0], target) if clips else None)
        # One count per run: `frames`, the brief, the timeline, the placements, the song and the clip cards.
        shot_len = shot_frames(seconds, media["shot_frames"], target, rate)
        shot_s = shot_len / rate
        warnings = list(media["warnings"])
        if luna_units.frames_warning(shot_len, grid):
            warnings.append("The shot: " + luna_units.frames_warning(shot_len, grid))
        # The canvas size when BOTH are wired; else None (picture 1's size, or 0, is used).
        canvas = (int(width), int(height)) if width and height and int(width) > 0 and int(height) > 0 else None
        if not paths and not cards and not clips:
            # Not an error: a graph can run before any card is filled. Everything is empty,
            # the refs are well-formed (RefsOut and the brief rebuild cope), and the note says why.
            print(f"[{WHO}] Nothing loaded: empty outputs. Press Upload and add pictures, clips or a song.")
            extra = [f"WARNING: {w}" for w in warnings]
            for line in extra:
                print(f"[{WHO}] {line}")
            brief = " ".join(["Nothing loaded."] + extra)
            refs = {"images": [], "roles": [], "videos": [], "video_audios": [], "audios": [], "width": 0,
                    "height": 0, "brief": brief, "song": None, "vocals": None, "keyframes": [],
                    "items": {"images": [], "videos": [], "video_audios": [], "audios": [], "song": "",
                              "vocals": "", "extra": extra},
                    "naming": make_naming(ROW)}
            timeline = {"version": 2, "fps": rate, "frames": shot_len, "items": []}
            outs = {"images": None, "width": canvas[0] if canvas else 0, "height": canvas[1] if canvas else 0,
                    "asset_brief": brief, "audio": None, "refs": refs, "timeline": timeline, "video": None,
                    "frames": shot_len, "overlap": overlap, "fps": rate, "vocals": None}
            outs.update({f"image_{i + 1}": None for i in range(MAX_IMAGES)})
            return {"luna_asset_audio": []}, outs

        # --- load everything -------------------------------------------------
        audio, song, song_line = self._song(cards, shot_s, shot, song_start, chain, overlap, rate)
        if song is not None:
            song["wired"] = seconds is not None and float(seconds) > 0
        vocals, vocals_line = self._vocals(cards, song)
        starts = self._clip_starts(clips, shot_s, shot, chain, overlap, rate)
        decoded = self._clips(clips, width, height, settings, probes, starts)
        extra_sounds = self._sounds(cards, shot_s)
        frames, dims, originals = [], [], []
        if paths:
            states = _parse_state(asset_state, len(paths))
            loaded = [_load(p, st) for p, st in zip(paths, states)]
            frames = [f for f, _, _ in loaded]
            originals = [o for _, o, _ in loaded]
            notes = [n for _, _, n in loaded]
            dims = [(int(f.shape[2]), int(f.shape[1])) for f in frames]  # (w, h) as delivered
        # A clip at another rate than the shot's: on H3 and LTX it does not fit the shot. Nothing is resampled
        # that the card did not ask for; the brief says so.
        for k, (c, d) in enumerate(zip(clips, decoded), 1):
            if d.get("warning"):
                warnings.append(f"Clip V{k} ({c['file']}): {d['warning']}")
            got = float(d["info"].get("fps", rate))
            if abs(got - rate) > 1e-3:
                warnings.append(
                    f"Clip V{k} ({c['file']}) is delivered at {got:g} fps, not the shot's {rate:g} fps"
                    + ({"h3": "; MiniMax H3 plays every clip at 24 fps, so it does not fit the shot.",
                        "ltx": "; LTX plays it at 25 fps, so it does not fit the shot."}.get(target, ".")))

        # --- roles, tags, placements --------------------------------------------
        pic = [media["pictures"][i] if i < len(media["pictures"]) else {"role": "ref", "at": 0}
               for i in range(len(paths))]
        pic_roles = [p["role"] for p in pic]
        clip_cards = [{"role": c["role"], "sound": d["sound"] is not None} for c, d in zip(clips, decoded)]
        tags = h3_tags(pic_roles, clip_cards, [c["role"] for c in cards])
        # The @ names: one per card, mapped to the tag H3 receives. The three lists say exactly which cards go
        # into refs["images"], ["videos"], ["video_audios"] and ["audios"] below (a reference sound that did not
        # load is not sent), so Luna-H3-RefsOut can recount them after its switches (luna_naming.remake).
        naming = make_naming(ROW, *naming_cards(
            pic_roles, clip_cards,
            [c["role"] if c["role"] != "ref" or extra_sounds.get(j) else "off" for j, c in enumerate(cards)]))
        placed = [(f"Picture card {i + 1}", p["at"], 1) for i, p in enumerate(pic) if p["role"] == "key"]
        placed += [(f"Clip V{k + 1} ({c['file']})", c["at"], int(d["batch"].shape[0]))
                   for k, (c, d) in enumerate(zip(clips, decoded)) if c["role"] == "guide"]
        placed += [(f"Sound A{j + 1} ({c['file']})", c["at"], 1)
                   for j, c in enumerate(cards) if c["role"] in ("guide", "lock")]
        errors = check_placements(shot_len, placed, target)
        if errors:
            raise RuntimeError(f"[{WHO}] " + " ".join(errors) + " Drag it on the shot timeline.")

        # --- references for MiniMax H3 ---------------------------------------------
        ref_pics = [i for i, p in enumerate(pic) if p["role"] == "ref"]
        ref_clips = [k for k, c in enumerate(clips) if c["role"] == "ref"]
        ref_sounds = [j for j, c in enumerate(cards) if c["role"] == "ref"]
        refs = {"images": [frames[i] for i in ref_pics],
                # One word per picture in `images`, Luna-ImageLoader's words: every Asset Loader picture sent is a
                # reference (a keyframe is not sent; it is in `keyframes`).
                "roles": ["reference"] * len(ref_pics),
                "videos": [decoded[k]["batch"] for k in ref_clips],
                "video_audios": [decoded[k]["sound"] for k in ref_clips],
                "audios": [extra_sounds[j][0] for j in ref_sounds if extra_sounds.get(j)],
                "width": dims[0][0] if dims else 0, "height": dims[0][1] if dims else 0,
                # The song window, so Luna-H3-RefsOut can hand it on (its `audio` output), and its vocal stem.
                "song": audio,
                "vocals": vocals,
                # The key pictures on the shot: picture card number (1-based), frame, picture.
                "keyframes": [{"card": i + 1, "at": p["at"], "image": frames[i]}
                              for i, p in enumerate(pic) if p["role"] == "key"],
                # The @ names (luna_naming.make, row minimax): which tag each card gets, or None when it is not sent.
                "naming": naming}
        # One description per item WITHOUT its <Tag>, per category. Luna-H3-RefsOut rebuilds the
        # brief from these with fresh numbering when it switches a category off (a clip sound
        # off moves the standalone sounds down a number), so the tags cannot be baked in here.
        items = {"images": [], "videos": [], "video_audios": [decoded[k]["sound"] is not None for k in ref_clips],
                 "audios": [], "song": song_line or "", "vocals": vocals_line or "", "extra": []}

        # --- the brief: luna_naming.inventory, the one text Luna-H3-RefsOut also writes -------------------------
        for i in ref_pics:
            (w, h), (ow, oh) = dims[i], originals[i]
            desc = _describe_ar(w, h)
            if len(ref_pics) != len(paths):
                desc += f" (picture card {i + 1})"
            if (w, h) != (ow, oh):
                # Say what it was as well as what it became. A prompt writer told
                # only the resized number cannot tell a deliberate crop from an
                # accident, and neither can you reading it back later.
                desc += f" [resized from {ow}x{oh}]"
            if notes[i]:
                desc += f" [{notes[i]}]"
            items["images"].append(desc)
        # FL2VA forces both frames to the exact generation size — the first with
        # crop disabled, so a mismatched aspect is STRETCHED, and the last with a
        # centre crop, so it is CUT. Neither errors, so say it here or nobody
        # finds out until they watch the render.
        # Relative, not absolute: snapping to a /32 grid moves the aspect ratio by
        # up to ~1% by itself, so a tight absolute threshold would warn on every
        # snapped pair. 2% is above that noise floor and still catches a stretch
        # you would see.
        if len(dims) >= 2:
            r0, r1 = dims[0][0] / dims[0][1], dims[1][0] / dims[1][1]
            if abs(r0 - r1) / max(r0, r1) > 0.02:
                warn = (f"WARNING: picture cards 1 and 2 have different aspect ratios "
                        f"({_describe_ar(*dims[0])} vs {_describe_ar(*dims[1])}). Fine for "
                        f"REF2VA, which keeps each aspect. In FL2VA the first frame gets "
                        f"stretched and the last centre-cropped.")
                print(f"[{WHO}] {warn}")
                # Kept apart from "extra": it is about the pictures, so it goes when they are switched off.
                items["picture_warning"] = warn
        for k in ref_clips:
            line = decoded[k]["line"]
            if clips[k]["sound"] == "keep" and decoded[k]["sound"] is None:
                line += "; it has no sound"   # kept in the item, so Refs Out's brief says it too
            items["videos"].append(line)
        for j in ref_sounds:
            if extra_sounds.get(j):
                items["audios"].append(extra_sounds[j][1])
        on_shot = self._shot_lines(pic, clips, decoded, cards, song)
        if on_shot:
            items["extra"].append(f"On the shot ({shot_len} frames, {shot_s:.2f} s): " + "; ".join(on_shot) + ".")
        # The loaders' warnings stay in every brief, whatever Refs Out switches off.
        for w in warnings:
            print(f"[{WHO}] WARNING: {w}")
            items["extra"].append(f"WARNING: {w}")
        brief = inventory(items, naming, ROW)
        refs["brief"] = brief
        refs["items"] = items
        print(f"[{WHO}] {brief}")

        timeline = self._timeline(shot_len, pic, frames, clips, decoded, cards, extra_sounds, audio, song,
                                  fps=rate, vocals=vocals)
        numbered: list = [frames[i] if i < len(frames) else None for i in range(MAX_IMAGES)]
        if canvas:
            # The canvas size wins whenever it is wired: one source of truth for the size.
            w, h = canvas
        elif dims:
            w, h = dims[0]
        else:
            # No pictures and nothing wired: a plain 0x0; `images` and image_N stay
            # empty (never black frames).
            w = h = 0
            print(f"[{WHO}] No pictures loaded and no width/height wired: width and height are 0.")
        outs = {"images": conform(frames) if frames else None, "width": w, "height": h, "asset_brief": brief,
                "audio": audio, "refs": refs, "timeline": timeline,
                "video": decoded[0]["batch"] if decoded else None, "frames": shot_len, "overlap": overlap,
                # The rate of the frames on `video` (clip 1, after its own fps choice and every-Nth), else the shot's.
                "fps": float(decoded[0]["info"].get("fps", rate)) if decoded else float(rate),
                "vocals": vocals}
        outs.update({f"image_{i + 1}": numbered[i] for i in range(MAX_IMAGES)})
        return {"luna_asset_audio": [song] if song else []}, outs

    @staticmethod
    def _probe(clips):
        """Each clip card's file facts (probe_media), in card order; refuses a missing file or one without picture."""
        out = []
        for k, c in enumerate(clips, 1):
            path = _media_file(c["file"])
            if not path:
                raise RuntimeError(f"[{WHO}] Clip not found in the input folder: {c['file']}")
            info = probe_media(path)
            if not info.get("video"):
                raise RuntimeError(f"[{WHO}] {c['file']} has no picture, only sound.")
            out.append(info)
        return out

    @staticmethod
    def _clip_starts(clips, shot_s, shot=None, chain=None, overlap=DEFAULT_OVERLAP, rate=FPS):
        """Where each clip card opens in its own file, in seconds. "free": its start. "shot": its start is shot 1's,
        and shot n opens (n - 1) x (shot length - overlap / rate) later, as the song does; n is the card's shot plus
        the chain's clip (Luna-H3-Chain), else the wired `shot`, else the card's shot."""
        out = []
        for k, c in enumerate(clips, 1):
            if c.get("position") != "shot":
                out.append(c["start"])
                continue
            if chain is not None:
                n = c["shot"] + int(chain["clip"])
            else:
                n = int(shot) if shot is not None and int(shot) > 0 else c["shot"]
            try:
                out.append(song_shot_start(c["start"], n, shot_s, overlap, rate))
            except ValueError as e:
                raise RuntimeError(f"[{WHO}] Clip V{k} ({c['file']}): {e} Lower the overlap.") from None
        return out

    @staticmethod
    def _clips(clips, width, height, settings=None, probes=None, starts=None):
        """Video cards -> [{"batch", "sound", "info", "line"}], in card order. probes: _probe's facts (made here
        when None); starts: where each card opens (_clip_starts; each card's own start when None)."""
        out = []
        settings = settings or {"target": "h3", "memory_gb": MEMORY_CAP / 1024 ** 3}
        cap, grid_t = settings["memory_gb"] * 1024 ** 3, target_grid(settings["target"])
        box = (int(width), int(height)) if width and height and int(width) > 0 and int(height) > 0 else None
        probes = probes if probes is not None else LunaAssetLoader._probe(clips)
        for k, c in enumerate(clips, 1):
            path = _media_file(c["file"])
            if not path:
                raise RuntimeError(f"[{WHO}] Clip not found in the input folder: {c['file']}")
            grid = grid_t if c["count"] == "target" else "free"
            bad = thinning_error("target" if grid != "free" else "free", c["nth"])
            if bad:
                raise RuntimeError(f"[{WHO}] Clip V{k} ({c['file']}): {bad}")
            info = probes[k - 1]
            fps = _card_fps(c, info, settings["target"])
            # The card's length, rounded UP onto the grid (no 362 cut); a file that ends first is cut down.
            frames = grid_frames(c["frames"], grid)
            if c["size"] == "canvas" and box:
                size, (ow, oh) = ("box", box[0], box[1]), box
            elif c["size"] == "source":
                size, (ow, oh) = ("source",), (info["width"], info["height"])
            else:   # 768 short edge, also the fallback when no canvas size is wired
                size = ("edge", 768)
                s = min(1.0, 768.0 / min(info["width"], info["height"]))
                ow, oh = int(round(info["width"] * s)), int(round(info["height"] * s))
                ow, oh = max(2, ow - ow % 2), max(2, oh - oh % 2)   # media_io._target_size's even rule
            cost = clip_bytes(-(-frames // c["nth"]), ow, oh, c["precision"])
            if cost > cap:
                raise RuntimeError(
                    f"[{WHO}] Clip V{k} ({c['file']}) would need {cost / 1024 ** 3:.1f} GB as "
                    f"{frames} frames at {ow}x{oh} {c['precision']}, more than the {cap / 1024 ** 3:g} GB "
                    f"this node allows (the Memory cap setting). Shorten it, use fp16 or a smaller size, or raise "
                    f"the cap" + ("." if grid != "free" else ", or take every 2nd frame."))
            start = starts[k - 1] if starts is not None else c["start"]
            batch, got = decode_clip(path, start, frames, fps, size, c["fit"], c["nth"], c["precision"], grid, who=WHO)
            sound = clip_soundtrack(path, got["start"], got["seconds"]) if c["sound"] == "keep" else None
            line = (f"{got['name']} from {_clock(got['start'])}, {got['frames']} frames "
                    f"({got['seconds']:.2f} s) at {got['fps']:g} fps, {got['width']}x{got['height']}")
            how = {"canvas": c["fit"] if box else "768 short edge", "edge": "768 short edge", "source": "as filmed"}[c["size"]]
            line += f" (from {got['source_size'][0]}x{got['source_size'][1]} at {got['source_fps']:g} fps, {how})"
            if got["short"]:
                line += f"; the clip ran out, so {got['frames'] * got['nth']} of the {frames} frames asked for"
            # The card's own count, as H3 receives it: past 362 a warning (luna_units), never a cut. _run puts it in
            # the console and the brief's extra lines, so Refs Out keeps it.
            warning = luna_units.frames_warning(int(batch.shape[0]), "h3") if grid_t == "h3" else ""
            out.append({"batch": batch, "sound": sound, "info": got, "line": line, "warning": warning})
        return out

    @staticmethod
    def _song(cards, shot_s, shot=None, song_start=None, chain=None, overlap=None, rate=FPS):
        """Sound card 1 -> (AUDIO or None, info dict or None, one brief line).

        The window is exactly the shot length, shot_s = frames / the shot's rate (one count per run). A wired
        song_start (Luna-H3-ShotList) places the window itself; the card's own start and its shot-number mode then
        stand aside. A wired chain (Luna-H3-Chain) moves the card's own shot on by the clip number; the card's Start
        stays. overlap: the node's overlap in frames (None = the card's own, from saves before the node setting);
        shot n starts (n - 1) x (shot_s - overlap / rate) after shot 1."""
        if chain is not None:
            if not cards:
                raise RuntimeError(f"[{WHO}] chain is wired but no song is loaded: add the song on "
                                   "sound card 1 and set its Position to 'By shot number'.")
            if cards[0]["position"] != "shot" or song_start is not None:
                raise RuntimeError(f"[{WHO}] chain is wired: set sound card 1's Position to 'By shot "
                                   "number' and leave song_start unwired, so the card's Start, Shot and the "
                                   "node's overlap place every clip.")
            shot = cards[0]["shot"] + int(chain["clip"])
        if not cards:
            return None, None, ""
        c = cards[0]
        ov = int(c["overlap"] if overlap is None else overlap)
        path = _media_file(c["file"])
        if not path:
            raise RuntimeError(f"[{WHO}] Song not found in the input folder: {c['file']}")
        want = float(shot_s)
        if want <= 0:
            raise RuntimeError(f"[{WHO}] The shot has no length: wire `seconds` or set the shot length.")
        start, describe, n = c["start"], None, None
        placed = song_start is not None
        if placed:
            start = max(0.0, float(song_start))

            def describe(total_s):
                return (f"[{WHO}] song_start puts this shot at {_clock(start)}, after "
                        f"{c['file']} ends ({_clock(total_s)}). Check the scene's start in the shot list.")
        elif c["position"] == "shot":
            n = int(shot) if shot is not None and int(shot) > 0 else c["shot"]
            try:
                start = song_shot_start(c["song_start"], n, want, ov, rate)
            except ValueError as e:
                raise RuntimeError(f"[{WHO}] {e} Lower the overlap.") from None

            def describe(total_s, _n=n):
                count = song_shot_count(total_s, c["song_start"], want, ov, rate)
                return (f"[{WHO}] Shot {_n} would start at {_clock(start)}, after "
                        f"{c['file']} ends ({_clock(total_s)}). This song has {count} shot(s) of "
                        f"{want:.2f} s with a {ov}-frame overlap, starting at "
                        f"{_clock(c['song_start'])}.")
        audio, info = load_audio_window(path, start, want, c["fill"], describe, who=WHO)
        info["start_wired"] = placed
        line = (f"Song: {info['name']} {_clock(info['start'])}-{_clock(info['start'] + info['seconds'])} "
                f"({info['seconds']:.2f} s{', placed by song_start' if placed else ''}).")
        if n is not None:
            info["shot"] = n
            info["shots"] = song_shot_count(info["total"], c["song_start"], want, ov, rate)
            info["overlap"] = ov
            line = (f"Song: {info['name']}, shot {n} of {info['shots']}, "
                    f"{_clock(info['start'])}-{_clock(info['start'] + info['seconds'])} "
                    f"({info['seconds']:.2f} s"
                    + (f", sharing its first {ov} frames with shot {n - 1}" if n > 1 and ov else "")
                    + ").")
        if info["filled"] > 0.0005:
            line += (f" The file ran out after {info['taken']:.2f} s; the last {info['filled']:.2f} s "
                     f"{'loop back to its start' if info['mode'] == 'loop' else 'are silence'}.")
            print(f"[{WHO}] {line}")
        # Its job (SPEC § 2). No <Audio n>: the number depends on the ref_audio socket it is wired to.
        line += (" The full mix: for the saved video, and H3's reference sound"
                 + (" for everything but the lip sync, which takes the vocal stem." if c.get("vocals") else "."))
        return audio, info, line

    @staticmethod
    def _vocals(cards, song):
        """Sound card 1's vocal stem -> (AUDIO or None, one brief line or ""): the song's window exactly (same
        start, same length, same fill), for H3's lip-sync lock."""
        if not cards or not cards[0].get("vocals") or not song:
            return None, ""
        c = cards[0]
        path = _media_file(c["vocals"])
        if not path:
            raise RuntimeError(f"[{WHO}] Vocal stem not found in the input folder: {c['vocals']}")
        audio, info = load_audio_window(path, song["start"], song["seconds"], c["fill"], None, who=WHO)
        line = (f"Vocal stem: {info['name']} {_clock(info['start'])}-{_clock(info['start'] + info['seconds'])} "
                f"({info['seconds']:.2f} s), the song's window: the vocal stem of the song, for the lip-sync lock.")
        if info["filled"] > 0.0005:
            line += (f" The stem ran out after {info['taken']:.2f} s; the last {info['filled']:.2f} s "
                     f"{'loop back to its start' if info['mode'] == 'loop' else 'are silence'}.")
        return audio, line

    @staticmethod
    def _sounds(cards, shot_s=None):
        """Sound cards 2-3 that are references or guides -> {index: (AUDIO, description)}.
        Cut to their window, never filled: a reference ends where it ends, and at most at the shot's end (shot_s)."""
        out = {}
        for j, c in enumerate(cards):
            # Sound 1 as a guide or locked uses the song window (`audio`); as a
            # reference it is loaded here like any other reference sound.
            if c["role"] not in ("ref", "guide") or (j == 0 and c["role"] != "ref"):
                continue
            path = _media_file(c["file"])
            if not path:
                raise RuntimeError(f"[{WHO}] Sound not found in the input folder: {c['file']}")
            want = c["duration"]
            if shot_s:
                want = float(shot_s) if want <= 0 else min(want, float(shot_s))
            clip, info = load_audio(path, c["start"], want, who=WHO)
            out[j] = (clip, f"{info['name']} {_clock(info['start'])}-{_clock(info['start'] + info['seconds'])} "
                            f"({info['seconds']:.2f} s)")
        return out

    @staticmethod
    def _shot_lines(pic, clips, decoded, cards, song):
        """What sits on the shot, in frame order, in words."""
        rows = []
        for i, p in enumerate(pic):
            if p["role"] == "key":
                rows.append((p["at"], f"frame {p['at']}: picture card {i + 1} as a keyframe"))
        for k, (c, d) in enumerate(zip(clips, decoded)):
            if c["role"] == "guide":
                n = int(d["batch"].shape[0])
                rows.append((c["at"], f"frames {c['at']}-{c['at'] + n - 1}: clip V{k + 1} ({d['info']['name']})"
                                      + (" with its sound" if d["sound"] is not None else "")))
        for j, c in enumerate(cards):
            if c["role"] == "lock":
                rows.append((c["at"], f"from frame {c['at']}: sound A{j + 1} locked: the picture follows it"))
            elif c["role"] == "guide":
                rows.append((c["at"], f"from frame {c['at']}: sound A{j + 1} as a guide"))
        return [text for _, text in sorted(rows, key=lambda r: r[0])]

    @staticmethod
    def _timeline(shot_len, pic, frames, clips, decoded, cards, extra_sounds, audio, song, fps=FPS, vocals=None):
        """Everything placed on the shot, with its frames/sound attached, for a node
        that applies it (guides, the locked song). Frames are at the shot's rate `fps` (H3 24, LTX 25). The song's
        lock or guide item also carries `vocals`, the vocal stem for the lip sync, when sound card 1 has one."""
        items = []
        for i, p in enumerate(pic):
            if p["role"] == "key":
                items.append({"kind": "picture", "card": i + 1, "at": p["at"], "image": frames[i]})
        for k, (c, d) in enumerate(zip(clips, decoded)):
            if c["role"] == "guide":
                items.append({"kind": "video", "card": k + 1, "at": c["at"], "file": d["info"]["name"],
                              "frames": d["batch"], "audio": d["sound"]})
        for j, c in enumerate(cards):
            if c["role"] not in ("lock", "guide"):
                continue
            item = {"kind": "audio", "role": c["role"], "card": j + 1, "at": c["at"], "file": c["file"],
                    "audio": audio if j == 0 else extra_sounds[j][0]}
            if j == 0 and vocals is not None:
                item["vocals"] = vocals
            if j == 0 and song:
                item.update({"start": round(song["start"], 4), "seconds": round(song["seconds"], 4)})
                if song.get("shot") is not None:
                    item.update({"shot": song["shot"], "shots": song["shots"], "overlap_frames": song["overlap"]})
            items.append(item)
        items.sort(key=lambda it: it["at"])
        return {"version": 2, "fps": fps, "frames": shot_len, "items": items}


def _inside(root_dir: str, rel: str, kind: str = "file"):
    """`rel` resolved inside `root_dir`, or None. The name is screened BEFORE
    anything touches the disk: on Windows merely resolving \\\\host\\share opens
    SMB, and the routes and /prompt take any caller's input. A junction or link
    that leads outside the folder (even onto another drive) is refused too."""
    if not isinstance(rel, str) or "\x00" in rel:
        return None
    if rel and (os.path.isabs(rel) or rel.startswith(("\\", "/")) or ":" in rel
                or ".." in rel.replace("\\", "/").split("/")):
        return None
    root = os.path.realpath(root_dir)
    path = os.path.realpath(os.path.join(root, rel))
    try:
        if os.path.commonpath([root, path]) != root:
            return None
    except ValueError:   # different drives: certainly not inside
        return None
    ok = os.path.isfile(path) if kind == "file" else os.path.isdir(path)
    return path if ok else None


def _input_file(name: str):
    """A file name from the browser -> a real file inside ComfyUI's input folder, or None."""
    if not name or not isinstance(name, str):
        return None
    return _inside(folder_paths.get_input_directory(), strip_annotation(name)[0])


def _media_file(name: str):
    """A sound or clip named in media_state -> a real file, or None. Plain names live
    in the input folder; a trailing ' [output]' / ' [temp]' tag (ComfyUI's own
    annotation) selects those folders. Nothing outside them is ever read."""
    if not name or not isinstance(name, str):
        return None
    bare, tag = strip_annotation(name)
    root = {"output": folder_paths.get_output_directory(),
            "temp": folder_paths.get_temp_directory()}.get(tag, folder_paths.get_input_directory())
    return _inside(root, bare)


import asyncio

try:
    from aiohttp import web
    from server import PromptServer
except ImportError:   # imported outside ComfyUI (tests): no routes
    PromptServer = None

if PromptServer is not None:
    @PromptServer.instance.routes.get("/lunasave/media-info")
    async def _luna_media_info(request):
        """Size, frame rate, length and sound of a clip, for the video card."""
        path = _input_file(request.query.get("file", ""))
        if not path:
            return web.json_response({"error": "not found"}, status=404)
        try:
            # Decoding off the event loop: a long file must not stall the whole server.
            return web.json_response(await asyncio.get_running_loop().run_in_executor(None, probe_media, path))
        except Exception as e:
            return web.json_response({"error": f"cannot read: {type(e).__name__}"}, status=415)

    @PromptServer.instance.routes.get("/lunasave/filmstrip")
    async def _luna_filmstrip(request):
        """Evenly spaced frames side by side, as one JPEG, for the film strip."""
        path = _input_file(request.query.get("file", ""))
        if not path:
            return web.Response(status=404)
        try:
            n = int(request.query.get("n", "12"))
        except ValueError:
            n = 12
        try:
            data = await asyncio.get_running_loop().run_in_executor(None, filmstrip_jpeg, path, n)
        except Exception:
            return web.Response(status=415)
        return web.Response(body=data, content_type="image/jpeg",
                            headers={"Cache-Control": "max-age=3600"})


NODE_CLASS_MAPPINGS = {"Luna-H3-AssetLoader": LunaAssetLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"Luna-H3-AssetLoader": "Luna-H3-AssetLoader"}

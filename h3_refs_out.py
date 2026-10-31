"""Luna-H3-RefsOut — one wire in, every MiniMax H3 reference socket out.

The Luna-H3-AssetLoader already knows what each thing it loaded is and what number
the model will give it. Carrying that across the canvas as eighteen wires throws
that knowledge away and makes you re-establish it by hand every time the loader
moves. This node takes the loader's single `refs` output and fans it back out
into sockets named exactly like H3's own, so the wiring is mechanical.

Deliberately NOT a generic bundle. The pattern this is modelled on
(PlagueKind's Bundle In / Bundle Out) uses `"*"` wildcard sockets on both ends
and maps them positionally, which works but throws type checking away with it —
nothing stops an AUDIO landing in `ref_image_3`, and nothing complains until the
sampler does. Here the payload's shape is known, so every socket carries its
real type and a mis-wire is refused by the frontend.

There is no Bundle In: the Luna-H3-AssetLoader IS the bundle in. A second node to
assemble what one node already assembled would only add a place for the order to
go wrong.

Switches. One per category (pictures, clips, clip sound, sounds, song). An off
category delivers nothing: its sockets return None, and H3 skips a socket that
delivers nothing (core nodes_minimax_h3.py skips None values). The brief is then
REBUILT from the loader's per-item descriptions (`refs["items"]`) with H3's numbering
recounted for what is still on: switch a clip's sound off and the standalone sounds
move down to <Audio 1>. Wire this node's `asset_brief`, not the loader's, into the
prompt writer. Refs without `items` (older data) fall back to the loader's own brief
when every switch is on, else a brief built from the counts.

Socket names are 0-based to match H3's (`ref_image_0`), while the prompt tags in
`asset_brief` are 1-based (`<Picture 1>`), because that is H3's own
inconsistency — its Autogrow sockets count from 0 and its tokenizer counts from
1. Absorbing it here is better than renumbering silently and having the brief
disagree with the graph.
"""

from __future__ import annotations

MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3


def _slot(items, i):
    """Item i, or None when fewer were loaded. Never a black frame — an empty
    socket leaves H3's Autogrow slot unconnected, which is what "not supplied"
    means; a zero tensor would be a silent black reference."""
    if not items or i >= len(items):
        return None
    return items[i]


class LunaH3RefsOut:
    DESCRIPTION = (
        "Unpacks the Luna-H3-AssetLoader's `refs` output into every MiniMax H3 reference "
        "socket. Park it beside the H3 node and the loader stays one wire away, wherever "
        "you move it. Socket names match H3's own, so each one drags straight across. "
        "Sockets past what was loaded stay empty.\n\n"
        "Five switches (ref_image, ref_video, ref_video_audio, ref_audio, audio) turn a category off: its sockets "
        "then deliver nothing, which H3 skips, and `asset_brief` is rebuilt for what is left with "
        "the numbering recounted. Wire this node's `asset_brief` to the prompt writer. `audio` is the "
        "song, for the save node and the prompt writer."
    )
    CATEGORY = "Luna-MinimaxH3"
    FUNCTION = "unpack"

    RETURN_TYPES = (
        ("IMAGE",) * MAX_IMAGES
        + ("IMAGE",) * MAX_VIDEOS
        + ("AUDIO",) * MAX_VIDEOS
        + ("AUDIO",) * MAX_AUDIOS
        + ("STRING",)
        + ("AUDIO",)     # LAST: saved links keep their slots
    )
    RETURN_NAMES = (
        tuple(f"ref_image_{i}" for i in range(MAX_IMAGES))
        + tuple(f"ref_video_{i}" for i in range(MAX_VIDEOS))
        + tuple(f"ref_video_audio_{i}" for i in range(MAX_VIDEOS))
        + tuple(f"ref_audio_{i}" for i in range(MAX_AUDIOS))
        + ("asset_brief",)
        + ("audio",)
    )
    OUTPUT_TOOLTIPS = (
        tuple(
            f"Picture {i + 1} at its original size, for H3's ref_image_{i} socket."
            for i in range(MAX_IMAGES)
        )
        + tuple(
            f"Video {i + 1} as a 24 fps frame batch on H3's 17k+5 grid, for ref_video_{i}."
            for i in range(MAX_VIDEOS)
        )
        + tuple(
            f"The soundtrack of video {i + 1}, trimmed to the same window, for "
            f"ref_video_audio_{i}. Empty if that clip had no audio or none was wanted."
            for i in range(MAX_VIDEOS)
        )
        + tuple(
            f"Standalone reference audio {i + 1}, for ref_audio_{i}."
            for i in range(MAX_AUDIOS)
        )
        + (
            "The inventory string, rebuilt for the categories switched on, with H3's numbering "
            "recounted. Feed it to the prompt writer from beside the H3 node.",
            "The song window (the loader's `audio`), when the song switch is on; else empty. For "
            "Luna-SaveVideo's audio and Luna-LLMPrompt's audio.",
        )
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refs": ("LUNA_H3_REFS", {
                    "tooltip": "The `refs` output of Luna-H3-AssetLoader.",
                }),
                "pictures": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off: the nine ref_image sockets deliver nothing (H3 skips an empty socket) and "
                               "the pictures leave asset_brief.",
                }),
                "clips": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off: the three ref_video sockets deliver nothing, and so do the clip "
                               "soundtracks (they ride with their clip). The clips leave asset_brief.",
                }),
                "clip_sound": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off: the three ref_video_audio sockets deliver nothing (H3 skips them). "
                               "The standalone sounds then count from <Audio 1> in asset_brief.",
                }),
                "sounds": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off: the three ref_audio sockets deliver nothing and the standalone "
                               "sounds leave asset_brief.",
                }),
                "song": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off: the `audio` output delivers nothing and the song line leaves "
                               "asset_brief.",
                }),
            },
        }

    def unpack(self, refs, pictures=True, clips=True, clip_sound=True, sounds=True, song=True):
        refs = refs if isinstance(refs, dict) else {}
        # An off category hands back None for every socket of it; clips off also silences
        # the clip soundtracks, which belong to their clip.
        images = (refs.get("images") or []) if pictures else []
        videos = (refs.get("videos") or []) if clips else []
        video_audios = (refs.get("video_audios") or []) if clips and clip_sound else []
        audios = (refs.get("audios") or []) if sounds else []
        brief = build_brief(refs, pictures, clips, clip_sound, sounds, song)
        return (
            tuple(_slot(images, i) for i in range(MAX_IMAGES))
            + tuple(_slot(videos, i) for i in range(MAX_VIDEOS))
            + tuple(_slot(video_audios, i) for i in range(MAX_VIDEOS))
            + tuple(_slot(audios, i) for i in range(MAX_AUDIOS))
            + (brief, refs.get("song") if song else None)
        )


def build_brief(refs, pictures=True, clips=True, clip_sound=True, sounds=True, song=True) -> str:
    """The inventory for what is switched on, numbered the way H3 numbers it.

    H3 gives each type its own 1-based count. A clip's soundtrack takes an <Audio>
    number IMMEDIATELY BEFORE its clip, and standalone sounds continue the count after
    every clip, so the numbers depend on which switches are on.
    """
    items = refs.get("items")
    if not isinstance(items, dict):
        return _fallback_brief(refs, pictures, clips, clip_sound, sounds, song)
    images = (items.get("images") or []) if pictures else []
    videos = (items.get("videos") or []) if clips else []
    v_audio = items.get("video_audios") or []
    standalone = (items.get("audios") or []) if sounds else []
    parts = []
    if images:
        parts.append("Reference images, in order: "
                     + "; ".join(f"<Picture {i}> = {d}" for i, d in enumerate(images, 1)) + ".")
        if items.get("picture_warning"):
            parts.append(items["picture_warning"])
    n_audio = 0
    for k, desc in enumerate(videos, 1):
        if clip_sound and k - 1 < len(v_audio) and v_audio[k - 1]:
            n_audio += 1
            parts.append(f"<Audio {n_audio}> = the soundtrack of <Video {k}>.")
        parts.append(f"<Video {k}> = {desc}.")
    for desc in standalone:
        n_audio += 1
        parts.append(f"<Audio {n_audio}> = {desc}.")
    if standalone and not images and not videos:
        parts.append("WARNING: MiniMax H3 does not accept reference sounds without a reference "
                     "picture or clip.")
    if song and items.get("song"):
        parts.append(items["song"])
    parts += [x for x in (items.get("extra") or []) if x]
    return " ".join(parts)


def _fallback_brief(refs, pictures, clips, clip_sound, sounds, song) -> str:
    """Refs from an older loader have no per-item descriptions: keep its brief when
    nothing is switched off, else say what is left from the counts alone."""
    if pictures and clips and clip_sound and sounds and song:
        return refs.get("brief", "")
    parts = []
    n_img = len(refs.get("images") or []) if pictures else 0
    if n_img:
        parts.append("Reference images, in order: " + "; ".join(f"<Picture {i}>" for i in range(1, n_img + 1)) + ".")
    n_audio = 0
    v_audio = refs.get("video_audios") or []
    for k in range(1, (len(refs.get("videos") or []) if clips else 0) + 1):
        if clip_sound and k - 1 < len(v_audio) and v_audio[k - 1] is not None:
            n_audio += 1
            parts.append(f"<Audio {n_audio}> = the soundtrack of <Video {k}>.")
        parts.append(f"<Video {k}>.")
    for _ in range(len(refs.get("audios") or []) if sounds else 0):
        n_audio += 1
        parts.append(f"<Audio {n_audio}>.")
    return " ".join(parts)


NODE_CLASS_MAPPINGS = {"Luna-H3-RefsOut": LunaH3RefsOut}
NODE_DISPLAY_NAME_MAPPINGS = {"Luna-H3-RefsOut": "Luna-H3-RefsOut"}

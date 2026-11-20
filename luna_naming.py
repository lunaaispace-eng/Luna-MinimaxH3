"""Shared naming for the Luna packs: the tag a target model gives each picture, clip and sound, and the user's @ names.

- One file, hand-copied and identical in Luna-LLM-Tools, Luna-Media-Tools and Luna-MinimaxH3. Each pack's
  `tests/test_luna_naming.py` compares the copies. Pure Python: no ComfyUI, torch or numpy import.
- The rule: Luna-MinimaxH3 `SPEC.md` § 2, Naming. The user may point at items with @imageN, @videoN, @audioN, N being the card
  number shown on the loader; nothing happens without them. No @ name reaches a model: the node that sends the prompt replaces
  each one with the target row's tag, and an @ name with no such item warns.
- A brief carries the map as one names line ("@ names: @image1 = <Picture 2>; …"). `read_names_line` reads it back, so the
  map a writer shows its model and the map it replaces with are the same one, also after Luna-H3-RefsOut's switches.
- `inventory` writes the brief of a loader's `refs["items"]` for what is sent and what the reader can take (Luna-H3-AssetLoader,
  Luna-H3-RefsOut and the writers use the one text).
- A position is an item's 1-based place among what is sent of its kind. Positions do not depend on the row; the row only
  renders them. Audio positions follow MiniMax's order (each clip's sound right before its clip, then the standalone sounds).
- Rows:
  - minimax (H3 video, MiniMax Image): <Picture n>, <Video n>, <Audio n>, from 1; edit base first (DOCUMENTED, ComfyUI
    `comfy_extras/nodes_minimax_h3.py`: tags, order, and H3 Image's image_1 as the edit base).
  - qwen (Qwen-Image 2.1): <imageN> from 1, pictures only (DOCUMENTED, preset `Qwen_Image_2.1_Edit.md`); edit base first
    (INFERRED: the preset's input image is <image1>).
  - cloud (GPT Image, Gemini, Flow, ChatGPT web): Image N from 1, pictures only, card order (PROPOSED).
  - grok (xAI Grok Imagine): <IMAGE_n> from 0, pictures only; the picture to edit first (DOCUMENTED, the Grok nodes' tooltips).
"""

from __future__ import annotations

import re

KINDS = ("image", "video", "audio")

ROWS = {
    "minimax": {"image": "<Picture {n}>", "video": "<Video {n}>", "audio": "<Audio {n}>", "first": 1, "edit_first": True},
    "qwen": {"image": "<image{n}>", "first": 1, "edit_first": True},
    "cloud": {"image": "Image {n}", "first": 1, "edit_first": False},
    "grok": {"image": "<IMAGE_{n}>", "first": 0, "edit_first": True},
}

# A loader's ⚙ Target or a preset's `labels:` value -> its row.
TARGET_ROWS = {
    "minimax": "minimax", "h3": "minimax", "minimax image": "minimax", "minimax_image": "minimax",
    "qwen": "qwen",
    "cloud": "cloud", "general": "cloud",
    "grok": "grok",
}

# What an @ name becomes when it has no tag: no such item, not sent, or a kind the row's model does not take.
WORDS = {"image": "the picture", "video": "the clip", "audio": "the sound"}
NOUNS = {"image": "picture", "video": "clip", "audio": "sound"}

# Roles of a picture card that is NOT sent to the model as a reference (a keyframe sits on the shot instead).
NOT_SENT = (None, "", "key", "off")

# An @ token: "@", a kind, and the rest of the word. Only ASCII digits after the kind make a name (@image2, @Image2's); any
# other tail (@image1st, @image, @image²) is no name: it warns and becomes plain words too, so no @ token reaches a model.
# The kind is matched case-blind in ASCII only (a Unicode case fold such as "İ" -> "i̇" would give a kind with no entry).
# A letter or "@" right before the "@" makes it no token (mail@image1.com); a digit does not (@image1@image2 is two names).
_KIND = r"([Ii][Mm][Aa][Gg][Ee]|[Vv][Ii][Dd][Ee][Oo]|[Aa][Uu][Dd][Ii][Oo])"
_AT = re.compile(r"(?<![^\W\d])(?<!@)@" + _KIND + r"(\w*)")
_NAME = re.compile(r"@" + _KIND + r"([0-9]+)")
_DIGITS = re.compile(r"[0-9]+")
# The names line inside any text (a brief joins its parts with spaces, so it need not stand on its own line).
_ENTRY = r"@(?:image|video|audio)[0-9]+ = [^;.\n]*"
_LINE = re.compile(rf"@ names: ({_ENTRY}(?:; {_ENTRY})*)\.")
# Each row's tag as a pattern, to read a names line back into positions.
_UNTAG = [(row, kind, re.compile(re.escape(spec[kind]).replace(r"\{n\}", r"(\d+)")), spec["first"])
          for row, spec in ROWS.items() for kind in KINDS if kind in spec]


def row_for(target) -> str | None:
    """A loader's ⚙ Target or a preset's `labels:` value -> row name, or None when it names no row."""
    return TARGET_ROWS.get(str(target or "").strip().lower())


def tag(row: str, kind: str, pos: int) -> str | None:
    """The row's tag for the pos-th item of a kind (pos 1-based), or None when the row's model does not take that kind."""
    spec = ROWS.get(row)
    if not spec or kind not in spec or pos < 1:
        return None
    return spec[kind].format(n=pos - 1 + spec["first"])


def assign(row: str, pictures=(), clips=(), sounds=()) -> dict:
    """Card index (0-based) -> position (1-based), per kind, for the cards that are sent.

    pictures: one role per picture card; "edit" is the edit base, a role in NOT_SENT is not sent, any other role is sent.
    clips: one {"sent": bool, "sound": bool} per video card; "sound" = its soundtrack is sent with it.
    sounds: one bool per sound card, True = sent as a reference sound.
    Returns {"image", "video", "video_audio", "audio": {card: pos}, "order": [picture cards in sent order]}."""
    sent = [i for i, role in enumerate(pictures) if role not in NOT_SENT]
    edit = [i for i in sent if pictures[i] == "edit"]
    if edit and ROWS.get(row, {}).get("edit_first"):
        sent = edit[:1] + [i for i in sent if i != edit[0]]
    image = {i: p for p, i in enumerate(sent, 1)}
    video, video_audio, audio, n_audio = {}, {}, {}, 0
    for k, c in enumerate(clips):
        if not c.get("sent"):
            continue
        if c.get("sound"):
            n_audio += 1
            video_audio[k] = n_audio
        video[k] = len(video) + 1
    for j, on in enumerate(sounds):
        if on:
            n_audio += 1
            audio[j] = n_audio
    return {"image": image, "video": video, "video_audio": video_audio, "audio": audio, "order": sent}


def make(row: str, pictures=(), clips=(), sounds=()) -> dict:
    """The `naming` payload a loader puts in `refs["naming"]` (and uses for its brief).

    {"row": row, "map": {"@image1": ("image", pos) or None, ...}, "counts": {kind: cards}, "order": [...],
    "cards": the three inputs, kept so `remake` can switch categories off}.
    A map value of None means the card exists but is not sent (a keyframe, a guide clip, a locked song)."""
    pictures, clips, sounds = list(pictures), [dict(c) for c in clips], [bool(s) for s in sounds]
    a = assign(row, pictures, clips, sounds)
    per_kind = {"image": (len(pictures), a["image"]), "video": (len(clips), a["video"]), "audio": (len(sounds), a["audio"])}
    names = {}
    for kind, (count, placed) in per_kind.items():
        for i in range(count):
            names[f"@{kind}{i + 1}"] = (kind, placed[i]) if i in placed else None
    return {"row": row, "map": names, "counts": {k: c for k, (c, _) in per_kind.items()}, "order": a["order"],
            "cards": {"pictures": pictures, "clips": clips, "sounds": sounds}}


def remake(naming: dict, pictures=True, clips=True, clip_sound=True, sounds=True) -> dict:
    """The naming after categories are switched off (Luna-H3-RefsOut): what is off is not sent, and what follows it moves
    up a number (a clip sound off makes the standalone sounds count from <Audio 1>)."""
    cards = naming.get("cards") or {}
    pics = [r if pictures else None for r in cards.get("pictures", [])]
    vids = [dict(c, sent=bool(c.get("sent")) and clips, sound=bool(c.get("sound")) and clip_sound)
            for c in cards.get("clips", [])]
    snds = [bool(s) and sounds for s in cards.get("sounds", [])]
    return make(naming.get("row") or "minimax", pics, vids, snds)


def without_sound(naming: dict | None) -> dict | None:
    """The naming for a reader that cannot take sound (a writer route without audio): every sound is not sent to it, and
    every picture and clip keeps its number. A naming read from a names line (no cards) gets the same, entry by entry."""
    if naming is None:
        return None
    if isinstance(naming.get("cards"), dict):
        return remake(naming, clip_sound=False, sounds=False)
    out = dict(naming)
    out["map"] = {k: (None if k.startswith("@audio") else v) for k, v in (naming.get("map") or {}).items()}
    return out


def placed(naming: dict | None, kind: str = "image", before: int = 0, keep: int | None = None) -> dict | None:
    """The naming after a node places the sent items of one kind among its own: `keep` = how many of them it really sends
    (the rest is dropped: not sent), `before` = how many items of that kind it sends ahead of them (a first frame), so each
    kept item's position moves up by `before`. Other kinds are unchanged. The result cannot be remade (no cards)."""
    if naming is None:
        return None
    out = dict(naming)
    m = {}
    for key, entry in (naming.get("map") or {}).items():
        if entry is not None and entry[0] == kind:
            pos = entry[1]
            entry = None if keep is not None and pos > keep else (kind, pos + int(before))
        m[key] = entry
    out["map"], out["cards"] = m, None
    return out


def from_refs(refs) -> dict | None:
    """The `naming` payload carried in a LUNA_H3_REFS dict, or None when there is none (older loaders)."""
    naming = refs.get("naming") if isinstance(refs, dict) else None
    if isinstance(naming, dict) and isinstance(naming.get("map"), dict) and naming.get("row") in ROWS:
        return naming
    return None


def _missing(name: str, kind: str, count) -> str:
    if count is None:
        return f"{name} names no {NOUNS[kind]}."
    return f"{name} names no {NOUNS[kind]}: {count} {NOUNS[kind]}(s) here."


def resolve(name: str, naming: dict | None = None, row: str | None = None, counts: dict | None = None):
    """One @ name -> (tag or None, warning or "").

    Without `naming`, @imageN is the N-th item sent (no reordering), and only when `counts` says how many items of each kind
    are sent (a kind it leaves out counts 0). With neither, there is nothing to point at: no tag, and a warning."""
    m = _NAME.fullmatch(name.strip())
    if not m:
        return None, f"{name} is not an @ name (@imageN, @videoN, @audioN)."
    kind, n = m.group(1).lower(), int(m.group(2))
    key = f"@{kind}{n}"
    row = row or (naming or {}).get("row") or "cloud"
    if naming is not None:
        if key not in naming.get("map", {}):
            return None, _missing(key, kind, (naming.get("counts") or {}).get(kind))
        entry = naming["map"][key]
        if entry is None:
            return None, (f"{key} is not sent to the model as a reference (it sits on the shot, is switched off, or is a "
                          f"sound on a route that takes none).")
        kind, pos = entry
    else:
        if counts is None:
            return None, f"{key}: nothing here says which {NOUNS[kind]}s are sent, so it is written as words."
        count = int(counts.get(kind) or 0)
        if n < 1 or n > count:
            return None, _missing(key, kind, count)
        pos = n
    t = tag(row, kind, pos)
    if t is None:
        return None, f"{key}: this model takes no {NOUNS[kind]}s."
    return t, ""


def replace(text: str, naming: dict | None = None, row: str | None = None, counts: dict | None = None):
    """Every @ name in `text` -> the row's tag; one with no tag becomes plain words. Returns (text, warnings).

    row: the row of the model the prompt goes to; default the naming's own row, else cloud.
    counts: without `naming`, how many items of each kind the node sends, after its own filtering ({"image": 2}); a kind
    left out counts 0. None = not known: every @ name then warns and becomes words."""
    warnings: list[str] = []

    def sub(m):
        kind, tail = m.group(1).lower(), m.group(2)
        if _DIGITS.fullmatch(tail):
            t, warn = resolve(m.group(0), naming, row, counts)
        else:
            t, warn = None, f"{m.group(0)} is not an @ name (@imageN, @videoN, @audioN), so it is written as words."
        if warn and warn not in warnings:
            warnings.append(warn)
        return t if t is not None else WORDS[kind]

    return _AT.sub(sub, text or ""), warnings


def names_line(naming: dict | None, row: str | None = None) -> str:
    """One brief line mapping each @ name to its tag: '@ names: @image1 = <Picture 2>; @image2 = <Picture 1>.' ("" if none)."""
    if not naming or not naming.get("map"):
        return ""
    row = row or naming.get("row") or "cloud"
    parts = []
    for key in naming["map"]:
        t, _ = resolve(key, naming, row)
        parts.append(f"{key} = {t}" if t is not None else f"{key} = not sent" if naming["map"][key] is None
                     else f"{key} = not taken by this model")
    return "@ names: " + "; ".join(parts) + "."


def _untag(kind: str, value: str):
    """A tag from a names line -> ((kind, pos), row), or (None, None) for "not sent" and anything else."""
    for row, k, pattern, first in _UNTAG:
        m = pattern.fullmatch(value) if k == kind else None
        if m:
            return (kind, int(m.group(1)) - first + 1), row
    return None, None


def read_names_line(text: str) -> dict | None:
    """The names line in a text (a brief, a PICTURES block) -> a naming payload, or None when the text has none.

    This is what the model is shown, so a writer replaces with it before any `refs` naming: Luna-H3-RefsOut's brief carries
    the map after its switches. Positions come from the tags, in whichever row the line was written; "not sent" and
    "not taken by this model" read as None. `row` is None when no tag shows it (every item not sent). `cards` is None: a read
    naming cannot be remade."""
    m = _LINE.search(text or "")
    if not m:
        return None
    names, counts, found = {}, {k: 0 for k in KINDS}, []
    for entry in m.group(1).split("; "):
        key, _, value = entry.partition(" = ")
        key = key.strip().lower()
        kind = _NAME.fullmatch(key).group(1)
        counts[kind] += 1
        names[key], row = _untag(kind, value.strip())
        if row:
            found.append(row)
    return {"row": found[0] if found else None, "map": names, "counts": counts, "order": [], "cards": None}


def pick(text: str | None = None, refs=None) -> dict | None:
    """The naming a writer uses: the names line in the text it shows the model wins, else the one in `refs`, else None.
    A line whose row no tag shows takes the row of the `refs` naming, when there is one."""
    read, carried = read_names_line(text), from_refs(refs)
    if read is None:
        return carried
    if read["row"] is None and carried is not None:
        read["row"] = carried["row"]
    return read


def drop_names_line(text: str) -> str:
    """`text` without its names line (to rebuild a brief around it); the text as it is when it has none."""
    return _LINE.sub("", text or "")


def put_names_line(text: str, naming: dict | None, row: str | None = None) -> str:
    """`text` with its names line rendered in `row` (the model's row): replaced where it is, else added on a new line."""
    line = names_line(naming, row)
    if not line:
        return text or ""
    if _LINE.search(text or ""):
        return _LINE.sub(lambda _m: line, text, count=1)
    return f"{text}\n{line}" if text else line


def inventory(items: dict, naming: dict | None = None, row: str = "minimax", *, pictures=True, clips=True,
              clip_sound=True, sounds=True, song=True, hear=True) -> str:
    """The brief of a loader's `refs["items"]`, in MiniMax's numbering: what the model receives, one line per item.

    items: one description per item WITHOUT its tag ({"images", "videos", "video_audios" (one bool per clip: it has a sound),
    "audios", "song", "vocals" (a line or ""), "picture_warning", "extra"}); a key left out is empty.
    pictures, clips, clip_sound, sounds, song: what the model receives (Luna-H3-RefsOut's switches). Off is not sent, and what
    follows it moves up a number, as `remake` does. The vocal stem follows `song`.
    hear: False when the reader of the brief cannot take sound (a writer route without audio). The sound lines (each clip's
    sound, the standalone sounds, the song, the vocal stem) leave the brief and the names line gives each sound as not sent
    (`without_sound`); every other number stays as the model receives it.
    naming: the map for the names line, already remade for the switches; rendered in `row`."""
    items = items if isinstance(items, dict) else {}
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
            if hear:
                parts.append(f"<Audio {n_audio}> = the soundtrack of <Video {k}>.")
        parts.append(f"<Video {k}> = {desc}.")
    for desc in standalone:
        n_audio += 1
        if hear:
            parts.append(f"<Audio {n_audio}> = {desc}.")
    if standalone and not images and not videos:
        parts.append("WARNING: MiniMax H3 does not accept reference sounds without a reference "
                     "picture or clip.")
    at_line = names_line(naming if hear else without_sound(naming), row) if naming else ""
    if at_line:
        parts.append(at_line)
    if song and hear:
        parts += [items[k] for k in ("song", "vocals") if items.get(k)]
    parts += [x for x in (items.get("extra") or []) if x]
    return " ".join(parts)


def mismatch(preset_row: str | None, refs_row: str | None) -> str:
    """A warning when a preset writes one row's tags and the loader named its items for another; "" when they agree."""
    if not preset_row or not refs_row or preset_row == refs_row:
        return ""
    return (f"The preset writes {tag(preset_row, 'image', 1)} tags, but the loader named its pictures for "
            f"{tag(refs_row, 'image', 1)} (its Target setting). The @ names are written as the preset's tags; set the loader's "
            f"Target to match the model.")

"""Luna-H3-AssetLoader — picture handling, pure PIL/numpy/torch (no ComfyUI imports).

Moved out of asset_loader.py on 2026-10-25 so it can be unit-tested
(tests/test_asset_images.py), and fixed the same day after a review:

- `to_rgb` fills transparent areas with a colour instead of dropping alpha (which
  showed whatever colour the file kept under them), and reads 16-bit greyscale
  instead of turning it solid white.
- `apply_state` snaps by cropping to the grid, never by stretching: rounding width
  and height separately changed the aspect by up to ~3% and could upscale a picture
  with `allow_upscale` off.
- `conform` letterboxes the `images` batch instead of stretching each picture to
  picture 1's shape.

The per-image state schema follows ComfyUI-Pixaroma's LoadImageMini state
(https://github.com/pixaroma/ComfyUI-Pixaroma, MIT, Copyright (c) 2026 pixaroma);
see asset_loader.py's header. The code is this pack's own.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image, ImageOps

# Pillow modes that carry transparency, and the 16/32-bit greyscale ones.
_ALPHA_MODES = ("RGBA", "LA", "PA", "RGBa", "La")
_WIDE_GREY = ("I;16", "I;16B", "I;16L", "I;16N", "I")

# Float slack for "w * scale" landing a hair under a grid line (1023.9999 -> 1024).
_EPS = 1e-6


def to_rgb(img: Image.Image, background: str = "#000000") -> tuple[Image.Image, str | None]:
    """Any Pillow picture -> 8-bit RGB, plus a note when something was filled.

    Transparent areas are laid on `background`; the note says so, because a cut-out
    character on black and the same one on white read differently to H3.
    """
    note = None
    if img.mode in _WIDE_GREY:
        arr = np.asarray(img).astype(np.float64)
        # 16-bit files fill 0..65535. Mode "I" is 32-bit and is 8-bit data as often
        # as 16-bit; its own range decides.
        top = 65535.0 if img.mode != "I" or arr.max(initial=0) > 255 else 255.0
        img = Image.fromarray(np.clip(arr / top * 255.0 + 0.5, 0, 255).astype(np.uint8))
    elif img.mode == "F":
        arr = np.asarray(img, dtype=np.float64)
        img = Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8))
    has_alpha = img.mode in _ALPHA_MODES or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        if rgba.getchannel("A").getextrema()[0] < 255:
            canvas = Image.new("RGBA", rgba.size, background or "#000000")
            canvas.alpha_composite(rgba)
            note = f"transparent areas filled with {background or '#000000'}"
            return canvas.convert("RGB"), note
        img = rgba
    return img.convert("RGB"), note


def _grid_floor(v: float, step: int) -> int:
    return max(step, int(math.floor(v / step + _EPS)) * step)


def snap_size(w: float, h: float, step: int) -> tuple[int, int]:
    """The size a picture of w x h (after scaling) ends at. With a grid step, each
    edge goes DOWN to the grid (never up: snapping must not upscale), and the
    difference is cropped, so the picture is never stretched."""
    if step and step > 1:
        return _grid_floor(w, step), _grid_floor(h, step)
    return max(1, round(w)), max(1, round(h))


def _anchored(span: int, keep: int, anchor: str, lo: str, hi: str) -> int:
    if lo in anchor:
        return 0
    if hi in anchor:
        return span - keep
    return (span - keep) // 2


def apply_state(img: Image.Image, st: dict) -> Image.Image:
    """Conform aspect first, then size, then snap. Order matters: cropping after
    scaling would undo the size you just asked for."""
    if st.get("mode") == "off" and not st.get("ratio") and not st.get("snap"):
        return img

    w, h = img.size
    anchor = str(st.get("crop_anchor") or "center")

    # 1. aspect — crop away or pad out to the requested ratio
    ratio = str(st.get("ratio") or "").strip()
    if ratio and ":" in ratio:
        try:
            rw, rh = (float(x) for x in ratio.split(":", 1))
            target = rw / rh
        except Exception:
            target = None
        if target and target > 0 and abs(w / h - target) > 0.001:
            if st.get("ratio_action") == "pad":
                nw, nh = (w, round(w / target)) if w / h > target else (round(h * target), h)
                canvas = Image.new("RGB", (max(nw, w), max(nh, h)), st.get("pad_color") or "#000000")
                canvas.paste(img, ((max(nw, w) - w) // 2, (max(nh, h) - h) // 2))
                img = canvas
            else:
                nw, nh = (round(h * target), h) if w / h > target else (w, round(w / target))
                left = _anchored(w, nw, anchor, "left", "right")
                top = _anchored(h, nh, anchor, "top", "bottom")
                img = img.crop((left, top, left + nw, top + nh))
            w, h = img.size

    # 2. size
    mode = str(st.get("mode") or "off")
    scale = 1.0
    if mode == "max_mp":
        budget = float(st.get("max_mp", 2.0)) * 1_000_000
        if budget > 0:
            scale = math.sqrt(budget / (w * h))
    elif mode == "longest_side":
        scale = float(st.get("longest_side", 1024)) / max(w, h)
    elif mode == "scale_factor":
        scale = float(st.get("scale_factor", 1.0))
    if not st.get("allow_upscale", False):
        scale = min(1.0, scale)

    # 3. snap — cover the snapped box at the picture's own aspect, crop the rest
    step = int(st.get("snap") or 0)
    tw, th = snap_size(w * scale, h * scale, step)
    if (tw, th) == (w, h):
        return img
    cover = max(tw / w, th / h)
    cw, ch = max(tw, math.ceil(w * cover - _EPS)), max(th, math.ceil(h * cover - _EPS))
    if (cw, ch) != (w, h):
        img = img.resize((cw, ch), Image.LANCZOS)
    if (cw, ch) != (tw, th):
        left = _anchored(cw, tw, anchor, "left", "right")
        top = _anchored(ch, th, anchor, "top", "bottom")
        img = img.crop((left, top, left + tw, top + th))
    return img


def conform(frames: list[torch.Tensor]) -> torch.Tensor:
    """Mixed sizes -> one batch at frame 1's size, each picture letterboxed (fitted
    whole, black bars) so none is stretched.

    Only the batch output is conformed. Its consumers (LLM Prompt, Luna Agent Chat)
    look at every frame, and a portrait squeezed into a landscape frame shows them a
    distorted picture; bars show them the real one.
    """
    h, w = int(frames[0].shape[1]), int(frames[0].shape[2])
    out = []
    for f in frames:
        if int(f.shape[1]) == h and int(f.shape[2]) == w:
            out.append(f)
            continue
        arr = (f[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        boxed = ImageOps.pad(Image.fromarray(arr), (w, h), method=Image.LANCZOS, color=(0, 0, 0))
        out.append(torch.from_numpy(np.asarray(boxed, dtype=np.float32) / 255.0)[None, ...])
    return torch.cat(out, dim=0)

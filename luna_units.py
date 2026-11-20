"""Shared units of the Luna video packs: the frame grid, the overlap list and the frame rate of each target.

- One file, hand-copied and identical in Luna-MinimaxH3 and Luna-Media-Tools. Each pack's `tests/test_luna_units.py`
  compares the copies. Pure Python: no ComfyUI, torch or numpy import.
- The rules: Luna-MinimaxH3 `SPEC.md` § 1 (frame count, fps) and § 2 (overlap); Luna-Media-Tools `SPEC.md` § 1.
- Frame count: every node that makes an H3 frame count rounds UP onto H3's `17k+5` grid; above 362 frames it warns and does
  not cut. A length in seconds becomes whole frames first (half up), so a length made from a grid count (124 / 24 s) gives
  that count back and a float's last digit never adds a grid step.
- Cutting to what exists (a clip that ends early) rounds DOWN instead: `frames_down`. That is a cut, not a count made.
- Overlap: one list on every node, in frames; any other value snaps to the nearest with a warning. A node may refuse a value
  of the list it cannot take (the chain refuses 22 without the song lock), with its own message.
- Every message is ASCII: a node prints it, and a console that is not UTF-8 must not stop the run.
"""

from __future__ import annotations

import math

H3_FPS = 24.0
LTX_FPS = 25.0
# H3's trained length (DOCUMENTED in both packs' media_io.py as TRAINED_MAX_FRAMES); past it a warning, not a cut.
H3_TRAINED_MAX = 362

# Grid name -> (step, phase): a count on the grid is step * k + phase. "free" is every whole frame.
GRIDS = {"h3": (17, 5), "ltx": (8, 1), "free": (1, 0)}

# Target -> grid and frame rate. Any other target (general) keeps the source's rate and has no grid.
TARGET_GRIDS = {"h3": "h3", "ltx": "ltx"}
TARGET_FPS = {"h3": H3_FPS, "ltx": LTX_FPS}

# The one overlap list (frames shared with the clip or scene before); 0 is the first clip or scene.
OVERLAPS = (0, 22, 39, 90, 141, 192)


def _grid(grid: str) -> tuple[int, int]:
    if grid not in GRIDS:
        raise ValueError(f"unknown frame grid {ascii(grid)}; one of {', '.join(GRIDS)}")
    return GRIDS[grid]


def _whole_up(n) -> int:
    """A frame count -> whole frames, a fraction counting as one more frame (124.9 -> 125); a float's last digit
    (124.00000000000001) does not."""
    return int(math.ceil(float(n) - 1e-9))


def first_frames(grid: str = "h3") -> int:
    """The smallest count on the grid: 5 on H3, 1 on LTX and free."""
    step, phase = _grid(grid)
    return phase if phase > 0 else 1


def on_grid(n: int, grid: str = "h3") -> bool:
    """True when n frames is a whole count on the grid."""
    step, phase = _grid(grid)
    if float(n) != int(n):
        return False
    n = int(n)
    return n >= first_frames(grid) and n % step == phase % step


def frames_up(n, grid: str = "h3") -> int:
    """n frames rounded UP onto the grid (a fraction of a frame counts as a frame); at least the grid's first count."""
    step, phase = _grid(grid)
    n = max(_whole_up(n), first_frames(grid))
    return n + ((phase - n) % step)


def frames_down(n, grid: str = "h3") -> int:
    """n frames rounded DOWN onto the grid (cut to what exists); 0 when n is below the grid's first count."""
    step, phase = _grid(grid)
    n = int(math.floor(float(n) + 1e-9))
    if n < first_frames(grid):
        return 0
    return n - ((n - phase) % step)


def whole_frames(seconds, fps: float = H3_FPS) -> int:
    """A length in seconds -> whole frames at fps, half up (5.0 s at 24 = 120; 124 / 24 s = 124). Exactly half a frame goes
    up, the direction of the rule (Canvas's old Python round sent 22.5 to 22, now 23 and so 39 frames)."""
    return int(math.floor(float(seconds) * float(fps) + 0.5))


def frames_for(seconds, fps: float = H3_FPS, grid: str = "h3") -> int:
    """A length in seconds -> the frame count on the grid: whole frames first, then rounded UP onto the grid."""
    return frames_up(whole_frames(seconds, fps), grid)


def frames_warning(frames, grid: str = "h3") -> str:
    """The warning for a count past H3's trained length, else ""."""
    if grid == "h3" and int(frames) > H3_TRAINED_MAX:
        return f"{int(frames)} frames is past the {H3_TRAINED_MAX} frames H3 was trained on; it is kept, not cut."
    return ""


def h3_frames(n) -> tuple[int, str]:
    """n frames -> (the H3 count, rounded UP onto 17k+5, and its warning or "")."""
    f = frames_up(n, "h3")
    return f, frames_warning(f, "h3")


def snap_overlap(value) -> tuple[int, str]:
    """An overlap in frames -> (the nearest value on OVERLAPS, warning or ""). A tie takes the smaller value."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"overlap must be a number of frames, not {ascii(value)}")
    best = min(OVERLAPS, key=lambda o: (abs(o - v), o))
    if best == v:
        return best, ""
    return best, (f"Overlap {v:g} is not on the list {', '.join(str(o) for o in OVERLAPS)} frames; "
                  f"{best} is used.")


def grid_for(target) -> str:
    """A loader's Target -> its grid: h3, ltx, else free."""
    return TARGET_GRIDS.get(str(target or "").strip().lower(), "free")


def fps_for(target, source=None) -> float:
    """The frame rate a target renders at: h3 24, ltx 25; any other target the source's rate (24 when not known)."""
    fixed = TARGET_FPS.get(str(target or "").strip().lower())
    if fixed is not None:
        return fixed
    try:
        rate = float(source)
    except (TypeError, ValueError, OverflowError):
        return H3_FPS
    return rate if math.isfinite(rate) and rate > 0 else H3_FPS

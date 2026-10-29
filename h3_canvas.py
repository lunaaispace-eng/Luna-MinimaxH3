"""Luna-H3-Canvas — one node for the three numbers H3 actually needs.

Ask for an aspect ratio and a duration in seconds; get back a canvas, a frame
count, and the two frame rates. Replaces the usual pile of a generic resolution
node plus hand-written maths expressions for frame count and fps.

Why a dedicated node rather than a generic one: **H3 does not want a megapixel
target.** Its canvas is fully determined by the aspect ratio — a fixed 768 short
edge under a fixed 768x1344 area cap, each axis rounded to a multiple of 32. A
generic node makes you hand-tune a megapixel figure to land back on the number
the model already defines (1.032 MP). The `H3 canvas` mode here skips that.

Frame count is not free either: H3 samples on a grid where the count satisfies
`n % 17 == 5`, so 5, 22, 39 ... 243, 362. Ask for 7.5 s and you get 192 frames,
which is 8.0 s — a generic node snaps you silently, this one reports it.

`fps` and `output_fps` both leave from here so a frame count and its playback
rate cannot drift apart. Feeding `output_fps` to the save node is what stops the
classic "interpolation off but the fps still doubled" desync.

The constants (768 short edge, 768x1344 cap, multiple of 32, 24 fps, the 17k+5
grid, the 124-362 trained range) are requirements of the MiniMax H3 model —
interface facts, observable from its ComfyUI node signatures. The implementation
below is this pack's own; no code is taken from ComfyUI core. Written under
Apache-2.0; the pack is GPL-3.0 since 2026-10-26.
"""

from __future__ import annotations

import math

from comfy_api.latest import io

# H3 model parameters.
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
FPS = 24.0
FRAME_GRID = 17          # frame count must satisfy n % FRAME_GRID == FRAME_PHASE
FRAME_PHASE = 5
TRAINED_MIN, TRAINED_MAX = 124, 362

# Landscape form only, widest first, with a `portrait` toggle to flip. Listing both
# orientations would mean two names for one canvas and an obvious trap: pick "9:16"
# AND tick portrait and you are back to landscape. One list, one switch.
#
# Flipped, these cover the social sizes: 16:9 -> 9:16 (Reels, Shorts, TikTok),
# 5:4 -> 4:5 (Instagram portrait), 3:2 -> 2:3, 4:3 -> 3:4.
# ASCII only — these labels are saved into workflows and echoed to the console.
ASPECT_RATIOS = {
    "2.39:1 (anamorphic)": (239, 100),
    "21:9 (ultrawide)": (21, 9),
    "2:1": (2, 1),
    "1.91:1 (link preview)": (191, 100),
    "16:9 (widescreen)": (16, 9),
    "16:10": (16, 10),
    "3:2 (photo)": (3, 2),
    "4:3": (4, 3),
    "5:4": (5, 4),
    "1:1 (square)": (1, 1),
}

SIZE_MODES = ["H3 canvas", "megapixels", "preset"]

# Grid the canvas edges are rounded to in `megapixels` and `preset` modes. 32 is H3's
# own (16x VAE times its 2x2 patches); 16 works because the model pads the latent to
# whole patches and crops back. 8 does not exist for H3: the latent is size // 16, so
# a /8 edge that is not /16 comes out 8 px smaller. It stays for image work, with a
# note in `info` saying what H3 really makes.
MULTIPLES = ["32", "16", "8"]
LATENT_MULTIPLE = 16

# `preset` lists every size between these areas, in decimal megapixels.
PRESET_MIN_PIXELS = 200_000
PRESET_MAX_PIXELS = 2_000_000


def _round_to(value: float, multiple: int = CANVAS_MULTIPLE) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _nearest_step(num: int, den: int, multiple: int) -> int:
    """num / den rounded half-up to a multiple, in integers. js/luna_h3_canvas.js does
    the same sum, so both sides list the very same sizes (Python's round() is
    banker's rounding, Math.round is not)."""
    return (2 * num + den * multiple) // (2 * den * multiple) * multiple


def preset_sizes(ratio_w: int, ratio_h: int, multiple: int) -> list[tuple[int, int]]:
    """Every landscape canvas for the ratio on the grid, 0.2 to 2 MP, smallest first.

    A size is listed when one edge is on the grid and the other is the grid step
    nearest the exact ratio. Walking both edges, not just the short one, keeps the
    model's own sizes in: 1344x768 is the nearest height for width 1344, but height
    768 alone would give width 1376. A finer grid also holds every size of the
    coarser ones (/8 holds /16 holds /32), so changing the grid keeps your size.
    """
    sizes = set()
    if multiple * 2 <= CANVAS_MULTIPLE:
        sizes.update(preset_sizes(ratio_w, ratio_h, multiple * 2))
    edge = multiple
    # Loop bounds have 2x headroom; rounding can pull an edge back under the cap.
    while edge * edge * ratio_w <= 2 * PRESET_MAX_PIXELS * ratio_h:  # edge = height
        sizes.add((max(edge, _nearest_step(edge * ratio_w, ratio_h, multiple)), edge))
        edge += multiple
    edge = multiple
    while edge * edge * ratio_h <= 2 * PRESET_MAX_PIXELS * ratio_w:  # edge = width
        height = max(multiple, _nearest_step(edge * ratio_h, ratio_w, multiple))
        sizes.add((edge, min(edge, height)))
        edge += multiple
    # The model's own canvas is always offered. At 2:1 its 1440x704 is neither edge's
    # nearest step (720 is a tie between 704 and 736), so the walk alone misses it.
    sizes.add(h3_canvas(ratio_w / ratio_h))
    return sorted((w, h) for w, h in sizes
                  if PRESET_MIN_PIXELS <= w * h <= PRESET_MAX_PIXELS)


def preset_label(width: int, height: int) -> str:
    # Hundredths of a megapixel in integers: w*h is a multiple of 64, never a tie.
    hundredths = (width * height + 5_000) // 10_000
    return f"{width}x{height} ({hundredths // 100}.{hundredths % 100:02d} MP)"


def preset_labels(ratio: str, multiple: str) -> list[str]:
    w_ratio, h_ratio = ASPECT_RATIOS[ratio]
    return [preset_label(w, h) for w, h in preset_sizes(w_ratio, h_ratio, int(multiple))]


def _label_size(label: str) -> tuple[int, int]:
    width, height = label.split(" ")[0].split("x")
    return int(width), int(height)


def align_frames(n: int) -> int:
    """Next frame count on H3's sampling grid, at or above n."""
    n = max(FRAME_PHASE, int(n))
    return n + ((FRAME_PHASE - n) % FRAME_GRID)


def h3_canvas(ratio: float) -> tuple[int, int]:
    """The model's own canvas for an aspect ratio: 768 short edge, area capped."""
    short, long = float(BASE_SHORT_EDGE), BASE_SHORT_EDGE * max(ratio, 1.0 / ratio)
    if short * long > MAX_PIXELS:
        # The cap binds on wide ratios: shrink both edges, keep the ratio exact.
        shrink = math.sqrt(MAX_PIXELS / (short * long))
        short, long = short * shrink, long * shrink
    width, height = (long, short) if ratio >= 1.0 else (short, long)
    return _round_to(width), _round_to(height)


def megapixel_canvas(ratio: float, megapixels: float,
                     multiple: int = CANVAS_MULTIPLE) -> tuple[int, int]:
    """Same aspect, a chosen area. For pushing past the model's default canvas."""
    area = max(0.01, float(megapixels)) * 1_000_000.0
    return (_round_to(math.sqrt(area * ratio), multiple),
            _round_to(math.sqrt(area / ratio), multiple))


def preset_canvas(ratio: str, multiple: str, resolution: str) -> tuple[int, int]:
    """The chosen preset, landscape. Refuses a label that is not on this ratio's list."""
    labels = preset_labels(ratio, multiple)
    if resolution not in labels:
        raise ValueError(
            f"Luna-H3-Canvas: '{resolution}' is not a {ratio} size on the /{multiple} "
            f"grid. Pick one from the resolution list ({labels[0]} to {labels[-1]}).")
    return _label_size(resolution)


# The combo has to accept every label any ratio and grid can produce; the frontend
# narrows the list to the chosen ratio and grid.
ALL_PRESETS = sorted(
    {label for r in ASPECT_RATIOS for m in MULTIPLES for label in preset_labels(r, m)},
    key=lambda label: (_label_size(label)[0] * _label_size(label)[1], _label_size(label)[0]),
)
DEFAULT_PRESET = "1344x768 (1.03 MP)"


class LunaMiniMaxH3Canvas(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Luna-H3-Canvas",
            display_name="Luna-H3-Canvas",
            category="Luna-MinimaxH3",
            description=(
                "Canvas, frame count and frame rates for MiniMax H3, from an aspect ratio "
                "and a duration in seconds.\n\n"
                "`H3 canvas` gives the model's own resolution for the ratio — no megapixel "
                "figure to guess. Switch to `megapixels` to push past it (H3 goes to 2K, at "
                "roughly four times the attention cost for double the area), or to `preset` "
                "to pick a fixed size: every size for the ratio from 0.2 to 2 MP on the "
                "chosen grid (32 = H3's own, 16 also works, 8 is for images: H3 rounds a /8 "
                "edge down to /16 and `info` says so).\n\n"
                "Frame count snaps up to H3's sampling grid, and `info` reports the duration "
                "you actually got. Feed `output_fps` to the save node so interpolation and "
                "playback rate can never disagree. `seconds` is the real duration (frames / 24), "
                "for nodes that think in seconds, such as Luna-H3-AssetLoader.\n\n"
                "Wire a picture into `image` (from Load Image or Luna-H3-ReferenceLoader) and the shape of "
                "the frame follows that picture instead of `ratio` and `portrait`: the model's own "
                "canvas and the megapixels mode use its exact aspect, preset mode uses the listed "
                "ratio nearest to it, turned to the picture's orientation."
            ),
            inputs=[
                # Named `ratio`, not `aspect_ratio`: the widget row shows name and value
                # on one line, and the longer name pushed the value into an ellipsis.
                io.Combo.Input(
                    "ratio", options=list(ASPECT_RATIOS), default="16:9 (widescreen)",
                    tooltip="Shape of the frame, landscape. The canvas follows from it.",
                ),
                io.Boolean.Input(
                    "portrait", default=False,
                    tooltip="Turn the ratio on its side. 16:9 becomes 9:16 for Reels and "
                            "Shorts, 5:4 becomes 4:5 for Instagram, 3:2 becomes 2:3.",
                ),
                io.Combo.Input(
                    "size_mode", options=SIZE_MODES, default="H3 canvas",
                    tooltip="H3 canvas: the model's own resolution (1.03 MP at 16:9). "
                            "megapixels: your own target area, same aspect. "
                            "preset: a fixed size from the resolution list (0.2 to 2 MP).",
                ),
                # New 2026-10-25, placed under size_mode. Saves made before then are
                # positional; js/luna_h3_canvas.js names their old order in
                # fallbackWidgetsValuesNames so they restore by name.
                io.Combo.Input(
                    "multiple", options=MULTIPLES, default="32",
                    tooltip="Grid for the edges in megapixels and preset modes. 32: H3's own "
                            "grid. 16: also exact for H3 (it pads and crops back). 8: for "
                            "images; H3 video rounds a /8 edge down to /16, info says the "
                            "size it really makes.",
                ),
                io.Combo.Input(
                    "resolution", options=ALL_PRESETS, default=DEFAULT_PRESET,
                    tooltip="Only used when size_mode is preset. Every size for the ratio "
                            "on the chosen grid, 0.2 to 2 MP, smallest first. Listed "
                            "landscape; portrait turns it on its side.",
                ),
                io.Float.Input(
                    "megapixels", default=1.03, min=0.1, max=8.0, step=0.01,
                    tooltip="Only used when size_mode is megapixels.",
                ),
                io.Float.Input(
                    "duration_seconds", default=10.0, min=0.2, max=150.0, step=0.1,
                    tooltip="Snapped up to H3's frame grid; info reports what you got.",
                ),
                io.Boolean.Input(
                    "interpolation_enabled", default=True,
                    tooltip="Wire this to the same boolean that gates your frame "
                            "interpolation nodes. When false the factor is forced to 1 "
                            "and output_fps drops back to 24, so bypassing the "
                            "interpolation cannot leave the frame rate doubled.",
                ),
                io.Int.Input(
                    "interpolation_factor", default=2, min=1, max=8,
                    tooltip="Frames per source frame. Drives both the interpolation "
                            "node and output_fps, so they cannot drift. Keeps its value "
                            "while interpolation_enabled is false.",
                ),
                # Added LAST: saved workflows restore widget values by position, and a
                # socket-only input adds none.
                io.Image.Input(
                    "image", optional=True,
                    tooltip="Optional. Wired (from Load Image or Luna-H3-ReferenceLoader), the shape "
                            "of the frame follows this picture instead of ratio and portrait: H3 "
                            "canvas and megapixels use its exact aspect; preset uses the listed "
                            "ratio nearest to it, turned to the picture's orientation. Only its "
                            "first frame counts. Not from Luna-H3-AssetLoader while the loader takes "
                            "this canvas's width, height or seconds: that is a circle ComfyUI refuses.",
                ),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                # Was "length" until 2026-10-29. Renamed, same slot. V3 gives an output
                # without an id the id "_{slot}_{type}_" (here "_2_INT_"), which does not
                # use display_name, so saved links (by slot index) and the graph file are
                # unaffected; only the label changes.
                io.Int.Output(display_name="frames",
                              tooltip="The frame count on H3's 17k+5 grid, at or above duration_seconds x 24. "
                                      "For the H3 node's length and Luna-LLMPrompt's frames."),
                io.Float.Output(display_name="fps"),
                io.Float.Output(display_name="output_fps"),
                io.Int.Output(display_name="interpolation_factor"),
                io.String.Output(display_name="info"),
                # Added LAST: saved workflows link outputs by slot number, so an
                # appended output moves nothing.
                io.Float.Output(display_name="seconds",
                                tooltip="The real duration, frames / 24: what the frame grid "
                                        "gave, not what was asked (5 s -> 124 frames -> 5.1667 s). "
                                        "Wire it to Luna-H3-AssetLoader's seconds so the song window "
                                        "matches the shot."),
            ],
        )

    @classmethod
    def execute(cls, ratio, portrait, size_mode, megapixels,
                duration_seconds, interpolation_enabled, interpolation_factor,
                multiple="32", resolution=DEFAULT_PRESET, image=None):
        w_ratio, h_ratio = ASPECT_RATIOS[ratio]
        if portrait:
            w_ratio, h_ratio = h_ratio, w_ratio
        aspect = w_ratio / h_ratio
        ratio_text = f"{w_ratio}:{h_ratio}"

        # A wired picture decides the shape: its exact aspect, first frame only. Preset
        # sizes exist only for the listed ratios, so that mode takes the nearest listed one
        # (landscape form, as ASPECT_RATIOS stores them) and turns it to the picture's side.
        if image is not None:
            img_h, img_w = int(image.shape[1]), int(image.shape[2])
            aspect = img_w / img_h
            portrait = aspect < 1.0
            ratio_text = f"ratio from image {img_w}x{img_h}"
            if size_mode == "preset":
                landscape = max(aspect, 1.0 / aspect)
                ratio = min(ASPECT_RATIOS, key=lambda k: abs(ASPECT_RATIOS[k][0] / ASPECT_RATIOS[k][1] - landscape))
                labels = preset_labels(ratio, multiple)
                if resolution not in labels:
                    # The chosen size belongs to the widget's ratio; take this ratio's size
                    # of the nearest area instead of refusing.
                    cw, ch = _label_size(resolution)
                    resolution = min(labels, key=lambda t: abs(_label_size(t)[0] * _label_size(t)[1] - cw * ch))

        if size_mode == "megapixels":
            width, height = megapixel_canvas(aspect, megapixels, int(multiple))
        elif size_mode == "preset":
            width, height = preset_canvas(ratio, multiple, resolution)
            if portrait:
                width, height = height, width
        else:
            width, height = h3_canvas(aspect)

        length = align_frames(round(duration_seconds * FPS))
        # Clamped once, then both returned: the factor that leaves this node is the
        # same one output_fps was computed from, so driving the interpolation node
        # from it cannot disagree with the frame rate. Gated off, the factor is 1
        # rather than the widget's value -- bypassing the interpolation nodes while
        # the rate stayed doubled is the exact failure this node exists to prevent.
        factor = max(1, int(interpolation_factor)) if interpolation_enabled else 1
        output_fps = FPS * factor
        actual_seconds = length / FPS

        # ASCII only: this string is printed, and a Windows console on cp1252
        # raises UnicodeEncodeError on the likes of a middle dot or an arrow.
        info = (f"{width}x{height} ({ratio_text}) | {width * height / 1e6:.2f} MP | "
                f"{length} frames | {actual_seconds:.2f}s @ {FPS:g} fps")
        if factor > 1:
            info += f" -> x{factor} -> {output_fps:g} fps out"
        elif not interpolation_enabled and int(interpolation_factor) > 1:
            info += f" | interpolation off (x{int(interpolation_factor)} held)"

        notes = []
        # The grid steps in 17-frame jumps, so a fraction of a second of snapping is
        # unavoidable and not worth flagging -- the line above already states the real
        # duration. Only say something when it lands more than half a step away.
        if abs(actual_seconds - duration_seconds) >= (FRAME_GRID / FPS) / 2:
            notes.append(f"asked {duration_seconds:g}s, grid gives {actual_seconds:.2f}s")
        if not TRAINED_MIN <= length <= TRAINED_MAX:
            notes.append(f"outside the trained {TRAINED_MIN}-{TRAINED_MAX} frame range")
        if width * height > MAX_PIXELS:
            notes.append(f"{width * height / MAX_PIXELS:.2f}x the model's default canvas")
        if width % LATENT_MULTIPLE or height % LATENT_MULTIPLE:
            notes.append(f"H3 video makes {width // LATENT_MULTIPLE * LATENT_MULTIPLE}x"
                         f"{height // LATENT_MULTIPLE * LATENT_MULTIPLE} (latent is /16)")
        if notes:
            info += "  (" + "; ".join(notes) + ")"

        print(f"[Luna-H3-Canvas] {info}")
        return io.NodeOutput(width, height, length, FPS, output_fps, factor, info, actual_seconds)

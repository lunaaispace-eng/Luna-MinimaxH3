"""Measure how much real fine detail a video holds at its own size (is a 2 MP render really 2 MP?).

For frames at fixed times (default 2, 4, 6, 8 s) it prints, averaged:
- `rt_db`: PSNR of the frame after a round trip through half its size (bicubic down, bicubic up). A frame that only
  holds half-size detail survives almost unchanged (high dB); real full-size detail is lost in the trip (lower dB).
  Lower = more real detail at the video's own size.
- `band`: share (%) of the luma spectrum's energy in the top 30 % of the frame's own frequency range (DC excluded).
  Higher = more fine detail.

Usage: python measure_detail.py <video> [<video> ...] [--times 2,4,6,8]
Needs ffmpeg/ffprobe on PATH; numpy and Pillow (ComfyUI's embedded Python has both).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def frame(video, t):
    w, h = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of",
         "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout.strip().split(",")[:2]
    w, h = int(w), int(h)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(video), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt",
         "rgb24", "-"], capture_output=True, check=True).stdout
    if len(raw) < w * h * 3:
        return None
    return np.frombuffer(raw[: w * h * 3], np.uint8).reshape(h, w, 3)


def luma(rgb):
    return rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def round_trip_db(rgb):
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb)
    back = np.asarray(img.resize((w // 2, h // 2), Image.BICUBIC).resize((w, h), Image.BICUBIC))
    mse = np.mean((luma(rgb) - luma(back)) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))


def band_share(rgb):
    y = luma(rgb)
    y = y - y.mean()
    p = np.abs(np.fft.fftshift(np.fft.fft2(y))) ** 2
    h, w = y.shape
    fy = (np.arange(h) - h // 2) / (h / 2)
    fx = (np.arange(w) - w // 2) / (w / 2)
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / np.sqrt(2)
    return 100.0 * p[r >= 0.7].sum() / p.sum()


def measure(video, times):
    rts, bands, size = [], [], None
    for t in times:
        f = frame(video, t)
        if f is None:
            continue
        size = f"{f.shape[1]}x{f.shape[0]}"
        rts.append(round_trip_db(f))
        bands.append(band_share(f))
    if not rts:
        return f"{Path(video).name}: no frames"
    return f"{size}\trt_db {np.mean(rts):.1f}\tband {np.mean(bands):.4f}\tn {len(rts)}\t{Path(video).name}"


if __name__ == "__main__":
    args = sys.argv[1:]
    times = [2, 4, 6, 8]
    if "--times" in args:
        i = args.index("--times")
        times = [float(x) for x in args[i + 1].split(",")]
        del args[i: i + 2]
    for v in args:
        print(measure(v, times), flush=True)

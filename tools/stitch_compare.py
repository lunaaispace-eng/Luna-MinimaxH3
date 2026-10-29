"""Stitch videos one after another, each with its caption burned in at the top, sound kept.

Made 2026-10-29 for Peter: "Can you make a stich, original kijai 2mp, time ram vram, ours with 2mp to 2 k and the
last 1.5 to 2k, mention our sampler".

Usage: python stitch_compare.py [--crf 18] [--preset medium] [--audio 192k] <out.mp4> <video1> <caption1> [...]
Best quality for sharing (Peter, 2026-10-29: "make best quality"): --crf 14 --preset slow --audio 256k.
A caption may hold several lines separated by "|". All videos must have the same size and an audio stream.
Needs ffmpeg on PATH; uses Arial from C:/Windows/Fonts.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

FONT = "C\\:/Windows/Fonts/arial.ttf"


def stitch(out, pairs, crf=18, preset="medium", audio_bitrate="192k"):
    with tempfile.TemporaryDirectory(dir=Path(out).parent) as tmp:
        inputs, filters, concat = [], [], ""
        for i, (video, caption) in enumerate(pairs):
            text = Path(tmp) / f"caption{i}.txt"
            # newline="\n": a Windows \r\n is drawn by drawtext as two line breaks.
            text.write_text("\n".join(line.strip() for line in caption.split("|")), encoding="utf-8", newline="\n")
            inputs += ["-i", str(video)]
            filters.append(
                f"[{i}:v]drawtext=fontfile='{FONT}':textfile='caption{i}.txt':fontsize=36:fontcolor=white:"
                f"line_spacing=10:box=1:boxcolor=black@0.6:boxborderw=24:x=40:y=40,setsar=1[v{i}]")
            concat += f"[v{i}][{i}:a]"
        graph = ";".join(filters) + f";{concat}concat=n={len(pairs)}:v=1:a=1[v][a]"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-b:a", audio_bitrate, "-movflags", "+faststart", str(Path(out).resolve())],
            check=True, cwd=tmp)


if __name__ == "__main__":
    args = sys.argv[1:]
    options = {}
    while args and args[0].startswith("--"):  # --crf 14 --preset slow --audio 256k
        key, value = args[0][2:], args[1]
        options[{"audio": "audio_bitrate"}.get(key, key)] = int(value) if key == "crf" else value
        args = args[2:]
    if len(args) < 3 or len(args) % 2 == 0:
        sys.exit(__doc__)
    stitch(args[0], list(zip(args[1::2], args[2::2])), **options)
    print(args[0])

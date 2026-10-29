"""How bright or muffled a video's sound is (made 2026-10-29 after Peter: "In the last i e the sound was a bit muffled").

Per file, from the decoded audio (mono mix, 48 kHz): RMS level (dBFS), spectral centroid (Hz, lower = darker), the
share of energy above 4 kHz and above 8 kHz (%), and the -20 dB roll-off frequency of the averaged spectrum (Hz,
lower = more muffled).

Usage: python measure_audio.py <video> [<video> ...]
Needs ffmpeg on PATH and numpy.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

RATE = 48000


def audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(RATE), "-f", "f32le",
                          "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def measure(x):
    n = 4096
    frames = [x[i:i + n] * np.hanning(n) for i in range(0, len(x) - n, n // 2)]
    power = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], axis=0)
    freqs = np.fft.rfftfreq(n, 1 / RATE)
    total = power[freqs > 30].sum()
    centroid = float((freqs * power).sum() / power.sum())
    above = lambda hz: 100 * float(power[freqs >= hz].sum() / total)
    db = 10 * np.log10(power / power.max() + 1e-20)
    rolloff = float(freqs[np.nonzero(db > -20)[0].max()])
    rms = 20 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-12)
    return rms, centroid, above(4000), above(8000), rolloff


if __name__ == "__main__":
    print("rms_dbfs\tcentroid_hz\t>4k_%\t>8k_%\trolloff20_hz\tfile")
    for path in sys.argv[1:]:
        rms, centroid, a4, a8, roll = measure(audio(path))
        print(f"{rms:.1f}\t{centroid:.0f}\t{a4:.2f}\t{a8:.3f}\t{roll:.0f}\t{Path(path).name}")

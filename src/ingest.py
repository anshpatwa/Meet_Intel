"""Step 1 - Ingestion.

Takes any meeting recording (mp3, mp4, m4a, mkv, wav, ...) and normalizes it
to a 16 kHz mono 16-bit PCM WAV - the format Whisper-family models expect.

Uses FFmpeg directly (robust with video containers and odd codecs).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Make ``import config`` work whether run as a module or a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise RuntimeError(
            f"'{tool}' was not found on PATH. Install FFmpeg "
            "(https://ffmpeg.org/download.html) and reopen your terminal."
        )


def probe_duration(path: str | Path) -> float | None:
    """Return media duration in seconds, or None if it can't be read."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return None


def processed_stem(name: str) -> str:
    """Capped output folder/file stem shared by all steps (Windows MAX_PATH guard)."""
    return name[:60].rstrip(" ._-")


def ingest(
    input_path: str | Path,
    out_dir: str | Path | None = None,
    sample_rate: int = config.SAMPLE_RATE,
) -> Path:
    """Convert ``input_path`` to a normalized WAV and return its path.

    Output lands in ``data/processed/<name>/<name>.wav`` by default.
    """
    _require("ffmpeg")
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Windows MAX_PATH guard (see processed_stem): the title is used for both the
    # folder and the files inside it, which can otherwise blow past 260 chars.
    stem = processed_stem(input_path.stem)
    out_dir = Path(out_dir) if out_dir else config.PROCESSED_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{stem}.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",                              # drop any video stream
        "-ac", str(config.AUDIO_CHANNELS),  # mono
        "-ar", str(sample_rate),            # 16 kHz
        "-c:a", "pcm_s16le",                # 16-bit PCM
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed to convert the file:\n{e.stderr}") from e

    return wav_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Normalize a recording to 16 kHz mono WAV.")
    ap.add_argument("input", help="Path to the meeting audio/video file.")
    args = ap.parse_args()

    wav = ingest(args.input)
    dur = probe_duration(wav)
    mins = f"{dur / 60:.1f} min" if dur else "unknown length"
    print(f"[ingest] wrote {wav}  ({mins})")

"""Optional Step 0 - fetch a recording from a URL (YouTube, etc.).

Uses yt-dlp to download the best available audio track. The downloaded file is
then handed to Step 1 (ingest), which normalizes it to 16 kHz mono WAV - so any
site yt-dlp supports works without extra code.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def is_url(text: str) -> bool:
    return str(text).lower().startswith(("http://", "https://"))


def download_audio(url: str, out_dir: str | Path = config.RAW_DIR) -> Path:
    """Download best-audio from ``url`` into ``out_dir`` and return the file path."""
    import yt_dlp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(title)s-%(id)s.%(ext)s"),
        "restrictfilenames": True,  # filesystem-safe names on Windows
        "noplaylist": True,          # one video, even if the URL carries a playlist
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))

    if not path.exists():  # extension can differ from the template guess
        matches = list(out_dir.glob(f"*{info.get('id', '')}*"))
        if not matches:
            raise RuntimeError(f"yt-dlp succeeded but no file was found for {url}")
        path = matches[0]

    dur = info.get("duration")
    mins = f"{dur / 60:.1f} min" if dur else "unknown length"
    print(f"[download] {info.get('title', '(untitled)')}  ({mins}) -> {path.name}")
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download audio from a URL via yt-dlp.")
    ap.add_argument("url")
    args = ap.parse_args()
    download_audio(args.url)

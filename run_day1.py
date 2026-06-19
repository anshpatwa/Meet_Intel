"""Day 1 end-to-end: Ingestion -> Captions.

    python run_day1.py "data/raw/meeting.mp4"
    python run_day1.py "C:/path/to/call.m4a" --model medium --language en

Outputs (SRT / VTT / TXT / JSON) land in data/processed/<name>/.

This stops at captions. The natural (LLM) Hinglish romanization is a SEPARATE
step - run ``run_hinglish.py`` afterward. Captioning and Hinglish are kept as two
independent scripts on purpose, so a problem in one never affects the other.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import config
from src.ingest import ingest, probe_duration
from src.transcribe import transcribe
from src.youtube import download_audio, is_url


def run(
    source: str,
    *,
    model: str = config.WHISPER_MODEL,
    language: str | None = config.LANGUAGE,
    device: str = config.DEVICE,
    compute_type: str = config.COMPUTE_TYPE,
    batch_size: int = config.BATCH_SIZE,
    align: bool = config.ALIGN,
) -> dict:
    """Run Day 1 (download -> ingest -> captions) on a URL or local file path.

    This is the single entry point shared by the command line (main) and the
    IDE helper (run_ide.py). It ends at captions; for natural LLM Hinglish run
    the separate run_hinglish.py on the transcript this produces.
    """
    t0 = time.perf_counter()

    if is_url(source):
        print("== Step 0: Download (yt-dlp) ==")
        source = download_audio(source, config.RAW_DIR)
        print()

    print("== Step 1: Ingestion ==")
    wav = ingest(source)
    dur = probe_duration(wav)
    print(f"[ingest] {wav}  ({(dur or 0) / 60:.1f} min audio)\n")

    print("== Step 2: Captions ==")
    result = transcribe(
        wav,
        model_name=model,
        language=language,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        align=align,
    )

    elapsed = time.perf_counter() - t0
    out_dir = Path(result["paths"]["srt"]).parent
    print(f"\nDone in {elapsed:.0f}s. Captions for {len(result['segments'])} "
          f"segments in:\n  {out_dir}")
    print("[next] For natural Hinglish, run:  python run_hinglish.py")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Day 1 - Ingestion + Captions")
    ap.add_argument("input", help="Meeting file (mp3, mp4, m4a, wav, ...) OR a video URL (YouTube, etc.).")
    ap.add_argument("--model", default=config.WHISPER_MODEL, help="Whisper model size.")
    ap.add_argument("--language", default=config.LANGUAGE, help="Force a language code, e.g. en.")
    ap.add_argument("--device", default=config.DEVICE, help="auto | cuda | cpu")
    ap.add_argument("--compute-type", default=config.COMPUTE_TYPE)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--no-align", action="store_true", help="Skip word-level alignment.")
    args = ap.parse_args()

    run(
        args.input,
        model=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        align=not args.no_align,
    )


if __name__ == "__main__":
    main()

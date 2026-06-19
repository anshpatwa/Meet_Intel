"""Natural (LLM-quality) Hinglish from a Hindi (Devanagari) transcript.

This is a SEPARATE step from captioning. First run run_ide.py (or run_day1.py)
to produce the Devanagari transcript, then run this to romanize it with the LLM.

    python run_hinglish.py                  # use the most recent Hindi transcript
    python run_hinglish.py "data/processed/<name>/<name>.txt"

Writes the preferred <name>.hinglish.txt next to the transcript - the high-quality
romanization (English words restored, correct schwa). The offline rule-based
<name>.hinglish.rule.txt stays as the fallback.

Needs GEMINI_API_KEY in .env.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import config
from src.analyze import to_natural_hinglish

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _latest_devanagari_txt() -> Path | None:
    cands = [p for p in config.PROCESSED_DIR.glob("*/*.txt")
             if ".hinglish" not in p.name and ".report" not in p.name]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:  # prefer the most recent transcript that actually contains Hindi
        if _DEVANAGARI.search(p.read_text(encoding="utf-8")):
            return p
    return cands[0] if cands else None


def run(transcript: str | None = None, model: str | None = None) -> Path:
    path = Path(transcript) if transcript else _latest_devanagari_txt()
    if not path or not path.exists():
        raise SystemExit("No transcript found. Run Day 1 on a Hindi video first, or pass a path.")

    text = path.read_text(encoding="utf-8")
    if not _DEVANAGARI.search(text):
        print(f"[hinglish] note: {path.name} has no Devanagari - is this really a Hindi transcript?")

    print(f"== Natural Hinglish (LLM: {model or config.LLM_MODEL}) ==")
    print(f"[hinglish] {path.name}  ({len(text.split())} words)")

    out = to_natural_hinglish(text, model=model)
    dest = path.with_name(path.stem + ".hinglish.txt")      # the preferred file
    dest.write_text(out + "\n", encoding="utf-8")
    print(f"[hinglish] wrote {dest}  (LLM - preferred)")
    return dest


def main():
    ap = argparse.ArgumentParser(description="LLM-quality Hinglish romanization")
    ap.add_argument("transcript", nargs="?",
                    help="Devanagari .txt (default: most recent Hindi transcript).")
    ap.add_argument("--model", default=None, help="Override the LLM model (default from .env).")
    args = ap.parse_args()
    run(args.transcript, model=args.model)


if __name__ == "__main__":
    main()

"""Day 3 - LLM analysis of a transcript.

By default it analyzes the transcript for the SAME video you set in run_ide.py
(its SOURCE), so Day 1 and Day 3 stay in sync. You can also pass a path
explicitly, and it falls back to the most recent transcript only if nothing
else can be determined.

    python run_day3.py                     # analyze the video set in run_ide.py
    python run_day3.py "data/processed/<name>/<name>.txt"
    python run_day3.py --model gemini-2.5-flash

Writes <name>.analysis.json and <name>.report.md next to the transcript.
Needs GEMINI_API_KEY in .env (free: https://aistudio.google.com/apikey).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import config
from src.analyze import analyze_text
from src.ingest import processed_stem
from src.report import to_markdown
from src.youtube import is_url

_YT_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def _transcript_path(stem: str) -> Path:
    return config.PROCESSED_DIR / stem / f"{stem}.txt"


def _transcript_for_source(source: str) -> Path | None:
    """Find the Day 1 transcript that corresponds to a URL or local file path."""
    if is_url(source):
        m = _YT_ID.search(source)
        if not m:
            return None
        vid = m.group(1)
        # The download in data/raw keeps the full name (incl. the video id);
        # derive the same capped folder name Day 1 used from it.
        for raw in config.RAW_DIR.glob(f"*{vid}*"):
            p = _transcript_path(processed_stem(raw.stem))
            if p.exists():
                return p
        return None
    p = _transcript_path(processed_stem(Path(source).stem))
    return p if p.exists() else None


def _latest_transcript() -> Path | None:
    cands = [p for p in config.PROCESSED_DIR.glob("*/*.txt")
             if ".hinglish" not in p.name and ".report" not in p.name]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _resolve(transcript: str | None) -> Path:
    if transcript:                                    # explicit path always wins
        return Path(transcript)
    try:                                              # otherwise follow run_ide.py
        from run_ide import SOURCE
        found = _transcript_for_source(SOURCE)
        if found:
            return found
        raise SystemExit(
            f"\nNo transcript yet for the link set in run_ide.py:\n  {SOURCE}\n"
            "Run  python run_ide.py  first to transcribe it (Day 1), then re-run this.\n"
        )
    except SystemExit:
        raise
    except Exception:
        pass
    latest = _latest_transcript()                     # last resort
    if latest:
        print(f"[analyze] couldn't read run_ide.py SOURCE; using most recent: {latest.name}")
        return latest
    raise SystemExit("No transcript found. Run  python run_ide.py  first.")


def run(transcript: str | None = None, model: str | None = None):
    path = _resolve(transcript)
    if not path.exists():
        raise SystemExit(f"Transcript not found: {path}")

    text = path.read_text(encoding="utf-8")
    print(f"== Step 4: LLM analysis ({model or config.LLM_MODEL}) ==")
    print(f"[analyze] {path.name}  ({len(text.split())} words)")

    analysis = analyze_text(text, model=model)

    out_dir, stem = path.parent, path.stem
    js = out_dir / f"{stem}.analysis.json"
    md = out_dir / f"{stem}.report.md"
    js.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    md.write_text(to_markdown(analysis), encoding="utf-8")

    resolved = sum(1 for i in analysis.issues if i.resolved)
    print(f"[analyze] {len(analysis.topics)} topics | {len(analysis.issues)} issues "
          f"({resolved} resolved) | {len(analysis.action_items)} action items")
    print(f"[analyze] wrote {js}")
    print(f"[analyze] wrote {md}")
    return analysis


def main():
    ap = argparse.ArgumentParser(description="Day 3 - LLM meeting analysis")
    ap.add_argument("transcript", nargs="?",
                    help="Path to a .txt transcript (default: the video set in run_ide.py).")
    ap.add_argument("--model", default=None, help="Override the LLM model (default from .env).")
    args = ap.parse_args()
    run(args.transcript, model=args.model)


if __name__ == "__main__":
    main()

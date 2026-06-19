"""Transcribe ONE audio file on the single GPU visible to this process.

Used for multi-GPU parallel transcription: a launcher pins each worker to one GPU
via CUDA_VISIBLE_DEVICES (so each process sees exactly one GPU as cuda:0), then runs
this on one slice of the audio. Caption files land next to the input wav, so the
launcher can merge the per-slice transcripts afterward.

    # GPU 0 on the first half, GPU 1 on the second half, in parallel:
    CUDA_VISIBLE_DEVICES=0 python gpu_worker.py part1.wav hi &
    CUDA_VISIBLE_DEVICES=1 python gpu_worker.py part2.wav hi &
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `src` importable

from src.transcribe import transcribe  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python gpu_worker.py <audio.wav> [language]")
    wav = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else "hi"
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
    print(f"[worker pid={os.getpid()} gpu={gpu}] transcribing {wav}", flush=True)
    transcribe(wav, language=language)
    print(f"[worker gpu={gpu}] done {wav}", flush=True)


if __name__ == "__main__":
    main()

"""Step 2 - Captions (ASR).

Produces highly accurate captions from a normalized WAV using WhisperX:
  * faster-whisper backend (Whisper large-v3) for transcription,
  * built-in VAD to suppress silence hallucinations,
  * wav2vec2 forced alignment for word-level timestamps.

Outputs SRT, VTT, plain text, and a rich JSON (with per-word timings) that
later stages - diarization and LLM analysis - build on.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


# --------------------------------------------------------------------------
# Device / precision resolution
# --------------------------------------------------------------------------
def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_compute_type(compute_type: str, device: str) -> str:
    if compute_type != "auto":
        return compute_type
    # int8_float16 keeps large-v3 inside ~4 GB of VRAM; int8 for CPU.
    return "int8_float16" if device == "cuda" else "int8"


def _free_gpu(model) -> None:
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------
# Subtitle formatting
# --------------------------------------------------------------------------
def _fmt_ts(seconds: float | None, vtt: bool = False) -> str:
    if seconds is None:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_srt(segments: list[dict], path: Path) -> None:
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines += [
            str(i),
            f"{_fmt_ts(seg.get('start'))} --> {_fmt_ts(seg.get('end'))}",
            text,
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[dict], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines += [
            f"{_fmt_ts(seg.get('start'), vtt=True)} --> {_fmt_ts(seg.get('end'), vtt=True)}",
            text,
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(segments: list[dict], path: Path) -> None:
    text = "\n".join((s.get("text") or "").strip() for s in segments if (s.get("text") or "").strip())
    path.write_text(text + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def transcribe(
    wav_path: str | Path,
    out_dir: str | Path | None = None,
    model_name: str = config.WHISPER_MODEL,
    language: str | None = config.LANGUAGE,
    device: str = config.DEVICE,
    compute_type: str = config.COMPUTE_TYPE,
    batch_size: int = config.BATCH_SIZE,
    align: bool = config.ALIGN,
) -> dict:
    """Transcribe ``wav_path`` and write caption files. Returns a result dict."""
    import whisperx

    wav_path = Path(wav_path)
    out_dir = Path(out_dir) if out_dir else wav_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(device)
    compute_type = _resolve_compute_type(compute_type, device)
    print(f"[captions] model={model_name} device={device} compute={compute_type}")

    # Reduce skipped speech: more sensitive VAD + keep borderline segments.
    vad_opts = {"vad_onset": config.VAD_ONSET, "vad_offset": config.VAD_OFFSET}
    asr_opts = {"no_speech_threshold": config.NO_SPEECH_THRESHOLD}

    audio = whisperx.load_audio(str(wav_path))

    # --- transcription -----------------------------------------------------
    # On a small GPU a large model can OOM at high batch sizes. Instead of
    # bailing to (slow) CPU right away, step the batch size down on the GPU
    # first; only fall back to CPU if even batch_size=1 won't fit.
    model = None
    result = None
    if device == "cuda":
        try:
            model = whisperx.load_model(model_name, "cuda", compute_type=compute_type,
                                        language=language, vad_options=vad_opts,
                                        asr_options=asr_opts)
            bs = batch_size
            while True:
                try:
                    result = model.transcribe(audio, batch_size=bs, print_progress=True)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and bs > 1:
                        bs = max(1, bs // 2)
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                        print(f"[captions] CUDA OOM -> retrying on GPU with batch_size={bs}")
                        continue
                    raise
            if bs != batch_size:
                print(f"[captions] settled on GPU batch_size={bs}")
        except RuntimeError as e:
            print(f"[captions] GPU unusable ({e}); falling back to CPU...")
            if model is not None:
                _free_gpu(model)
                model = None
            device, compute_type = "cpu", "int8"

    if result is None:  # CPU path (GPU fell back, or device was CPU to begin with)
        device, compute_type = "cpu", "int8"
        model = whisperx.load_model(model_name, "cpu", compute_type=compute_type,
                                    language=language, vad_options=vad_opts,
                                    asr_options=asr_opts)
        result = model.transcribe(audio, batch_size=max(1, batch_size // 2), print_progress=True)

    detected = result.get("language", language or "en")
    _free_gpu(model)  # release ASR model before loading the alignment model

    # --- word-level alignment (sharper timestamps) -------------------------
    if align:
        try:
            model_a, metadata = whisperx.load_align_model(language_code=detected, device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                                    return_char_alignments=False)
            _free_gpu(model_a)
        except Exception as e:
            print(f"[captions] alignment skipped ({e}); using segment-level timestamps.")

    segments = result.get("segments", [])

    # --- write outputs -----------------------------------------------------
    stem = wav_path.stem
    paths = {
        "srt": out_dir / f"{stem}.srt",
        "vtt": out_dir / f"{stem}.vtt",
        "txt": out_dir / f"{stem}.txt",
        "json": out_dir / f"{stem}.captions.json",
    }
    write_srt(segments, paths["srt"])
    write_vtt(segments, paths["vtt"])
    write_txt(segments, paths["txt"])
    paths["json"].write_text(
        json.dumps({"language": detected, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Offline, rule-based romanized "Hinglish" when the transcript is Devanagari.
    # This always-available file is the fallback; run_day1 adds the preferred
    # natural LLM romanization on top as <stem>.hinglish.txt.
    try:
        from src.romanize import can_romanize, to_hinglish
        if can_romanize(detected):
            body = "\n".join((s.get("text") or "").strip()
                             for s in segments if (s.get("text") or "").strip())
            paths["hinglish_rule"] = out_dir / f"{stem}.hinglish.rule.txt"
            paths["hinglish_rule"].write_text(to_hinglish(body) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"[captions] romanization skipped ({e})")

    print(f"[captions] language={detected}  segments={len(segments)}")
    for p in paths.values():
        print(f"[captions] wrote {p}")

    return {"language": detected, "segments": segments, "paths": paths}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate captions from a normalized WAV.")
    ap.add_argument("wav", help="Path to a 16 kHz mono WAV (output of ingest.py).")
    ap.add_argument("--model", default=config.WHISPER_MODEL)
    ap.add_argument("--language", default=config.LANGUAGE)
    ap.add_argument("--no-align", action="store_true", help="Skip word-level alignment.")
    args = ap.parse_args()

    transcribe(args.wav, model_name=args.model, language=args.language, align=not args.no_align)

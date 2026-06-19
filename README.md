# Meeting Intelligence

Turn a meeting recording into an accurate, speaker-aware transcript and a
structured brief (summary, topics, issues, resolved issues).

This repo is built in four daily milestones:

| Day | Milestone | Status |
|-----|-----------|--------|
| **1** | **Ingestion + Captions** | ✅ this commit |
| 2 | Speaker diarization (who spoke when) | ⬜ next |
| 3 | LLM analysis (summary / topics / issues / resolved) | ⬜ |
| 4 | UI + report export | ⬜ |

## Day 1 — what's here

```
Meeting Documentation/
├── run_day1.py          # one command: ingest -> captions
├── config.py            # all settings (env-overridable)
├── src/
│   ├── ingest.py        # Step 1: any recording -> 16 kHz mono WAV (FFmpeg)
│   └── transcribe.py    # Step 2: WhisperX captions -> SRT/VTT/TXT/JSON
├── data/raw/            # drop input meetings here
└── data/processed/      # outputs land here
```

**Step 1 – Ingestion** normalizes any audio/video to the 16 kHz mono WAV that
Whisper expects. **Step 2 – Captions** runs WhisperX (`large-v3`) with VAD to
suppress silence hallucinations and wav2vec2 forced alignment for word-level
timestamps — giving subtitles that are tight enough to feed Day 2's diarization.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 1) PyTorch with CUDA (RTX 3050 -> cu121):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2) The rest:
pip install -r requirements.txt
```

FFmpeg must be on PATH (`ffmpeg -version` to check).

## Run

```powershell
# Put a recording in data/raw, then:
python run_day1.py "data/raw/meeting.mp4"

# Options:
python run_day1.py "data/raw/call.m4a" --model medium --language en
python run_day1.py "data/raw/call.m4a" --no-align     # faster, looser timestamps
```

Outputs in `data/processed/<name>/`:

| File | Use |
|------|-----|
| `<name>.srt` / `.vtt` | subtitles for players / video |
| `<name>.txt` | plain transcript |
| `<name>.hinglish.txt` | **(Hindi) preferred** — natural LLM Hinglish, produced in the same run |
| `<name>.hinglish.rule.txt` | (Hindi) offline rule-based Hinglish — fallback |
| `<name>.captions.json` | segments **with per-word timings** → input for Day 2 |

## Notes for a 4 GB GPU (RTX 3050 Laptop)

`large-v3` fits with the default `COMPUTE_TYPE=int8_float16`. If you ever see a
CUDA out-of-memory error, the pipeline automatically retries on CPU; you can
also lower `BATCH_SIZE` or pass `--model medium` for a lighter run.

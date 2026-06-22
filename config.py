"""Central configuration for the Meeting Intelligence pipeline.

All tunables are read from the environment (optionally a local ``.env`` file)
so nothing sensitive or machine-specific is hard-coded. See ``.env.example``.
"""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

# --- Quiet known-harmless third-party noise (keeps the run console readable) ---
# WhisperX hands audio to pyannote in memory, so torchcodec is never used; its
# scary "libtorchcodec not found" warning is irrelevant here. The TF32 and
# Lightning checkpoint-upgrade notes are informational only.
warnings.filterwarnings("ignore", module=r"pyannote.*")
warnings.filterwarnings("ignore", module=r"torchaudio.*")
for _name in ("pytorch_lightning.utilities.migration",
              "lightning.pytorch.utilities.migration",
              "lightning_fabric.utilities.migration"):
    logging.getLogger(_name).setLevel(logging.ERROR)

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional; env vars still work without it
    def load_dotenv(*_a, **_k):  # type: ignore
        return False

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- Project paths --------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # drop input meetings here
PROCESSED_DIR = DATA_DIR / "processed"  # outputs (wav, captions, json)

for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Step 1: audio normalization -----------------------------------------
# Whisper-family models expect 16 kHz mono audio.
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
AUDIO_CHANNELS = 1

# --- Step 2: captions / ASR ----------------------------------------------
# Model size trades accuracy for VRAM. On a 4 GB GPU (e.g. RTX 3050 Laptop)
# "large-v3" fits comfortably with COMPUTE_TYPE=int8_float16.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")

# None => auto-detect spoken language. Set e.g. "en" to skip detection.
LANGUAGE = os.getenv("LANGUAGE") or None

# auto => use CUDA when available, else CPU.
DEVICE = os.getenv("DEVICE", "auto")

# auto => int8_float16 on GPU (fits 4 GB), int8 on CPU.
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "auto")

# Lower this if you hit out-of-memory errors.
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

# Forced alignment gives word-level timestamps => sharper subtitles and the
# foundation for Day 2 (speaker diarization). Disable with ALIGN=false.
ALIGN = os.getenv("ALIGN", "true").lower() in ("1", "true", "yes", "on")

# VAD sensitivity: lower = catches more speech / fewer skipped lines (too low can
# hallucinate over music or silence). whisperx defaults are 0.500 / 0.363.
VAD_ONSET = float(os.getenv("VAD_ONSET", "0.35"))
VAD_OFFSET = float(os.getenv("VAD_OFFSET", "0.25"))
# How readily Whisper discards a segment as "no speech" (higher = keeps more).
NO_SPEECH_THRESHOLD = float(os.getenv("NO_SPEECH_THRESHOLD", "0.8"))

# --- Step 4: LLM analysis (Day 3) ----------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")     # currently supported: gemini
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")  # higher free-tier quota than 2.5-flash
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")       # free key: https://aistudio.google.com/apikey

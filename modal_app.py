"""Public web app: WhisperX (GPU) + Gemini, served on Modal.

The GPU function reuses the exact src/ pipeline (large-v3 captions -> Gemini
Hinglish -> Gemini analysis). A small CPU web server hosts the liquid-glass page
and a /process endpoint; it calls the GPU function on demand, which scales to
zero when idle (so you only pay while a file is actually being processed).

ONE-TIME SETUP (on your machine):
    pip install modal
    modal token new                                   # browser auth
    modal secret create gemini GEMINI_API_KEY=<your-key>

DEPLOY (prints a public https://...modal.run URL):
    modal deploy modal_app.py

Iterate locally first if you like:
    modal serve modal_app.py
"""
import modal

app = modal.App("meeting-intelligence")

# CUDA + cuDNN8 base so ctranslate2 4.4.x has matching libs (the Kaggle lesson:
# ct2 4.5+ needs cuDNN9; pairing cuDNN8 with ct2 4.4.x avoids the precision/DLL
# errors). torch brings its own CUDA libs, so it coexists fine.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch", "torchaudio",
        "whisperx>=3.1.1",
        "faster-whisper>=1.0.0",
        "ctranslate2==4.4.0",
        "google-genai>=1.0.0",
        "pydantic>=2.0",
        "indic-transliteration>=2.3.0",
        "fastapi[standard]>=0.110",
    )
    .add_local_dir("src", "/app/src")
    .add_local_file("config.py", "/app/config.py")
    .add_local_dir("web", "/app/web")
)

# Persist HuggingFace model downloads (large-v3 + VAD + alignment) across cold
# starts so they download once, not every time the GPU spins up.
model_cache = modal.Volume.from_name("mi-model-cache", create_if_missing=True)
CACHE_DIR = "/cache"

GPU_TYPE = "T4"  # 16 GB, cheapest; large-v3 int8_float16 fits comfortably.


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=1800,
    volumes={CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("gemini")],
)
def process_audio(audio_bytes: bytes, filename: str) -> dict:
    """Run the full pipeline on one uploaded file. Returns {hinglish, analysis}."""
    import os
    import sys
    import tempfile
    from pathlib import Path

    os.environ.setdefault("HF_HOME", CACHE_DIR)
    os.environ.setdefault("HF_HUB_CACHE", f"{CACHE_DIR}/hub")
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from src.ingest import ingest
    from src.transcribe import transcribe
    from src.analyze import to_natural_hinglish, analyze_text
    from src.romanize import can_romanize

    work = Path(tempfile.mkdtemp())
    src_path = work / (filename or "upload.bin")
    src_path.write_bytes(audio_bytes)

    wav = ingest(str(src_path))                      # -> 16 kHz mono wav
    result = transcribe(wav, language="hi")          # WhisperX large-v3 on the GPU
    txt = Path(result["paths"]["txt"]).read_text(encoding="utf-8")

    hinglish = to_natural_hinglish(txt) if can_romanize(result.get("language")) else txt
    analysis = analyze_text(txt)                     # Gemini (with 503 retry)

    model_cache.commit()                             # keep the downloaded models
    return {"hinglish": hinglish, "analysis": analysis.model_dump()}


@app.function(image=image)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse

    api = FastAPI(title="Meeting Intelligence")

    @api.get("/")
    def index():
        with open("/app/web/index.html", encoding="utf-8") as f:
            return HTMLResponse(f.read())

    @api.post("/process")
    async def process(file: UploadFile = File(...)):
        data = await file.read()
        try:
            return JSONResponse(process_audio.remote(data, file.filename))
        except Exception as e:  # surface a readable error to the UI
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    return api

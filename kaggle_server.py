"""Run the liquid-glass site on KAGGLE's GPU (free) instead of Modal.

A tiny FastAPI server that serves web/index.html at "/" and runs the SAME src/
pipeline (WhisperX large-v3 -> Gemini Hinglish -> Gemini analysis) at "/process",
all on the Kaggle notebook's GPU. Expose it publicly with a tunnel (cloudflared) -
see the run cell in the chat / README.

    uvicorn kaggle_server:app --host 127.0.0.1 --port 8000

Note: this is only live while the Kaggle notebook is running. For an always-on
public site, use the Modal deployment (modal_app.py) instead.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fastapi import FastAPI, File, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

app = FastAPI(title="Meeting Intelligence (Kaggle)")


@app.get("/")
def index():
    return HTMLResponse((HERE / "web" / "index.html").read_text(encoding="utf-8"))


@app.post("/process")
async def process(file: UploadFile = File(...)):
    data = await file.read()
    try:
        from src.ingest import ingest
        from src.transcribe import transcribe
        from src.analyze import to_natural_hinglish, analyze_text
        from src.romanize import can_romanize

        work = Path(tempfile.mkdtemp())
        src_path = work / (file.filename or "upload.bin")
        src_path.write_bytes(data)

        wav = ingest(str(src_path))
        result = transcribe(wav, language="hi")          # WhisperX large-v3 on the Kaggle GPU
        txt = Path(result["paths"]["txt"]).read_text(encoding="utf-8")
        hinglish = to_natural_hinglish(txt) if can_romanize(result.get("language")) else txt
        analysis = analyze_text(txt)                     # Gemini (with 503 retry)
        return JSONResponse({"hinglish": hinglish, "analysis": analysis.model_dump()})
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

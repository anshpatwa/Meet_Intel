"""Step 4 - LLM analysis.

Feeds a meeting transcript to an LLM and returns a validated MeetingAnalysis
(summary, topics, issues + which were resolved, decisions, action items).

Backend is pluggable via LLM_PROVIDER; only "gemini" is wired up so far. The
provider call is isolated in _gemini_* so adding Claude/Ollama later is local.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.schemas import MeetingAnalysis  # noqa: E402

_SYSTEM = (
    "You are an expert meeting analyst. Read the meeting transcript and produce a "
    "faithful, concise analysis. Never invent facts that are not supported by the "
    "transcript. If the meeting is in Hindi or Hinglish, still write ALL output in "
    "clear English. Carefully separate issues that were RESOLVED during the meeting "
    "from those left open, and only fill 'resolution' when it was actually resolved."
)


def _gemini_client():
    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai") from e
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and add it to your .env file."
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


def _generate_with_retry(client, *, tries: int = 6, **kwargs):
    """Call generate_content, retrying transient Gemini 5xx (e.g. 503 "high demand").

    Those overload errors are temporary, so retry with exponential backoff instead
    of failing the whole run. Non-transient errors (bad request, auth) propagate.
    """
    import time
    from google.genai import errors

    for attempt in range(tries):
        try:
            return client.models.generate_content(**kwargs)
        except errors.ServerError as e:          # 5xx incl. 503 "high demand"
            if attempt == tries - 1:
                raise
            wait = 2 ** attempt                   # 1, 2, 4, 8, 16 s
            print(f"[gemini] busy ({e}); retry {attempt + 1}/{tries} in {wait}s...", flush=True)
            time.sleep(wait)


def analyze_text(transcript: str, model: str | None = None) -> MeetingAnalysis:
    """Analyze a transcript and return a validated MeetingAnalysis."""
    if config.LLM_PROVIDER.lower() != "gemini":
        raise NotImplementedError(
            f"LLM_PROVIDER '{config.LLM_PROVIDER}' is not wired up yet (only 'gemini')."
        )
    from google.genai import types

    client = _gemini_client()
    resp = _generate_with_retry(
        client,
        model=model or config.LLM_MODEL,
        contents=f"Meeting transcript:\n\n{transcript}",
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM,
            response_mime_type="application/json",
            response_schema=MeetingAnalysis,
            temperature=0.2,
        ),
    )
    parsed = getattr(resp, "parsed", None)
    return parsed if parsed is not None else MeetingAnalysis.model_validate_json(resp.text)


_HINGLISH_RULES = (
    "Convert the following Hindi (Devanagari) transcript into natural romanized "
    "Hinglish, exactly as an Indian professional would type it.\n"
    "Rules:\n"
    "- Keep English words in correct English spelling (meeting, deadline, project, deploy).\n"
    "- Romanize Hindi words in everyday Latin spelling (aaj, theek hai, karna, baat).\n"
    "- Preserve line breaks and meaning. Output ONLY the converted text, nothing else.\n\n"
)


def _hinglish_chunk(text: str, model: str | None) -> str:
    from google.genai import types

    client = _gemini_client()
    resp = _generate_with_retry(
        client,
        model=model or config.LLM_MODEL,
        contents=_HINGLISH_RULES + text,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=8192),
    )
    return (resp.text or "").strip()


def to_natural_hinglish(text: str, model: str | None = None, max_chars: int = 6000) -> str:
    """Convert a Devanagari transcript into natural romanized Hinglish via the LLM.

    Unlike rule-based romanization, the LLM restores English words to their real
    spelling and applies correct schwa handling. Long transcripts are split into
    line-aligned chunks so the model output is never truncated.
    """
    lines = text.splitlines() or [text]
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for ln in lines:
        if cur and n + len(ln) + 1 > max_chars:
            chunks.append("\n".join(cur))
            cur, n = [], 0
        cur.append(ln)
        n += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))

    out = []
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"[hinglish] chunk {i}/{len(chunks)}")
        out.append(_hinglish_chunk(chunk, model))
    return "\n".join(out)

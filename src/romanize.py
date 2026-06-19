"""Optional post-step: romanize a Devanagari transcript into Latin "Hinglish".

Whisper transcribes Hindi in Devanagari (e.g. "मीटिंग"); this converts it to a
readable Latin "Hinglish" file next to the Devanagari one.

It goes Devanagari -> IAST (precise) -> readable Latin, applying two rules that
plain transliteration skips, which is what makes it look natural:
  * long vowels kept  (आ -> "aa", ई -> "ee")        -> "laal" not "lala"
  * Hindi word-final schwa deletion  (बात -> "baat") -> "baat" not "bata"

Ceiling of any rule-based approach: English words spoken in the meeting were
written by Whisper in Devanagari, so they come back phonetic ("मीटिंग" -> roughly
"meeting", "डेडलाइन" -> "dedlain"), and medial-schwa cases stay approximate.
For *natural* Hinglish that recovers English spelling, use the LLM stage (Day 3).
"""
from __future__ import annotations

import re
import unicodedata

# Devanagari-script languages worth romanizing (Whisper language codes).
_DEVANAGARI_LANGS = {"hi", "mr", "ne", "sa", "kok", "doi", "mai"}

# Nuqta letters -> base form, so ड़/ढ़ romanize as d/dh (the common Hinglish choice).
_NUQTA = {"ड़": "ड", "ढ़": "ढ"}

# IAST diacritic letters -> ASCII Hinglish.
_IAST_MAP = {
    "ā": "aa", "ī": "ee", "ū": "oo", "ṝ": "ri", "ṛ": "ri", "ḷ": "l",
    "ṭ": "t", "ḍ": "d", "ṇ": "n", "ṅ": "ng", "ñ": "ny",
    "ś": "sh", "ṣ": "sh", "ṁ": "n", "ḥ": "h",
}
_IAST_VOWELS = set("aāiīuūeoṛṝḷ")


def can_romanize(language_code: str | None) -> bool:
    """True if the transcript language uses Devanagari (so romanizing makes sense)."""
    return (language_code or "").split("-")[0].lower() in _DEVANAGARI_LANGS


def _strip_final_schwa(token: str) -> str:
    """Drop a trailing inherent short-'a' (Hindi word-final schwa deletion)."""
    lead = trail = ""
    while token and not token[0].isalpha():
        lead, token = lead + token[0], token[1:]
    while token and not token[-1].isalpha():
        trail, token = token[-1] + trail, token[:-1]
    if len(token) >= 2 and token[-1] == "a" and token[-2] not in _IAST_VOWELS:
        token = token[:-1]
    return lead + token + trail


def _to_ascii(text: str) -> str:
    text = re.sub(r"ṃ(?=[pbm])", "m", text)  # anusvara -> m before labials, else n
    text = text.replace("ṃ", "n")
    for k, v in _IAST_MAP.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)  # strip any leftover combining marks
    return "".join(c for c in text if not unicodedata.combining(c))


def to_hinglish(text: str) -> str:
    """Devanagari -> readable Latin. Returns the input unchanged if the lib is absent."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        return text

    for k, v in _NUQTA.items():
        text = text.replace(k, v)
    iast = transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)

    parts = re.split(r"(\s+)", iast)
    parts = [p if (p == "" or p.isspace()) else _strip_final_schwa(p) for p in parts]
    out = _to_ascii("".join(parts))
    return out.replace("|", ".").replace("।", ".").lower()

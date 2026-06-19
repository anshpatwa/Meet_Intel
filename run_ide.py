"""Run Day 1 from your IDE - no command line needed.

HOW TO RUN (do this once):
  1. Point your IDE's Python interpreter at the project venv:
        .venv\\Scripts\\python.exe
     - VS Code : Ctrl+Shift+P -> "Python: Select Interpreter" -> pick .venv
     - PyCharm : Settings -> Project -> Python Interpreter -> Add -> Existing
                 -> select .venv\\Scripts\\python.exe

EVERY TIME:
  2. Paste your YouTube link (or a local file path) into SOURCE below.
  3. Run this file (green Run arrow / Shift+F10 in PyCharm / F5 or "Run Python File" in VS Code).

Outputs land in  data\\processed\\<name>\\  as .srt / .vtt / .txt / .captions.json
For Hindi audio you also get the offline .hinglish.rule.txt (rule-based romanization).

The natural LLM Hinglish (.hinglish.txt) is a SEPARATE step: after this run
finishes, run  run_hinglish.py  to produce it. Captioning and Hinglish are kept
as two independent scripts on purpose.
"""
from run_day1 import run

# ==========================================================================
#  >>> PASTE YOUR YOUTUBE LINK HERE <<<
#  (a local file works too, e.g.  SOURCE = r"data\raw\meeting.mp4")
# ==========================================================================
SOURCE = r"C:\Users\ANSH\Downloads\bacche_interns.mpeg"

# Language (set this per video):
#   "hi"  = Hindi / Hinglish  <-- USE THIS for Hindi audio.
#   "en"  = English.
#   None  = auto-detect, BUT it often mistakes Hindi for Urdu and outputs the
#           wrong (Perso-Arabic) script. For Hindi content, always set "hi".
# With "hi" you get a Devanagari transcript and the offline "*.hinglish.rule.txt".
# For the natural LLM "*.hinglish.txt", run run_hinglish.py after this finishes.
LANGUAGE = "hi"
# ==========================================================================


if __name__ == "__main__":
    run(SOURCE, language=LANGUAGE)

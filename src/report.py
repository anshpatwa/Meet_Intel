"""Render a MeetingAnalysis into a readable Markdown report."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.schemas import MeetingAnalysis  # noqa: E402


def to_markdown(a: MeetingAnalysis) -> str:
    open_issues = [i for i in a.issues if not i.resolved]
    resolved = [i for i in a.issues if i.resolved]
    L: list[str] = [f"# {a.title}", ""]

    L += ["## Summary", "", a.summary, ""]

    if a.participants:
        L += ["## Participants", "", ", ".join(a.participants), ""]

    L += ["## Topics covered", ""]
    L += [f"- {t}" for t in a.topics] or ["- (none identified)"]
    L += [""]

    L += [f"## Issues raised ({len(a.issues)})", ""]
    if open_issues:
        L += ["### Open", ""]
        for i in open_issues:
            who = f" _(raised by {i.raised_by})_" if i.raised_by else ""
            tail = f" — {i.detail}" if i.detail else ""
            L.append(f"- **{i.title}**{who}{tail}")
        L += [""]
    if resolved:
        L += ["### Resolved", ""]
        for i in resolved:
            tail = f" — {i.detail}" if i.detail else ""
            res = f" → _{i.resolution}_" if i.resolution else ""
            L.append(f"- **{i.title}**{tail}{res}")
        L += [""]
    if not a.issues:
        L += ["- (none identified)", ""]

    if a.decisions:
        L += ["## Decisions", ""]
        L += [f"- {d}" for d in a.decisions]
        L += [""]

    if a.action_items:
        L += ["## Action items", ""]
        for ai in a.action_items:
            meta = [m for m in (f"owner: {ai.owner}" if ai.owner else "",
                                f"due: {ai.due}" if ai.due else "") if m]
            suffix = f"  _({', '.join(meta)})_" if meta else ""
            L.append(f"- [ ] {ai.task}{suffix}")
        L += [""]

    return "\n".join(L)

"""Pydantic schemas for the Day 3 LLM analysis (structured, validated output)."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Issue(BaseModel):
    title: str = Field(description="Short title of the problem/issue raised in the meeting.")
    detail: str = Field(default="", description="One or two sentences of detail.")
    raised_by: str = Field(default="", description="Person/speaker who raised it, if identifiable.")
    resolved: bool = Field(description="True if this issue was resolved during the meeting.")
    resolution: str = Field(default="", description="How it was resolved (only if resolved).")


class ActionItem(BaseModel):
    task: str = Field(description="The action to be done.")
    owner: str = Field(default="", description="Who owns it, if stated.")
    due: str = Field(default="", description="Deadline/timeframe, if stated.")


class MeetingAnalysis(BaseModel):
    title: str = Field(description="A short, descriptive title for this meeting.")
    summary: str = Field(description="A concise paragraph summarizing the meeting.")
    topics: List[str] = Field(description="Main topics discussed, as short phrases.")
    issues: List[Issue] = Field(
        description="Issues/problems raised, each flagged as resolved or still open."
    )
    decisions: List[str] = Field(default_factory=list, description="Concrete decisions made.")
    action_items: List[ActionItem] = Field(default_factory=list)
    participants: List[str] = Field(
        default_factory=list, description="Distinct speakers/people, if identifiable."
    )

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _strip(value: str) -> str:
    return value.strip()


def _validate_string_list(values: List[str]) -> List[str]:
    cleaned: List[str] = []
    for item in values:
        item = _strip(item)
        if not item:
            raise ValueError("Empty strings are not allowed")
        cleaned.append(item)
    return cleaned


class Exercise(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exercise: str
    sets: Optional[int] = None
    reps: Optional[str | int] = None

    _strip_exercise = field_validator("exercise", mode="before")(_strip)


class Day(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    exercises: List[Exercise]

    _strip_name = field_validator("name", mode="before")(_strip)


class Routine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    days: List[Day]


class ActionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    why: str
    how: List[str] = Field(min_length=1)

    _strip_title = field_validator("title", mode="before")(_strip)
    _strip_why = field_validator("why", mode="before")(_strip)
    _strip_how = field_validator("how", mode="before")(_validate_string_list)


class FeedbackResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strengths: List[str] = Field(default_factory=list)
    feedback: List[ActionItem] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)

    _strip_strengths = field_validator("strengths", mode="before")(_validate_string_list)
    _strip_questions = field_validator("questions", mode="before")(_validate_string_list)

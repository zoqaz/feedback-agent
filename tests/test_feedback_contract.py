import json
import os
from pathlib import Path

import pytest

from feedback import generate_feedback_for_routine
from providers.base import FeedbackProvider
from schemas import FeedbackResultV1, Routine

FIXTURE_PATH = Path("tests") / "fixtures" / "routine.json"


class FakeProvider(FeedbackProvider):
    def __init__(self, response: str) -> None:
        self.response = response
        self.model = "fake-model"
        self.retry_count = 0

    def generate_feedback(self, prompt: str) -> str:
        return self.response


def _load_routine() -> Routine:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return Routine.model_validate(payload)


def test_feedback_schema_validates():
    payload = {
        "strengths": ["Balanced exercise selection."],
        "feedback": [
            {
                "title": "Progressive overload plan",
                "why": "Ensure continued adaptation.",
                "how": ["Add 1 rep per set weekly"],
            }
        ],
        "questions": ["What is your primary goal?"],
    }
    FeedbackResultV1.model_validate(payload)


def test_empty_strings_rejected():
    payload = {
        "strengths": [""],
        "feedback": [],
        "questions": [],
    }
    with pytest.raises(ValueError):
        FeedbackResultV1.model_validate(payload)


def test_list_truncation(tmp_path: Path):
    routine = _load_routine()
    payload = {
        "strengths": [f"s{i}" for i in range(10)],
        "feedback": [
            {
                "title": f"t{i}",
                "why": "because",
                "how": [f"h{j}" for j in range(10)],
            }
            for i in range(10)
        ],
        "questions": [f"q{i}" for i in range(10)],
    }
    provider = FakeProvider(json.dumps(payload))
    result = generate_feedback_for_routine(
        routine=routine,
        context={},
        provider=provider,
        output_dir=tmp_path,
    )
    assert len(result.strengths) == 2
    assert len(result.feedback) == 6
    assert len(result.feedback[0].how) == 6
    assert len(result.questions) == 6


def test_orchestrator_writes_artifacts(tmp_path: Path):
    routine = _load_routine()
    payload = {
        "strengths": [],
        "feedback": [],
        "questions": [],
    }
    provider = FakeProvider(json.dumps(payload))
    generate_feedback_for_routine(
        routine=routine,
        context={},
        provider=provider,
        output_dir=tmp_path,
    )
    assert (tmp_path / "feedback_v1.json").exists()
    assert (tmp_path / "feedback_meta.json").exists()


def test_live_claude_optional(tmp_path: Path):
    if os.environ.get("RUN_LIVE_TESTS") != "1":
        pytest.skip("Live test disabled")
    from providers.claude import ClaudeProvider

    routine = _load_routine()
    provider = ClaudeProvider()
    result = generate_feedback_for_routine(
        routine=routine,
        context={},
        provider=provider,
        output_dir=tmp_path,
    )
    assert result.feedback is not None

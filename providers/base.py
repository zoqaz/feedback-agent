from typing import Protocol


class FeedbackProvider(Protocol):
    def generate_feedback(self, prompt: str) -> str:
        ...

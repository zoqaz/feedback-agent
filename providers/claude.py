from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import requests

from providers.base import FeedbackProvider

_DEFAULT_MODEL = "claude-sonnet-4-5"
_API_URL = "https://api.anthropic.com/v1/messages"


class ClaudeProvider(FeedbackProvider):
    def __init__(self) -> None:
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        self.model = os.environ.get("LLM_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "1200"))
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
        self.timeout_s = int(os.environ.get("LLM_TIMEOUT_S", "60"))
        self.retry_count = 0

    def generate_feedback(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        return self._request(payload)

    def _request(self, payload: Dict[str, Any]) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        self.retry_count = 0
        response = self._post(headers, payload)
        if response is not None:
            return response

        self.retry_count = 1
        response = self._post(headers, payload, retry=True)
        if response is not None:
            return response

        raise RuntimeError("Claude request failed after retry")

    def _post(
        self,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        retry: bool = False,
    ) -> Optional[str]:
        try:
            response = requests.post(
                _API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            if retry:
                raise RuntimeError("Claude request failed") from exc
            return None

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    time.sleep(float(retry_after))
                except ValueError:
                    time.sleep(1.0)
            return None

        if 500 <= response.status_code < 600:
            return None if not retry else None

        if response.status_code >= 400:
            raise RuntimeError(
                f"Claude request failed: {response.status_code} {response.text}"
            )

        data = response.json()
        return _extract_text(data)


def _extract_text(data: Dict[str, Any]) -> str:
    content = data.get("content")
    if not isinstance(content, list):
        raise RuntimeError("Claude response missing content")
    parts = [part.get("text", "") for part in content if part.get("type") == "text"]
    if not parts:
        raise RuntimeError("Claude response contained no text")
    return "\n".join(parts).strip()

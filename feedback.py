from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from providers.base import FeedbackProvider
from schemas import FeedbackResultV1, Routine

_MAX_ITEMS = 6
_MAX_HOW = 6
_MAX_STRENGTHS = 2


def _anonymize_day_names(routine: Routine) -> tuple[Routine, dict[str, str]]:
    dumped = routine.model_dump()
    day_name_map: dict[str, str] = {}
    for idx, day in enumerate(dumped.get("days", [])):
        original = str(day.get("name", "")).strip()
        anonymized = f"Day {chr(ord('A') + idx)}"
        day_name_map[anonymized] = original
        day["name"] = anonymized
    return Routine.model_validate(dumped), day_name_map


def generate_feedback_for_routine(
    routine: Routine,
    context: Dict[str, Any],
    provider: FeedbackProvider,
    output_dir: Path,
) -> FeedbackResultV1:
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path("prompts") / "feedback_v1.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt_hash = _hash_text(prompt_text)

    anonymized_routine, day_name_map = _anonymize_day_names(routine)
    routine_json = json.dumps(anonymized_routine.model_dump(), ensure_ascii=False, separators=(",", ":"))
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    prompt = prompt_text.format(routine_json=routine_json, context_json=context_json)

    request_id = str(uuid.uuid4())
    start = time.time()
    raw_output = provider.generate_feedback(prompt)
    latency_ms = int((time.time() - start) * 1000)

    try:
        payload = _extract_json(raw_output)
    except Exception as exc:
        raise RuntimeError(f"Model output was not valid JSON. See {raw_path}") from exc

    payload = _post_process(payload)

    result = FeedbackResultV1.model_validate(payload)

    feedback_path = output_dir / "feedback_v1.json"
    feedback_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    meta = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": getattr(provider, "model", "unknown"),
        "input_tokens": getattr(provider, "input_tokens", None),
        "output_tokens": getattr(provider, "output_tokens", None),
        "prompt_version": prompt_hash,
        "day_name_map": day_name_map,
        "schema_version": "v1",
        "latency_ms": latency_ms,
        "retry_count": getattr(provider, "retry_count", 0),
    }
    meta_path = output_dir / "feedback_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[INFO] feedback request_id={request_id}")
    print(f"[INFO] wrote {feedback_path}")
    print(f"[INFO] wrote {meta_path}")

    return result


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_json(raw: str) -> Dict[str, Any]:
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")
    slice_text = raw[start:]
    end = _find_json_end(slice_text)
    if end is not None:
        slice_text = slice_text[:end]
    return json.loads(slice_text)


def _find_json_end(text: str) -> int | None:
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx + 1
    return None


def _post_process(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload["strengths"] = _trim_list(payload.get("strengths", []), limit=_MAX_STRENGTHS)
    payload["questions"] = _trim_list(payload.get("questions", []))

    payload["feedback"] = _trim_action_items(payload.get("feedback", []))

    return payload


def _trim_list(values: Any, limit: int = _MAX_ITEMS) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("Expected list")
    trimmed: list[str] = []
    for item in values:
        item = _strip(str(item))
        if not item:
            raise ValueError("Empty strings are not allowed")
        trimmed.append(item)
        if len(trimmed) >= limit:
            break
    return trimmed


def _trim_action_items(values: Any, limit: int = _MAX_ITEMS) -> list[dict]:
    if not isinstance(values, list):
        raise ValueError("Expected list")
    trimmed: list[dict] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("Expected object for action item")
        title = _strip(str(item.get("title", "")))
        why = _strip(str(item.get("why", "")))
        how = item.get("how", [])
        if not title or not why:
            raise ValueError("Action item requires title and why")
        how_list = _trim_list(how)[:_MAX_HOW]
        trimmed.append({"title": title, "why": why, "how": how_list})
        if len(trimmed) >= limit:
            break
    return trimmed


def _strip(value: str) -> str:
    return value.strip()

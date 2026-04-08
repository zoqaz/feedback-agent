from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from feedback import generate_feedback_for_routine
from providers.claude import ClaudeProvider
from schemas import Routine


def _build_context(args: argparse.Namespace) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {}
    if args.injuries:
        constraints["injuries"] = args.injuries

    return {
        "goal": args.goal or "unknown",
        "constraints": constraints,
    }


def _determine_output_dir(input_path: Path) -> Path:
    if input_path.name == "parsed_structure.json":
        return input_path.parent
    return input_path.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate feedback for a routine JSON.")
    parser.add_argument("--input", required=True, type=Path, help="Path to routine JSON")
    parser.add_argument("--goal", type=str, default="unknown")
    parser.add_argument("--injuries", type=str)

    args = parser.parse_args()

    input_path = args.input
    routine_payload = json.loads(input_path.read_text(encoding="utf-8"))
    routine = Routine.model_validate(routine_payload)

    context = _build_context(args)
    output_dir = _determine_output_dir(input_path)

    provider = ClaudeProvider()
    result = generate_feedback_for_routine(
        routine=routine,
        context=context,
        provider=provider,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

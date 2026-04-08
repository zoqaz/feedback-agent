import os
import re
import json
from pathlib import Path
from typing import List, Optional
import logging

from pydantic import BaseModel, ValidationError
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class Exercise(BaseModel):
    name: str
    sets: Optional[int]
    reps: Optional[int | str]


class Day(BaseModel):
    name: str
    exercises: List[Exercise]


class Routine(BaseModel):
    days: List[Day]


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt template from the prompts/ directory.
    """
    return Path("prompts", prompt_name).read_text()


MODEL_PATH = os.path.abspath(
    "models/qwen2.5-7b-gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
)

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,#4096,
    n_threads=8,
    n_batch=512,
    temperature=0.0,
    verbose=False
)


def _build_prompt_stats(clean_text: str) -> tuple[str, int, int]:
    template = load_prompt("structural_parse.txt")
    prompt = template.format(clean_text=clean_text)

    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
    max_tokens = min(2200, max(1000, int(prompt_tokens * 2.2)))
    return prompt, prompt_tokens, max_tokens


def parse_routine_local_with_stats(clean_text: str) -> tuple[Routine, dict]:
    """
    Parse cleaned workout text and return routine + stats.
    """
    prompt, prompt_tokens, max_tokens = _build_prompt_stats(clean_text)
    logger.info(f"Prompt tokens: {prompt_tokens}")
    logger.info(f"Max token threshold: {max_tokens}")

    response = llm(
        prompt,
        max_tokens=max_tokens,
        stop=["</s>"],
        stream=False
    )

    raw = response["choices"][0]["text"].strip()  # type: ignore

    try:
        json_str = extract_and_close_json(raw)
        data = json.loads(json_str)
        routine = Routine.model_validate(data)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Local LLM returned invalid JSON:\n{raw}"
        ) from e
    except ValidationError as e:
        raise RuntimeError(
            f"Schema validation failed:\n{e}\nRaw output:\n{raw}"
        ) from e

    stats = {
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
    }
    return routine, stats



def parse_routine_local(clean_text: str) -> Routine:
    """
    Parse cleaned workout text into a validated Routine using a local LLM.
    """
    routine, _ = parse_routine_local_with_stats(clean_text)
    return routine


# Defensive repair for truncated local-LLM JSON
def extract_and_close_json(text: str) -> str:
    """
    Extract the first JSON object and deterministically close missing brackets/braces.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")

    json_str = text[start:]
    end = _find_json_end(json_str)
    if end is not None:
        json_str = json_str[:end]

    # Balance brackets and braces
    open_curly = json_str.count("{")
    close_curly = json_str.count("}")
    open_square = json_str.count("[")
    close_square = json_str.count("]")

    if close_square < open_square:
        json_str += "]" * (open_square - close_square)

    if close_curly < open_curly:
        json_str += "}" * (open_curly - close_curly)

    return json_str


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

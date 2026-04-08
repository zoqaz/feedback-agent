import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import pandas as pd

from core.config import load_config
from core.logging_config import configure_logging

configure_logging(logging.INFO)

logger = logging.getLogger(__name__)

_CONFIG = load_config()
CATALOG_PATH = Path(_CONFIG["paths"]["catalog_csv"])
DEFAULT_ROUTINE_PATH = Path(_CONFIG["paths"]["default_routine_json"])
FUZZY_THRESHOLD = float(_CONFIG["matching"]["fuzzy_threshold"])


@dataclass
class MatchResult:
    routine_name: str
    routine_clean: str
    catalog_display_name: str
    catalog_clean: str
    score: float
    top_matches: List[Tuple[str, float]] = field(default_factory=list)


def normalise_name(name: str) -> str:
    """
    Lowercase the string and remove digits/whitespace/special characters.
    """
    if not name:
        return ""
    normalised = re.sub(r"[^a-z]", "", name.lower())
    return normalised


def load_json(path: Path) -> dict:
    logger.info("Loading JSON from %s", path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    logger.info("Writing matched JSON to %s", path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def extract_routine_names(routine_json: dict) -> List[str]:
    """
    Collect exercise display names from the nested routine JSON.
    """
    names: List[str] = []
    for day in routine_json.get("days", []):
        for exercise in day.get("exercises", []):
            raw_name = (exercise.get("name") or "").strip()
            if raw_name:
                names.append(raw_name)
    return names


def build_catalog_index(catalog: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Prepare (display_name, cleaned_name) tuples for matching.
    """
    catalog = catalog.copy()
    catalog["clean_name"] = catalog["display_name"].fillna("").map(normalise_name)
    indexed = list(zip(catalog["display_name"], catalog["clean_name"]))
    logger.info("Prepared %d catalog rows for fuzzy matching", len(indexed))
    return indexed


def fuzzy_match(
    target_clean: str, catalog_index: List[Tuple[str, str]]
) -> MatchResult:
    best_score = -1.0
    best_display = ""
    best_clean = ""
    scored_matches: List[Tuple[str, float]] = []

    for display_name, catalog_clean in catalog_index:
        if not catalog_clean:
            continue
        score = SequenceMatcher(None, target_clean, catalog_clean).ratio()
        scored_matches.append((display_name, score))
        if score > best_score:
            best_score = score
            best_display = display_name
            best_clean = catalog_clean

    scored_matches.sort(key=lambda item: item[1], reverse=True)
    top_matches = scored_matches[:3]
    return MatchResult(
        routine_name="",
        routine_clean=target_clean,
        catalog_display_name=best_display,
        catalog_clean=best_clean,
        score=best_score,
        top_matches=top_matches,
    )


def match_routine_to_catalog(
    routine_names: Iterable[str], catalog_index: List[Tuple[str, str]]
) -> List[MatchResult]:
    results: List[MatchResult] = []
    seen: Set[str] = set()

    for raw_name in routine_names:
        clean_name = normalise_name(raw_name)
        if not clean_name or clean_name in seen:
            continue
        seen.add(clean_name)

        match = fuzzy_match(clean_name, catalog_index)
        match.routine_name = raw_name
        results.append(match)

    flagged = [r for r in results if r.score < FUZZY_THRESHOLD]
    if not flagged:
        logger.info("All routine exercises have confident matches.")
        return results

    flagged_exc = [result.routine_name for result in flagged]
    logger.warning("Unmatched exercises: %d/%d", len(flagged_exc), len(results))
    logger.warning(flagged_exc)
    return results


def apply_matches_to_routine(
    routine_json: dict, catalog_index: List[Tuple[str, str]]
) -> dict:
    """
    Add matched_exercise field to each exercise using the best catalog match.
    """
    matched = json.loads(json.dumps(routine_json))
    for day in matched.get("days", []):
        for exercise in day.get("exercises", []):
            raw_name = (exercise.get("name") or "").strip()
            clean_name = normalise_name(raw_name)
            if not clean_name:
                exercise["matched_exercise"] = None
                exercise["match_status"] = "needs_review"
                exercise.pop("matched_candidates", None)
                continue
            match = fuzzy_match(clean_name, catalog_index)
            if match.score < FUZZY_THRESHOLD:
                exercise["matched_exercise"] = None
                exercise["match_status"] = "needs_review"
                exercise["matched_candidates"] = [
                    {"name": name, "score": round(score, 2)}
                    for name, score in match.top_matches
                ]
            else:
                exercise["matched_exercise"] = match.catalog_display_name
                exercise["match_status"] = "auto"
                exercise.pop("matched_candidates", None)
    return matched


def main(routine_path: Path) -> None:
    catalog = pd.read_csv(CATALOG_PATH)
    routine_json = load_json(routine_path)

    routine_names = extract_routine_names(routine_json)
    logger.info("Collected %d exercise mentions from routine JSON", len(routine_names))
    logger.info("There are %d unique exercises in the routine JSON", len(set(routine_names)))

    catalog_index = build_catalog_index(catalog)
    match_routine_to_catalog(routine_names, catalog_index)
    matched_routine = apply_matches_to_routine(routine_json, catalog_index)
    output_path = routine_path.with_name("matched_extract.json")
    write_json(output_path, matched_routine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean routine JSON exercise names and fuzzy match them against the exercise catalog."
    )
    parser.add_argument(
        "--routine-json",
        type=Path,
        default=DEFAULT_ROUTINE_PATH,
        help=f"Path to the parsed routine JSON (default: {DEFAULT_ROUTINE_PATH})",
    )
    args = parser.parse_args()
    main(args.routine_json)

# add muscle group to display name for exercise catalog
# next step - leave in all routine names which dont have a matched exercise from the catalog. to be passed onto the LLM for judgment. can update to do more analytics on the workout later as the exercise metadata is improved
# need to include alternative exercise names for our catalog to ensure matches are also performed against alternative exercise names.

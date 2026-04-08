import argparse
import csv
from pathlib import Path

from core.config import load_config
from core.mapping import load_json, write_json

COLOR_RESET = "\x1b[0m"
COLOR_TITLE = "\x1b[36m"
COLOR_CANDIDATE = "\x1b[33m"
COLOR_WARNING = "\x1b[31m"

import questionary


def load_exercises(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_exercises(path: Path, new_items: list[str]) -> None:
    if not new_items:
        return
    with path.open("a", encoding="utf-8") as f:
        for item in new_items:
            f.write(item)
            f.write("\n")


def review_matches(payload: dict, exercises_path: Path) -> dict:
    updated = payload
    existing = load_exercises(exercises_path)
    to_append: list[str] = []
    config = load_config()
    catalog_path = Path(config["paths"]["catalog_csv"])
    catalog_names = load_catalog_names(catalog_path)
    decisions: dict[str, dict[str, str]] = {}

    for day in updated.get("days", []):
        for exercise in day.get("exercises", []):
            if exercise.get("match_status") != "needs_review":
                continue

            routine_name = (exercise.get("name") or "").strip()
            candidates = exercise.get("matched_candidates", [])
            key = routine_name.lower()
            if key in decisions:
                decision = decisions[key]
                exercise["matched_exercise"] = decision["matched_exercise"]
                exercise["match_status"] = decision["match_status"]
                exercise.pop("matched_candidates", None)
                continue

            print(f"\n{COLOR_TITLE}Routine exercise:{COLOR_RESET} {routine_name}")
            if candidates:
                print(f"{COLOR_CANDIDATE}Candidates:{COLOR_RESET}")
                for candidate in candidates:
                    print(f"  - {candidate['name']} ({candidate['score']:.2f})")
            else:
                print(f"{COLOR_CANDIDATE}Candidates:{COLOR_RESET} none")

            while True:
                choice = input(
                    "Select 1 manual, 2 catalog_add, 3 skip, x exit: "
                ).strip().lower()
                if choice == "x":
                    raise SystemExit(0)

                if choice == "1":
                    selected = prompt_catalog_selection(catalog_names)
                    if selected is None:
                        print("No selection made. Returning to menu.")
                        continue
                    exercise["matched_exercise"] = selected
                    exercise["match_status"] = "selected"
                    exercise.pop("matched_candidates", None)
                    decisions[key] = {
                        "matched_exercise": exercise["matched_exercise"],
                        "match_status": exercise["match_status"],
                    }
                    break

                if choice == "2":
                    if routine_name and routine_name.lower() not in existing:
                        existing.add(routine_name.lower())
                        to_append.append(routine_name)
                    exercise["matched_exercise"] = routine_name
                    exercise["match_status"] = "skipped"
                    exercise.pop("matched_candidates", None)
                    decisions[key] = {
                        "matched_exercise": exercise["matched_exercise"],
                        "match_status": exercise["match_status"],
                    }
                    break

                if choice == "3":
                    exercise["matched_exercise"] = routine_name
                    exercise["match_status"] = "skipped"
                    exercise.pop("matched_candidates", None)
                    decisions[key] = {
                        "matched_exercise": exercise["matched_exercise"],
                        "match_status": exercise["match_status"],
                    }
                    break

                print(f"{COLOR_WARNING}Invalid choice. Please select 1, 2, 3, or x.{COLOR_RESET}")

    append_exercises(exercises_path, to_append)
    return updated


def load_catalog_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        names = [row.get("display_name", "").strip() for row in reader]
    return [name for name in names if name]


def prompt_catalog_selection(catalog_names: list[str]) -> str | None:
    if not catalog_names:
        print(f"{COLOR_WARNING}Catalog is empty; cannot select.{COLOR_RESET}")
        return None

    choices = ["[Back]"] + catalog_names
    selected = questionary.autocomplete(
        "Select exercise from catalog:",
        choices=choices,
        ignore_case=True,
        match_middle=True,
    ).ask()
    if selected in (None, "", "[Back]"):
        return None
    if selected not in catalog_names:
        print(f"{COLOR_WARNING}Selection must be from the catalog.{COLOR_RESET}")
        return None
    return selected


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Review and approve matched exercises.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(config["paths"]["default_routine_json"]).with_name(
            config["outputs"]["matched_filename"]
        ),
        help="Path to matched_extract.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(config["paths"]["default_routine_json"]).with_name(
            config["outputs"]["reviewed_filename"]
        ),
        help="Path to save reviewed JSON",
    )
    args = parser.parse_args()

    exercises_path = Path(config["review"]["exercises_txt"])
    payload = load_json(args.input_json)
    reviewed = review_matches(payload, exercises_path)
    write_json(args.output_json, reviewed)
    print(f"\nReviewed JSON saved to {args.output_json}")


if __name__ == "__main__":
    main()

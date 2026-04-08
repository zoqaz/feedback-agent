import pandas as pd

from core import mapping


def make_catalog(names):
    return pd.DataFrame({"display_name": names})


def make_routine(names):
    return {
        "days": [
            {
                "name": "Day 1",
                "exercises": [{"name": name, "sets": 1, "reps": 10} for name in names],
            }
        ]
    }


def test_apply_matches_high_score():
    catalog = make_catalog(["Face Pull (cable)", "Bench Press"])
    catalog_index = mapping.build_catalog_index(catalog)
    routine_json = make_routine(["Face Pull"])

    matched = mapping.apply_matches_to_routine(routine_json, catalog_index)
    exercise = matched["days"][0]["exercises"][0]

    assert exercise["matched_exercise"] == "Face Pull (cable)"
    assert exercise["match_status"] == "auto"
    assert "matched_candidates" not in exercise


def test_apply_matches_low_score():
    catalog = make_catalog(["Squat", "Bench Press", "Deadlift"])
    catalog_index = mapping.build_catalog_index(catalog)
    routine_json = make_routine(["Zzzz"])

    matched = mapping.apply_matches_to_routine(routine_json, catalog_index)
    exercise = matched["days"][0]["exercises"][0]

    assert exercise["matched_exercise"] is None
    assert exercise["match_status"] == "needs_review"
    candidates = exercise.get("matched_candidates")
    assert candidates is not None
    assert len(candidates) == 3
    assert all("name" in item and "score" in item for item in candidates)


def test_match_routine_dedupes():
    catalog = make_catalog(["Face Pull (cable)"])
    catalog_index = mapping.build_catalog_index(catalog)
    routine_names = ["Face Pull", "Face Pull!!"]

    results = mapping.match_routine_to_catalog(routine_names, catalog_index)

    assert len(results) == 1

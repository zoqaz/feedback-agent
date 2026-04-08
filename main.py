import json
import time
from PIL import Image
import pytesseract
import os
import logging
import pandas as pd
from pathlib import Path

from core.config import load_config
from core.logging_config import configure_logging
from core.mapping import (
    apply_matches_to_routine,
    build_catalog_index,
    match_routine_to_catalog,
    write_json,
    CATALOG_PATH,
)
from core.ocr import clean_text
from core.parse import parse_routine_local
from core.review_matches import review_matches

_CONFIG = load_config()
IMAGE_PATH = _CONFIG["paths"]["image_path"]
custom_config = _CONFIG["ocr"]["tesseract_config"]

configure_logging(logging.INFO)

logger = logging.getLogger(__name__)

def extract_text_from_image(image_path: str) -> str:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(
        img,
        lang="eng",
        config=custom_config
    )
    return clean_text(text)

def build_final_plan(reviewed_routine: dict) -> dict:
    final_plan = json.loads(json.dumps(reviewed_routine))
    for day in final_plan.get("days", []):
        for exercise in day.get("exercises", []):
            exercise.pop("match_status", None)
            exercise.pop("name", None)
            if "matched_exercise" in exercise:
                exercise["exercise"] = exercise.pop("matched_exercise")
    return final_plan

if __name__ == "__main__":
    logger.info("Extracting text from image")
    text = extract_text_from_image(IMAGE_PATH)
    
    base_name = os.path.basename(IMAGE_PATH)
    output_name = os.path.splitext(base_name)[0]
    output_dir = os.path.join(_CONFIG["outputs"]["base_dir"], output_name)

    os.makedirs(output_dir, exist_ok=True)
    logger.debug("Output directory ensured: %s", output_dir)

    raw_path = f"{output_dir}/raw_extract.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("OCR text saved to %s", raw_path)

    logger.info("Parsing routine with LLM")
    start = time.time()
    routine = parse_routine_local(text)
    elapsed = time.time() - start

    # print(routine.model_dump_json(indent=2))
    logger.info("LLM parse completed in %.1fs", elapsed)

    parsed_path = os.path.join(output_dir, _CONFIG["outputs"]["parsed_filename"])
    with open(parsed_path, "w", encoding="utf-8") as f:
        f.write(routine.model_dump_json(indent=2))
    logger.info("Parsed routine JSON saved to %s", parsed_path)

    logger.info("Matching routine to catalog")
    catalog = pd.read_csv(CATALOG_PATH)
    catalog_index = build_catalog_index(catalog)
    routine_json = json.loads(routine.model_dump_json())
    routine_names = [ex["name"] for day in routine_json.get("days", []) for ex in day.get("exercises", [])]
    match_routine_to_catalog(routine_names, catalog_index)
    matched_routine = apply_matches_to_routine(routine_json, catalog_index)
    matched_path = Path(output_dir) / _CONFIG["outputs"]["matched_filename"]
    write_json(matched_path, matched_routine)
    logger.info("Matched routine JSON saved to %s", matched_path)

    logger.info("Starting manual review")
    exercises_path = Path(_CONFIG["review"]["exercises_txt"])
    reviewed_routine = review_matches(matched_routine, exercises_path)
    reviewed_path = Path(output_dir) / _CONFIG["outputs"]["reviewed_filename"]
    write_json(reviewed_path, reviewed_routine)
    logger.info("Reviewed routine JSON saved to %s", reviewed_path)

    final_plan = build_final_plan(reviewed_routine)
    final_path = Path(output_dir) / "final_payload.json"
    write_json(final_path, final_plan)
    logger.info("Final payload JSON saved to %s", final_path)

    # - weekly sets per muscle, quad vs ham, anterior vs. posterior shoulder, redundancy, ordering, rep-intent
#     {
#   "what_works": [...],
#   "high_priority_changes": [...],
#   "medium_priority": [...],
#   "notes": [...],
#   "confidence": 0.82
# }

import json
import logging
from typing import Any, Dict

import pandas as pd
from PIL import Image
import pytesseract

from core.config import load_config
from core.logging_config import configure_logging
from core.mapping import (
    apply_matches_to_routine,
    build_catalog_index,
    extract_routine_names,
    match_routine_to_catalog,
    CATALOG_PATH,
)
from core.ocr import clean_text
from core.parse import parse_routine_local

configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

_CONFIG = load_config()
_TESSERACT_CONFIG = _CONFIG["ocr"]["tesseract_config"]


def extract_text_from_image(image_path: str) -> str:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(
        img,
        lang="eng",
        config=_TESSERACT_CONFIG,
    )
    return clean_text(text)


def run_pipeline_from_text(text: str) -> Dict[str, Any]:
    logger.info("Parsing routine with local LLM")
    routine = parse_routine_local(text)
    routine_json = json.loads(routine.model_dump_json())

    logger.info("Matching routine to catalog")
    catalog = pd.read_csv(CATALOG_PATH)
    catalog_index = build_catalog_index(catalog)
    routine_names = extract_routine_names(routine_json)
    match_routine_to_catalog(routine_names, catalog_index)
    matched_routine = apply_matches_to_routine(routine_json, catalog_index)

    # print(matched_routine)

    artifact: Dict[str, Any] = {
        "artifactVersion": 1,
        "canonicalPlan": matched_routine,
        "findings": {},
        "recommendations": [],
    }
    return artifact


def run_pipeline_from_image(image_path: str) -> Dict[str, Any]:
    logger.info("Extracting text from image: %s", image_path)
    text = extract_text_from_image(image_path)
    return run_pipeline_from_text(text)

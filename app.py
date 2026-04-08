import json
import time
import os
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

try:
    from streamlit_sortables import sort_items
except ImportError:
    sort_items = None
from PIL import Image
import pytesseract

from core.config import load_config
from core.mapping import (
    apply_matches_to_routine,
    build_catalog_index,
    match_routine_to_catalog,
    CATALOG_PATH,
)
from core.ocr import clean_text
from core.parse import parse_routine_local_with_stats
from feedback import generate_feedback_for_routine
from providers.claude import ClaudeProvider
from schemas import Routine

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

OUTPUTS_DIR = Path("outputs")
CONFIG = load_config()
TESSERACT_CONFIG = CONFIG["ocr"]["tesseract_config"]


st.set_page_config(page_title="Feedback Agent Dev", layout="wide")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "job"


def _job_id_from_files(files: List[Any]) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = _safe_slug(Path(files[0].name).stem) if files else "job"
    return f"{stamp}_{base}"


def _job_state_path(job_dir: Path) -> Path:
    return job_dir / "job_state.json"


def _load_state(job_dir: Path) -> Dict[str, Any]:
    path = _job_state_path(job_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_state(job_dir: Path, state: Dict[str, Any]) -> None:
    _job_state_path(job_dir).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _append_backlog(items: List[str]) -> None:
    if not items:
        return
    backlog_path = Path("exercise_backlog.txt")
    existing = set()
    if backlog_path.exists():
        existing = {
            line.strip().lower()
            for line in backlog_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    with backlog_path.open("a", encoding="utf-8") as handle:
        for item in items:
            item = item.strip()
            if not item:
                continue
            if item.lower() in existing:
                continue
            handle.write(item)
            handle.write("\n")


def _extract_text_from_image(path: Path) -> str:
    img = Image.open(path)
    text = pytesseract.image_to_string(img, lang="eng", config=TESSERACT_CONFIG)
    return clean_text(text)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_final_payload(reviewed: Dict[str, Any], order_by_day: Dict[str, List[int]]) -> Dict[str, Any]:
    final_payload = {"days": []}
    for day_idx, day in enumerate(reviewed.get("days", [])):
        exercises = day.get("exercises", [])
        order = order_by_day.get(str(day_idx)) or list(range(len(exercises)))
        reordered: List[Dict[str, Any]] = []
        for idx in order:
            if idx >= len(exercises):
                continue
            ex = deepcopy(exercises[idx])
            name_value = ex.get("matched_exercise") or ex.get("name")
            ex.pop("match_status", None)
            ex.pop("matched_candidates", None)
            ex.pop("name", None)
            ex["exercise"] = name_value
            ex.pop("matched_exercise", None)
            reordered.append(ex)
        final_payload["days"].append({"name": day.get("name", ""), "exercises": reordered})
    return final_payload




def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        _rerun()

def _model_label(model_name: str) -> str:
    if "haiku" in model_name:
        return "haiku"
    if "sonnet" in model_name:
        return "sonnet"
    return _safe_slug(model_name)


st.title("Feedback Agent Dev")

jobs = sorted([p for p in OUTPUTS_DIR.glob("*") if p.is_dir()], reverse=True)
job_names = [p.name for p in jobs]

with st.sidebar:
    st.header("Jobs")
    if "selected_job" not in st.session_state:
        st.session_state["selected_job"] = "<new>"
    if "next_job" in st.session_state:
        st.session_state["selected_job"] = st.session_state.pop("next_job")
    selected = st.selectbox("Select job", ["<new>"] + job_names, key="selected_job")

job_dir: Path | None = None
state: Dict[str, Any] = {}

if selected == "<new>":
    st.subheader("Create new job")
    uploaded = st.file_uploader("Upload images", accept_multiple_files=True, type=["png", "jpg", "jpeg", "webp"])
    if uploaded and st.button("Create job"):
        job_id = _job_id_from_files(uploaded)
        job_dir = OUTPUTS_DIR / job_id
        input_dir = job_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        image_names = []
        for file in uploaded:
            dest = input_dir / file.name
            dest.write_bytes(file.getbuffer())
            image_names.append(file.name)
        state = {
            "job_id": job_id,
            "images": image_names,
            "image_order": image_names,
        }
        _save_state(job_dir, state)
        st.session_state["job_id"] = job_id
        st.success(f"Created job {job_id}")
        st.session_state["next_job"] = job_id
        _rerun()
else:
    job_dir = OUTPUTS_DIR / selected
    state = _load_state(job_dir)
    st.session_state["job_id"] = selected

if job_dir is None:
    st.stop()

st.caption(f"Job: {job_dir.name}")


state.setdefault("images", [])
state.setdefault("image_order", state.get("images", []))
state.setdefault("order_by_day", {})

_save_state(job_dir, state)

upload_tab, ocr_tab, parse_tab, match_tab, reorder_tab, edit_tab, feedback_tab = st.tabs(
    ["Upload", "OCR", "Parse", "Match", "Reorder", "Edit", "Feedback"]
)

with upload_tab:
    st.subheader("Images")
    if not state["images"]:
        st.info("No images found in this job.")
    else:
        for idx, name in enumerate(state["image_order"]):
            cols = st.columns([5, 1, 1])
            cols[0].write(name)
            if cols[1].button("Up", key=f"img_up_{idx}"):
                if idx > 0:
                    order = state["image_order"]
                    order[idx - 1], order[idx] = order[idx], order[idx - 1]
                    state["image_order"] = order
                    _save_state(job_dir, state)
                    _rerun()
            if cols[2].button("Down", key=f"img_down_{idx}"):
                if idx < len(state["image_order"]) - 1:
                    order = state["image_order"]
                    order[idx + 1], order[idx] = order[idx], order[idx + 1]
                    state["image_order"] = order
                    _save_state(job_dir, state)
                    _rerun()

with ocr_tab:
    st.subheader("OCR")
    if st.button("Run OCR"):
        ocr_by_image: Dict[str, str] = {}
        for name in state["image_order"]:
            img_path = job_dir / "input" / name
            ocr_by_image[name] = _extract_text_from_image(img_path)
        merged = "\n\n".join(ocr_by_image[name] for name in state["image_order"])
        (job_dir / "raw_extract.txt").write_text(merged, encoding="utf-8")
        state["ocr_by_image"] = ocr_by_image
        _save_state(job_dir, state)
        st.success("OCR complete")

    left, right = st.columns([2, 3])

    with left:
        input_dir = job_dir / "input"
        images = state.get("image_order") or state.get("images") or []
        if images:
            for name in images:
                img_path = input_dir / name
                if img_path.exists():
                    st.image(str(img_path), caption=name, use_column_width=True)

    with right:
        raw_path = job_dir / "raw_extract.txt"
    if raw_path.exists():
        edited_text = st.text_area(
            "Merged OCR text (editable)",
            value=raw_path.read_text(encoding="utf-8"),
            height=300,
            key="ocr_edit_text",
        )
        if st.button("Save OCR text"):
            raw_path.write_text(edited_text, encoding="utf-8")
            st.success("Saved OCR text")

with parse_tab:
    st.subheader("Parse routine")
    raw_path = job_dir / "raw_extract.txt"
    if not raw_path.exists():
        st.info("Run OCR first.")
    else:
        if st.button("Run Parse"):
            text = raw_path.read_text(encoding="utf-8")
            start = time.perf_counter()
            routine, stats = parse_routine_local_with_stats(text)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            stats["elapsed_ms"] = elapsed_ms
            state["parse_stats"] = stats
            _save_state(job_dir, state)
            parsed_path = job_dir / "parsed_extract.json"
            parsed_path.write_text(routine.model_dump_json(indent=2), encoding="utf-8")
            st.success("Parsing complete")
        parsed_path = job_dir / "parsed_extract.json"
        if parsed_path.exists():
            stats = state.get("parse_stats")
            if stats:
                st.info(
                    f"Parse time: {stats.get('elapsed_ms')} ms | "
                    f"prompt tokens: {stats.get('prompt_tokens')} | "
                    f"max tokens: {stats.get('max_tokens')}"
                )
            st.json(_read_json(parsed_path))

with match_tab:
    st.subheader("Match exercises")
    parsed_path = job_dir / "parsed_extract.json"
    matched_path = job_dir / "matched_extract.json"
    reviewed_path = job_dir / "matched_extract_reviewed.json"

    if parsed_path.exists() and st.button("Run Matching"):
        routine_json = _read_json(parsed_path)
        catalog = pd.read_csv(CATALOG_PATH)
        catalog_index = build_catalog_index(catalog)
        routine_names = [
            ex["name"]
            for day in routine_json.get("days", [])
            for ex in day.get("exercises", [])
        ]
        match_routine_to_catalog(routine_names, catalog_index)
        matched = apply_matches_to_routine(routine_json, catalog_index)
        _write_json(matched_path, matched)
        st.success("Matching complete")

    if matched_path.exists():
        matched = _read_json(matched_path)

        total = 0
        needs_review = 0
        for day in matched.get("days", []):
            for ex in day.get("exercises", []):
                total += 1
                if ex.get("match_status") == "needs_review":
                    needs_review += 1

        catalog_df = pd.read_csv(CATALOG_PATH)
        catalog_names: List[str] = []
        if "display_name" in catalog_df.columns:
            catalog_names = sorted(
                {
                    str(name).strip()
                    for name in catalog_df["display_name"].tolist()
                    if str(name).strip()
                }
            )

        st.caption(f"Needs review: {needs_review} of {total} exercises")
        show_auto = st.checkbox("Show auto matches", value=False)
        apply_all = st.checkbox("Group duplicates (same raw name)", value=True)

        reviewed = deepcopy(matched)
        backlog_items: List[str] = []

        def should_show(exercise: Dict[str, Any]) -> bool:
            if exercise.get("match_status") == "needs_review":
                return True
            return show_auto

        def key_for(day_idx: int, ex_idx: int, raw_name: str) -> str:
            if apply_all:
                return raw_name.strip().lower() or f"{day_idx}:{ex_idx}"
            return f"{day_idx}:{ex_idx}"

        def apply_choice(name_key: str, chosen: str, status: str) -> None:
            for d in reviewed.get("days", []):
                for item in d.get("exercises", []):
                    if apply_all:
                        if (item.get("name") or "").strip().lower() != name_key:
                            continue
                    else:
                        continue
                    item["matched_exercise"] = chosen
                    item["match_status"] = status
                    item.pop("matched_candidates", None)

        st.markdown("### Review")
        st.caption("Changes update immediately; click Save matches to write artifacts.")

        seen: set[str] = set()
        for day_idx, day in enumerate(matched.get("days", [])):
            day_name = day.get("name", "Day")
            day_exercises = day.get("exercises", [])

            if not any(should_show(ex) for ex in day_exercises):
                continue

            st.markdown(f"**{day_name}**")
            for ex_idx, ex in enumerate(day_exercises):
                if not should_show(ex):
                    continue

                raw_name = (ex.get("name") or "").strip()
                name_key = raw_name.lower()
                row_key = key_for(day_idx, ex_idx, raw_name)

                if apply_all and row_key in seen:
                    continue
                seen.add(row_key)

                orig_match = ex.get("matched_exercise")

                options: List[str] = []
                if orig_match:
                    options.append(orig_match)
                for cand in ex.get("matched_candidates", []):
                    cand_name = cand.get("name")
                    if cand_name and cand_name not in options:
                        options.append(cand_name)
                options += ["Catalog", "Custom", "Skip", "Delete"]

                current = orig_match
                default = options.index(current) if current in options else 0

                # Layout: raw exercise on left, picker controls on right.
                c0, c1, c2, c3, c4 = st.columns([2.2, 1.8, 1.4, 3.2, 0.8])

                with c0:
                    st.write(raw_name)

                with c1:
                    selection = st.selectbox(
                        "Pick",
                        options,
                        index=default,
                        key=f"match_sel::{row_key}",
                        label_visibility="collapsed",
                    )

                chosen = None
                status = "approved"

                with c2:
                    search_text = ""
                    if selection == "Catalog":
                        search_text = st.text_input(
                            "Search",
                            key=f"match_search::{row_key}",
                            label_visibility="collapsed",
                            placeholder="Search catalog…",
                        )
                    else:
                        st.empty()

                with c3:
                    if selection == "Catalog":
                        if not catalog_names:
                            st.info("Catalog empty")
                            chosen = raw_name
                            status = "defaulted"
                        else:
                            if search_text:
                                filtered = [
                                    name
                                    for name in catalog_names
                                    if search_text.lower() in name.lower()
                                ]
                            else:
                                filtered = catalog_names

                            if not filtered:
                                st.warning("No matches")
                                chosen = raw_name
                                status = "defaulted"
                            else:
                                initial = filtered.index(current) if current in filtered else 0
                                picked = st.selectbox(
                                    "Catalog",
                                    filtered,
                                    index=initial,
                                    key=f"match_catalog::{row_key}",
                                    label_visibility="collapsed",
                                )
                                chosen = picked
                                status = "approved"
                    elif selection == "Custom":
                        custom_value = st.text_input(
                            "Custom exercise",
                            value=current or raw_name,
                            key=f"match_custom::{row_key}",
                            label_visibility="collapsed",
                        )
                        chosen = (custom_value or raw_name).strip()
                        status = "custom"
                    elif selection == "Skip":
                        chosen = raw_name
                        status = "defaulted"
                    elif selection == "Delete":
                        chosen = raw_name
                        status = "deleted"
                    else:
                        chosen = selection
                        status = "approved"

                with c4:
                    add_backlog = st.checkbox(
                        "Backlog",
                        key=f"match_backlog::{row_key}",
                    )

                if chosen is None:
                    chosen = raw_name
                    status = "defaulted"

                if apply_all:
                    apply_choice(name_key, chosen, status)
                else:
                    target = reviewed["days"][day_idx]["exercises"][ex_idx]
                    target["matched_exercise"] = chosen
                    target["match_status"] = status
                    target.pop("matched_candidates", None)

                if add_backlog:
                    backlog_items.append(chosen)

        col_save_1, col_save_2 = st.columns([1, 3])
        if col_save_1.button("Save matches"):
            cleaned = deepcopy(reviewed)
            for day in cleaned.get("days", []):
                day["exercises"] = [
                    ex for ex in day.get("exercises", [])
                    if ex.get("match_status") != "deleted"
                ]
            _write_json(reviewed_path, cleaned)
            _append_backlog([item for item in backlog_items if item])
            st.success(f"Saved {reviewed_path}")

        if reviewed_path.exists():
            with col_save_2:
                st.caption("Current saved reviewed JSON")
            st.json(_read_json(reviewed_path))

with reorder_tab:
    st.subheader("Reorder exercises")
    left, right = st.columns([2, 3])

    with left:
        input_dir = job_dir / "input"
        images = state.get("image_order") or state.get("images") or []
        if images:
            for name in images:
                img_path = input_dir / name
                if img_path.exists():
                    st.image(str(img_path), caption=name, use_column_width=True)

    with right:
        reviewed_path = job_dir / "matched_extract_reviewed.json"
        if not reviewed_path.exists():
            st.info("Complete match review first.")
        else:
            reviewed = _read_json(reviewed_path)

            st.markdown(
                """
<style>
/* Best-effort left alignment for sortable items */
iframe[title="streamlit_sortables.sort_items"] { width: 100% !important; }
</style>
""",
                unsafe_allow_html=True,
            )

            days = list(enumerate(reviewed.get("days", [])))
            if not days:
                st.info("No days found in routine.")
            else:
                cols_count = 3 if len(days) >= 3 else len(days)
                cols = st.columns(cols_count)

                for idx, (day_idx, day) in enumerate(days):
                    col = cols[idx % cols_count]
                    with col:
                        st.markdown(f"**{day.get('name', 'Day')}**")
                        exercises = day.get("exercises", [])
                        order = state["order_by_day"].get(str(day_idx))
                        if not order:
                            order = list(range(len(exercises)))
                            state["order_by_day"][str(day_idx)] = order

                        labels = [
                            (exercises[i].get("matched_exercise") or exercises[i].get("name") or "").strip()
                            for i in order
                        ]

                        if sort_items:
                            sorted_items = sort_items(
                                labels,
                                direction="vertical",
                                key=f"sort_{day_idx}",
                            )
                            if sorted_items:
                                label_to_indices: dict[str, list[int]] = {}
                                for label, ex_idx in zip(labels, order):
                                    label_to_indices.setdefault(label, []).append(ex_idx)

                                new_order: list[int] = []
                                for label in sorted_items:
                                    bucket = label_to_indices.get(label)
                                    if not bucket:
                                        continue
                                    new_order.append(bucket.pop(0))

                                if new_order and new_order != order:
                                    state["order_by_day"][str(day_idx)] = new_order
                                    _save_state(job_dir, state)
                                    order = new_order
                        else:
                            for position, ex_idx in enumerate(order):
                                ex = exercises[ex_idx]
                                row = st.columns([6, 1, 1])
                                row[0].write(ex.get("matched_exercise") or ex.get("name"))
                                if row[1].button("↑", key=f"re_up_{day_idx}_{position}"):
                                    if position > 0:
                                        order[position - 1], order[position] = order[position], order[position - 1]
                                        state["order_by_day"][str(day_idx)] = order
                                        _save_state(job_dir, state)
                                        _rerun()
                                if row[2].button("↓", key=f"re_down_{day_idx}_{position}"):
                                    if position < len(order) - 1:
                                        order[position + 1], order[position] = order[position], order[position + 1]
                                        state["order_by_day"][str(day_idx)] = order
                                        _save_state(job_dir, state)
                                        _rerun()

            if st.button("Save final payload"):
                final_payload = _build_final_payload(reviewed, state.get("order_by_day", {}))
                _write_json(job_dir / "final_payload.json", final_payload)
                _save_state(job_dir, state)
                st.success("Saved final_payload.json")

            final_path = job_dir / "final_payload.json"
            if final_path.exists():
                st.json(_read_json(final_path))

with edit_tab:
    st.subheader("Fill missing sets/reps")
    reviewed_path = job_dir / "matched_extract_reviewed.json"
    if not reviewed_path.exists():
        st.info("Complete matching first.")
    else:
        left, right = st.columns([2, 3])

        with left:
            st.markdown("**Reference**")
            input_dir = job_dir / "input"
            images = state.get("image_order") or state.get("images") or []
            if images:
                for name in images:
                    img_path = input_dir / name
                    if img_path.exists():
                        st.image(str(img_path), caption=name, use_column_width=True)
            raw_path = job_dir / "raw_extract.txt"
            if raw_path.exists():
                st.text_area(
                    "Merged OCR text",
                    raw_path.read_text(encoding="utf-8"),
                    height=220,
                )

        with right:
            reviewed = _read_json(reviewed_path)
            pending_updates: list[tuple[list[str], str, str]] = []

            show_all = st.checkbox("Show all exercises", value=False)
            st.caption("Edit only fields that are currently null. Click Save to write updates.")

            for day_idx, day in enumerate(reviewed.get("days", [])):
                st.markdown(f"**{day.get('name', 'Day')}**")
                exercises = day.get("exercises", [])
                for ex_idx, ex in enumerate(exercises):
                    label = (ex.get("matched_exercise") or ex.get("name") or "").strip()
                    sets_val = ex.get("sets")
                    reps_val = ex.get("reps")

                    if not show_all and sets_val is not None and reps_val is not None:
                        continue

                    row = st.columns([3, 1.2, 1.8])
                    row[0].write(label)

                    sets_key = f"edit_sets::{day_idx}::{ex_idx}"
                    reps_key = f"edit_reps::{day_idx}::{ex_idx}"

                    if sets_val is None:
                        sets_in = row[1].text_input(
                            "Sets",
                            key=sets_key,
                            label_visibility="collapsed",
                            placeholder="sets",
                        ).strip()
                    else:
                        row[1].write(str(sets_val))
                        sets_in = ""

                    if reps_val is None:
                        reps_in = row[2].text_input(
                            "Reps",
                            key=reps_key,
                            label_visibility="collapsed",
                            placeholder="reps",
                        ).strip()
                    else:
                        row[2].write(str(reps_val))
                        reps_in = ""

                    if sets_val is None and sets_in:
                        pending_updates.append(([
                            "days", str(day_idx), "exercises", str(ex_idx), "sets"
                        ], sets_in, "int"))
                    if reps_val is None and reps_in:
                        pending_updates.append(([
                            "days", str(day_idx), "exercises", str(ex_idx), "reps"
                        ], reps_in, "str"))

            if st.button("Save sets/reps"):
                updated = deepcopy(reviewed)
                errors: list[str] = []

                def set_path(obj: Any, path_parts: list[str], value: Any) -> None:
                    cur = obj
                    for part in path_parts[:-1]:
                        cur = cur[int(part)] if part.isdigit() else cur[part]
                    last = path_parts[-1]
                    if last.isdigit():
                        cur[int(last)] = value
                    else:
                        cur[last] = value

                for path_parts, raw_value, kind in pending_updates:
                    if kind == "int":
                        try:
                            val = int(raw_value)
                        except ValueError:
                            errors.append(f"Invalid sets value: {raw_value}")
                            continue
                        set_path(updated, path_parts, val)
                    else:
                        set_path(updated, path_parts, raw_value)

                if errors:
                    st.error("; ".join(errors))
                else:
                    _write_json(reviewed_path, updated)
                    final_path = job_dir / "final_payload.json"
                    if final_path.exists():
                        final_payload = _build_final_payload(updated, state.get("order_by_day", {}))
                        _write_json(final_path, final_payload)
                    st.success("Saved updates")

with feedback_tab:
    st.subheader("Claude feedback")
    final_path = job_dir / "final_payload.json"
    if not final_path.exists():
        st.info("Generate final payload first.")
    else:
        additional_context = st.text_area(
            "Additional context (optional)",
            help="Anything you want the model to consider. Keep it brief and specific.",
            height=120,
        )

        if st.button("Generate feedback"):
            routine_payload = _read_json(final_path)
            routine = Routine.model_validate(routine_payload)
            context = {
                "additional_context": (additional_context or "").strip(),
            }
            provider = ClaudeProvider()
            output_dir = job_dir / _model_label(getattr(provider, "model", "claude"))
            result = generate_feedback_for_routine(
                routine=routine,
                context=context,
                provider=provider,
                output_dir=output_dir,
            )
            st.success("Feedback generated")
            st.json(result.model_dump())

        meta_paths = list(job_dir.glob("*/feedback_meta.json"))
        if meta_paths:
            st.caption("Existing feedback artifacts:")
            for meta_path in meta_paths:
                st.write(str(meta_path))

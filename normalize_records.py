#!/usr/bin/env python3
"""
Normalize all Colorado JSON retention schedule files to the standard import format.

Output format per record:
  record_number, record_title, record_description,
  record_custodian_preservation_destruction_requirement,
  minimum_retention_period, regulatory_citation_statutes_rules_notations,
  page_number, document_title

Usage:
  python normalize_records.py
  → Reads "Colorado JSONs/", writes normalized files to "Colorado JSONs Normalized/"
"""

import json
import os
import re
from pathlib import Path

INPUT_DIR = Path(__file__).parent / "Colorado JSONs"
OUTPUT_DIR = Path(__file__).parent / "Colorado JSONs Normalized"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def to_str(v) -> str:
    return "" if v is None else str(v).strip()

def to_page(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None

def pdf_filename_to_title(filename: str) -> str:
    """Convert 'Governor_Jared_Polis_Email_Retention_Policy.pdf' → readable title."""
    name = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    return name.replace("_", " ").strip()


# ── Title-case / alternate key mappings ──────────────────────────────────────

FIELD_MAP = {
    # Title Case variants (SCHEDULE 75, Congressional 3/4, etc.)
    "Record Title":                                       "record_title",
    "Record Description":                                 "record_description",
    "Record Custodian/Preservation/Destruction Requirement": "record_custodian_preservation_destruction_requirement",
    "Minimum Retention Period":                           "minimum_retention_period",
    "Regulatory Citation/Statutes/Rules/Notations":       "regulatory_citation_statutes_rules_notations",
    "Document Title":                                     "document_title",
    "Page Number":                                        "page_number",
    # lowercase variants (Congressional 2/3, Schedule 17)
    "document title":                                     "document_title",
    "page number":                                        "page_number",
    # record_number aliases
    "section_number":   "record_number",    # SCHEDULE 15, 10 Cemetery
    "item_number":      "record_number",    # Schedule 11 District/Personnel
    "section":          "record_number",    # SCHEDULE 105, 100, SCHEDULE 30 "Section"
    "Section":          "record_number",    # SCHEDULE 30 (Title Case)
    "schedule_item_no": "record_number",    # Colorado Sheriff
    # regulatory_citation shorthand (Schedule 7 GA/Financial, Schedule 8/9/10)
    "regulatory_citation": "regulatory_citation_statutes_rules_notations",
}

# Keys to drop (metadata, duplicates, non-standard)
DROP_KEYS = {
    "source_document",   # PDF filename in entry (Governor Polis / LCS / DPA / OEDIT)
    "source file",       # Congressional 3
    "schedule_section",  # County Treasurer / Public Trustees
    "source_document_title",  # handled below
}


def normalize_entry(entry: dict, fallback_document_title: str = "") -> dict:
    """Map any entry dict to the 8 standard fields."""
    mapped: dict = {}
    for k, v in entry.items():
        if k in DROP_KEYS:
            continue
        canonical = FIELD_MAP.get(k, k)
        # Don't overwrite if already set (e.g., both 'section' and 'record_number')
        if canonical not in mapped:
            mapped[canonical] = v

    # source_document_title → document_title (County Treasurer / Public Trustees)
    if "source_document_title" in entry and "document_title" not in mapped:
        mapped["document_title"] = entry["source_document_title"]

    # Fill document_title from fallback if missing
    if not mapped.get("document_title") and fallback_document_title:
        mapped["document_title"] = fallback_document_title

    doc_title = to_str(mapped.get("document_title", ""))
    # If document_title is a PDF filename (e.g. "DOR_Email_Retention_Policy.pdf"), clean it
    if doc_title.lower().endswith(".pdf"):
        doc_title = pdf_filename_to_title(doc_title)
    if not doc_title and fallback_document_title:
        doc_title = fallback_document_title

    return {
        "record_number":   to_str(mapped.get("record_number", "")),
        "record_title":    to_str(mapped.get("record_title", "")),
        "record_description": to_str(mapped.get("record_description", "")),
        "record_custodian_preservation_destruction_requirement":
            to_str(mapped.get("record_custodian_preservation_destruction_requirement", "")),
        "minimum_retention_period": to_str(mapped.get("minimum_retention_period", "")),
        "regulatory_citation_statutes_rules_notations":
            to_str(mapped.get("regulatory_citation_statutes_rules_notations", "")),
        "page_number":     to_page(mapped.get("page_number")),
        "document_title":  doc_title,
    }


def get_document_title(data: dict, json_filename: str) -> str:
    """Extract document title from various top-level dict structures."""
    # Direct key
    if isinstance(data.get("document_title"), str):
        return data["document_title"]
    # Nested under document_metadata
    if isinstance(data.get("document_metadata"), dict):
        t = data["document_metadata"].get("document_title", "")
        if t:
            return t
    # Nested under schedule_metadata
    if isinstance(data.get("schedule_metadata"), dict):
        t = data["schedule_metadata"].get("document_title", "")
        if t:
            return t
    # 'schedule' key can be either the title string or a dict
    sched = data.get("schedule")
    if isinstance(sched, str) and sched:
        return sched
    if isinstance(sched, dict):
        t = sched.get("document_title", "")
        if t:
            return t
    # 'source_document' as a plain title string (Public Trustees, County Treasurer,
    # Schedule 16 — NOT the LCS/DPA/OEDIT case where it's a PDF filename)
    src = data.get("source_document")
    if isinstance(src, str) and src and not src.endswith(".pdf"):
        return src
    # source_document is a PDF filename → convert to readable title
    if isinstance(src, str) and src.endswith(".pdf"):
        return pdf_filename_to_title(src)
    # Last resort: use JSON filename without extension
    return Path(json_filename).stem


def get_entries(data: dict | list) -> list[dict]:
    """Extract the raw list of record entries from any top-level structure."""
    if isinstance(data, list):
        return data
    for key in [
        "entries",
        "records",
        "retention_schedule_entries",
        "email_retention_entries",
        "records_retention_entries",
        "records_retention_schedules",
    ]:
        if key in data and isinstance(data[key], list):
            return data[key]
    return []


# ── Per-file processing ───────────────────────────────────────────────────────

def process_file(input_path: Path, output_path: Path) -> int:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_entries = get_entries(data)
    if not raw_entries:
        print(f"  SKIP (no entries): {input_path.name}")
        return 0

    # document_title fallback for entries that don't carry one
    fallback_title = get_document_title(data, input_path.name) if isinstance(data, dict) else ""

    normalized = [normalize_entry(e, fallback_title) for e in raw_entries]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    return len(normalized)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    files = sorted(INPUT_DIR.glob("*.json"))
    total_records = 0
    for fp in files:
        out = OUTPUT_DIR / fp.name
        count = process_file(fp, out)
        if count:
            print(f"  {count:>4} records  {fp.name}")
            total_records += count
    print(f"\nDone. {len(files)} files, {total_records} total records → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

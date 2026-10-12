"""
metadata_extractor.py - Extracts structured metadata from JKBOSE question paper headers
and filenames using regex-based parsing.
"""
import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ─── Filename pattern ────────────────────────────────────────────────
# Example: jkbose-12th-physics-2024-x-1108_questions.txt
FILENAME_PATTERN = re.compile(
    r"jkbose-(\d+)th-([a-z\-]+)-(\d+)-([a-z\-]*)-?(\d+)?_questions\.txt",
    re.IGNORECASE,
)

# ─── Header patterns for parsing the structured text content ─────────
HEADER_PATTERNS = {
    "exam_title": re.compile(
        r"EXAM\s+PAPER\s+TITLE\s*:\s*(.+)", re.IGNORECASE
    ),
    "subject": re.compile(
        r"SUBJECT\s*:\s*(.+)", re.IGNORECASE
    ),
    "class_level": re.compile(
        r"CLASS\s*/?\s*LEVEL\s*:\s*(?:Class\s+)?(\d+)", re.IGNORECASE
    ),
    "year": re.compile(
        r"EXAMINATION\s+YEAR\s*:\s*(\d+)", re.IGNORECASE
    ),
    "set_code": re.compile(
        r"SERIES\s*/?\s*SET\s*:\s*(?:SET\s+)?([A-Z\-]+)", re.IGNORECASE
    ),
    "paper_code": re.compile(
        r"QUESTION\s+CODE\s*:\s*(\w+)", re.IGNORECASE
    ),
    "max_marks": re.compile(
        r"MAXIMUM\s+MARKS\s*:\s*(\d+)", re.IGNORECASE
    ),
    "time_allowed": re.compile(
        r"TIME\s+ALLOWED\s*:\s*(.+)", re.IGNORECASE
    ),
}

# ─── Section pattern ──────────────────────────────────────────────────
SECTION_PATTERN = re.compile(
    r"SECTION\s+([A-Z]+)\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE
)

# ─── Question pattern ────────────────────────────────────────────────
QUESTION_PATTERN = re.compile(
    r"Q\.(\d+)\s+\[(\d+)\s+Marks?\]\s*\(([^)]+)\)\s*:\s*\n\s+(.+?)(?=\n\nQ\.\d|\n\n\s*-{5,}|\n\n={5,}|\Z)",
    re.DOTALL | re.IGNORECASE,
)

# Simpler fallback for question detection
QUESTION_SIMPLE_PATTERN = re.compile(
    r"Q\.(\d+)\s+\[(\d+)\s+Marks?\]",
    re.IGNORECASE,
)


def extract_metadata_from_filename(filename: str) -> Dict:
    """Extract metadata from the structured filename."""
    meta = {}
    m = FILENAME_PATTERN.match(filename)
    if m:
        meta["class_level"] = m.group(1)
        meta["subject"] = normalize_subject(m.group(2))
        meta["year"] = m.group(3)
        meta["set_code"] = m.group(4).upper() if m.group(4) and m.group(4) != "-" else None
        meta["paper_code"] = m.group(5) if m.group(5) else None
    return meta


def extract_metadata_from_header(content: str) -> Dict:
    """Parse the structured header block of a question paper."""
    meta = {}
    # Only search in the first 800 characters (header area)
    header_block = content[:800]

    for field, pattern in HEADER_PATTERNS.items():
        match = pattern.search(header_block)
        if match:
            value = match.group(1).strip().rstrip("\r")
            if field == "subject":
                value = normalize_subject(value)
            elif field == "class_level":
                value = value.replace("TH", "").strip()
            elif field == "set_code":
                value = value.upper().strip()
            meta[field] = value

    # Determine exam type from header
    if "HIGHER SECONDARY" in header_block.upper():
        meta["exam_type"] = "Higher Secondary"
    elif "SECONDARY SCHOOL" in header_block.upper():
        meta["exam_type"] = "Secondary School"
    else:
        meta["exam_type"] = None

    return meta


def normalize_subject(raw: str) -> str:
    """Normalize subject names for consistency."""
    raw = raw.lower().strip().replace("-", " ").replace("_", " ")
    raw = re.sub(r"\s*\(class\s+\d+.*?\)", "", raw, flags=re.IGNORECASE)
    raw = raw.strip()

    mapping = {
        "physics": "Physics",
        "chemistry": "Chemistry",
        "biology": "Biology",
        "mathematics": "Mathematics",
        "math": "Mathematics",
        "science": "Science",
        "environmental science": "Environmental Science",
        "env science": "Environmental Science",
    }

    for key, val in mapping.items():
        if key in raw:
            return val
    # Title-case fallback
    return raw.title()


def extract_paper_metadata(doc: Dict) -> Dict:
    """
    Combine filename-based and header-based metadata for a document.
    Header values take precedence when both are available.
    """
    fname_meta = extract_metadata_from_filename(doc["filename"])
    header_meta = extract_metadata_from_header(doc["content"])

    # Merge: header takes priority
    merged = {**fname_meta, **{k: v for k, v in header_meta.items() if v is not None}}

    # Ensure required fields exist
    merged.setdefault("source_file", doc["filename"])
    merged.setdefault("class_level", None)
    merged.setdefault("subject", None)
    merged.setdefault("year", None)
    merged.setdefault("set_code", None)
    merged.setdefault("paper_code", None)
    merged.setdefault("exam_type", None)
    merged.setdefault("max_marks", None)
    merged.setdefault("time_allowed", None)

    return merged


def classify_question_type(marks: str, text: str) -> str:
    """Classify question type based on marks and content."""
    marks_int = int(marks) if marks and marks.isdigit() else 0

    text_lower = text.lower()

    # Check for MCQ indicators
    if any(marker in text_lower for marker in ["(a)", "(b)", "(c)", "(d)"]):
        if marks_int <= 1:
            return "MCQ"

    # Fill in the blank
    if "________" in text or "fill in" in text_lower:
        return "Fill in the Blank"

    # Classify by marks
    if marks_int <= 1:
        return "Very Short Answer"
    elif marks_int == 2:
        return "Short Answer Type I"
    elif marks_int == 3:
        return "Short Answer Type II"
    elif marks_int == 4:
        return "Long Answer"
    elif marks_int >= 5:
        return "Long Answer (Choice)"

    return "Unknown"

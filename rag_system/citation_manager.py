"""
citation_manager.py - Generates and validates source citations for retrieved results.
Ensures every factual claim has a traceable source.
"""
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


def format_citation(metadata: Dict) -> str:
    """
    Format a citation string from chunk metadata.
    Uses the format: [Source: filename | year | Question N | M marks]
    Only includes fields that are actually available.
    """
    parts = []

    source_file = metadata.get("source_file")
    if source_file:
        parts.append(source_file)

    year = metadata.get("year")
    if year:
        parts.append(f"Year {year}")

    class_level = metadata.get("class_level")
    if class_level:
        parts.append(f"Class {class_level}")

    subject = metadata.get("subject")
    if subject:
        parts.append(subject)

    set_code = metadata.get("set_code")
    if set_code and set_code not in ("-", "None", "null"):
        parts.append(f"Set {set_code}")

    paper_code = metadata.get("paper_code")
    if paper_code:
        parts.append(f"Code {paper_code}")

    question_number = metadata.get("question_number")
    if question_number:
        parts.append(f"Q.{question_number}")

    marks = metadata.get("marks")
    if marks:
        parts.append(f"{marks} Mark(s)")

    if not parts:
        return "[Source: metadata unavailable]"

    return "[Source: " + " | ".join(parts) + "]"


def format_short_citation(metadata: Dict) -> str:
    """Compact citation for inline use."""
    parts = []
    source_file = metadata.get("source_file", "")
    if source_file:
        # Shorten filename
        short = source_file.replace("jkbose-", "").replace("_questions.txt", "")
        parts.append(short)
    q = metadata.get("question_number")
    if q:
        parts.append(f"Q.{q}")
    return "[" + " | ".join(parts) + "]" if parts else "[Source unavailable]"


def validate_citation(result: Dict) -> bool:
    """
    Validate that a citation corresponds to actual retrieved data.
    Returns True if the citation has real source information.
    """
    meta = result.get("chunk", {}).get("metadata", {})
    if not meta:
        return False
    # Must have at least a source file
    return bool(meta.get("source_file"))


def build_cited_response(results: List[Dict]) -> List[Dict]:
    """
    Build a list of cited results, each with formatted citation.
    Filters out results with invalid citations.
    """
    cited = []
    for r in results:
        if not validate_citation(r):
            continue
        meta = r["chunk"]["metadata"]
        cited.append({
            "text": r["chunk"]["text"],
            "citation": format_citation(meta),
            "short_citation": format_short_citation(meta),
            "score": r.get("score", 0),
            "rank": r.get("rank", 0),
            "metadata": meta,
        })
    return cited

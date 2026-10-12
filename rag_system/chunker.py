"""
chunker.py - Splits question papers into semantically meaningful chunks.
Preserves question boundaries, OR-choice groupings, and section context.
Each chunk is a single question (or question + OR choice).
"""
import re
import logging
from typing import List, Dict, Optional
from rag_system.metadata_extractor import (
    extract_paper_metadata,
    classify_question_type,
    SECTION_PATTERN,
)

logger = logging.getLogger(__name__)

# Matches a full question block including text and optional OR choice
QUESTION_BLOCK_PATTERN = re.compile(
    r"(Q\.(\d+)\s+\[(\d+)\s+Marks?\]\s*\(([^)]*)\)\s*:"
    r"(.*?)"
    r"(?=Q\.\d+\s+\[|={10,}|\Z))",
    re.DOTALL,
)

# Pattern to detect consolidated file paper boundaries
PAPER_BOUNDARY_PATTERN = re.compile(
    r"^={10,}\s*$\s*(?:JAMMU AND KASHMIR|JKBOSE)",
    re.MULTILINE | re.IGNORECASE,
)

PAPER_TITLE_PATTERN = re.compile(
    r"EXAM\s+PAPER\s+TITLE\s*:\s*(.+?)(?:\r?\n)",
    re.IGNORECASE,
)


def split_consolidated_file(content: str) -> List[str]:
    """
    Split a consolidated master file into individual paper sections.
    Uses the ===...=== + JAMMU AND KASHMIR pattern to find paper boundaries.
    """
    # Find all paper start positions
    starts = []
    for m in re.finditer(r"={60,}", content):
        # Check if this is a paper header (followed by JAMMU or JKBOSE within 200 chars)
        after = content[m.start():m.start() + 300]
        if re.search(r"JAMMU AND KASHMIR|JKBOSE.*EXAMINATION", after, re.IGNORECASE):
            starts.append(m.start())

    if not starts:
        return [content]

    papers = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(content)
        papers.append(content[start:end])

    return papers


def chunk_single_paper(content: str, paper_metadata: Dict) -> List[Dict]:
    """
    Chunk a single paper into question-level chunks.
    Each chunk contains one question block (including OR choice if present).
    """
    chunks = []
    current_section = None
    current_section_title = None

    # Detect sections
    sections = {}
    for m in SECTION_PATTERN.finditer(content):
        sections[m.start()] = {
            "section_letter": m.group(1).upper(),
            "section_title": m.group(2).strip().rstrip("\r"),
        }

    section_positions = sorted(sections.keys())

    # Find all question blocks
    for match in QUESTION_BLOCK_PATTERN.finditer(content):
        full_block = match.group(1).strip()
        q_num = match.group(2).strip()
        marks = match.group(3).strip()
        topic = match.group(4).strip().rstrip("\r")
        question_text = match.group(5).strip()

        # Determine which section this question belongs to
        q_pos = match.start()
        for sp in section_positions:
            if sp <= q_pos:
                current_section = sections[sp]["section_letter"]
                current_section_title = sections[sp]["section_title"]

        # Clean up question text
        question_text = re.sub(r"\s+", " ", question_text).strip()
        # Remove trailing section delimiters
        question_text = re.sub(r"\s*={5,}.*$", "", question_text).strip()

        # Check for OR choice
        has_or_choice = bool(re.search(r"\[OR\s+CHOICE\]|^\s*OR\s*$", full_block, re.MULTILINE | re.IGNORECASE))

        # Determine question type
        q_type = classify_question_type(marks, question_text)

        # Build the chunk
        chunk = {
            "chunk_id": f"{paper_metadata.get('source_file', 'unknown')}::Q{q_num}",
            "text": question_text,
            "full_block": full_block,
            "metadata": {
                **paper_metadata,
                "section": current_section,
                "section_title": current_section_title,
                "question_number": q_num,
                "marks": marks,
                "topic": topic,
                "question_type": q_type,
                "has_or_choice": has_or_choice,
                "page": None,  # text files don't have pages
            },
        }
        chunks.append(chunk)

    if not chunks:
        # Fallback: if regex didn't match, create one chunk per paper
        logger.warning(
            f"No questions parsed from {paper_metadata.get('source_file', 'unknown')}. "
            f"Creating single fallback chunk."
        )
        chunks.append({
            "chunk_id": f"{paper_metadata.get('source_file', 'unknown')}::FULL",
            "text": content[:2000],  # First 2000 chars as fallback
            "full_block": content[:3000],
            "metadata": {
                **paper_metadata,
                "section": None,
                "section_title": None,
                "question_number": None,
                "marks": None,
                "topic": None,
                "question_type": None,
                "has_or_choice": False,
                "page": None,
            },
        })

    return chunks


def chunk_document(doc: Dict) -> List[Dict]:
    """
    Main entry point: chunk a loaded document into question-level chunks.
    Handles both individual papers and consolidated master files.
    """
    filename = doc["filename"]
    content = doc["content"]

    all_chunks = []

    if "_all_questions_" in filename:
        # Consolidated file: split into individual papers first
        paper_sections = split_consolidated_file(content)
        logger.info(f"Split {filename} into {len(paper_sections)} papers")

        for paper_content in paper_sections:
            paper_doc = {
                "filename": filename,
                "content": paper_content,
                "filepath": doc["filepath"],
            }
            paper_meta = extract_paper_metadata(paper_doc)
            # Try to get the specific paper title from content
            title_match = PAPER_TITLE_PATTERN.search(paper_content)
            if title_match:
                paper_meta["exam_title"] = title_match.group(1).strip().rstrip("\r")
            chunks = chunk_single_paper(paper_content, paper_meta)
            all_chunks.extend(chunks)
    else:
        # Individual paper file
        paper_meta = extract_paper_metadata(doc)
        chunks = chunk_single_paper(content, paper_meta)
        all_chunks.extend(chunks)

    logger.info(f"Chunked {filename} → {len(all_chunks)} chunks")
    return all_chunks

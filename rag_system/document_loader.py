"""
document_loader.py - Discovers and loads all JKBOSE question paper files.
Handles individual .txt files and consolidated master files.
"""
import os
import glob
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def discover_files(data_dir: str) -> List[str]:
    """
    Recursively discover all .txt question paper files in the data directory.
    Prefers individual_papers/ directories over consolidated master files
    to avoid double-counting questions.
    """
    individual_files = []
    consolidated_files = []

    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            full_path = os.path.join(root, fname)
            if "individual_papers" in root:
                individual_files.append(full_path)
            elif "_all_questions_" in fname:
                consolidated_files.append(full_path)

    # Log summary
    logger.info(f"Discovered {len(individual_files)} individual paper files")
    logger.info(f"Discovered {len(consolidated_files)} consolidated files")

    # We process individual files AND consolidated files.
    # The consolidated files contain papers that may not have individual files (class 10 has none consolidated).
    # To avoid duplication: we track paper titles and deduplicate at the chunk level.
    return individual_files + consolidated_files


def load_file(filepath: str) -> Dict:
    """
    Load a single text file and return its content with basic file metadata.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "content": content,
            "size_bytes": os.path.getsize(filepath),
        }
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None


def load_all_documents(data_dir: str) -> List[Dict]:
    """
    Discover and load all question paper files.
    Returns a list of document dicts.
    """
    filepaths = discover_files(data_dir)
    documents = []
    for fp in filepaths:
        doc = load_file(fp)
        if doc is not None:
            documents.append(doc)
    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents

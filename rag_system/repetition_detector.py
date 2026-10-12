"""
repetition_detector.py - Detects repeated and similar questions across papers.
Uses both semantic similarity (embeddings) and normalized text similarity.
Classifies pairs as EXACT_REPEAT, NEAR_REPEAT, SEMANTICALLY_SIMILAR, RELATED_TOPIC, or NOT_SIMILAR.
"""
import re
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from rag_system.config import (
    REPEAT_EXACT_THRESHOLD,
    REPEAT_NEAR_THRESHOLD,
    REPEAT_SEMANTIC_THRESHOLD,
    REPEAT_RELATED_THRESHOLD,
    TEXT_EXACT_THRESHOLD,
    TEXT_NEAR_THRESHOLD,
)
from rag_system.embedding_service import embed_texts

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize question text for comparison."""
    text = text.lower().strip()
    # Remove question numbering
    text = re.sub(r"^q\.\s*\d+\s*", "", text)
    # Remove marks info
    text = re.sub(r"\[\d+\s*marks?\]", "", text, flags=re.IGNORECASE)
    # Remove topic annotations
    text = re.sub(r"\([^)]*\)\s*:", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_similarity(a: str, b: str) -> float:
    """
    Compute normalized text similarity using character-level n-gram overlap.
    Faster than full Levenshtein for our use case.
    """
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return 0.0

    if a_norm == b_norm:
        return 1.0

    # Character trigram overlap (Jaccard-like)
    def trigrams(s):
        return set(s[i:i+3] for i in range(len(s) - 2)) if len(s) >= 3 else {s}

    a_tri = trigrams(a_norm)
    b_tri = trigrams(b_norm)

    if not a_tri or not b_tri:
        return 0.0

    intersection = len(a_tri & b_tri)
    union = len(a_tri | b_tri)

    return intersection / union if union > 0 else 0.0


def classify_similarity(semantic_sim: float, text_sim: float) -> str:
    """
    Classify the similarity between two questions.
    Uses both semantic and text similarity for robust classification.
    """
    # EXACT_REPEAT: very high text similarity AND high semantic similarity
    if text_sim >= TEXT_EXACT_THRESHOLD and semantic_sim >= REPEAT_NEAR_THRESHOLD:
        return "EXACT_REPEAT"

    # NEAR_REPEAT: high text similarity OR very high semantic + moderate text
    if text_sim >= TEXT_NEAR_THRESHOLD and semantic_sim >= REPEAT_NEAR_THRESHOLD:
        return "NEAR_REPEAT"
    if semantic_sim >= REPEAT_EXACT_THRESHOLD and text_sim >= 0.5:
        return "NEAR_REPEAT"

    # SEMANTICALLY_SIMILAR: high semantic but moderate text
    if semantic_sim >= REPEAT_SEMANTIC_THRESHOLD:
        return "SEMANTICALLY_SIMILAR"

    # RELATED_TOPIC: moderate semantic similarity
    if semantic_sim >= REPEAT_RELATED_THRESHOLD:
        return "RELATED_TOPIC"

    return "NOT_SIMILAR"


class RepetitionDetector:
    """Detects repeated and similar questions across the dataset."""

    def __init__(self):
        self._cache: Dict = {}

    def find_repeated_questions(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        metadata_filters: Optional[Dict] = None,
        min_similarity_type: str = "NEAR_REPEAT",
    ) -> List[Dict]:
        """
        Find groups of repeated/similar questions in the dataset.
        
        Returns a list of question groups, each containing:
        {
            "representative_text": str,
            "similarity_type": str,
            "occurrences": [
                {
                    "text": str,
                    "year": str,
                    "source_file": str,
                    "question_number": str,
                    "marks": str,
                    "set_code": str,
                    "paper_code": str,
                }
            ],
            "years_found": List[str],
            "occurrence_count": int,
        }
        """
        if len(chunks) == 0:
            return []

        # Filter chunks/embeddings by metadata
        if metadata_filters:
            indices = []
            for i, c in enumerate(chunks):
                meta = c.get("metadata", {})
                match = True
                for k, v in metadata_filters.items():
                    if v is None:
                        continue
                    cv = meta.get(k)
                    if cv is None or (isinstance(v, str) and isinstance(cv, str) and v.lower() != cv.lower()):
                        match = False
                        break
                if match:
                    indices.append(i)
            if not indices:
                return []
            chunks = [chunks[i] for i in indices]
            embeddings = embeddings[indices]

        n = len(chunks)
        if n < 2:
            return []

        logger.info(f"Computing similarity matrix for {n} chunks...")

        # Compute pairwise semantic similarity matrix
        sim_matrix = np.dot(embeddings, embeddings.T)

        # Union-Find for grouping similar questions
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Determine minimum semantic threshold based on requested type
        type_thresholds = {
            "EXACT_REPEAT": REPEAT_EXACT_THRESHOLD,
            "NEAR_REPEAT": REPEAT_NEAR_THRESHOLD,
            "SEMANTICALLY_SIMILAR": REPEAT_SEMANTIC_THRESHOLD,
            "RELATED_TOPIC": REPEAT_RELATED_THRESHOLD,
        }
        min_threshold = type_thresholds.get(min_similarity_type, REPEAT_NEAR_THRESHOLD)

        # Find pairs above threshold
        # Only compare across DIFFERENT papers (same paper = same set of questions)
        for i in range(n):
            for j in range(i + 1, n):
                # Skip if same paper
                src_i = chunks[i].get("metadata", {}).get("source_file", "")
                src_j = chunks[j].get("metadata", {}).get("source_file", "")
                if src_i == src_j:
                    continue

                semantic_sim = float(sim_matrix[i, j])
                if semantic_sim < min_threshold * 0.9:  # Quick filter
                    continue

                text_sim = text_similarity(chunks[i]["text"], chunks[j]["text"])
                sim_type = classify_similarity(semantic_sim, text_sim)

                if self._type_meets_minimum(sim_type, min_similarity_type):
                    union(i, j)

        # Group by connected component
        groups_map = defaultdict(list)
        for i in range(n):
            groups_map[find(i)].append(i)

        # Build result groups (only groups with 2+ members = repeated)
        results = []
        for members in groups_map.values():
            if len(members) < 2:
                continue

            # Determine the strongest similarity type in the group
            best_type = "RELATED_TOPIC"
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    mi, mj = members[i], members[j]
                    ss = float(sim_matrix[mi, mj])
                    ts = text_similarity(chunks[mi]["text"], chunks[mj]["text"])
                    st = classify_similarity(ss, ts)
                    if self._type_rank(st) > self._type_rank(best_type):
                        best_type = st

            if not self._type_meets_minimum(best_type, min_similarity_type):
                continue

            occurrences = []
            years_found = set()
            for idx in members:
                meta = chunks[idx].get("metadata", {})
                year = meta.get("year", "Unknown")
                years_found.add(str(year))
                occurrences.append({
                    "text": chunks[idx]["text"],
                    "year": str(year),
                    "source_file": meta.get("source_file", "Unknown"),
                    "question_number": meta.get("question_number", "Unknown"),
                    "marks": meta.get("marks", "Unknown"),
                    "set_code": meta.get("set_code", "Unknown"),
                    "paper_code": meta.get("paper_code", "Unknown"),
                    "subject": meta.get("subject", "Unknown"),
                })

            # Sort occurrences by year
            occurrences.sort(key=lambda x: str(x.get("year", "0")))

            results.append({
                "representative_text": chunks[members[0]]["text"][:300],
                "similarity_type": best_type,
                "occurrences": occurrences,
                "years_found": sorted(years_found),
                "occurrence_count": len(occurrences),
            })

        # Sort by occurrence count (most repeated first)
        results.sort(key=lambda x: x["occurrence_count"], reverse=True)

        logger.info(f"Found {len(results)} groups of repeated questions")
        return results

    def find_similar_to_query(
        self,
        query_text: str,
        chunks: List[Dict],
        embeddings: np.ndarray,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Find questions similar to a reference query/question text.
        Returns matches with similarity type classification.
        """
        from rag_system.embedding_service import embed_query

        query_emb = embed_query(query_text)
        similarities = np.dot(embeddings, query_emb)

        # Get top candidates
        top_indices = np.argsort(-similarities)[:top_k * 3]  # Over-retrieve for filtering

        results = []
        for idx in top_indices:
            semantic_sim = float(similarities[idx])
            text_sim = text_similarity(query_text, chunks[idx]["text"])
            sim_type = classify_similarity(semantic_sim, text_sim)

            if sim_type == "NOT_SIMILAR":
                continue

            meta = chunks[idx].get("metadata", {})
            results.append({
                "text": chunks[idx]["text"],
                "similarity_type": sim_type,
                "semantic_similarity": round(semantic_sim, 4),
                "text_similarity": round(text_sim, 4),
                "year": meta.get("year", "Unknown"),
                "source_file": meta.get("source_file", "Unknown"),
                "question_number": meta.get("question_number", "Unknown"),
                "marks": meta.get("marks", "Unknown"),
                "subject": meta.get("subject", "Unknown"),
                "class_level": meta.get("class_level", "Unknown"),
            })

            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _type_rank(sim_type: str) -> int:
        """Numeric rank for similarity types (higher = more similar)."""
        ranks = {
            "EXACT_REPEAT": 4,
            "NEAR_REPEAT": 3,
            "SEMANTICALLY_SIMILAR": 2,
            "RELATED_TOPIC": 1,
            "NOT_SIMILAR": 0,
        }
        return ranks.get(sim_type, 0)

    @staticmethod
    def _type_meets_minimum(actual_type: str, minimum_type: str) -> bool:
        """Check if actual type meets the minimum threshold."""
        ranks = {
            "EXACT_REPEAT": 4,
            "NEAR_REPEAT": 3,
            "SEMANTICALLY_SIMILAR": 2,
            "RELATED_TOPIC": 1,
            "NOT_SIMILAR": 0,
        }
        return ranks.get(actual_type, 0) >= ranks.get(minimum_type, 0)

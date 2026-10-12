"""
keyword_retriever.py - BM25-based keyword retrieval for the JKBOSE RAG system.
Uses rank_bm25 library for TF-IDF-style keyword matching.
Provides complementary retrieval to semantic search.
"""
import re
import logging
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    text = text.lower()
    # Replace common math/science symbols with spaces
    text = re.sub(r"[=+\-*/^(){}[\]<>|\\@#$%&~`\"':;,!?.]", " ", text)
    tokens = text.split()
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


class KeywordRetriever:
    """BM25-based keyword retrieval over question chunks."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict] = []
        self._built = False

    def build(self, chunks: List[Dict]):
        """Build the BM25 index from chunks."""
        self.chunks = chunks
        corpus = [tokenize(chunk.get("text", "")) for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)
        self._built = True
        logger.info(f"BM25 index built with {len(chunks)} documents")

    def search(
        self,
        query: str,
        top_k: int = 20,
        metadata_filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search using BM25 scoring.
        Returns list of dicts with keys: chunk, score, rank.
        """
        if not self._built:
            logger.warning("BM25 index not built")
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Apply metadata filters
        candidate_results = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            if metadata_filters and not self._matches_filters(chunk, metadata_filters):
                continue
            candidate_results.append({"chunk": chunk, "score": float(score), "rank": 0})

        # Sort by score descending
        candidate_results.sort(key=lambda x: x["score"], reverse=True)

        # Assign ranks and trim
        for i, r in enumerate(candidate_results[:top_k]):
            r["rank"] = i + 1

        return candidate_results[:top_k]

    @staticmethod
    def _matches_filters(chunk: Dict, filters: Dict) -> bool:
        """Check if a chunk matches all metadata filters."""
        meta = chunk.get("metadata", {})
        for key, value in filters.items():
            if value is None:
                continue
            chunk_val = meta.get(key)
            if chunk_val is None:
                return False
            if isinstance(value, str) and isinstance(chunk_val, str):
                if value.lower() != chunk_val.lower():
                    return False
            elif isinstance(value, list):
                if isinstance(chunk_val, str):
                    if chunk_val.lower() not in [v.lower() for v in value if isinstance(v, str)]:
                        return False
            else:
                if str(value) != str(chunk_val):
                    return False
        return True

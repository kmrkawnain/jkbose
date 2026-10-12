"""
vector_store.py - In-memory vector store using numpy for cosine similarity search.
Stores chunks with embeddings and metadata. Supports serialization to disk.

Why numpy instead of FAISS/ChromaDB:
- Zero external service dependencies (no Docker, no server)
- The dataset is ~5000-15000 chunks, well within RAM
- L2-normalized embeddings make cosine similarity = dot product (fast with numpy)
- Full control over metadata filtering without adapting to external DB schemas
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

from rag_system.config import INDEX_DIR, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class VectorStore:
    """In-memory vector store with disk persistence."""

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, dim)
        self.chunks: List[Dict] = []                   # parallel list of chunk dicts
        self._built = False

    @property
    def size(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray):
        """Add chunks and their embeddings to the store."""
        if self.embeddings is None:
            self.embeddings = embeddings
            self.chunks = chunks
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.chunks.extend(chunks)
        self._built = True
        logger.info(f"VectorStore now contains {self.size} chunks")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        metadata_filters: Optional[Dict] = None,
        similarity_threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Search for the most similar chunks to the query embedding.
        Applies metadata filters BEFORE ranking.
        
        Returns a list of dicts with keys: chunk, score, rank.
        """
        if not self._built or self.embeddings is None:
            logger.warning("VectorStore is empty - no results")
            return []

        # Apply metadata filters to get candidate indices
        candidate_indices = self._filter_by_metadata(metadata_filters)

        if not candidate_indices:
            return []

        # Compute cosine similarity (embeddings are L2-normalized, so dot product = cosine sim)
        candidate_embeddings = self.embeddings[candidate_indices]
        similarities = np.dot(candidate_embeddings, query_embedding)

        # Apply threshold
        mask = similarities >= similarity_threshold
        filtered_indices = np.array(candidate_indices)[mask]
        filtered_scores = similarities[mask]

        if len(filtered_indices) == 0:
            return []

        # Sort by similarity descending
        sorted_order = np.argsort(-filtered_scores)
        top_indices = sorted_order[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            original_idx = filtered_indices[idx]
            results.append({
                "chunk": self.chunks[original_idx],
                "score": float(filtered_scores[idx]),
                "rank": rank + 1,
            })

        return results

    def get_all_chunks(self, metadata_filters: Optional[Dict] = None) -> List[Dict]:
        """Return all chunks, optionally filtered by metadata."""
        indices = self._filter_by_metadata(metadata_filters)
        return [self.chunks[i] for i in indices]

    def get_all_embeddings_and_chunks(
        self, metadata_filters: Optional[Dict] = None
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Return embeddings and chunks arrays, optionally filtered."""
        indices = self._filter_by_metadata(metadata_filters)
        if not indices:
            return np.array([]), []
        embeddings = self.embeddings[indices]
        chunks = [self.chunks[i] for i in indices]
        return embeddings, chunks

    def _filter_by_metadata(self, filters: Optional[Dict]) -> List[int]:
        """Return indices of chunks matching all metadata filters."""
        if not filters:
            return list(range(len(self.chunks)))

        indices = []
        for i, chunk in enumerate(self.chunks):
            meta = chunk.get("metadata", {})
            match = True
            for key, value in filters.items():
                if value is None:
                    continue
                chunk_val = meta.get(key)
                if chunk_val is None:
                    match = False
                    break
                # Case-insensitive string comparison
                if isinstance(value, str) and isinstance(chunk_val, str):
                    if value.lower() != chunk_val.lower():
                        match = False
                        break
                elif isinstance(value, list):
                    # Support list of allowed values
                    if isinstance(chunk_val, str):
                        if chunk_val.lower() not in [v.lower() for v in value if isinstance(v, str)]:
                            match = False
                            break
                    elif chunk_val not in value:
                        match = False
                        break
                else:
                    if str(value) != str(chunk_val):
                        match = False
                        break
            if match:
                indices.append(i)
        return indices

    def save(self, directory: Optional[str] = None):
        """Persist the vector store to disk."""
        save_dir = directory or INDEX_DIR
        os.makedirs(save_dir, exist_ok=True)

        # Save embeddings
        emb_path = os.path.join(save_dir, "embeddings.npy")
        np.save(emb_path, self.embeddings)

        # Save chunks (metadata only, text is included)
        chunks_path = os.path.join(save_dir, "chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=1)

        logger.info(f"VectorStore saved to {save_dir} ({self.size} chunks)")

    def load(self, directory: Optional[str] = None) -> bool:
        """Load the vector store from disk. Returns True if successful."""
        load_dir = directory or INDEX_DIR
        emb_path = os.path.join(load_dir, "embeddings.npy")
        chunks_path = os.path.join(load_dir, "chunks.json")

        if not os.path.exists(emb_path) or not os.path.exists(chunks_path):
            logger.warning(f"No saved index found at {load_dir}")
            return False

        try:
            self.embeddings = np.load(emb_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self._built = True
            logger.info(f"VectorStore loaded from {load_dir} ({self.size} chunks)")
            return True
        except Exception as e:
            logger.error(f"Failed to load VectorStore: {e}")
            return False

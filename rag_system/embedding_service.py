"""
embedding_service.py - Generates embeddings using Sentence Transformers.
Uses all-MiniLM-L6-v2 for efficient, high-quality semantic embeddings (384-dim).
"""
import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

_model = None  # Lazy-loaded singleton


def get_model():
    """Lazy-load the sentence-transformers model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from rag_system.config import EMBEDDING_MODEL_NAME
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully")
    return _model


def embed_texts(texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    Returns an ndarray of shape (len(texts), EMBEDDING_DIMENSION).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2-normalize for cosine similarity = dot product
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Generate embedding for a single query string."""
    model = get_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding[0]

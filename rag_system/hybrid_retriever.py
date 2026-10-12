"""
hybrid_retriever.py - Combines semantic (vector) and keyword (BM25) retrieval.
Implements reciprocal rank fusion and configurable weight blending.
Includes a cross-encoder-style reranker using embedding similarity refinement.
"""
import logging
import numpy as np
from typing import List, Dict, Optional

from rag_system.config import (
    DEFAULT_TOP_K,
    DEFAULT_TOP_N,
    BM25_WEIGHT,
    SEMANTIC_WEIGHT,
    SIMILARITY_THRESHOLD,
    RERANK_ENABLED,
)
from rag_system.vector_store import VectorStore
from rag_system.keyword_retriever import KeywordRetriever
from rag_system.embedding_service import embed_query

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines vector-based semantic search and BM25 keyword search.
    Uses reciprocal rank fusion to merge results and optional reranking.
    """

    def __init__(self, vector_store: VectorStore, keyword_retriever: KeywordRetriever):
        self.vector_store = vector_store
        self.keyword_retriever = keyword_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        top_n: int = DEFAULT_TOP_N,
        metadata_filters: Optional[Dict] = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        bm25_weight: float = BM25_WEIGHT,
        semantic_weight: float = SEMANTIC_WEIGHT,
        rerank: bool = RERANK_ENABLED,
    ) -> List[Dict]:
        """
        Perform hybrid retrieval:
        1. Semantic search via vector store
        2. BM25 keyword search
        3. Reciprocal rank fusion to combine
        4. Optional reranking
        5. Return top_n results
        """
        query_embedding = embed_query(query)

        # 1. Semantic search
        semantic_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Over-retrieve for fusion
            metadata_filters=metadata_filters,
            similarity_threshold=similarity_threshold * 0.5,  # Looser threshold for initial retrieval
        )

        # 2. BM25 search
        bm25_results = self.keyword_retriever.search(
            query=query,
            top_k=top_k * 2,
            metadata_filters=metadata_filters,
        )

        # 3. Fuse results using Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            semantic_results, bm25_results,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
        )

        # 4. Rerank using the original semantic scores as a proxy
        if rerank and fused:
            fused = self._rerank(fused, query_embedding, top_n)
        else:
            fused = fused[:top_n]

        # 5. Apply final similarity threshold
        fused = [r for r in fused if r["score"] >= similarity_threshold]

        # Update ranks
        for i, r in enumerate(fused):
            r["rank"] = i + 1

        logger.info(
            f"Hybrid retrieval: query='{query[:60]}...' → "
            f"{len(semantic_results)} semantic + {len(bm25_results)} BM25 → "
            f"{len(fused)} final results"
        )

        return fused

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        semantic_weight: float,
        bm25_weight: float,
        k: int = 60,
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) to combine ranked lists.
        RRF score = w_semantic * 1/(k + rank_semantic) + w_bm25 * 1/(k + rank_bm25)
        """
        # Index by chunk_id for deduplication
        chunk_map = {}  # chunk_id -> {chunk, rrf_score, semantic_score}

        for r in semantic_results:
            cid = r["chunk"]["chunk_id"]
            rrf = semantic_weight * (1.0 / (k + r["rank"]))
            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk": r["chunk"],
                    "rrf_score": rrf,
                    "semantic_score": r["score"],
                }
            else:
                chunk_map[cid]["rrf_score"] += rrf
                chunk_map[cid]["semantic_score"] = max(
                    chunk_map[cid]["semantic_score"], r["score"]
                )

        for r in bm25_results:
            cid = r["chunk"]["chunk_id"]
            rrf = bm25_weight * (1.0 / (k + r["rank"]))
            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk": r["chunk"],
                    "rrf_score": rrf,
                    "semantic_score": 0.0,
                }
            else:
                chunk_map[cid]["rrf_score"] += rrf

        # Sort by RRF score
        fused = sorted(chunk_map.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Convert to standard result format
        results = []
        for item in fused:
            results.append({
                "chunk": item["chunk"],
                "score": item["semantic_score"],  # Use semantic score as primary
                "rrf_score": item["rrf_score"],
                "rank": 0,
            })

        return results

    def _rerank(
        self, results: List[Dict], query_embedding: np.ndarray, top_n: int
    ) -> List[Dict]:
        """
        Rerank results using a more refined scoring approach.
        Re-computes cosine similarity with the query for all candidates
        and combines with RRF score.
        """
        from rag_system.embedding_service import embed_texts

        # Get texts for reranking
        texts = [r["chunk"]["text"] for r in results]

        if not texts:
            return results

        # Re-embed all candidate texts
        candidate_embeddings = embed_texts(texts)

        # Compute fresh similarities
        similarities = np.dot(candidate_embeddings, query_embedding)

        for i, r in enumerate(results):
            # Combine: 70% fresh semantic similarity + 30% RRF score (normalized)
            r["score"] = float(similarities[i])
            r["rerank_score"] = 0.7 * float(similarities[i]) + 0.3 * r.get("rrf_score", 0) * 100

        # Sort by rerank score
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return results[:top_n]

"""
rag_engine.py - Main RAG engine that orchestrates all components.
Provides a unified interface for querying the JKBOSE dataset.
"""
import logging
import numpy as np
from typing import Dict, List, Optional

from rag_system.config import (
    DEFAULT_TOP_K,
    DEFAULT_TOP_N,
    SIMILARITY_THRESHOLD,
)
from rag_system.vector_store import VectorStore
from rag_system.keyword_retriever import KeywordRetriever
from rag_system.hybrid_retriever import HybridRetriever
from rag_system.query_analyzer import QueryAnalyzer
from rag_system.repetition_detector import RepetitionDetector
from rag_system.answer_generator import AnswerGenerator, compute_statistics
from rag_system.citation_manager import format_citation

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Main RAG engine that ties all components together.
    Manages the full query → retrieve → generate cycle.
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.keyword_retriever = KeywordRetriever()
        self.hybrid_retriever = None  # Built after loading index
        self.query_analyzer = QueryAnalyzer()
        self.repetition_detector = RepetitionDetector()
        self.answer_generator = AnswerGenerator()
        self._initialized = False

    def initialize(self) -> bool:
        """Load the index and build retrievers. Returns True if successful."""
        # Load vector store from disk
        if not self.vector_store.load():
            logger.error("Failed to load vector store. Run ingest.py first.")
            return False

        # Build BM25 index from loaded chunks
        self.keyword_retriever.build(self.vector_store.chunks)

        # Create hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            self.vector_store, self.keyword_retriever
        )

        self._initialized = True
        logger.info(
            f"RAG Engine initialized with {self.vector_store.size} chunks"
        )
        return True

    def query(
        self,
        user_query: str,
        top_k: int = DEFAULT_TOP_K,
        top_n: int = DEFAULT_TOP_N,
    ) -> Dict:
        """
        Process a user query through the full RAG pipeline.
        
        Returns a structured response dict.
        """
        if not self._initialized:
            return {
                "answer": "System not initialized. Please run the ingestion pipeline first.",
                "questions": [],
                "confidence": "NONE",
                "error": True,
            }

        # 1. Analyze the query
        analysis = self.query_analyzer.analyze(user_query)
        logger.info(
            f"Query analysis: intent={analysis['intent']}, "
            f"filters={analysis['filters']}"
        )

        # 2. Prepare metadata filters (remove None values)
        filters = {k: v for k, v in analysis["filters"].items() if v is not None}

        # Handle "last_N" year filter
        if "year" in filters and isinstance(filters["year"], str) and filters["year"].startswith("last_"):
            try:
                n = int(filters["year"].split("_")[1])
                # Get all unique years and take the last N
                all_years = set()
                for c in self.vector_store.chunks:
                    y = c.get("metadata", {}).get("year")
                    if y and y.isdigit():
                        all_years.add(y)
                sorted_years = sorted(all_years, reverse=True)[:n]
                filters["year"] = sorted_years  # List filter
            except (ValueError, IndexError):
                del filters["year"]

        # 3. Dispatch based on intent
        intent = analysis["intent"]

        repeated_groups = None
        similar_results = None
        stats = None

        if intent == "repeated":
            # Find repeated questions
            embeddings, chunks = self.vector_store.get_all_embeddings_and_chunks(filters if filters else None)
            if len(chunks) > 0:
                repeated_groups = self.repetition_detector.find_repeated_questions(
                    chunks, embeddings,
                    metadata_filters=None,  # Already filtered
                    min_similarity_type="NEAR_REPEAT",
                )

        elif intent == "similar":
            # Find similar questions
            ref = analysis.get("reference_question") or analysis.get("clean_query", "")
            embeddings, chunks = self.vector_store.get_all_embeddings_and_chunks(filters if filters else None)
            if len(chunks) > 0:
                similar_results = self.repetition_detector.find_similar_to_query(
                    ref, chunks, embeddings, top_k=top_n,
                )

        elif intent == "statistics":
            # Compute statistics
            stats = compute_statistics(self.vector_store.chunks, filters if filters else None)

        elif intent == "important":
            # For important questions, find repeated and sort by frequency
            embeddings, chunks = self.vector_store.get_all_embeddings_and_chunks(filters if filters else None)
            if len(chunks) > 0:
                repeated_groups = self.repetition_detector.find_repeated_questions(
                    chunks, embeddings,
                    metadata_filters=None,
                    min_similarity_type="SEMANTICALLY_SIMILAR",
                )

        # 4. Retrieve relevant chunks (for all intents)
        retrieved = self.hybrid_retriever.retrieve(
            query=analysis.get("reference_question") or user_query,
            top_k=top_k,
            top_n=top_n,
            metadata_filters=filters if filters else None,
            similarity_threshold=SIMILARITY_THRESHOLD,
        )

        # 5. Generate answer
        response = self.answer_generator.generate(
            query_analysis=analysis,
            retrieved_results=retrieved,
            repeated_groups=repeated_groups,
            similar_results=similar_results,
            stats=stats,
        )

        # 6. Add analysis metadata to response
        response["query_analysis"] = {
            "intent": analysis["intent"],
            "filters": analysis["filters"],
            "clean_query": analysis["clean_query"],
        }

        # 7. Add similar results to response if applicable
        if similar_results:
            response["similar_results"] = similar_results

        return response

    def get_statistics(self, filters: Optional[Dict] = None) -> Dict:
        """Get dataset statistics."""
        if not self._initialized:
            return {}
        clean_filters = {k: v for k, v in (filters or {}).items() if v is not None}
        return compute_statistics(self.vector_store.chunks, clean_filters if clean_filters else None)

    def reset_conversation(self):
        """Reset conversation context."""
        self.query_analyzer.reset_context()

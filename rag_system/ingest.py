"""
ingest.py - Full ingestion pipeline: load → parse → chunk → embed → index.
Run this script to build the vector store from the JKBOSE dataset.
"""
import os
import sys
import time
import logging

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.config import DATA_DIR, INDEX_DIR
from rag_system.document_loader import load_all_documents
from rag_system.chunker import chunk_document
from rag_system.embedding_service import embed_texts
from rag_system.vector_store import VectorStore
from rag_system.keyword_retriever import KeywordRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


def run_ingestion():
    """Execute the full ingestion pipeline."""
    start_time = time.time()

    # ─── Step 1: Load documents ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading documents...")
    logger.info("=" * 60)
    documents = load_all_documents(DATA_DIR)
    logger.info(f"Loaded {len(documents)} document files")

    # ─── Step 2: Chunk documents ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Chunking documents into questions...")
    logger.info("=" * 60)
    all_chunks = []
    seen_chunk_ids = set()
    skipped_duplicates = 0

    for doc in documents:
        try:
            chunks = chunk_document(doc)
            for chunk in chunks:
                cid = chunk["chunk_id"]
                # Deduplicate: if the same chunk_id appears (from consolidated + individual),
                # prefer the individual file version
                if cid in seen_chunk_ids:
                    skipped_duplicates += 1
                    continue
                seen_chunk_ids.add(cid)
                all_chunks.append(chunk)
        except Exception as e:
            logger.error(f"Failed to chunk {doc['filename']}: {e}")

    logger.info(f"Total chunks: {len(all_chunks)} (skipped {skipped_duplicates} duplicates)")

    if not all_chunks:
        logger.error("No chunks generated! Check the dataset path and format.")
        sys.exit(1)

    # Log metadata distribution
    _log_metadata_stats(all_chunks)

    # ─── Step 3: Generate embeddings ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Generating embeddings...")
    logger.info("=" * 60)
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embed_texts(texts, batch_size=128, show_progress=True)
    logger.info(f"Generated embeddings: shape={embeddings.shape}")

    # ─── Step 4: Build and save vector store ─────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Building vector store...")
    logger.info("=" * 60)
    store = VectorStore()
    store.add_chunks(all_chunks, embeddings)
    store.save()
    logger.info(f"Vector store saved to {INDEX_DIR}")

    # ─── Step 5: Build BM25 index (saved as part of the store) ───────
    logger.info("=" * 60)
    logger.info("STEP 5: Building BM25 keyword index...")
    logger.info("=" * 60)
    kw_retriever = KeywordRetriever()
    kw_retriever.build(all_chunks)
    logger.info("BM25 index built (in-memory, rebuilt on server start)")

    # ─── Done ────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"INGESTION COMPLETE in {elapsed:.1f} seconds")
    logger.info(f"  Total documents loaded: {len(documents)}")
    logger.info(f"  Total chunks indexed:   {len(all_chunks)}")
    logger.info(f"  Embedding dimensions:   {embeddings.shape[1]}")
    logger.info(f"  Index saved to:         {INDEX_DIR}")
    logger.info("=" * 60)


def _log_metadata_stats(chunks):
    """Log distribution statistics for indexed chunks."""
    from collections import Counter

    classes = Counter()
    subjects = Counter()
    years = Counter()
    marks = Counter()

    for c in chunks:
        meta = c.get("metadata", {})
        cl = meta.get("class_level")
        if cl:
            classes[f"Class {cl}"] += 1
        s = meta.get("subject")
        if s:
            subjects[s] += 1
        y = meta.get("year")
        if y:
            years[str(y)] += 1
        m = meta.get("marks")
        if m:
            marks[f"{m}-mark"] += 1

    logger.info("--- Metadata Distribution ---")
    logger.info(f"  Classes: {dict(classes.most_common())}")
    logger.info(f"  Subjects: {dict(subjects.most_common())}")
    logger.info(f"  Years: {dict(sorted(years.items())[:10])} ... ({len(years)} unique years)")
    logger.info(f"  Marks: {dict(marks.most_common())}")


if __name__ == "__main__":
    run_ingestion()

"""
Central configuration for the JKBOSE RAG system.
All tunable parameters are defined here for consistency.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = PROJECT_ROOT  # root of jkbose folder
INDEX_DIR = os.path.join(BASE_DIR, "index_store")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Subdirectories containing question papers
CLASS_DIRS = {
    "10": os.path.join(DATA_DIR, "class_10th"),
    "11": os.path.join(DATA_DIR, "class_11th"),
    "12": os.path.join(DATA_DIR, "class_12th"),
}

# ─── Embedding Model ────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Matches all-MiniLM-L6-v2

# ─── Retrieval Parameters ───────────────────────────────────────────
DEFAULT_TOP_K = 20          # Initial retrieval pool size
DEFAULT_TOP_N = 10          # After reranking
SIMILARITY_THRESHOLD = 0.25  # Minimum cosine similarity to consider relevant
BM25_WEIGHT = 0.35           # Weight for BM25 in hybrid scoring
SEMANTIC_WEIGHT = 0.65       # Weight for semantic search in hybrid scoring

# ─── Reranking ───────────────────────────────────────────────────────
RERANK_ENABLED = True

# ─── Repeated Question Detection ────────────────────────────────────
REPEAT_EXACT_THRESHOLD = 0.96       # Cosine similarity for EXACT_REPEAT
REPEAT_NEAR_THRESHOLD = 0.88        # For NEAR_REPEAT
REPEAT_SEMANTIC_THRESHOLD = 0.78    # For SEMANTICALLY_SIMILAR
REPEAT_RELATED_THRESHOLD = 0.65     # For RELATED_TOPIC
# Below REPEAT_RELATED_THRESHOLD → NOT_SIMILAR

# Text similarity thresholds (normalized Levenshtein-based)
TEXT_EXACT_THRESHOLD = 0.92
TEXT_NEAR_THRESHOLD = 0.75

# ─── Server ──────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# ─── Metadata Fields ────────────────────────────────────────────────
METADATA_FIELDS = [
    "source_file",
    "page",
    "class_level",
    "subject",
    "year",
    "exam_type",
    "paper_code",
    "set_code",
    "section",
    "question_number",
    "marks",
    "question_type",
    "topic",
    "max_marks",
    "time_allowed",
]

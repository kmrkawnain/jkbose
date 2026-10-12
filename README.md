# JKBOSE RAG System - Question Paper Intelligence

A fully functional, self-consistent Retrieval-Augmented Generation (RAG) system built exclusively for the JKBOSE question paper dataset.

## Architecture

*   **Ingestion Pipeline:** Loads `.txt` question papers, intelligently chunks by question boundary, extracts metadata, and generates embeddings.
*   **Vector Store:** Custom Numpy-based in-memory vector store for fast semantic search (L2 normalized cosine similarity) without external database dependencies.
*   **Keyword Retriever:** BM25 (TF-IDF) indexing for lexical search.
*   **Hybrid Retriever:** Combines Semantic and BM25 results using Reciprocal Rank Fusion (RRF), followed by a re-ranking step.
*   **Query Analyzer:** Parses natural language and slash commands to extract filters (Class, Subject, Year, Marks) and detect intent (Search, Repeated, Statistics).
*   **Repetition Detector:** Uses semantic matrix dot-products and normalized text trigram Jaccard similarity to identify EXACT_REPEAT, NEAR_REPEAT, and SEMANTICALLY_SIMILAR questions.
*   **Answer Generator:** Formats strict, grounded responses with proper citations and hallucination prevention.
*   **Student AI Chatbot:** Interactive multi-turn study mentor offering model exam answers tailored to JKBOSE marking schemes (1M–5M), intuitive conceptual breakdowns, past-paper frequency trends, practice quizzes, and local Ollama LLM integration.
*   **Web Frontend:** Modern Glassmorphism UI built with vanilla JS, CSS, and HTML featuring dual-view navigation (Student AI Chatbot & Paper Explorer), voice recognition, and text-to-speech.

## Folder Structure

```
jkbose/
├── rag_system/
│   ├── __init__.py
│   ├── config.py                # Central tuning parameters
│   ├── chatbot.py               # Student AI Chatbot & Exam Answer Engine
│   ├── document_loader.py       # Reads TXT files
│   ├── metadata_extractor.py    # Regex parsing of filenames and headers
│   ├── chunker.py               # Splits papers into Q-blocks preserving OR logic
│   ├── embedding_service.py     # SentenceTransformers all-MiniLM-L6-v2
│   ├── vector_store.py          # Numpy similarity search
│   ├── keyword_retriever.py     # BM25 lexical search
│   ├── hybrid_retriever.py      # RRF fusion and reranking
│   ├── query_analyzer.py        # NLP filter and intent extraction
│   ├── repetition_detector.py   # Similarity matrix and union-find clustering
│   ├── citation_manager.py      # Citation generation
│   ├── answer_generator.py      # Response formatting & grounding
│   ├── ingest.py                # Script to build the index
│   ├── rag_engine.py            # Main RAG orchestrator
│   ├── server.py                # Flask REST API
│   └── evaluation.py            # Evaluation testing framework
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── index_store/                 # Generated embeddings and metadata (created on ingest)
├── class_10th/                  # Dataset files
├── class_11th/                  # Dataset files
└── class_12th/                  # Dataset files
```

## Setup Instructions

1.  **Install Dependencies:**
    The system requires Python 3.8+ and the following packages:
    `pip install flask flask-cors numpy scikit-learn sentence-transformers rank-bm25`

2.  **Build the Index:**
    Before querying, the system must parse the dataset and generate embeddings.
    Run from the root `jkbose` directory:
    ```bash
    python -m rag_system.ingest
    ```
    This takes a few minutes and creates `index_store/embeddings.npy` and `index_store/chunks.json`.

3.  **Start the Server:**
    Run the Flask backend which also serves the frontend UI.
    ```bash
    python rag_system/server.py
    ```

4.  **Access the UI:**
    Open a web browser and navigate to `http://localhost:5000`

5.  **Run Evaluation:**
    ```bash
    python -m rag_system.evaluation
    ```

## Example Queries

*   `Show me 5 mark questions from Class 12 Physics 2024`
*   `Find repeated questions in Class 12 Chemistry`
*   `/important Class 10 Science`
*   `Which topics appear most frequently in Mathematics?`
*   `Find questions similar to Gauss's law`

## Known Limitations

*   **Metadata Consistency:** Older question papers have inconsistent structures, which might lead to some missing metadata (e.g., marks or topic).
*   **Embedding Model:** Uses `all-MiniLM-L6-v2` which is fast and compact but might miss complex physics/math notation nuances compared to larger models.
*   **Memory:** The vector store is entirely in RAM (Numpy arrays). This works well for the current JKBOSE dataset (~10,000 to 20,000 chunks) but would require transitioning to Qdrant or ChromaDB if scaling to millions of questions.

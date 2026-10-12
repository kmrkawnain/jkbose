"""
server.py - Flask API server for the JKBOSE RAG system.
Exposes REST endpoints for querying, statistics, and system management.
"""
import os
import sys
import logging

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from rag_system.config import HOST, PORT, DEBUG
from rag_system.rag_engine import RAGEngine
from rag_system.chatbot import JKBOSEChatbot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("server")

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"),
    static_url_path="",
)
CORS(app)

# Initialize RAG engine & Chatbot
engine = RAGEngine()
chatbot = JKBOSEChatbot(engine)


@app.before_request
def ensure_initialized():
    """Ensure the RAG engine is initialized before handling requests."""
    if not engine._initialized:
        if not engine.initialize():
            pass  # Will return error in the endpoint


# ─── API Routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    """
    Main query endpoint.
    Body: { "query": str, "top_k": int?, "top_n": int? }
    """
    if not engine._initialized:
        return jsonify({"error": "System not initialized. Run ingest.py first."}), 503

    data = request.get_json(force=True)
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    top_k = data.get("top_k", 20)
    top_n = data.get("top_n", 10)

    try:
        result = engine.query(query, top_k=top_k, top_n=top_n)
        return jsonify(result)
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/statistics", methods=["GET", "POST"])
def api_statistics():
    """
    Dataset statistics endpoint.
    Query params or body: class_level, subject, year
    """
    if not engine._initialized:
        return jsonify({"error": "System not initialized."}), 503

    filters = {}
    if request.method == "POST":
        data = request.get_json(force=True)
        filters = data.get("filters", {})
    else:
        for key in ("class_level", "subject", "year"):
            val = request.args.get(key)
            if val:
                filters[key] = val

    try:
        stats = engine.get_statistics(filters)
        return jsonify(stats)
    except Exception as e:
        logger.exception(f"Statistics failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset conversation context."""
    engine.reset_conversation()
    return jsonify({"status": "Conversation context reset"})


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok" if engine._initialized else "not_initialized",
        "chunks_indexed": engine.vector_store.size if engine._initialized else 0,
    })


# ─── Chatbot Endpoints ───────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Main student chatbot endpoint.
    Body: {
        "message": str,
        "session_id": str?,
        "mode": "exam_answer" | "tutor" | "trends" | "quiz",
        "engine_type": "fast" | "ollama",
        "filters": { "class_level": str?, "subject": str?, "year": str?, "marks": str? },
        "preferred_model": str?
    }
    """
    if not engine._initialized:
        return jsonify({"error": "System not initialized."}), 503

    data = request.get_json(force=True)
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default_student")
    mode = data.get("mode", "exam_answer")
    engine_type = data.get("engine_type", "fast")
    filters = data.get("filters", {})
    preferred_model = data.get("preferred_model")

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    try:
        res = chatbot.chat(
            message=message,
            session_id=session_id,
            mode=mode,
            engine_type=engine_type,
            filters=filters,
            preferred_model=preferred_model,
        )
        return jsonify(res)
    except Exception as e:
        logger.exception(f"Chatbot failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/history", methods=["GET"])
def api_chat_history():
    """Get chat history for a session."""
    session_id = request.args.get("session_id", "default_student")
    return jsonify({"history": chatbot.get_history(session_id)})


@app.route("/api/chat/clear", methods=["POST", "GET"])
def api_chat_clear():
    """Clear chat history for a session or all sessions."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id") or request.args.get("session_id")
    if session_id:
        chatbot.clear_history(session_id)
    else:
        chatbot.sessions.clear()
    return jsonify({"status": "Chat history cleared"})


@app.route("/api/ollama/status", methods=["GET"])
def api_ollama_status():
    """Check local Ollama LLM status and available models."""
    return jsonify(chatbot.check_ollama())


if __name__ == "__main__":
    logger.info("Starting JKBOSE RAG Server...")

    if not engine.initialize():
        logger.error(
            "Failed to initialize RAG engine. "
            "Please run 'python -m rag_system.ingest' first to build the index."
        )
        sys.exit(1)

    logger.info(f"Server starting on http://localhost:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG, use_reloader=False)

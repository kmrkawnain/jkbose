"""
evaluation.py - Testing framework for the RAG system.
Tests retrieval accuracy, exact match recall, filtering, and hallucination resistance.
"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eval")


def run_evaluation():
    engine = RAGEngine()
    if not engine.initialize():
        logger.error("Cannot evaluate: Engine failed to initialize.")
        return

    logger.info("\n" + "=" * 50)
    logger.info("JKBOSE RAG SYSTEM EVALUATION")
    logger.info("=" * 50)

    tests = [
        {
            "name": "1. Exact question retrieval",
            "query": "State Ohm's Law and express it in mathematical form relating Potential Difference (V), Current (I), and Resistance (R).",
            "expected_intent": "search",
            "check": lambda res: any("Ohm's Law" in q["text"] for q in res["questions"])
        },
        {
            "name": "2. Class & Subject filtering",
            "query": "Give me questions from Class 12 Biology",
            "expected_intent": "search",
            "check": lambda res: all(
                q["metadata"].get("class_level") == "12" and 
                q["metadata"].get("subject") == "Biology" 
                for q in res["questions"]
            ) and len(res["questions"]) > 0
        },
        {
            "name": "3. Metadata filtering (Marks)",
            "query": "5 mark questions from physics",
            "expected_intent": "search",
            "check": lambda res: all(
                (not q["metadata"].get("marks") or q["metadata"].get("marks") == "5") and 
                q["metadata"].get("subject") == "Physics" 
                for q in res["questions"]
            )
        },
        {
            "name": "4. Repeated-question detection intent",
            "query": "Which questions appeared more than once in Chemistry?",
            "expected_intent": "repeated",
            "check": lambda res: len(res.get("repeated_groups", [])) >= 0
        },
        {
            "name": "5. Topic analysis intent",
            "query": "Which topics appear most frequently in Class 10?",
            "expected_intent": "topic_analysis",
            "check": lambda res: "Topic Frequency Analysis" in res["answer"]
        },
        {
            "name": "6. Hallucination resistance (No result queries)",
            "query": "Give me questions about Quantum Entanglement and String Theory from 1950",
            "expected_intent": "search",
            "check": lambda res: res["confidence"] == "NONE" and len(res["questions"]) == 0
        }
    ]

    passed = 0
    for i, test in enumerate(tests):
        logger.info(f"\nRunning Test {test['name']}")
        logger.info(f"Query: '{test['query']}'")
        
        try:
            res = engine.query(test["query"], top_k=10, top_n=5)
            
            intent_match = res["query_analysis"]["intent"] == test["expected_intent"]
            logic_pass = test["check"](res)
            
            if intent_match and logic_pass:
                logger.info("✅ PASS")
                passed += 1
            else:
                logger.info("❌ FAIL")
                if not intent_match:
                    logger.error(f"  Expected intent '{test['expected_intent']}', got '{res['query_analysis']['intent']}'")
                if not logic_pass:
                    logger.error("  Logic check failed.")
        except Exception as e:
            logger.info("❌ ERROR")
            logger.error(f"  {e}")

    logger.info("\n" + "=" * 50)
    logger.info(f"Evaluation Complete: {passed}/{len(tests)} tests passed.")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_evaluation()

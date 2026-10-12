"""
query_analyzer.py - Understands user queries and extracts structured filters.
Handles natural language queries, slash commands, and contextual follow-ups.
"""
import re
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

# ─── Subject aliases ─────────────────────────────────────────────────
SUBJECT_ALIASES = {
    "physics": "Physics",
    "phy": "Physics",
    "phys": "Physics",
    "chemistry": "Chemistry",
    "chem": "Chemistry",
    "biology": "Biology",
    "bio": "Biology",
    "botany": "Biology",
    "zoology": "Biology",
    "mathematics": "Mathematics",
    "math": "Mathematics",
    "maths": "Mathematics",
    "science": "Science",
    "environmental science": "Environmental Science",
    "env science": "Environmental Science",
    "evs": "Environmental Science",
    "environmental": "Environmental Science",
}

# ─── Intent detection patterns ───────────────────────────────────────
INTENT_PATTERNS = {
    "repeated": re.compile(
        r"\b(repeat(?:ed)?|recurring|appeared\s+(?:more\s+than\s+once|multiple|again|frequently))\b",
        re.IGNORECASE,
    ),
    "similar": re.compile(
        r"\b(similar|like|resembl|paraphras)\b", re.IGNORECASE
    ),
    "important": re.compile(
        r"\b(important|most\s+asked|frequent|common|high[\s-]?yield)\b",
        re.IGNORECASE,
    ),
    "statistics": re.compile(
        r"\b(statistic|how\s+many|count|total|number\s+of|frequenc)\b",
        re.IGNORECASE,
    ),
    "topic_analysis": re.compile(
        r"\b(topic|chapter|unit)\b.*\b(most|frequent|highest|appear)\b",
        re.IGNORECASE,
    ),
    "search": re.compile(
        r"\b(find|search|show|give|list|get|retrieve|display|provide)\b",
        re.IGNORECASE,
    ),
    "explain": re.compile(
        r"\b(explain|describe|elaborate|detail|what\s+is|define)\b",
        re.IGNORECASE,
    ),
}

# ─── Slash command mapping ───────────────────────────────────────────
SLASH_COMMANDS = {
    "/search": "search",
    "/find": "search",
    "/repeated": "repeated",
    "/similar": "similar",
    "/topic": "topic_analysis",
    "/year": "search",
    "/important": "important",
    "/explain": "explain",
    "/statistics": "statistics",
    "/stats": "statistics",
}


class QueryAnalyzer:
    """Analyzes user queries and extracts structured intent + filters."""

    def __init__(self):
        self._conversation_context: Dict = {}

    def analyze(self, query: str) -> Dict:
        """
        Analyze a query and return structured analysis.
        Returns:
            {
                "original_query": str,
                "clean_query": str,         # query without slash commands/filters
                "intent": str,              # detected intent
                "filters": {                # metadata filters
                    "class_level": str|None,
                    "subject": str|None,
                    "year": str|None,
                    "marks": str|None,
                    "question_type": str|None,
                    "paper_code": str|None,
                    "set_code": str|None,
                },
                "requested_count": int|None, # "give me 5 questions"
                "reference_question": str|None, # for similarity search
            }
        """
        result = {
            "original_query": query,
            "clean_query": query,
            "intent": "search",
            "filters": {},
            "requested_count": None,
            "reference_question": None,
        }

        # 1. Handle slash commands
        query, slash_intent = self._extract_slash_command(query)
        if slash_intent:
            result["intent"] = slash_intent

        # 2. Extract class level
        class_level = self._extract_class(query)
        if class_level:
            result["filters"]["class_level"] = class_level

        # 3. Extract subject
        subject = self._extract_subject(query)
        if subject:
            result["filters"]["subject"] = subject

        # 4. Extract year
        year = self._extract_year(query)
        if year:
            result["filters"]["year"] = year

        # 5. Extract marks
        marks = self._extract_marks(query)
        if marks:
            result["filters"]["marks"] = marks

        # 6. Extract question type
        q_type = self._extract_question_type(query)
        if q_type:
            result["filters"]["question_type"] = q_type

        # 7. Extract paper code
        paper_code = self._extract_paper_code(query)
        if paper_code:
            result["filters"]["paper_code"] = paper_code

        # 8. Extract requested count
        count = self._extract_count(query)
        if count:
            result["requested_count"] = count

        # 9. Detect intent from patterns (if not already set by slash command)
        if not slash_intent:
            result["intent"] = self._detect_intent(query)

        # 10. Extract reference question for similarity
        ref = self._extract_reference_question(query)
        if ref:
            result["reference_question"] = ref

        # 11. Clean the query (remove filter tokens)
        result["clean_query"] = self._clean_query(query)

        # 12. Apply conversation context (inherit previous filters if not overridden)
        result = self._apply_context(result)

        # 13. Update conversation context
        self._update_context(result)

        return result

    def reset_context(self):
        """Clear conversation context."""
        self._conversation_context = {}

    def _extract_slash_command(self, query: str) -> Tuple[str, Optional[str]]:
        """Extract slash command if present."""
        for cmd, intent in SLASH_COMMANDS.items():
            if query.strip().lower().startswith(cmd):
                clean = query.strip()[len(cmd):].strip()
                return clean, intent
        return query, None

    def _extract_class(self, query: str) -> Optional[str]:
        """Extract class level (10, 11, 12)."""
        patterns = [
            r"\bclass\s*(\d{1,2})(?:th)?\b",
            r"\b(\d{1,2})(?:th)?\s*class\b",
            r"\b(\d{1,2})th\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                level = m.group(1)
                if level in ("10", "11", "12"):
                    return level
        return None

    def _extract_subject(self, query: str) -> Optional[str]:
        """Extract subject from query."""
        query_lower = query.lower()
        # Check multi-word subjects first
        if "environmental science" in query_lower or "env science" in query_lower:
            return "Environmental Science"
        for alias, subject in SUBJECT_ALIASES.items():
            # Use word boundary matching
            if re.search(r"\b" + re.escape(alias) + r"\b", query_lower):
                return subject
        return None

    def _extract_year(self, query: str) -> Optional[str]:
        """Extract year from query."""
        # Match 4-digit years in reasonable range
        m = re.search(r"\b(19\d{2}|20[0-3]\d)\b", query)
        if m:
            return m.group(1)
        # Match "last N years" pattern
        m = re.search(r"last\s+(\d+)\s+years?", query, re.IGNORECASE)
        if m:
            # This returns a special marker that the retriever can interpret
            return f"last_{m.group(1)}"
        return None

    def _extract_marks(self, query: str) -> Optional[str]:
        """Extract marks filter."""
        patterns = [
            r"(\d+)\s*[-–]?\s*marks?\b",
            r"\bmarks?\s*[-–:]?\s*(\d+)\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                marks = m.group(1)
                if marks in ("1", "2", "3", "4", "5"):
                    return marks
        return None

    def _extract_question_type(self, query: str) -> Optional[str]:
        """Extract question type filter."""
        query_lower = query.lower()
        if "mcq" in query_lower or "multiple choice" in query_lower:
            return "MCQ"
        if "fill in" in query_lower or "blank" in query_lower:
            return "Fill in the Blank"
        if "long answer" in query_lower or "long question" in query_lower:
            return "Long Answer"
        if "short answer" in query_lower or "short question" in query_lower:
            return "Short Answer"  # Will match both Type I and II
        if "numerical" in query_lower or "problem" in query_lower:
            return "numerical"  # Special handling in retriever
        return None

    def _extract_paper_code(self, query: str) -> Optional[str]:
        """Extract specific paper code."""
        m = re.search(r"\bcode\s*[-:]?\s*(\d{3,6})\b", query, re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    def _extract_count(self, query: str) -> Optional[int]:
        """Extract requested number of results."""
        patterns = [
            r"\b(?:give|show|list|find|get)\s+(?:me\s+)?(\d+)\b",
            r"\btop\s+(\d+)\b",
            r"\b(\d+)\s+questions?\b",
        ]
        for pattern in patterns:
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                count = int(m.group(1))
                if 1 <= count <= 100:
                    return count
        return None

    def _detect_intent(self, query: str) -> str:
        """Detect query intent from patterns."""
        for intent, pattern in INTENT_PATTERNS.items():
            if pattern.search(query):
                return intent
        return "search"

    def _extract_reference_question(self, query: str) -> Optional[str]:
        """Extract a reference question for similarity search."""
        patterns = [
            r"similar\s+to\s*[:\"]?\s*(.{15,}?)(?:\"|$)",
            r"like\s*[:\"]?\s*(.{15,}?)(?:\"|$)",
            r"questions?\s+(?:about|on|related\s+to)\s+(.{5,}?)(?:\?|$)",
        ]
        for pattern in patterns:
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip("\"'.,")
        return None

    def _clean_query(self, query: str) -> str:
        """Remove filter-related tokens from query to get the search core."""
        # Remove slash commands
        for cmd in SLASH_COMMANDS:
            query = re.sub(re.escape(cmd), "", query, flags=re.IGNORECASE)
        # Remove class/year/marks references (they're already in filters)
        query = re.sub(r"\bclass\s*\d{1,2}(?:th)?\b", "", query, flags=re.IGNORECASE)
        query = re.sub(r"\b\d{4}\b", "", query)
        query = re.sub(r"\b\d+\s*[-–]?\s*marks?\b", "", query, flags=re.IGNORECASE)
        # Remove common request words
        query = re.sub(
            r"\b(give|show|find|list|get|me|all|the|from|of|in|questions?|papers?)\b",
            "", query, flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", query).strip()

    def _apply_context(self, result: Dict) -> Dict:
        """Inherit filters from conversation context if not overridden."""
        for key in ("class_level", "subject", "year"):
            if key not in result["filters"] or result["filters"][key] is None:
                if key in self._conversation_context:
                    result["filters"][key] = self._conversation_context[key]
        return result

    def _update_context(self, result: Dict):
        """Update conversation context with current query's filters."""
        for key, value in result["filters"].items():
            if value is not None:
                self._conversation_context[key] = value

"""
answer_generator.py - Generates structured answers from retrieved context.
Implements strict grounding rules and hallucination prevention.
"""
import logging
from typing import Dict, List, Optional
from collections import Counter, defaultdict

from rag_system.citation_manager import format_citation, build_cited_response
from rag_system.config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates grounded answers from retrieved chunks."""

    def generate(
        self,
        query_analysis: Dict,
        retrieved_results: List[Dict],
        repeated_groups: Optional[List[Dict]] = None,
        similar_results: Optional[List[Dict]] = None,
        stats: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate a structured answer based on intent and retrieved context.
        
        Returns:
            {
                "answer": str,               # Main answer text
                "questions": List[Dict],      # Retrieved questions with citations
                "repeated_groups": List[Dict], # Repeated question groups
                "statistics": Dict,           # Statistics if requested
                "confidence": str,            # HIGH, MEDIUM, LOW, NONE
                "source_count": int,          # Number of source documents used
                "warning": str|None,          # Hallucination prevention warnings
            }
        """
        intent = query_analysis.get("intent", "search")
        filters = query_analysis.get("filters", {})
        requested_count = query_analysis.get("requested_count")

        # Build cited results
        cited = build_cited_response(retrieved_results)

        # Apply count limit
        if requested_count and cited:
            cited = cited[:requested_count]

        # Determine confidence
        confidence = self._assess_confidence(cited, retrieved_results)

        # Build response based on intent
        response = {
            "answer": "",
            "questions": cited,
            "repeated_groups": repeated_groups or [],
            "statistics": stats or {},
            "confidence": confidence,
            "source_count": len(set(
                r.get("metadata", {}).get("source_file", "") for r in cited
            )),
            "warning": None,
        }

        if confidence == "NONE":
            response["answer"] = (
                "I could not find sufficiently relevant information in the "
                "uploaded JKBOSE dataset for your query."
            )
            response["warning"] = "No relevant results found in the dataset."
            return response

        # Generate intent-specific answer
        if intent == "repeated":
            response["answer"] = self._format_repeated_answer(
                repeated_groups or [], filters
            )
        elif intent == "similar":
            response["answer"] = self._format_similar_answer(
                similar_results or [], query_analysis
            )
        elif intent == "important":
            response["answer"] = self._format_important_answer(cited, filters)
        elif intent == "statistics":
            response["answer"] = self._format_statistics_answer(stats or {}, filters)
        elif intent == "topic_analysis":
            response["answer"] = self._format_topic_answer(cited, filters)
        else:
            response["answer"] = self._format_search_answer(
                cited, filters, query_analysis
            )

        if confidence == "LOW":
            response["warning"] = (
                "Low confidence results. The retrieved information may not "
                "precisely match your query."
            )

        return response

    def _assess_confidence(self, cited: List[Dict], raw_results: List[Dict]) -> str:
        """Assess answer confidence based on retrieval quality."""
        if not cited and not raw_results:
            return "NONE"
        if not cited:
            return "NONE"

        top_score = cited[0].get("score", 0) if cited else 0

        if top_score >= 0.7:
            return "HIGH"
        elif top_score >= 0.45:
            return "MEDIUM"
        elif top_score >= SIMILARITY_THRESHOLD:
            return "LOW"
        return "NONE"

    def _format_search_answer(
        self, cited: List[Dict], filters: Dict, analysis: Dict
    ) -> str:
        """Format a standard search result."""
        lines = []
        filter_desc = self._describe_filters(filters)
        if filter_desc:
            lines.append(f"**Results for: {filter_desc}**\n")

        lines.append(f"Found **{len(cited)}** relevant question(s) from the JKBOSE dataset.\n")
        return "\n".join(lines)

    def _format_repeated_answer(
        self, groups: List[Dict], filters: Dict
    ) -> str:
        """Format repeated question detection results."""
        if not groups:
            return (
                "No repeated questions found matching your criteria in the "
                "uploaded JKBOSE dataset."
            )

        lines = []
        filter_desc = self._describe_filters(filters)
        lines.append(
            f"**Dataset-Based Repeated Question Analysis**"
            + (f" ({filter_desc})" if filter_desc else "")
        )
        lines.append(
            f"\nFound **{len(groups)}** group(s) of repeated/similar questions.\n"
        )
        lines.append(
            "*Note: This analysis is based on semantic and textual similarity "
            "detection across the uploaded papers. Questions classified as "
            "repeated appeared in multiple examination papers.*"
        )
        return "\n".join(lines)

    def _format_similar_answer(
        self, results: List[Dict], analysis: Dict
    ) -> str:
        """Format similar question search results."""
        ref = analysis.get("reference_question", analysis.get("clean_query", ""))
        if not results:
            return f'No similar questions found for: "{ref}"'

        lines = [
            f'**Questions similar to:** "{ref[:100]}..."' if len(ref) > 100 else f'**Questions similar to:** "{ref}"',
            f"\nFound **{len(results)}** similar question(s).\n",
        ]
        return "\n".join(lines)

    def _format_important_answer(self, cited: List[Dict], filters: Dict) -> str:
        """Format important questions analysis."""
        lines = [
            "**Dataset-Based Frequency Analysis (Important Questions)**\n",
            "*Disclaimer: These questions are identified as 'important' based on "
            "their frequency of appearance across multiple examination papers in "
            "the uploaded dataset. This is a data-driven analysis and does NOT "
            "constitute an official JKBOSE recommendation. There is no guarantee "
            "that these questions will appear in future examinations.*\n",
        ]
        filter_desc = self._describe_filters(filters)
        if filter_desc:
            lines.append(f"**Filters applied:** {filter_desc}\n")

        lines.append(f"Showing **{len(cited)}** frequently appearing question(s).\n")
        return "\n".join(lines)

    def _format_statistics_answer(self, stats: Dict, filters: Dict) -> str:
        """Format statistical analysis results."""
        if not stats:
            return "No statistics available for the given criteria."

        lines = ["**Dataset Statistics**\n"]
        filter_desc = self._describe_filters(filters)
        if filter_desc:
            lines.append(f"**Filters:** {filter_desc}\n")

        for key, value in stats.items():
            lines.append(f"- **{key}:** {value}")
        return "\n".join(lines)

    def _format_topic_answer(self, cited: List[Dict], filters: Dict) -> str:
        """Format topic analysis results."""
        if not cited:
            return "No topic analysis available for the given criteria."

        # Count topics
        topic_counts = Counter()
        for r in cited:
            topic = r.get("metadata", {}).get("topic", "Unknown")
            if topic and topic != "Unknown":
                topic_counts[topic] += 1

        lines = ["**Topic Frequency Analysis**\n"]
        for topic, count in topic_counts.most_common(20):
            lines.append(f"- **{topic}**: {count} question(s)")

        return "\n".join(lines)

    @staticmethod
    def _describe_filters(filters: Dict) -> str:
        """Human-readable description of active filters."""
        parts = []
        if filters.get("class_level"):
            parts.append(f"Class {filters['class_level']}")
        if filters.get("subject"):
            parts.append(filters["subject"])
        if filters.get("year"):
            parts.append(f"Year {filters['year']}")
        if filters.get("marks"):
            parts.append(f"{filters['marks']}-mark")
        if filters.get("question_type"):
            parts.append(filters["question_type"])
        if filters.get("paper_code"):
            parts.append(f"Code {filters['paper_code']}")
        return ", ".join(parts)


def compute_statistics(chunks: List[Dict], filters: Optional[Dict] = None) -> Dict:
    """Compute dataset statistics from chunks."""
    # Filter chunks
    filtered = chunks
    if filters:
        filtered = []
        for c in chunks:
            meta = c.get("metadata", {})
            match = True
            for k, v in filters.items():
                if v is None:
                    continue
                cv = meta.get(k)
                if cv is None:
                    match = False
                    break
                if isinstance(v, str) and isinstance(cv, str) and v.lower() != cv.lower():
                    match = False
                    break
            if match:
                filtered.append(c)

    if not filtered:
        return {"total_questions": 0}

    # Gather stats
    years = set()
    subjects = Counter()
    marks_dist = Counter()
    question_types = Counter()
    topics = Counter()
    papers = set()

    for c in filtered:
        meta = c.get("metadata", {})
        y = meta.get("year")
        if y:
            years.add(str(y))
        s = meta.get("subject")
        if s:
            subjects[s] += 1
        m = meta.get("marks")
        if m:
            marks_dist[f"{m}-mark"] += 1
        qt = meta.get("question_type")
        if qt:
            question_types[qt] += 1
        t = meta.get("topic")
        if t and t not in ("Unknown", None):
            topics[t] += 1
        sf = meta.get("source_file")
        if sf:
            papers.add(sf)

    stats = {
        "total_questions": len(filtered),
        "total_papers": len(papers),
        "years_covered": sorted(years),
        "year_range": f"{min(years)} - {max(years)}" if years else "N/A",
        "subjects": dict(subjects.most_common()),
        "marks_distribution": dict(marks_dist.most_common()),
        "question_types": dict(question_types.most_common()),
        "top_topics": dict(topics.most_common(15)),
    }

    return stats

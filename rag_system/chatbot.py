"""
chatbot.py - Intelligent Exam & Study Chatbot for JKBOSE Students.
Provides grounded exam answers, concept explanations, marking scheme breakdowns,
past-year trend analysis, and interactive revision quizzes using the JKBOSE dataset.
"""

import json
import logging
import re
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_MODEL = "qwen2:1.5b"


class JKBOSEChatbot:
    """
    Intelligent student chatbot for JKBOSE examination preparation.
    Integrates with RAGEngine for grounding and supports both fast expert synthesis
    and local Ollama LLM generation.
    """

    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.sessions.get(session_id, [])

    def clear_history(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id] = []

    def check_ollama(self) -> Dict[str, Any]:
        """Check if local Ollama server is running and which models are available."""
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    models = [m.get("name", "") for m in data.get("models", [])]
                    preferred = "qwen3:0.6b" if "qwen3:0.6b" in models else ("qwen2:1.5b" if "qwen2:1.5b" in models else (models[0] if models else None))
                    return {"available": True, "models": models, "active_model": preferred}
        except Exception as e:
            logger.debug(f"Ollama check failed: {e}")
        return {"available": False, "models": [], "active_model": None}

    def chat(
        self,
        message: str,
        session_id: str = "default",
        mode: str = "exam_answer",      # exam_answer | tutor | trends | quiz
        engine_type: str = "fast",       # fast | ollama
        filters: Optional[Dict] = None,
        preferred_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a student chat message and return a comprehensive, exam-ready response.
        """
        message = message.strip()
        if not message:
            return {
                "reply": "Please ask a question or enter a topic from your JKBOSE syllabus!",
                "sources": [],
                "suggested_followups": ["State Ohm's Law", "Important Class 12 Physics questions", "Class 10 Light chapter"],
                "mode": mode,
            }

        # Initialize session history
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        # 1. Retrieve relevant questions from the JKBOSE dataset
        retrieval_query = message
        # Augment retrieval query if pronoun / follow-up detected
        history = self.sessions[session_id]
        if len(history) >= 2 and any(w in message.lower().split() for w in ["it", "this", "formula", "derivation", "units", "marks", "diagram"]):
            last_user_msg = history[-2]["content"]
            retrieval_query = f"{last_user_msg} {message}"

        # Run RAG retrieval (top 15 candidates)
        rag_res = self.rag.query(retrieval_query, top_k=15, top_n=8)
        retrieved_questions = rag_res.get("questions", [])
        clean_filters = rag_res.get("query_analysis", {}).get("filters", {})
        if filters:
            clean_filters.update({k: v for k, v in filters.items() if v})

        # 2. Extract exam statistics & repetition facts
        exam_stats = self._extract_exam_stats(retrieved_questions)

        # 3. Handle Quiz Mode specifically
        if mode == "quiz" or "quiz" in message.lower() or "test me" in message.lower():
            return self._generate_quiz_response(message, retrieved_questions, clean_filters, session_id)

        # 4. Generate Answer via selected engine
        reply_text = ""
        engine_used = "fast"

        if engine_type == "ollama":
            ollama_status = self.check_ollama()
            if ollama_status["available"]:
                model_to_use = preferred_model or ollama_status["active_model"] or DEFAULT_OLLAMA_MODEL
                try:
                    reply_text = self._generate_with_ollama(
                        message=message,
                        history=history,
                        retrieved=retrieved_questions,
                        mode=mode,
                        model=model_to_use,
                    )
                    engine_used = f"ollama ({model_to_use})"
                except Exception as e:
                    logger.warning(f"Ollama generation error: {e}. Falling back to Fast Expert Engine.")
                    reply_text = ""

        # If fast engine chosen or Ollama failed/skipped
        if not reply_text:
            reply_text = self._generate_fast_expert_answer(
                message=message,
                retrieved=retrieved_questions,
                exam_stats=exam_stats,
                mode=mode,
                filters=clean_filters,
            )
            engine_used = "fast_expert"

        # 5. Format sources for student review
        sources_list = []
        for q in retrieved_questions[:6]:
            meta = q.get("metadata", {})
            sources_list.append({
                "source_file": meta.get("source_file", "JKBOSE Question Paper"),
                "year": meta.get("year", "N/A"),
                "class_level": meta.get("class_level", "N/A"),
                "subject": meta.get("subject", "N/A"),
                "marks": meta.get("marks", "N/A"),
                "paper_code": meta.get("paper_code", "N/A"),
                "set_code": meta.get("set_code", ""),
                "question_number": meta.get("question_number", ""),
                "text": q.get("text", ""),
                "citation": q.get("short_citation") or q.get("citation", ""),
            })

        # 6. Generate intelligent student follow-up suggestions
        suggested_followups = self._generate_followups(message, retrieved_questions, clean_filters)

        # 7. Update conversation history
        self.sessions[session_id].append({"role": "user", "content": message})
        self.sessions[session_id].append({"role": "assistant", "content": reply_text})
        if len(self.sessions[session_id]) > 16:
            self.sessions[session_id] = self.sessions[session_id][-16:]

        return {
            "reply": reply_text,
            "mode": mode,
            "engine_used": engine_used,
            "sources": sources_list,
            "exam_stats": exam_stats,
            "suggested_followups": suggested_followups,
            "confidence": rag_res.get("confidence", "HIGH"),
        }

    # ─────────────────────────────────────────────────────────────────
    # Fast Educational Synthesizer (<500ms)
    # ─────────────────────────────────────────────────────────────────

    def _generate_fast_expert_answer(
        self,
        message: str,
        retrieved: List[Dict],
        exam_stats: Dict[str, Any],
        mode: str,
        filters: Dict[str, Any],
    ) -> str:
        """
        Synthesizes an exam-tailored, curriculum-accurate answer based on retrieved papers.
        Formats answer with clear marking breakdown, definitions, derivations, and examiner tips.
        """
        topic_title = self._clean_topic_title(message)
        primary_q = retrieved[0] if retrieved else None
        primary_text = primary_q.get("text", "") if primary_q else ""
        primary_meta = primary_q.get("metadata", {}) if primary_q else {}

        subject = primary_meta.get("subject") or filters.get("subject") or "JKBOSE Examination"
        class_lvl = primary_meta.get("class_level") or filters.get("class_level") or "10/12"
        marks = primary_meta.get("marks") or filters.get("marks") or "3"

        sections = []

        # ── Mode: Exam Trends / High Yield ──
        if mode == "trends" or any(w in message.lower() for w in ["trend", "repeat", "frequent", "important"]):
            return self._build_trends_answer(topic_title, retrieved, exam_stats, subject, class_lvl)

        # ── Mode: Tutor / Simple Explanation ──
        if mode == "tutor":
            return self._build_tutor_explanation(topic_title, primary_text, retrieved, subject, class_lvl)

        # ── Mode: Standard Model Exam Answer ──
        # Header Badge
        sections.append(f"### 📘 Model Exam Solution: **{topic_title}**\n")
        sections.append(f"> **Target Level:** JKBOSE Class {class_lvl} {subject} | **Exam Mark Weight:** {marks} Mark(s)\n")

        # 1. Authentic JKBOSE Paper Match
        if primary_text:
            cleaned_q = re.sub(r'^\s*Q\.?\s*\d+[\.\:\)]?\s*', '', primary_text).strip()
            sections.append(f"**Past Exam Question:**\n*\"{cleaned_q}\"*\n")
            if exam_stats.get("frequency", 0) > 1:
                years_str = ", ".join(str(y) for y in exam_stats.get("years", []))
                sections.append(f"⭐ **High-Yield Question**: Appeared across **{exam_stats['frequency']}** JKBOSE examination papers ({years_str}).\n")

        # 2. Structured Model Answer
        sections.append("#### ✍️ Standard Model Answer for Evaluation:")
        core_solution = self._get_educational_solution(topic_title, primary_text, subject)
        sections.append(core_solution)

        # 3. Mark Scheme Guide
        sections.append("\n#### 🎯 JKBOSE Marking Scheme Breakdown:")
        sections.append(self._get_marking_breakdown(marks, subject))

        # 4. Examiner Advice
        sections.append("\n#### 💡 Examiner's Scoring Tips:")
        sections.append(
            "- **Always highlight final units:** For numericals, write the SI unit explicitly (e.g. $\\Omega$, $V$, $J$, $m/s$).\n"
            "- **Label Diagrams:** Draw neat pencil diagrams with proper annotations where required.\n"
            "- **Bullet points over essays:** JKBOSE examiners award marks point-by-point based on the official answer key."
        )

        return "\n".join(sections)

    def _build_trends_answer(
        self,
        topic: str,
        retrieved: List[Dict],
        stats: Dict[str, Any],
        subject: str,
        class_lvl: str,
    ) -> str:
        """Builds an analysis of repeated questions and examination trends."""
        lines = [
            f"### 📊 JKBOSE Exam Trend Analysis: **{topic}**\n",
            f"Analysis based on indexed question papers for **Class {class_lvl} {subject}**:\n",
        ]

        if not retrieved:
            lines.append("No historical questions found in the archive matching this query.")
            return "\n".join(lines)

        lines.append(f"- **Total Question Occurrences in Archive:** {len(retrieved)}")
        if stats.get("years"):
            years_str = ", ".join(str(y) for y in sorted(stats["years"]))
            lines.append(f"- **Academic Sessions Seen:** {years_str}")
        if stats.get("marks_distribution"):
            marks_str = ", ".join(f"{m} Mark ({c}x)" for m, c in stats["marks_distribution"].items())
            lines.append(f"- **Common Mark Distribution:** {marks_str}")

        lines.append("\n#### 🔁 Key Past Examination Questions:")
        for idx, q in enumerate(retrieved[:5], 1):
            meta = q.get("metadata", {})
            q_text = re.sub(r'^\s*Q\.?\s*\d+[\.\:\)]?\s*', '', q.get("text", "")).strip()
            year = meta.get("year", "Unknown Year")
            p_code = meta.get("paper_code", "")
            set_c = meta.get("set_code", "")
            marks = meta.get("marks", "?")
            lines.append(f"**{idx}. [{year} | Set {set_c} | Code {p_code} | {marks} Marks]**")
            lines.append(f"> \"{q_text}\"\n")

        lines.append("#### 📌 Recommendation for Students:")
        lines.append("Focus on memorizing the core definitions and practicing the standard 3-mark and 5-mark derivations from these exact years, as JKBOSE frequently alternates question sets across series X, Y, and Z.")

        return "\n".join(lines)

    def _build_tutor_explanation(
        self,
        topic: str,
        primary_text: str,
        retrieved: List[Dict],
        subject: str,
        class_lvl: str,
    ) -> str:
        """Builds an intuitive, student-friendly explanation."""
        lines = [
            f"### 💡 Friendly Tutor Guide: **{topic}**\n",
            f"*Designed for JKBOSE Class {class_lvl} {subject} students*\n",
        ]

        solution = self._get_educational_solution(topic, primary_text, subject)
        lines.append(solution)

        lines.append("\n#### 🧠 Memory Mnemonic & Intuition:")
        lines.append(
            "- **Concept Summary:** Break the definition into 2 key conditions: what is given, and what happens.\n"
            "- **Real-Life Analogy:** Think of how this applies in everyday devices or nature.\n"
            "- **Self-Check Question:** If you double the input or change the condition, what happens to the output?"
        )

        lines.append("\n#### 📚 Past JKBOSE Question on this Concept:")
        if primary_text:
            lines.append(f"> *\"{primary_text}\"*")

        return "\n".join(lines)

    def _get_marking_breakdown(self, marks: Any, subject: str) -> str:
        """Returns standard JKBOSE marking breakdown guide."""
        try:
            m = int(marks)
        except (ValueError, TypeError):
            m = 3

        if m <= 1:
            return (
                "- **1 Mark (Objective / Very Short):**\n"
                "  - Direct definition or single formula with correct SI unit.\n"
                "  - No lengthy explanation needed; 1–2 crisp sentences will fetch full marks."
            )
        elif m == 2:
            return (
                "- **2 Marks (Short Answer):**\n"
                "  - **Point 1 (1 Mark):** Accurate scientific definition / law statement.\n"
                "  - **Point 2 (1 Mark):** Mathematical equation with SI unit OR 2 key points of distinction."
            )
        elif m == 3:
            return (
                "- **3 Marks (Short Answer Type-II):**\n"
                "  - **Step 1 (1 Mark):** Fundamental statement / principle.\n"
                "  - **Step 2 (1 Mark):** Key derivation step / chemical equation / labelled diagram.\n"
                "  - **Step 3 (1 Mark):** Practical application / consequence / limiting condition."
            )
        else:
            return (
                f"- **{m} Marks (Long Answer Question):**\n"
                "  - **Part A (1 Mark):** Clear definition / statement of law or principle.\n"
                "  - **Part B (2.5 Marks):** Full step-by-step mathematical derivation or procedural explanation.\n"
                "  - **Part C (1 Mark):** Neatly labelled diagram with directional arrows / circuit symbols.\n"
                "  - **Part D (0.5 Mark):** Specific conditions, precautions, or special cases."
            )

    def _get_educational_solution(self, topic: str, question_text: str, subject: str) -> str:
        """
        Generates rich, curriculum-accurate answers for standard JKBOSE science/math concepts.
        """
        lower = (topic + " " + question_text).lower()

        # ── OHM'S LAW ──
        if "ohm" in lower:
            return (
                "**1. Statement of Ohm's Law:**\n"
                "At constant physical conditions (especially temperature), the electric current ($I$) flowing through a conductor is directly proportional to the potential difference ($V$) applied across its ends.\n\n"
                "**2. Mathematical Expression:**\n"
                "$$V \\propto I \\implies V = I \\times R$$\n"
                "Where:\n"
                "- $V$ = Potential Difference across conductor (in **Volts, V**)\n"
                "- $I$ = Electric Current flowing (in **Amperes, A**)\n"
                "- $R$ = Constant of proportionality known as **Resistance** (in **Ohms, $\\Omega$**)\n\n"
                "**3. Definition of 1 Ohm ($1\\Omega$):**\n"
                "A conductor has a resistance of $1\\,\\Omega$ if a potential difference of $1\\text{ Volt}$ across its ends causes a current of $1\\text{ Ampere}$ to flow through it ($1\\,\\Omega = 1\\text{ V} / 1\\text{ A}$).\n\n"
                "**4. Circuit & V-I Graph:**\n"
                "- For an ohmic conductor, the $V-I$ graph is a straight line passing through the origin. The slope of the $V-I$ graph gives resistance ($R = \\Delta V / \\Delta I$)."
            )

        # ── COULOMB'S LAW ──
        if "coulomb" in lower:
            return (
                "**1. Statement:**\n"
                "The electrostatic force of attraction or repulsion between two stationary point charges is directly proportional to the product of the magnitudes of the charges and inversely proportional to the square of the distance between them.\n\n"
                "**2. Mathematical Expression:**\n"
                "$$F = \\frac{1}{4\\pi\\varepsilon_0} \\cdot \\frac{q_1 q_2}{r^2}$$\n"
                "Where:\n"
                "- $q_1, q_2$ = Magnitudes of charges (in Coulombs, C)\n"
                "- $r$ = Distance between charges (in meters, m)\n"
                "- $\\frac{1}{4\\pi\\varepsilon_0} = 9 \\times 10^9 \\text{ N}\\cdot\\text{m}^2/\\text{C}^2$ in vacuum/free space.\n\n"
                "**3. Vector Form & Key Characteristics:**\n"
                "- Acts along the line joining the centers of the two charges (central force).\n"
                "- Obeys Newton's third law of motion: $\\vec{F}_{12} = -\\vec{F}_{21}$."
            )

        # ── GAUSS'S LAW ──
        if "gauss" in lower:
            return (
                "**1. Statement:**\n"
                "Gauss's law states that the total electric flux ($\\Phi_E$) passing through any closed Gaussian surface is equal to $\\frac{1}{\\varepsilon_0}$ times the total charge ($q_{enc}$) enclosed by that surface.\n\n"
                "**2. Mathematical Expression:**\n"
                "$$\\Phi_E = \\oint \\vec{E} \\cdot d\\vec{A} = \\frac{q_{\\text{enclosed}}}{\\varepsilon_0}$$\n\n"
                "**3. Applications for JKBOSE Exams:**\n"
                "- Electric field due to an infinitely long straight charged wire: $E = \\frac{\\lambda}{2\\pi\\varepsilon_0 r}$\n"
                "- Electric field due to an infinite uniformly charged plane sheet: $E = \\frac{\\sigma}{2\\varepsilon_0}$\n"
                "- Field inside a charged thin spherical shell is **Zero** ($E = 0$ for $r < R$)."
            )

        # ── KIRCHHOFF'S LAWS ──
        if "kirchhoff" in lower:
            return (
                "**1. Kirchhoff's First Law (Junction Rule / KCL):**\n"
                "- *Statement:* The algebraic sum of all electric currents entering and leaving any electrical node or junction is zero ($\\sum I = 0$).\n"
                "- *Law of Conservation:* Based on the **Conservation of Electric Charge**.\n\n"
                "**2. Kirchhoff's Second Law (Loop Rule / KVL):**\n"
                "- *Statement:* In any closed electrical loop or mesh, the algebraic sum of changes in potential (including EMFs and potential drops $IR$) is zero ($\\sum \\Delta V = \\sum \\mathcal{E} - \\sum IR = 0$).\n"
                "- *Law of Conservation:* Based on the **Conservation of Energy**."
            )

        # ── PHOTOELECTRIC EFFECT ──
        if "photoelectric" in lower:
            return (
                "**1. Definition:**\n"
                "The phenomenon of emission of electrons from a metal surface when light (electromagnetic radiation) of suitable frequency falls upon it.\n\n"
                "**2. Einstein's Photoelectric Equation:**\n"
                "$$h\\nu = \\Phi_0 + K_{\\text{max}} = h\\nu_0 + \\frac{1}{2}m v_{\\text{max}}^2$$\n"
                "Where $h\\nu$ = Energy of incident photon, $\\Phi_0 = h\\nu_0$ = Work function, $K_{\\text{max}} = e V_0$ = Max kinetic energy.\n\n"
                "**3. Three Key Laws of Photoelectric Emission:**\n"
                "- Emission occurs only if frequency $\\nu \\ge \\nu_0$ (Threshold frequency).\n"
                "- Number of photoelectrons emitted per second is directly proportional to the intensity of incident radiation.\n"
                "- Kinetic energy of photoelectrons depends only on frequency, not intensity.\n"
                "- It is an instantaneous process ($< 10^{-9}$ seconds)."
            )

        # ── PHOTOSYNTHESIS ──
        if "photosynthesis" in lower:
            return (
                "**1. Definition:**\n"
                "The biochemical process by which autotrophic green plants synthesize organic food (glucose) from carbon dioxide and water using sunlight trapped by chlorophyll pigments.\n\n"
                "**2. Balanced Chemical Equation:**\n"
                "$$6\\text{CO}_2 + 6\\text{H}_2\\text{O} \\xrightarrow[\\text{Chlorophyll}]{\\text{Sunlight}} \\text{C}_6\\text{H}_{12}\\text{O}_6 + 6\\text{O}_2$$\n\n"
                "**3. Major Stages for Class 10/11 Exams:**\n"
                "- **Light Reaction (Granum):** Absorption of light energy $\\to$ Photolysis of water ($2\\text{H}_2\\text{O} \\to 4\\text{H}^+ + \\text{O}_2 + 4e^-$) $\\to$ Formation of ATP and NADPH.\n"
                "- **Dark Reaction / Calvin Cycle (Stroma):** Reduction of $\\text{CO}_2$ into carbohydrates using ATP and NADPH."
            )

        # ── DISPERSION OF LIGHT ──
        if "dispersion" in lower or "prism" in lower:
            return (
                "**1. Definition:**\n"
                "The splitting of white light into its constituent seven spectral colors (VIBGYOR) when passing through a refracting medium like a glass prism.\n\n"
                "**2. Cause of Dispersion:**\n"
                "Different colors of light have different wavelengths. Since refractive index ($n$) varies inversely with wavelength ($\\lambda$), each color travels with a different speed and deviates by a different angle ($\\delta$).\n"
                "- **Violet:** Shortest wavelength $\\to$ Maximum deviation.\n"
                "- **Red:** Longest wavelength $\\to$ Minimum deviation."
            )

        # ── MITOSIS VS MEIOSIS ──
        if "mitosis" in lower or "meiosis" in lower:
            return (
                "**Comparison Table for JKBOSE Examinations (3 to 5 Marks):**\n\n"
                "| Feature | Mitosis (Equational) | Meiosis (Reductional) |\n"
                "| :--- | :--- | :--- |\n"
                "| **Site of Occurrence** | Somatic cells (body cells) | Germ / Reproductive cells |\n"
                "| **Number of Divisions**| 1 nuclear division | 2 successive divisions (I & II) |\n"
                "| **Daughter Cells** | 2 genetically identical cells | 4 genetically diverse cells |\n"
                "| **Chromosome Count** | Remains Diploid ($2n \\to 2n$) | Becomes Haploid ($2n \\to n$) |\n"
                "| **Crossing Over** | Absent | Present in Pachytene of Prophase I |\n"
                "| **Primary Role** | Growth, repair, regeneration | Gametogenesis & genetic variation |"
            )

        # ── GENERIC CURRICULUM FALLBACK ──
        return (
            f"**1. Core Academic Definition:**\n"
            f"{topic} is a fundamental concept in {subject}. In JKBOSE examinations, questions on this topic test clear theoretical definitions, key governing principles, and standard practical applications.\n\n"
            "**2. Essential Examination Points:**\n"
            "- Clearly state the fundamental law or definition in unambiguous scientific terms.\n"
            "- Write the relevant mathematical formula with explanation of each symbol and its proper SI unit.\n"
            "- Mention necessary assumptions, limiting conditions, or real-life examples.\n\n"
            "**3. Standard Steps to Full Marks:**\n"
            "- Define the primary term in 1–2 precise sentences.\n"
            "- Supplement with a clean, labeled diagram or circuit schema.\n"
            "- Conclude with 2 practical applications or consequences."
        )

    def _extract_exam_stats(self, retrieved: List[Dict]) -> Dict[str, Any]:
        """Calculates repetition and year distribution from retrieved question chunks."""
        years = set()
        marks_dist = {}
        for q in retrieved:
            meta = q.get("metadata", {})
            y = meta.get("year")
            if y and str(y).isdigit():
                years.add(int(y))
            m = meta.get("marks")
            if m:
                marks_dist[str(m)] = marks_dist.get(str(m), 0) + 1

        return {
            "frequency": len(retrieved),
            "years": sorted(list(years)),
            "marks_distribution": marks_dist,
            "is_repeated": len(years) > 1,
        }

    def _generate_quiz_response(
        self,
        message: str,
        retrieved: List[Dict],
        filters: Dict[str, Any],
        session_id: str,
    ) -> Dict[str, Any]:
        """Presents an authentic past-paper exam question as a quiz."""
        if not retrieved:
            reply = "I couldn't find a question in the current filtered archive for a quiz. Try asking: `Quiz me on Class 10 Science` or `Test me on Class 12 Physics`!"
            return {"reply": reply, "sources": [], "suggested_followups": ["Quiz Class 10 Math", "Quiz Class 12 Physics"], "mode": "quiz"}

        q_item = retrieved[0]
        meta = q_item.get("metadata", {})
        q_text = re.sub(r'^\s*Q\.?\s*\d+[\.\:\)]?\s*', '', q_item.get("text", "")).strip()

        year = meta.get("year", "Past Paper")
        subject = meta.get("subject", "Science")
        class_lvl = meta.get("class_level", "10")
        marks = meta.get("marks", "3")
        paper_code = meta.get("paper_code", "")

        reply = (
            f"### 🎯 JKBOSE Exam Practice Challenge!\n\n"
            f"> **Paper:** Class {class_lvl} {subject} ({year}) | Code: `{paper_code}` | **{marks} Mark(s)**\n\n"
            f"**Question:**\n"
            f"### \"{q_text}\"\n\n"
            f"---\n"
            f"⏱️ **Take 2 minutes to write your answer down!**\n\n"
            f"<details>\n"
            f"<summary>🔍 <b>Click to Reveal Official Model Answer & Mark Scheme</b></summary>\n\n"
            f"{self._get_educational_solution(q_text, q_text, subject)}\n\n"
            f"**Scoring Criteria ({marks} Marks):**\n"
            f"{self._get_marking_breakdown(marks, subject)}\n"
            f"</details>\n\n"
            f"How did you do? Click below to try another question or ask for a detailed derivation!"
        )

        followups = [
            f"Explain {q_text[:30]}...",
            f"Another Class {class_lvl} {subject} question",
            "What marks did this carry in 2024?",
        ]

        return {
            "reply": reply,
            "mode": "quiz",
            "engine_used": "fast_quiz",
            "sources": [{
                "source_file": meta.get("source_file", ""),
                "year": year,
                "class_level": class_lvl,
                "subject": subject,
                "marks": marks,
                "text": q_text,
                "citation": q_item.get("short_citation") or q_item.get("citation", ""),
            }],
            "exam_stats": {"frequency": len(retrieved), "years": [year]},
            "suggested_followups": followups,
            "confidence": "HIGH",
        }

    # ─────────────────────────────────────────────────────────────────
    # Ollama Generation Engine
    # ─────────────────────────────────────────────────────────────────

    def _generate_with_ollama(
        self,
        message: str,
        history: List[Dict[str, str]],
        retrieved: List[Dict],
        mode: str,
        model: str,
    ) -> str:
        """Call local Ollama instance for deep conversational AI explanation."""
        # Format grounded context from retrieved papers
        context_snippets = []
        for idx, q in enumerate(retrieved[:4], 1):
            meta = q.get("metadata", {})
            q_clean = re.sub(r'^\s*Q\.?\s*\d+[\.\:\)]?\s*', '', q.get("text", "")).strip()
            context_snippets.append(
                f"[Source {idx}: JKBOSE Class {meta.get('class_level')} {meta.get('subject')} {meta.get('year')}, Marks: {meta.get('marks')}]:\n{q_clean}"
            )
        grounded_context = "\n\n".join(context_snippets)

        system_prompt = (
            "You are the master JKBOSE Exam Assistant and Tutor for students preparing for Jammu & Kashmir Board (Class 10, 11, and 12) examinations.\n"
            "Your goal is to help students score 100% in their exams by giving accurate, curriculum-aligned, well-structured answers.\n"
            "Format your answers with:\n"
            "1. Precise statement / definition with formulas, SI units, and equations.\n"
            "2. Step-by-step points suitable for 1-mark, 3-mark, and 5-mark evaluation.\n"
            "3. Mention any diagrams or graphs the student should draw.\n"
            "4. Be encouraging, clear, and academically rigorous.\n"
            f"Here are authentic retrieved questions from the JKBOSE question paper archive to ground your answer:\n\n{grounded_context}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_predict": 450,
            }
        }

        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "").strip()

    def _clean_topic_title(self, text: str) -> str:
        """Extract a readable topic title from user message."""
        clean = re.sub(r'(?i)(what is|explain|define|state|give me|how does|solve|question on|show me|tell me about)\s+', '', text).strip()
        clean = clean.strip("? .!").title()
        return clean if clean else "JKBOSE Exam Topic"

    def _generate_followups(self, message: str, retrieved: List[Dict], filters: Dict[str, Any]) -> List[str]:
        """Generates 3 contextual student follow-up queries."""
        topic = self._clean_topic_title(message)
        subject = filters.get("subject", "Physics")
        class_lvl = filters.get("class_level", "12")

        followups = [
            f"Give me 5-mark answer for {topic}",
            f"Show repeated questions in Class {class_lvl} {subject}",
            f"Test me with a quiz question on {topic}",
        ]
        return followups

#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          EduPlanner AI  ·  Local Student Assistant & Quiz Generator          ║
║                                                                              ║
║  Stack  : Python 3.10+ · Streamlit · Google Gemini 2.5 Flash · SQLite3      ║
║  Run    : streamlit run app.py                                               ║
║  Setup  : pip install streamlit google-generativeai                          ║
║  API Key: Set GOOGLE_API_KEY env var OR enter it in the sidebar              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import os
import re
import json
import sqlite3
from datetime import datetime
from typing import Optional
from pypdf import PdfReader


import streamlit as st

# Graceful import of google-generativeai so the app can at least load and show
# a helpful error message if the package is missing.
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
DB_PATH         = "eduplanner.db"   
GEMINI_MODEL    = "gemini-2.5-flash"
MAX_INPUT_CHARS = 8_000             
MAX_QUESTIONS   = 5                 


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  ── must be the very first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="EduPlanner AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://aistudio.google.com/app/apikey",
        "About": "EduPlanner AI – Powered by Gemini 1.5 Flash · SQLite3 · Streamlit",
    },
)


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 1 ──────────────────────────────────────────────────────────────────
# DATABASE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def init_db() -> None:
    """
    Run once at startup.
    Creates `eduplanner.db` and the `quiz_history` table if they don't exist.
    Uses a context manager so the connection is always closed cleanly.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quiz_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                topic           TEXT    NOT NULL,
                score           INTEGER NOT NULL,
                total_questions INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def db_save_result(topic: str, score: int, total: int) -> None:
    """Insert one completed quiz record into quiz_history."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO quiz_history (timestamp, topic, score, total_questions) VALUES (?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), topic, score, total),
        )
        conn.commit()


def db_fetch_history() -> list[tuple]:
    """Return all quiz_history rows, newest first."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT id, timestamp, topic, score, total_questions "
            "FROM quiz_history ORDER BY id DESC"
        )
        return cursor.fetchall()


def db_clear_history() -> None:
    """Delete every row in quiz_history (user-triggered reset)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM quiz_history")
        conn.commit()

# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 2 ──────────────────────────────────────────────────────────────────
# AI INTEGRATION LAYER
# Prompts → API call → parse/validate JSON
# ═══════════════════════════════════════════════════════════════════════════════




PLANNER_PROMPT = """\
You are an expert educational content analyst and study coach.

Analyze the TEXT below and determine its difficulty level:
  • Beginner   – assumes little prior knowledge, simple vocabulary
  • Intermediate – requires some foundational understanding
  • Advanced   – dense, technical, or requires significant prior knowledge

Then produce a structured **3-Step Study Plan** to help a student master the material.

## Strict output format (Markdown only – no extra prose before/after):

### 📊 Difficulty Assessment
*One sentence naming the level and justifying it.*

---

### 🗺️ Step 1 — [Engaging Step Title]
**Goal:** What the student will achieve in this step.

**How to do it:**
- Action-item 1
- Action-item 2
- Action-item 3

---

### 🔍 Step 2 — [Engaging Step Title]
**Goal:** …
**How to do it:**
- …

---

### 🚀 Step 3 — [Engaging Step Title]
**Goal:** …
**How to do it:**
- …

---
*Pro Tip: [One memorable study tip relevant to this specific material.]*

TEXT TO ANALYSE:
---
{text}
---"""


QUIZ_PROMPT = """\
You are an expert quiz designer. Using the TEXT below, create exactly {n} multiple-choice questions.

## ABSOLUTE RULES (violation = task failure):
1. Return ONLY a raw JSON array — no markdown fences, no preamble, no trailing text whatsoever.
2. Each question must have exactly 4 options (full meaningful phrases, NOT single letters).
3. The "answer" field must be the EXACT verbatim copy of one element inside "options".
4. Vary difficulty: {easy} easy, {medium} medium, {hard} hard question(s).
5. Questions must be directly testable from the given text (no outside knowledge).

## Required JSON schema:
[
  {{
    "question":    "Full question sentence ending with a question mark?",
    "options":     ["Option A text", "Option B text", "Option C text", "Option D text"],
    "answer":      "Exact text of the correct option (copied verbatim from options)",
    "explanation": "1–2 sentence explanation of why the answer is correct."
  }},
  ...
]

TEXT:
---
{text}
---"""


# ── API Key Resolution ─────────────────────────────────────────────────────────

def resolve_api_key() -> Optional[str]:
    """
    Resolve Gemini API key with priority:
        1. GOOGLE_API_KEY environment variable
        2. Key typed into the sidebar text_input (session_state["sidebar_api_key"])

    Returns the key string, or None if no key is present.
    """
    env_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if env_key:
        return env_key
    sidebar_key = st.session_state.get("sidebar_api_key", "").strip()
    return sidebar_key or None


# ── Low-level Gemini call ──────────────────────────────────────────────────────

def _call_gemini(prompt: str, api_key: str) -> Optional[str]:
    """
    Configure Gemini with the given key, send `prompt`, return response text.
    Raises exceptions (caught by callers) on network/auth/API errors.
    """
    if not GENAI_AVAILABLE:
        raise RuntimeError(
            "`google-generativeai` is not installed. "
            "Run: pip install google-generativeai"
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=genai.types.GenerationConfig(
            temperature=0.65,   
            max_output_tokens=4096,
        ),
    )
    response = model.generate_content(prompt)
    return response.text.strip()


# ── High-level AI functions ────────────────────────────────────────────────────

def ai_study_plan(text: str, api_key: str) -> str:
    """
    Generate a Markdown 3-step study plan via Gemini.
    Returns the plan string, or an error message prefixed with '❌'.
    """
    try:
        prompt = PLANNER_PROMPT.format(text=text[:MAX_INPUT_CHARS])
        result = _call_gemini(prompt, api_key)
        return result or "❌ AI returned an empty response. Please try again."
    except Exception as exc:
        return f"❌ **Gemini error:** {exc}"


def _extract_json_array(raw: str) -> Optional[list]:
    """
    Three-strategy parser to pull a valid JSON list from AI output.
    The AI sometimes wraps output in markdown fences or adds preamble text,
    so we try progressively more aggressive extraction before giving up.

    Strategy 1 – Direct parse:    fastest path when the model is well-behaved.
    Strategy 2 – Regex extraction: finds the first '[…]' block in dirty output.
    Strategy 3 – Fence stripping:  removes ```json … ``` wrapper then retries.
    """
    
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    
    match = re.search(r'\[[\s\S]*\]', raw)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    
    cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`').strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    return None  


def ai_quiz(text: str, api_key: str) -> Optional[list]:
    """
    Generate quiz questions via Gemini and validate the JSON response.

    Returns a list of valid question dicts, or None if generation/parsing fails.

    Each returned dict is guaranteed to have:
        question  (str)
        options   (list of 4 str)
        answer    (str that exists in options)
        explanation (str)
    """
    try:
        n = MAX_QUESTIONS
        prompt = QUIZ_PROMPT.format(
            n=n, easy=2, medium=2, hard=1,
            text=text[:MAX_INPUT_CHARS],
        )
        raw = _call_gemini(prompt, api_key)
        if not raw:
            st.error("❌ AI returned an empty response for the quiz.")
            return None

        quiz_list = _extract_json_array(raw)

        if quiz_list is None:
            
            with st.expander("🔍 Raw AI output (debug)"):
                st.code(raw[:600], language="text")
            st.error(
                "❌ Could not parse valid JSON from AI response. "
                "Try regenerating – occasional formatting failures happen."
            )
            return None

        
        
        
        valid: list[dict] = []
        required_keys = {"question", "options", "answer", "explanation"}

        for idx, q in enumerate(quiz_list, start=1):
            if not isinstance(q, dict):
                st.warning(f"⚠️ Q{idx}: not a dict – skipped.")
                continue
            if not required_keys.issubset(q.keys()):
                missing = required_keys - q.keys()
                st.warning(f"⚠️ Q{idx}: missing keys {missing} – skipped.")
                continue
            if not isinstance(q["options"], list) or len(q["options"]) < 2:
                st.warning(f"⚠️ Q{idx}: 'options' must be a list of ≥2 items – skipped.")
                continue
            if q["answer"] not in q["options"]:
                st.warning(
                    f"⚠️ Q{idx}: answer '{q['answer'][:40]}…' not found in options – skipped."
                )
                continue
            valid.append(q)

        if not valid:
            st.error("❌ All generated questions failed validation. Please try regenerating.")
            return None

        return valid

    except Exception as exc:
        st.error(f"❌ **Quiz generation error:** {exc}")
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 3 ──────────────────────────────────────────────────────────────────
# SESSION STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════



_STATE_DEFAULTS: dict = {
    "study_text":       "",     
    "quiz_topic":       "",     
    "generated_plan":   None,   
    "quiz_data":        None,   
    "quiz_submitted":   False,  
    "current_score":    0,      
    "quiz_saved":       False,  
    "sidebar_api_key":  "",     
    
}


def init_session_state() -> None:
    """Ensure all required session_state keys exist with their defaults."""
    for key, default in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _reset_quiz() -> None:
    """
    Clear quiz-related state so a brand-new quiz can be generated.
    Also deletes the individual radio-button keys (q_0, q_1, …) so
    no stale selections carry over to the next quiz attempt.
    """
    st.session_state["quiz_data"]      = None
    st.session_state["quiz_submitted"] = False
    st.session_state["current_score"]  = 0
    st.session_state["quiz_saved"]     = False
    
    for i in range(MAX_QUESTIONS + 5):
        st.session_state.pop(f"q_{i}", None)


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 4 ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    """
    Sidebar layout:
        ┌─────────────────────────┐
        │  🎓 EduPlanner AI       │
        │  ── API Key section ─── │
        │  ── Progress Tracker ── │
        └─────────────────────────┘
    """
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────────────
        st.markdown("## 🎓 EduPlanner AI")
        st.caption("Your personal AI study companion")
        st.divider()

        # ── API Key ────────────────────────────────────────────────────────────
        st.markdown("### 🔑 API Configuration")
        if os.environ.get("GOOGLE_API_KEY"):
            st.success("✅ Key loaded from environment variable")
        else:
            st.text_input(
                "Google Gemini API Key",
                type="password",
                key="sidebar_api_key",       
                placeholder="AIzaSy...",
                help="Free key at: https://aistudio.google.com/app/apikey",
            )
            if st.session_state.get("sidebar_api_key"):
                st.success("✅ Key entered")
            else:
                st.warning("⚠️ An API key is required to use AI features")

        st.divider()

        # ── Progress Tracker ───────────────────────────────────────────────────
        st.markdown("### 📊 Progress Tracker")
        history = db_fetch_history()

        if not history:
            st.info("No quiz history yet.\nComplete a quiz to track your progress!")
        else:
            
            total_quizzes   = len(history)
            total_correct   = sum(r[3] for r in history)
            total_questions = sum(r[4] for r in history)
            avg_pct         = (total_correct / total_questions * 100) if total_questions else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Quizzes", total_quizzes)
            c2.metric("Correct",  total_correct)
            c3.metric("Avg %",    f"{avg_pct:.0f}%")

            
            st.caption("**Recent sessions:**")
            for _, ts, topic, score, total in history[:10]:
                pct   = (score / total * 100) if total else 0
                
                dot   = "🟢" if pct >= 80 else "🟡" if pct >= 50 else "🔴"
                label = (topic[:22] + "…") if len(topic) > 24 else topic
                st.markdown(
                    f"{dot} **{label}** — {score}/{total} ({pct:.0f}%)  \n"
                    f"<small>🕒 {ts}</small>",
                    unsafe_allow_html=True,
                )

            st.divider()

            
            if st.button("🗑️ Clear All History", use_container_width=True):
                db_clear_history()
                st.rerun()

        st.divider()
        st.caption("Built with Streamlit · Gemini 1.5 Flash · SQLite3")


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 5 ──────────────────────────────────────────────────────────────────
# TAB 1 – STUDY PLANNER (updated version: PDF SUPPORT)
# ═══════════════════════════════════════════════════════════════════════════════

def render_planner_tab() -> None:
    """
    Tab 1 — Context ingestion (Text or PDF) + AI study plan generation.
    """
    st.markdown("## 📖 Study Planner")
    st.write(
        "Paste text or **upload a PDF**, and Gemini will build a personalized "
        "3-Step Study Plan based on the material."
    )

    # ── 1. Input Method Selection ──────────────────────────────────────────────
    input_method = st.radio(
        "Choose input method:",
        ["✍️ Paste Text", "📂 Upload PDF"],
        horizontal=True,
        label_visibility="collapsed"
    )

    study_text = ""

    # ── 2. Handling Inputs ─────────────────────────────────────────────────────
    if input_method == "✍️ Paste Text":
        study_text = st.text_area(
            "Paste your study material here:",
            value=st.session_state.get("study_text", ""),  
            height=270,
            placeholder="Example: Photosynthesis is the biological process...",
            key="text_area_input"
        )

    else:  
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

        if uploaded_file is not None:
            try:
                with st.spinner("📄 Reading PDF file..."):
                    reader = PdfReader(uploaded_file)
                    extracted_text = ""

                    
                    max_pages = min(len(reader.pages), 15)

                    for i in range(max_pages):
                        page_text = reader.pages[i].extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"

                    study_text = extracted_text

                    
                    st.success(f"✅ PDF Processed: {len(reader.pages)} pages found (analyzing first {max_pages}).")
                    with st.expander("👀 View Extracted Text"):
                        st.text(study_text[:1000] + "...")

            except Exception as e:
                st.error(f"Error reading PDF: {e}")

    
    if study_text:
        st.session_state["study_text"] = study_text

    # ── 3. Topic Label & Stats ─────────────────────────────────────────────────
    col_topic, col_hint = st.columns([3, 2])
    with col_topic:
        topic_name = st.text_input(
            "📌 Topic label (shown in tracker):",
            value=st.session_state.get("quiz_topic", ""),
            placeholder="e.g. Python Generators · Cell Biology",
            key="topic_input_box"
        )

    with col_hint:
        char_count = len(study_text.strip())
        if char_count > 0:
            st.metric(
                "Characters",
                f"{char_count:,}",
                delta=f"≤{MAX_INPUT_CHARS:,} sent to AI" if char_count > MAX_INPUT_CHARS else None,
                delta_color="off",
            )

    # ── 4. Action Button ───────────────────────────────────────────────────────
    if st.button("🚀 Analyse & Generate Study Plan", type="primary", use_container_width=True):
        if not study_text.strip():
            st.warning("⚠️ Please provide some content (Text or PDF) before analysing.")
            return

        api_key = resolve_api_key()
        if not api_key:
            st.error("❌ No API key found. Please enter your Gemini key in the sidebar.")
            return

        
        st.session_state["study_text"] = study_text.strip()
        st.session_state["quiz_topic"] = topic_name.strip() if topic_name.strip() else "Study Session"

        
        st.session_state["quiz_data"] = None
        st.session_state["generated_plan"] = None

        
        with st.spinner("🤖 Analysing difficulty and crafting your study plan..."):
            plan = ai_study_plan(study_text.strip(), api_key)
            st.session_state["generated_plan"] = plan
            st.rerun()

    # ── 5. Display Result (with download button) ────────────
    plan = st.session_state.get("generated_plan")

    if plan:
        st.divider()
        if plan.startswith("❌"):
            st.error(plan)
        else:
            st.markdown("### 📋 Your Personalised Study Plan")
            st.markdown(plan)

            
            
            safe_topic = topic_name.strip().replace(" ", "_") or "Study_Plan"
            file_name = f"{safe_topic}_Plan.md"

            col_download, col_go_quiz = st.columns([1, 2])

            with col_download:
                st.download_button(
                    label="📥 Download Plan (Markdown)",
                    data=plan,
                    file_name=file_name,
                    mime="text/markdown",
                    use_container_width=True
                )

            with col_go_quiz:
                st.success("✅ Plan saved? Now test yourself in the **Knowledge Quiz** tab! 👉")


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 6 ──────────────────────────────────────────────────────────────────
# TAB 2 – INTERACTIVE QUIZ
# ═══════════════════════════════════════════════════════════════════════════════

def _render_single_question(idx: int, q: dict, submitted: bool) -> None:
    """
    Render one MCQ question block.

    Before submission:
        • Shows a radio button group (no pre-selection).
        • On selection → shows ✅ with explanation OR ❌ hint (no answer revealed yet).

    After submission:
        • Shows each option labelled as correct / your-correct / your-wrong / neutral.
        • Always shows the full explanation so the student can learn.

    Args:
        idx       : 0-based question index
        q         : question dict with keys: question, options, answer, explanation
        submitted : True once the user has clicked "Submit Quiz"
    """
    question_text = q.get("question", f"Question {idx + 1}")
    options       = q.get("options", [])
    correct       = q.get("answer", "")
    explanation   = q.get("explanation", "")

    st.markdown(f"**Q{idx + 1}. {question_text}**")

    if not submitted:
        
        
        
        selected = st.radio(
            label=f"_q{idx}_label",   
            options=options,
            key=f"q_{idx}",
            index=None,
            label_visibility="collapsed",
        )

        
        if selected is not None:
            if selected == correct:
                st.success(f"✅ **Correct!** — {explanation}")
            else:
                
                
                st.error(
                    "❌ **Not quite.** You can change your answer before submitting. "
                    "*(Full explanation revealed after you submit.)*"
                )

    else:
        
        user_choice = st.session_state.get(f"q_{idx}")   

        for opt in options:
            is_correct  = (opt == correct)
            is_selected = (opt == user_choice)

            if is_correct and is_selected:
                st.markdown(f"✅ **{opt}** ← *your answer – correct!*")
            elif is_correct:
                st.markdown(f"✅ **{opt}** ← *correct answer*")
            elif is_selected:
                st.markdown(f"❌ ~~{opt}~~ ← *your answer*")
            else:
                st.markdown(f"　◦ {opt}")

        if not user_choice:
            st.warning("⚠️ You did not answer this question.")

        st.info(f"💡 **Explanation:** {explanation}")

    st.divider()


def render_quiz_tab() -> None:
    """
    Tab 2 — AI quiz generation, interactive answering, and results.

    Flow:
        ① Gate check: study text must exist (directs user to Tab 1 if not).
        ② If no quiz_data: show "Generate" button → call ai_quiz() → store & rerun.
        ③ Render all questions with _render_single_question().
        ④ Progress bar counts how many questions have been answered.
        ⑤ "Submit Quiz" → calculate score → save to SQLite → rerun.
        ⑥ Score card shown after submission; "Try Again" clears quiz state.
    """
    st.markdown("## 🧠 Knowledge Quiz")

    
    if not st.session_state.get("study_text"):
        st.info(
            "👈 Go to the **📖 Study Planner** tab, paste your study material, "
            "and click **Analyse** — then come back here for the quiz!"
        )
        return

    topic = st.session_state.get("quiz_topic", "Study Session")
    st.markdown(f"**Topic:** `{topic}`")
    st.markdown("")

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase A: Quiz not yet generated
    # ═══════════════════════════════════════════════════════════════════════════
    if not st.session_state.get("quiz_data"):
        st.write(
            f"Click below to generate **{MAX_QUESTIONS} multiple-choice questions** "
            "based on your study material (2 easy · 2 medium · 1 hard)."
        )

        col_gen, col_rst = st.columns([4, 1])
        with col_gen:
            gen_clicked = st.button(
                f"⚡ Generate {MAX_QUESTIONS}-Question Quiz",
                type="primary",
                use_container_width=True,
            )
        with col_rst:
            if st.button("🔄 Reset", use_container_width=True, help="Clear quiz state"):
                _reset_quiz()
                st.rerun()

        if gen_clicked:
            api_key = resolve_api_key()
            if not api_key:
                st.error("❌ API key is missing. Add it in the sidebar.")
                return

            with st.spinner(
                "🤖 Generating your quiz — Gemini is crafting the questions (≈10 s)…"
            ):
                quiz_data = ai_quiz(st.session_state["study_text"], api_key)

            if quiz_data:
                st.session_state["quiz_data"]      = quiz_data
                st.session_state["quiz_submitted"] = False
                st.session_state["current_score"]  = 0
                st.session_state["quiz_saved"]     = False
                st.rerun()
            
        return  

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase B: Quiz ready — render questions
    # ═══════════════════════════════════════════════════════════════════════════
    quiz_data  = st.session_state["quiz_data"]
    submitted  = st.session_state.get("quiz_submitted", False)
    n_q        = len(quiz_data)

    
    if not submitted:
        col_prog, col_new = st.columns([5, 1])
        with col_new:
            if st.button("🔄 New Quiz", help="Discard this quiz and generate a fresh one"):
                _reset_quiz()
                st.rerun()

        
        answered = sum(
            1 for i in range(n_q)
            if st.session_state.get(f"q_{i}") is not None
        )
        with col_prog:
            st.progress(
                answered / n_q if n_q else 0,
                text=f"Progress: {answered} / {n_q} questions answered",
            )
        st.markdown("")

    
    for i, q in enumerate(quiz_data):
        _render_single_question(i, q, submitted)

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase C: Submit button (shown only before submission)
    # ═══════════════════════════════════════════════════════════════════════════
    if not submitted:
        answered_count = sum(
            1 for i in range(n_q)
            if st.session_state.get(f"q_{i}") is not None
        )
        if answered_count < n_q:
            st.caption(
                f"💬 You've answered {answered_count}/{n_q} questions. "
                "You can still submit with unanswered questions."
            )

        if st.button("📊 Submit Quiz & See Results", type="primary", use_container_width=True):
            
            final_score = sum(
                1 for i, q in enumerate(quiz_data)
                if st.session_state.get(f"q_{i}") == q.get("answer")
            )
            st.session_state["current_score"]  = final_score
            st.session_state["quiz_submitted"] = True

            
            if not st.session_state.get("quiz_saved"):
                db_save_result(topic, final_score, n_q)
                st.session_state["quiz_saved"] = True

            st.rerun()

    # ═══════════════════════════════════════════════════════════════════════════
    # Phase D: Score card (shown after submission)
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        score = st.session_state.get("current_score", 0)
        pct   = (score / n_q * 100) if n_q else 0

        
        if pct >= 80:
            st.balloons()
            badge  = "🏆"
            msg    = "Outstanding! You've truly mastered this material."
            accent = "#22c55e"    
        elif pct >= 60:
            badge  = "🌟"
            msg    = "Great work! A little more review and you'll ace it."
            accent = "#3b82f6"    
        elif pct >= 40:
            badge  = "📚"
            msg    = "Decent effort. Re-read the material and try again."
            accent = "#f97316"    
        else:
            badge  = "💪"
            msg    = "Keep going — every attempt makes you stronger!"
            accent = "#ef4444"    

        
        st.markdown(
            f"""
            <div style="
                background : linear-gradient(135deg, #0f0f23 0%, #1a1040 100%);
                border     : 1px solid {accent};
                border-radius : 16px;
                padding    : 32px 24px;
                text-align : center;
                margin-top : 12px;
            ">
                <div style="font-size:3.6em; line-height:1.1;">{badge}</div>
                <h2 style="color:#ffffff; margin: 10px 0 4px;">Quiz Complete!</h2>
                <p  style="color:#94a3b8; font-size:0.9em; margin:0 0 16px;">
                    Topic: {topic}
                </p>
                <div style="
                    font-size  : 3.5em;
                    font-weight: 700;
                    color      : {accent};
                    line-height: 1;
                    margin-bottom: 4px;
                ">{score} / {n_q}</div>
                <p style="color:#cbd5e1; font-size:1.1em; margin:6px 0;">{pct:.0f}% Correct</p>
                <p style="color:#e2e8f0; margin:10px 0;">{msg}</p>
                <p style="color:#64748b; font-size:0.82em; margin-top:16px;">
                    ✅ Result saved to your Progress Tracker
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⚡ Try Another Quiz", type="primary", use_container_width=True):
                _reset_quiz()
                st.rerun()
        with col_b:
            if st.button("📖 New Study Material", type="secondary", use_container_width=True):
                
                st.session_state["study_text"]     = ""
                st.session_state["generated_plan"] = None
                _reset_quiz()
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 7 ──────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════

def inject_css() -> None:
    """
    Minimal, targeted CSS overrides.
    Keeps the look native-Streamlit while polishing key elements.
    Does NOT replace Streamlit's theme system.
    """
    st.markdown(
        """
        <style>
        /* ── Tab strip ───────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] { gap: 6px; }
        .stTabs [data-baseweb="tab"] {
            padding: 6px 20px;
            border-radius: 8px 8px 0 0;
            font-weight: 500;
        }

        /* ── Radio option labels – slightly larger text ───────── */
        .stRadio > div > label { font-size: 0.96rem; line-height: 1.5; }

        /* ── Metric boxes – faint background tint ─────────────── */
        [data-testid="metric-container"] {
            background    : rgba(148, 163, 184, 0.06);
            border-radius : 8px;
            padding       : 8px 10px;
        }

        /* ── Slightly tighter paragraph spacing inside the plan ── */
        .stMarkdown p { margin-bottom: 0.5rem; }

        /* ── Primary button uniform height ───────────────────── */
        .stButton > button[kind="primary"] { min-height: 2.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ── SECTION 8 ──────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Application bootstrap.

    Execution order:
        1. init_db()           – ensure database + table exist
        2. init_session_state() – ensure all session keys are initialised
        3. inject_css()         – style overrides
        4. render_sidebar()     – API key + progress tracker
        5. Tab layout           – Study Planner | Knowledge Quiz
    """
    
    init_db()
    init_session_state()

    
    if not GENAI_AVAILABLE:
        st.warning(
            "⚠️ `google-generativeai` package is not installed.  \n"
            "Run `pip install google-generativeai` and restart the app.",
            icon="⚠️",
        )

    
    inject_css()

    
    render_sidebar()

    
    tab_plan, tab_quiz = st.tabs(["📖 Study Planner", "🧠 Knowledge Quiz"])

    with tab_plan:
        render_planner_tab()

    with tab_quiz:
        render_quiz_tab()



if __name__ == "__main__":
    main()

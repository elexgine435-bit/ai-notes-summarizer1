"""SmartNotes AI - Your Intelligent Study Assistant.

Run locally with:  streamlit run app.py
"""

import copy
import html
import os
import re

import streamlit as st

from utils.ai_engine import DEFAULT_MODELS, AIEngine, AIEngineError, was_truncated
from utils.pdf_reader import PDFReadError, extract_text_from_pdf

st.set_page_config(
    page_title="SmartNotes AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Page names (used for the sidebar menu)
# ---------------------------------------------------------------------------
PAGE_HOME = "🏠 Home"
PAGE_SUMMARY = "📝 Summary"
PAGE_POINTS = "⭐ Important Points"
PAGE_MCQ = "❓ MCQ Quiz"
PAGE_CHAT = "💬 Ask Your Notes"
PAGE_CARDS = "🃏 Flashcards"
PAGE_DOWNLOADS = "⬇️ Downloads"
PAGES = [PAGE_HOME, PAGE_SUMMARY, PAGE_POINTS, PAGE_MCQ, PAGE_CHAT, PAGE_CARDS, PAGE_DOWNLOADS]

# ---------------------------------------------------------------------------
# Styling (CSS)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
.stApp, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label,
.stApp button, .stApp input, .stApp textarea { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #F5F8FF; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1100px; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1E3F 0%, #081530 100%); }
section[data-testid="stSidebar"] * { color: #E6EEFF; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }
.brand { display: flex; align-items: center; gap: 0.75rem; margin: 0.4rem 0 1.2rem 0; }
.brand-mark { background: #2563EB; border-radius: 12px; width: 42px; height: 42px; display: flex;
  align-items: center; justify-content: center; font-size: 1.3rem; }
.brand-name { font-weight: 800; font-size: 1.2rem; color: #FFFFFF; line-height: 1.1; }
.brand-sub { font-size: 0.75rem; color: #9DB4E0; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.2rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label { padding: 0.55rem 0.8rem; border-radius: 10px;
  width: 100%; cursor: pointer; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(255,255,255,0.08); }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) { background: #2563EB; }
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * { color: #FFFFFF; font-weight: 700; }
.doc-card { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.15); border-radius: 12px;
  padding: 0.8rem 0.9rem; margin-top: 0.4rem; }
.doc-name { font-weight: 700; font-size: 0.9rem; word-break: break-word; color: #FFFFFF; }
.doc-meta { font-size: 0.78rem; color: #9DB4E0; margin-top: 0.2rem; }
section[data-testid="stSidebar"] .stButton > button { background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.25); color: #FFFFFF; }
section[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,0.16); border-color: #FFFFFF; }

/* Landing hero */
.hero { background: linear-gradient(135deg, #0B1E3F 0%, #12306B 55%, #2563EB 100%); border-radius: 20px;
  padding: 2.8rem 2.6rem; margin-bottom: 1.6rem; }
.stApp .hero h1 { color: #FFFFFF; font-size: 2.5rem; font-weight: 800; line-height: 1.15; margin: 0 0 0.8rem 0;
  max-width: 680px; padding: 0; }
.stApp .hero p { color: #CFE0FF; font-size: 1.05rem; line-height: 1.6; max-width: 600px; margin: 0; }

/* Cards */
.feature { background: #FFFFFF; border: 1px solid #DFE8F8; border-radius: 14px; padding: 1.1rem 1.2rem;
  min-height: 150px; margin-bottom: 0.6rem; }
.feature .icon { width: 40px; height: 40px; border-radius: 10px; background: #DBEAFE; display: flex;
  align-items: center; justify-content: center; font-size: 1.2rem; }
.stApp .feature h4 { color: #0B1E3F; margin: 0.7rem 0 0.3rem 0; padding: 0; font-size: 1.05rem; font-weight: 700; }
.stApp .feature p { color: #5B6B8A; font-size: 0.9rem; margin: 0; line-height: 1.5; }
.step { background: #FFFFFF; border: 1px solid #DFE8F8; border-left: 4px solid #2563EB; border-radius: 12px;
  padding: 1rem 1.1rem; min-height: 110px; }
.stApp .step h4 { color: #0B1E3F; margin: 0 0 0.3rem 0; padding: 0; font-size: 1rem; }
.stApp .step p { color: #5B6B8A; font-size: 0.9rem; margin: 0; }
.empty { background: #FFFFFF; border: 1px dashed #A9C0EA; border-radius: 14px; padding: 2rem; text-align: center;
  margin-bottom: 1rem; }
.stApp .empty h4 { color: #0B1E3F; margin: 0 0 0.3rem 0; }
.stApp .empty p { color: #5B6B8A; margin: 0; }

/* Page header */
.page-head { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.4rem; }
.page-icon { background: #0B1E3F; border-radius: 14px; width: 52px; height: 52px; display: flex;
  align-items: center; justify-content: center; font-size: 1.5rem; flex-shrink: 0; }
.stApp .page-head h2 { color: #0B1E3F; font-size: 1.7rem; font-weight: 800; margin: 0; padding: 0; line-height: 1.2; }
.stApp .page-head p { color: #5B6B8A; margin: 0.15rem 0 0 0; }

/* Flashcards */
.fc-label { font-size: 0.82rem; font-weight: 700; color: #2563EB; margin-bottom: 0.4rem; }
.fc-question { background: #0B1E3F; border-radius: 16px; padding: 2rem 2rem; min-height: 150px;
  display: flex; align-items: center; }
.stApp .fc-question p { color: #FFFFFF; font-size: 1.25rem; font-weight: 600; margin: 0; line-height: 1.5; }
.fc-answer { background: #FFFFFF; border: 2px solid #2563EB; border-radius: 16px; padding: 1.6rem 2rem;
  margin-top: 0.8rem; }
.stApp .fc-answer p { color: #0B1E3F; font-size: 1.05rem; margin: 0; line-height: 1.6; }

/* Buttons */
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; padding: 0.5rem 1.2rem; }

@media (max-width: 768px) {
  .hero { padding: 1.7rem 1.3rem; }
  .stApp .hero h1 { font-size: 1.7rem; }
  .stApp .fc-question p { font-size: 1.05rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state: the app's "memory" while the student is using it
# ---------------------------------------------------------------------------
DEFAULTS = {
    "page": PAGE_HOME,
    "uploader_n": 0,
    "doc_id": None,
    "doc_name": "",
    "doc_text": "",
    "doc_info": {},
    "bad_id": None,
    "bad_message": "",
    "summary_length": "Medium",
    "summary": "",
    "points": "",
    "mcqs": [],
    "mcq_submitted": False,
    "mcq_round": 0,
    "flashcards": [],
    "fc_index": 0,
    "fc_show": False,
    "chat": [],
}

# These are cleared whenever a new document is uploaded.
RESULT_KEYS = ["summary", "points", "mcqs", "mcq_submitted", "flashcards", "fc_index", "fc_show", "chat"]


def init_state():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)


def reset_results():
    for key in RESULT_KEYS:
        st.session_state[key] = copy.deepcopy(DEFAULTS[key])
    st.session_state.mcq_round += 1


def remove_document():
    reset_results()
    for key in ("doc_id", "doc_name", "doc_text", "doc_info"):
        st.session_state[key] = copy.deepcopy(DEFAULTS[key])
    st.session_state.uploader_n += 1  # gives the upload box a new identity so it empties
    st.session_state.page = PAGE_HOME


def go(page):
    """Button callback: jump to another page."""
    st.session_state.page = page


init_state()

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def require_engine():
    """Create the AI engine from the Streamlit secret. Shows setup help if the key is missing."""
    api_key = ""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.environ.get("GROQ_API_KEY", "")

    models = None
    try:
        custom_model = st.secrets["GROQ_MODEL"]
        models = [custom_model] + [m for m in DEFAULT_MODELS if m != custom_model]
    except Exception:
        pass

    try:
        return AIEngine(api_key, models)
    except AIEngineError:
        st.error("The Groq API key is not set up yet, so the AI features are switched off.")
        st.markdown(
            "Add your key in **`.streamlit/secrets.toml`** (on your computer) or in the "
            "**Secrets** box (on Streamlit Community Cloud) like this:"
        )
        st.code('GROQ_API_KEY = "paste_your_key_here"', language="toml")
        st.stop()


def require_document():
    """Stop the page with a friendly message if no PDF has been uploaded yet."""
    if not st.session_state.doc_text:
        st.markdown(
            '<div class="empty"><h4>No notes uploaded yet</h4>'
            "<p>Upload a PDF on the Home page and this feature will be ready.</p></div>",
            unsafe_allow_html=True,
        )
        st.button("Upload your notes", type="primary", on_click=go, args=(PAGE_HOME,))
        st.stop()


def run_ai(task, message):
    """Run an AI task with a spinner. Shows a friendly error instead of crashing."""
    try:
        with st.spinner(message):
            return task()
    except AIEngineError as error:
        st.error(str(error))
        return None
    except Exception:
        st.error("Something unexpected went wrong. Please try again.")
        return None


def note_if_truncated():
    if was_truncated(st.session_state.doc_text):
        st.caption(
            "Your notes are long, so SmartNotes AI reads samples from across the whole document "
            "to stay within the free AI limits."
        )


def page_header(icon, title, subtitle):
    st.markdown(
        f'<div class="page-head"><div class="page-icon">{icon}</div>'
        f"<div><h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div></div>",
        unsafe_allow_html=True,
    )


def safe_file_stem():
    stem = os.path.splitext(st.session_state.doc_name or "notes")[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "notes"


def download_button(label, text, suffix, key):
    st.download_button(
        label,
        data=text,
        file_name=f"{safe_file_stem()}_{suffix}.txt",
        mime="text/plain",
        key=key,
    )


# ---------------------------------------------------------------------------
# Text builders for downloads
# ---------------------------------------------------------------------------

def mcqs_to_text(mcqs):
    lines = []
    for number, q in enumerate(mcqs, start=1):
        lines.append(f"Q{number}. {q['question']}")
        for letter, option in zip("ABCD", q["options"]):
            lines.append(f"   {letter}) {option}")
        lines.append(f"   Correct answer: {'ABCD'[q['answer_index']]}) {q['options'][q['answer_index']]}")
        lines.append(f"   Explanation: {q['explanation']}")
        lines.append("")
    return "\n".join(lines).strip()


def flashcards_to_text(cards):
    lines = []
    for number, card in enumerate(cards, start=1):
        lines.append(f"Card {number}")
        lines.append(f"Question: {card['question']}")
        lines.append(f"Answer: {card['answer']}")
        lines.append("")
    return "\n".join(lines).strip()


def chat_to_text(chat):
    lines = []
    for message in chat:
        speaker = "You" if message["role"] == "user" else "SmartNotes AI"
        lines.append(f"{speaker}: {message['content']}")
        lines.append("")
    return "\n".join(lines).strip()


def build_study_pack():
    s = st.session_state
    sections = [("SUMMARY", s.summary), ("IMPORTANT POINTS", s.points)]
    if s.mcqs:
        sections.append(("MCQ PRACTICE QUESTIONS", mcqs_to_text(s.mcqs)))
    if s.flashcards:
        sections.append(("FLASHCARDS", flashcards_to_text(s.flashcards)))
    if s.chat:
        sections.append(("QUESTIONS AND ANSWERS", chat_to_text(s.chat)))

    parts = [f"SmartNotes AI - Study Pack\nDocument: {s.doc_name}\n"]
    for title, body in sections:
        if body:
            parts.append(f"{'=' * 60}\n{title}\n{'=' * 60}\n{body}\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="brand-mark">🎓</div><div>'
            '<div class="brand-name">SmartNotes AI</div>'
            '<div class="brand-sub">Your intelligent study assistant</div></div></div>',
            unsafe_allow_html=True,
        )
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
        st.divider()

        if st.session_state.doc_text:
            info = st.session_state.doc_info
            st.markdown(
                f'<div class="doc-card"><div class="doc-name">📄 {html.escape(st.session_state.doc_name)}</div>'
                f'<div class="doc-meta">{info["pages"]} pages, {info["words"]:,} words</div></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.button("Remove document", key="remove_doc", on_click=remove_document)
        else:
            st.markdown(
                '<div class="doc-card"><div class="doc-name">No document yet</div>'
                '<div class="doc-meta">Upload a PDF on the Home page.</div></div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
FEATURES = [
    ("📝", "AI Summary", "Short, medium or detailed summaries in simple English, organised with headings.", PAGE_SUMMARY),
    ("⭐", "Important Points", "Key concepts, definitions and key terms pulled out of your notes.", PAGE_POINTS),
    ("❓", "MCQ Quiz", "Practice questions with four options, correct answers and short explanations.", PAGE_MCQ),
    ("💬", "Ask Your Notes", "Chat with your PDF. Answers come only from your own material.", PAGE_CHAT),
    ("🃏", "Flashcards", "Question and answer cards that are ready for exam revision.", PAGE_CARDS),
    ("⬇️", "Downloads", "Save your summary, quiz and flashcards as simple TXT files.", PAGE_DOWNLOADS),
]


def handle_upload():
    """Show the upload box and read the PDF when a new file is chosen."""
    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key=f"uploader_{st.session_state.uploader_n}",
        help="Use a PDF with selectable text. Scanned pages (pictures) cannot be read.",
    )
    if uploaded is None:
        return

    doc_id = f"{uploaded.name}-{uploaded.size}"
    if doc_id == st.session_state.doc_id:
        return

    if doc_id == st.session_state.bad_id:
        st.error(st.session_state.bad_message)
        return

    try:
        with st.spinner("Reading your PDF..."):
            info = extract_text_from_pdf(uploaded)
    except PDFReadError as error:
        st.session_state.bad_id = doc_id
        st.session_state.bad_message = str(error)
        st.error(str(error))
        return
    except Exception:
        st.session_state.bad_id = doc_id
        st.session_state.bad_message = "Something went wrong while reading this PDF. Please try another file."
        st.error(st.session_state.bad_message)
        return

    reset_results()
    st.session_state.doc_id = doc_id
    st.session_state.doc_name = uploaded.name
    st.session_state.doc_text = info["text"]
    st.session_state.doc_info = info
    st.rerun()  # refresh so the sidebar shows the new document


def render_home():
    st.markdown(
        '<div class="hero"><h1>Turn your lecture notes into summaries, quizzes and flashcards</h1>'
        "<p>Upload a PDF and SmartNotes AI helps you understand it, test yourself and revise faster, "
        "using only your own study material.</p></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Upload your PDF notes")
    handle_upload()

    if st.session_state.doc_text:
        info = st.session_state.doc_info
        st.success(f"Ready: **{st.session_state.doc_name}** has been loaded.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pages", info["pages"])
        col2.metric("Words", f"{info['words']:,}")
        col3.metric("Characters", f"{info['characters']:,}")
        st.button("Start with a summary", type="primary", on_click=go, args=(PAGE_SUMMARY,))

    st.write("")
    st.subheader("What you can do")
    for start in range(0, len(FEATURES), 3):
        columns = st.columns(3)
        for column, (icon, title, text, page) in zip(columns, FEATURES[start : start + 3]):
            with column:
                st.markdown(
                    f'<div class="feature"><div class="icon">{icon}</div>'
                    f"<h4>{html.escape(title)}</h4><p>{html.escape(text)}</p></div>",
                    unsafe_allow_html=True,
                )
                st.button(f"Open {title}", key=f"open_{title}", on_click=go, args=(page,))

    st.write("")
    st.subheader("How it works")
    steps = [
        ("1. Upload", "Add your PDF notes. The text is read in a few seconds."),
        ("2. Study", "Create summaries, important points, quizzes and flashcards."),
        ("3. Revise", "Ask questions about your notes and download your study material."),
    ]
    for column, (title, text) in zip(st.columns(3), steps):
        with column:
            st.markdown(f'<div class="step"><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)


def render_summary():
    page_header("📝", "AI Summary", "Choose how long you want your summary to be.")
    require_document()
    engine = require_engine()
    note_if_truncated()

    st.radio("Summary length", ["Short", "Medium", "Detailed"], key="summary_length", horizontal=True)
    st.caption("Short: quick overview. Medium: main topics. Detailed: full explanation with a recap.")

    if st.button("Generate summary", type="primary"):
        result = run_ai(
            lambda: engine.summarize(st.session_state.doc_text, st.session_state.summary_length),
            "Writing your summary...",
        )
        if result:
            st.session_state.summary = result

    if st.session_state.summary:
        with st.container(border=True):
            st.markdown(st.session_state.summary)
        download_button("Download summary (TXT)", st.session_state.summary, "summary", "dl_summary_page")


def render_points():
    page_header("⭐", "Important Points", "Key concepts, definitions and key terms from your notes.")
    require_document()
    engine = require_engine()
    note_if_truncated()

    if st.button("Find important points", type="primary"):
        result = run_ai(lambda: engine.important_points(st.session_state.doc_text), "Finding the key ideas...")
        if result:
            st.session_state.points = result

    if st.session_state.points:
        with st.container(border=True):
            st.markdown(st.session_state.points)
        download_button("Download important points (TXT)", st.session_state.points, "important_points", "dl_points_page")


def submit_quiz():
    st.session_state.mcq_submitted = True


def retry_quiz():
    st.session_state.mcq_submitted = False
    st.session_state.mcq_round += 1  # new widget keys clear the old answers


def render_mcq():
    page_header("❓", "MCQ Quiz", "Test your knowledge with questions made from your notes.")
    require_document()
    engine = require_engine()
    note_if_truncated()

    count = st.slider("Number of questions", min_value=3, max_value=15, value=5)
    if st.button("Generate MCQs", type="primary"):
        result = run_ai(lambda: engine.generate_mcqs(st.session_state.doc_text, count), "Creating your questions...")
        if result:
            st.session_state.mcqs = result
            st.session_state.mcq_submitted = False
            st.session_state.mcq_round += 1

    mcqs = st.session_state.mcqs
    if not mcqs:
        st.info("Choose how many questions you want and click **Generate MCQs**.")
        return

    submitted = st.session_state.mcq_submitted
    round_id = st.session_state.mcq_round

    if submitted:
        score = sum(
            1
            for i, q in enumerate(mcqs)
            if st.session_state.get(f"mcq_{round_id}_{i}") == q["options"][q["answer_index"]]
        )
        st.success(f"Your score: **{score} out of {len(mcqs)}**")

    for i, q in enumerate(mcqs):
        key = f"mcq_{round_id}_{i}"
        with st.container(border=True):
            st.markdown(f"**Q{i + 1}. {q['question']}**")
            st.radio("Choose one answer", q["options"], index=None, key=key, disabled=submitted,
                     label_visibility="collapsed")
            if submitted:
                chosen = st.session_state.get(key)
                correct = q["options"][q["answer_index"]]
                if chosen == correct:
                    st.success(f"Correct. {q['explanation']}")
                elif chosen is None:
                    st.warning(f"Not answered. The correct answer is **{correct}**. {q['explanation']}")
                else:
                    st.error(f"Not quite. The correct answer is **{correct}**. {q['explanation']}")

    if submitted:
        st.button("Try the same questions again", on_click=retry_quiz)
    else:
        st.button("Check my answers", type="primary", on_click=submit_quiz)

    download_button("Download MCQs with answers (TXT)", mcqs_to_text(mcqs), "mcqs", "dl_mcq_page")


def render_chat():
    page_header("💬", "Ask Your Notes", "Ask any question. Answers come only from your uploaded PDF.")
    require_document()
    engine = require_engine()

    if not st.session_state.chat:
        st.info("Try: *What is the main idea of these notes?* or *Explain the first topic in simple words.*")

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your notes...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            history = list(st.session_state.chat)
            answer = run_ai(
                lambda: engine.answer_question(st.session_state.doc_text, question, history),
                "Searching your notes...",
            )
            if answer:
                st.markdown(answer)
        if answer:
            st.session_state.chat.append({"role": "user", "content": question})
            st.session_state.chat.append({"role": "assistant", "content": answer})

    if st.session_state.chat:
        def clear_chat():
            st.session_state.chat = []

        st.button("Clear chat", on_click=clear_chat)


def flip_card():
    st.session_state.fc_show = not st.session_state.fc_show


def move_card(step, total):
    st.session_state.fc_index = (st.session_state.fc_index + step) % total
    st.session_state.fc_show = False


def render_cards():
    page_header("🃏", "Flashcards", "Study one card at a time. Try to answer before you flip it.")
    require_document()
    engine = require_engine()
    note_if_truncated()

    count = st.slider("Number of flashcards", min_value=5, max_value=20, value=10)
    if st.button("Generate flashcards", type="primary"):
        result = run_ai(lambda: engine.generate_flashcards(st.session_state.doc_text, count), "Making your flashcards...")
        if result:
            st.session_state.flashcards = result
            st.session_state.fc_index = 0
            st.session_state.fc_show = False

    cards = st.session_state.flashcards
    if not cards:
        st.info("Choose how many cards you want and click **Generate flashcards**.")
        return

    total = len(cards)
    index = min(st.session_state.fc_index, total - 1)
    card = cards[index]

    st.markdown(f'<div class="fc-label">Question {index + 1} of {total}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="fc-question"><p>{html.escape(card["question"])}</p></div>', unsafe_allow_html=True)

    if st.session_state.fc_show:
        st.markdown(f'<div class="fc-answer"><div class="fc-label">Answer</div><p>{html.escape(card["answer"])}</p></div>',
                    unsafe_allow_html=True)

    st.write("")
    left, middle, right = st.columns(3)
    left.button("Previous", key="fc_prev", on_click=move_card, args=(-1, total))
    middle.button("Hide answer" if st.session_state.fc_show else "Show answer", key="fc_flip",
                  type="primary", on_click=flip_card)
    right.button("Next", key="fc_next", on_click=move_card, args=(1, total))

    with st.expander("See all flashcards"):
        for number, item in enumerate(cards, start=1):
            st.markdown(f"**{number}. {item['question']}**")
            st.markdown(item["answer"])
            st.divider()

    download_button("Download flashcards (TXT)", flashcards_to_text(cards), "flashcards", "dl_cards_page")


def render_downloads():
    page_header("⬇️", "Downloads", "Save your study material as TXT files.")
    require_document()
    s = st.session_state

    items = [
        ("Summary", s.summary, "summary"),
        ("Important points", s.points, "important_points"),
        ("MCQs with answers", mcqs_to_text(s.mcqs) if s.mcqs else "", "mcqs"),
        ("Flashcards", flashcards_to_text(s.flashcards) if s.flashcards else "", "flashcards"),
        ("Chat history", chat_to_text(s.chat) if s.chat else "", "chat"),
    ]

    available = [item for item in items if item[1]]
    if not available:
        st.info("Nothing to download yet. Create a summary, quiz or flashcards first, then come back here.")
        return

    st.markdown("**Complete study pack**")
    st.caption("Everything you have created for this document in one file.")
    download_button("Download complete study pack (TXT)", build_study_pack(), "study_pack", "dl_pack")

    st.divider()
    st.markdown("**Individual files**")
    for label, text, suffix in available:
        download_button(f"Download {label.lower()} (TXT)", text, suffix, f"dl_{suffix}")

    missing = [label for label, text, _ in items if not text]
    if missing:
        st.caption("Not created yet: " + ", ".join(missing) + ".")


# ---------------------------------------------------------------------------
# Run the app
# ---------------------------------------------------------------------------
render_sidebar()

PAGE_FUNCTIONS = {
    PAGE_HOME: render_home,
    PAGE_SUMMARY: render_summary,
    PAGE_POINTS: render_points,
    PAGE_MCQ: render_mcq,
    PAGE_CHAT: render_chat,
    PAGE_CARDS: render_cards,
    PAGE_DOWNLOADS: render_downloads,
}
PAGE_FUNCTIONS[st.session_state.page]()

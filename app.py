"""Streamlit interface for the Local AI Study Assistant.

This module only handles presentation: collecting input, calling the service
layer in ``src/local_study_assistant`` and rendering the results. All document
processing and model interaction happens locally.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# Make the src directory available for Python imports
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from local_study_assistant.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    MAX_CONTEXT_CHARACTERS,
    AppConfig,
)
from local_study_assistant.document_loader import (
    LoadedDocument,
    load_pasted_text,
    load_pdf_document,
    total_character_count,
)
from local_study_assistant.exceptions import (
    DocumentError,
    ResponseParsingError,
    StudyAssistantError,
)
from local_study_assistant.ollama_client import OllamaClient
from local_study_assistant.response_parser import (
    CHOICE_LABELS,
    Flashcard,
    QuizQuestion,
    flashcards_to_markdown,
    quiz_to_markdown,
    to_json,
)
from local_study_assistant.study_service import AnswerResult, StudyService

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

PAGE_TITLE = "Local AI Study Assistant"
MODEL_LIST_CACHE_SECONDS = 20

MODE_SUMMARY = "Summary"
MODE_FLASHCARDS = "Flashcards"
MODE_QUIZ = "Multiple-choice quiz"
MODE_ASK = "Ask the documents"
MODES = (MODE_SUMMARY, MODE_FLASHCARDS, MODE_QUIZ, MODE_ASK)

FLASHCARD_COUNTS = (5, 10, 15, 20)
QUIZ_COUNTS = (5, 10, 15)

# Session-state keys, named once so a typo cannot silently create a new entry.
STATE_RESULT = "result"
STATE_RESULT_MODE = "result_mode"
STATE_RAW_RESPONSE = "raw_response"
STATE_PASTED_TEXT = "pasted_text"
STATE_QUESTION = "question"


# --------------------------------------------------------------------------- #
# Cached helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def extract_pdf(file_name: str, data: bytes) -> LoadedDocument:
    """Extract text from PDF bytes, cached so reruns do not re-parse the file."""
    return load_pdf_document(file_name, data)


@st.cache_data(show_spinner=False, ttl=MODEL_LIST_CACHE_SECONDS)
def fetch_models(base_url: str, timeout: int) -> tuple[list[str], str]:
    """Return the installed models and an error message (empty when fine)."""
    client = OllamaClient(base_url=base_url, timeout=timeout)
    try:
        return client.list_models(), ""
    except StudyAssistantError as exc:
        return [], str(exc)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(config: AppConfig) -> tuple[str, AppConfig]:
    """Render the sidebar and return the selected model and effective settings."""
    st.sidebar.header("Local model")

    models, error = fetch_models(config.ollama_base_url, config.ollama_timeout)

    if error:
        st.sidebar.error(error)
        st.sidebar.caption(
            "Install Ollama from https://ollama.com/download, then run "
            "`ollama pull qwen2.5:7b` in a terminal."
        )
        selected_model = ""
    elif not models:
        st.sidebar.warning(
            "Ollama is running but no model is installed. "
            "Run `ollama pull qwen2.5:7b` in a terminal, then refresh."
        )
        selected_model = ""
    else:
        st.sidebar.success(f"Connected to Ollama · {len(models)} model(s) available")
        default_index = models.index(config.ollama_model) if config.ollama_model in models else 0
        selected_model = st.sidebar.selectbox("Model", models, index=default_index)

    if st.sidebar.button("Refresh model list", width="stretch"):
        fetch_models.clear()
        st.rerun()

    st.sidebar.caption(f"Endpoint: {config.ollama_base_url}")

    with st.sidebar.expander("Retrieval settings"):
        chunk_size = st.slider("Chunk size (characters)", 500, 3000, DEFAULT_CHUNK_SIZE, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, DEFAULT_CHUNK_OVERLAP, step=50)
        top_k = st.slider("Passages sent to the model", 1, 10, DEFAULT_TOP_K)

    st.sidebar.divider()
    if st.sidebar.button("Clear session", width="stretch"):
        clear_session()

    effective_config = AppConfig(
        ollama_base_url=config.ollama_base_url,
        ollama_model=selected_model,
        ollama_timeout=config.ollama_timeout,
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, chunk_size - 1),
        top_k=top_k,
    )
    return selected_model, effective_config


def clear_session() -> None:
    """Drop every generated result and cached extraction, then rerun."""
    for key in (STATE_RESULT, STATE_RESULT_MODE, STATE_RAW_RESPONSE):
        st.session_state.pop(key, None)
    st.session_state[STATE_PASTED_TEXT] = ""
    st.session_state[STATE_QUESTION] = ""
    extract_pdf.clear()
    st.rerun()


# --------------------------------------------------------------------------- #
# Input collection
# --------------------------------------------------------------------------- #
def collect_documents() -> list[LoadedDocument]:
    """Read uploaded PDFs and pasted text into in-memory documents."""
    documents: list[LoadedDocument] = []

    uploads = st.file_uploader(
        "Lecture PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are read in memory and never saved to disk.",
    )
    for upload in uploads or []:
        try:
            documents.append(extract_pdf(upload.name, upload.getvalue()))
        except DocumentError as exc:
            st.warning(str(exc))

    pasted = st.text_area(
        "Or paste text",
        key=STATE_PASTED_TEXT,
        height=160,
        placeholder="Paste lecture notes here. Pasted text is combined with any uploaded PDFs.",
    )
    if pasted and pasted.strip():
        try:
            documents.append(load_pasted_text(pasted))
        except DocumentError as exc:
            st.warning(str(exc))

    return documents


def render_document_overview(documents: list[LoadedDocument]) -> None:
    """Show the loaded documents and their approximate sizes."""
    if not documents:
        st.info("Upload at least one PDF or paste some text to get started.")
        return

    total = total_character_count(documents)
    st.success(f"{len(documents)} document(s) loaded · about {total:,} characters of text")

    if total > MAX_CONTEXT_CHARACTERS:
        st.info(
            f"Summary, flashcards and quiz use the first {MAX_CONTEXT_CHARACTERS:,} characters so "
            "the request fits the local model's context window. "
            "'Ask the documents' searches all of the text."
        )

    with st.expander("Loaded documents"):
        for document in documents:
            pages = f"{document.page_count} page(s) · " if document.page_count else ""
            st.write(f"**{document.name}** — {pages}{document.character_count:,} characters")


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #
def render_summary(summary: str) -> None:
    """Render a Markdown summary with a download button."""
    st.markdown(summary)
    st.download_button(
        "Download summary (Markdown)",
        data=summary,
        file_name="summary.md",
        mime="text/markdown",
    )


def render_flashcards(cards: list[Flashcard]) -> None:
    """Render flashcards with hidden answers and download buttons."""
    st.caption(f"{len(cards)} flashcard(s) generated.")
    for number, card in enumerate(cards, start=1):
        with st.expander(f"{number}. {card.question}"):
            st.write(card.answer)

    left, right = st.columns(2)
    left.download_button(
        "Download JSON",
        data=to_json(cards),
        file_name="flashcards.json",
        mime="application/json",
    )
    right.download_button(
        "Download Markdown",
        data=flashcards_to_markdown(cards),
        file_name="flashcards.md",
        mime="text/markdown",
    )


def render_quiz(questions: list[QuizQuestion]) -> None:
    """Render quiz questions, keeping each answer behind an expander."""
    st.caption(f"{len(questions)} question(s) generated.")
    for number, question in enumerate(questions, start=1):
        st.markdown(f"**{number}. {question.question}**")
        for label, choice in zip(CHOICE_LABELS, question.choices, strict=False):
            st.markdown(f"- **{label})** {choice}")
        with st.expander("Show answer"):
            correct_label = CHOICE_LABELS[question.correct_index]
            st.markdown(f"**{correct_label}) {question.correct_answer}**")
            st.caption(question.explanation)
        st.divider()

    left, right = st.columns(2)
    left.download_button(
        "Download JSON",
        data=to_json(questions),
        file_name="quiz.json",
        mime="application/json",
    )
    right.download_button(
        "Download Markdown",
        data=quiz_to_markdown(questions),
        file_name="quiz.md",
        mime="text/markdown",
    )


def render_answer(result: AnswerResult) -> None:
    """Render an answer plus the retrieved passages and their scores."""
    st.markdown(result.answer)

    if result.passages:
        with st.expander(f"Retrieved passages ({len(result.passages)})"):
            for number, retrieved in enumerate(result.passages, start=1):
                st.markdown(
                    f"**Passage {number}** · source: `{retrieved.source}` · "
                    f"similarity: {retrieved.score:.3f}"
                )
                st.text(retrieved.text)
                st.divider()
    else:
        st.caption("No passage in your documents matched this question.")

    st.download_button(
        "Download answer (Markdown)",
        data=result.answer,
        file_name="answer.md",
        mime="text/markdown",
    )


def render_result() -> None:
    """Render whatever result is currently held in session state."""
    result = st.session_state.get(STATE_RESULT)
    if result is None:
        return

    mode = st.session_state.get(STATE_RESULT_MODE)
    st.subheader(f"Result — {mode}")

    if mode == MODE_SUMMARY:
        render_summary(result)
    elif mode == MODE_FLASHCARDS:
        render_flashcards(result)
    elif mode == MODE_QUIZ:
        render_quiz(result)
    elif mode == MODE_ASK:
        render_answer(result)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate(
    service: StudyService,
    mode: str,
    documents: list[LoadedDocument],
    model: str,
    item_count: int,
    question: str,
) -> object:
    """Dispatch to the service method matching the selected mode."""
    if mode == MODE_SUMMARY:
        return service.summarise(documents, model)
    if mode == MODE_FLASHCARDS:
        return service.create_flashcards(documents, model, item_count)
    if mode == MODE_QUIZ:
        return service.create_quiz(documents, model, item_count)
    return service.answer_question(documents, model, question)


def run_generation(
    service: StudyService,
    mode: str,
    documents: list[LoadedDocument],
    model: str,
    item_count: int,
    question: str,
) -> None:
    """Run a generation request and store the result or a friendly error."""
    st.session_state.pop(STATE_RESULT, None)
    st.session_state.pop(STATE_RAW_RESPONSE, None)

    try:
        with st.spinner(f"Generating with {model}. This runs locally and may take a moment…"):
            result = generate(service, mode, documents, model, item_count, question)
    except ResponseParsingError as exc:
        st.error(str(exc))
        st.session_state[STATE_RAW_RESPONSE] = exc.raw_response
    except StudyAssistantError as exc:
        st.error(str(exc))
    else:
        st.session_state[STATE_RESULT] = result
        st.session_state[STATE_RESULT_MODE] = mode
        st.toast("Study material ready", icon="✅")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    """Compose and run the Streamlit page."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon="📚", layout="wide")
    config = AppConfig.from_env()

    st.title(PAGE_TITLE)
    st.write(
        "Turn lecture PDFs and notes into summaries, flashcards, practice quizzes and "
        "grounded answers — using a language model running on this machine."
    )
    st.caption(
        "Privacy: documents are processed in memory, sent only to your local Ollama server, "
        "and never uploaded, stored or logged."
    )
    st.divider()

    model, effective_config = render_sidebar(config)
    client = OllamaClient(
        base_url=effective_config.ollama_base_url,
        timeout=effective_config.ollama_timeout,
    )
    service = StudyService(client, effective_config)

    input_column, action_column = st.columns([3, 2], gap="large")

    with input_column:
        st.subheader("1. Study material")
        documents = collect_documents()
        render_document_overview(documents)

    with action_column:
        st.subheader("2. Study mode")
        mode = st.radio("Mode", MODES, label_visibility="collapsed")

        item_count = 0
        question = ""
        if mode == MODE_FLASHCARDS:
            item_count = st.select_slider("Number of flashcards", FLASHCARD_COUNTS, value=10)
        elif mode == MODE_QUIZ:
            item_count = st.select_slider("Number of questions", QUIZ_COUNTS, value=5)
        elif mode == MODE_ASK:
            question = st.text_input(
                "Your question",
                key=STATE_QUESTION,
                placeholder="What is the difference between a process and a thread?",
            )

        blocked = not documents or not model or (mode == MODE_ASK and not question.strip())
        if st.button("Generate", type="primary", width="stretch", disabled=blocked):
            run_generation(service, mode, documents, model, item_count, question)

        if not model:
            st.caption("Select an installed Ollama model in the sidebar to enable generation.")

    st.divider()
    render_result()

    raw_response = st.session_state.get(STATE_RAW_RESPONSE)
    if raw_response:
        with st.expander("Raw model response (could not be parsed)"):
            st.code(raw_response)

    with st.expander("Technical details"):
        st.markdown(
            f"""
            - **Model endpoint:** `{effective_config.ollama_base_url}` (local HTTP API)
            - **Selected model:** `{model or "none"}`
            - **Request timeout:** {effective_config.ollama_timeout}s
            - **Chunking:** {effective_config.chunk_size} characters,
              {effective_config.chunk_overlap} overlap
            - **Retrieval:** TF-IDF term weighting with cosine similarity
              (lexical keyword matching, not semantic embeddings),
              top {effective_config.top_k} passages
            - **Persistence:** none — documents live in memory for this session only
            """
        )


if __name__ == "__main__":
    main()

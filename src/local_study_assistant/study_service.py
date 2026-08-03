"""Application layer that turns documents into study material.

This module is the only place that knows how the pieces fit together: it builds
the context, picks the prompt, calls the local model and parses the reply. The
Streamlit interface calls these methods and renders the results, so the business
logic stays testable and independent of the UI framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import (
    MAX_CONTEXT_CHARACTERS,
    STRUCTURED_TASK_TEMPERATURE,
    SUMMARY_TEMPERATURE,
    AppConfig,
)
from .document_loader import LoadedDocument
from .exceptions import EmptyDocumentError
from .ollama_client import OllamaClient
from .prompts import (
    build_flashcard_prompt,
    build_question_answering_prompt,
    build_quiz_prompt,
    build_summary_prompt,
)
from .response_parser import (
    Flashcard,
    QuizQuestion,
    parse_flashcards,
    parse_quiz_questions,
)
from .retrieval import RetrievedPassage, TfidfRetriever, build_passages

LOGGER = logging.getLogger(__name__)

_TRUNCATION_NOTICE = "\n\n[Material truncated to fit the local model's context window.]"

#: Answer returned verbatim when retrieval finds no matching passage, so the
#: model is never asked to answer from an empty context.
NO_CONTEXT_ANSWER = "The answer was not found in the provided documents."


@dataclass(frozen=True)
class DocumentContext:
    """Concatenated study material prepared for a whole-document task."""

    text: str
    truncated: bool


@dataclass(frozen=True)
class AnswerResult:
    """An answer together with the passages it was based on."""

    answer: str
    passages: list[RetrievedPassage]


def build_document_context(
    documents: list[LoadedDocument],
    max_characters: int = MAX_CONTEXT_CHARACTERS,
) -> DocumentContext:
    """Join documents into one context string, truncating if needed.

    Each document keeps a heading so the model can attribute content to a source.

    Raises:
        EmptyDocumentError: If no document carries any text.
    """
    sections = [
        f"### Document: {document.name}\n{document.text}"
        for document in documents
        if document.text.strip()
    ]
    if not sections:
        raise EmptyDocumentError("Add a PDF or paste some text before generating study material.")

    combined = "\n\n".join(sections)
    if len(combined) <= max_characters:
        return DocumentContext(text=combined, truncated=False)

    LOGGER.info("Truncating context from %d to %d characters", len(combined), max_characters)
    return DocumentContext(text=combined[:max_characters] + _TRUNCATION_NOTICE, truncated=True)


def format_passages_for_prompt(passages: list[RetrievedPassage]) -> str:
    """Render retrieved passages as a numbered, attributed context block."""
    return "\n\n".join(
        f"[Passage {number} — source: {retrieved.source}]\n{retrieved.text}"
        for number, retrieved in enumerate(passages, start=1)
    )


class StudyService:
    """Generates summaries, flashcards, quizzes and grounded answers."""

    def __init__(self, client: OllamaClient, config: AppConfig) -> None:
        """Store the model client and the runtime configuration."""
        self._client = client
        self._config = config

    def summarise(self, documents: list[LoadedDocument], model: str) -> str:
        """Return a Markdown revision summary of the documents."""
        context = build_document_context(documents)
        prompt = build_summary_prompt(context.text)
        return self._client.chat(prompt, model=model, temperature=SUMMARY_TEMPERATURE)

    def create_flashcards(
        self, documents: list[LoadedDocument], model: str, card_count: int
    ) -> list[Flashcard]:
        """Generate and parse flashcards from the documents."""
        context = build_document_context(documents)
        prompt = build_flashcard_prompt(context.text, card_count)
        raw = self._client.chat(prompt, model=model, temperature=STRUCTURED_TASK_TEMPERATURE)
        return parse_flashcards(raw)

    def create_quiz(
        self, documents: list[LoadedDocument], model: str, question_count: int
    ) -> list[QuizQuestion]:
        """Generate and parse multiple-choice questions from the documents."""
        context = build_document_context(documents)
        prompt = build_quiz_prompt(context.text, question_count)
        raw = self._client.chat(prompt, model=model, temperature=STRUCTURED_TASK_TEMPERATURE)
        return parse_quiz_questions(raw)

    def answer_question(
        self,
        documents: list[LoadedDocument],
        model: str,
        question: str,
        top_k: int | None = None,
    ) -> AnswerResult:
        """Answer a question using TF-IDF retrieval over the documents.

        Only the highest-scoring passages are sent to the model. If retrieval
        finds nothing, the fixed "not found" answer is returned without calling
        the model at all.
        """
        passages = build_passages(
            documents,
            chunk_size=self._config.chunk_size,
            overlap=self._config.chunk_overlap,
        )
        retriever = TfidfRetriever(passages)
        retrieved = retriever.retrieve(question, top_k=top_k or self._config.top_k)

        if not retrieved:
            LOGGER.info("No passage matched the question; skipping the model call")
            return AnswerResult(answer=NO_CONTEXT_ANSWER, passages=[])

        prompt = build_question_answering_prompt(format_passages_for_prompt(retrieved), question)
        answer = self._client.chat(prompt, model=model, temperature=STRUCTURED_TASK_TEMPERATURE)
        return AnswerResult(answer=answer, passages=retrieved)

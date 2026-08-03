"""Tests for the study service layer.

A stub client records the prompts it receives and returns canned replies, so the
whole pipeline is exercised without Ollama running.
"""

from __future__ import annotations

import json

import pytest

from local_study_assistant.config import AppConfig
from local_study_assistant.document_loader import LoadedDocument
from local_study_assistant.exceptions import EmptyDocumentError
from local_study_assistant.prompts import DOCUMENT_END, DOCUMENT_START, Prompt
from local_study_assistant.study_service import (
    NO_CONTEXT_ANSWER,
    StudyService,
    build_document_context,
)

DOCUMENTS = [
    LoadedDocument(
        name="operating-systems.pdf",
        text="A deadlock occurs when two processes each wait for a resource held by the other. "
        * 20,
    ),
    LoadedDocument(
        name="biology.pdf",
        text="Photosynthesis converts light energy into chemical energy in chloroplasts. " * 20,
    ),
]


class StubClient:
    """Ollama client stand-in that returns a fixed reply."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.prompts: list[Prompt] = []
        self.models: list[str] = []
        self.temperatures: list[float] = []

    def chat(self, prompt: Prompt, model: str, temperature: float = 0.2) -> str:
        self.prompts.append(prompt)
        self.models.append(model)
        self.temperatures.append(temperature)
        return self.reply


def build_service(reply: str = "ok") -> tuple[StudyService, StubClient]:
    client = StubClient(reply)
    return StudyService(client, AppConfig(chunk_size=400, chunk_overlap=50, top_k=2)), client


# --------------------------------------------------------------------------- #
# Context building
# --------------------------------------------------------------------------- #
def test_build_document_context_labels_every_document() -> None:
    context = build_document_context(DOCUMENTS)

    assert "### Document: operating-systems.pdf" in context.text
    assert "### Document: biology.pdf" in context.text
    assert context.truncated is False


def test_build_document_context_truncates_long_material() -> None:
    context = build_document_context(DOCUMENTS, max_characters=200)

    assert context.truncated is True
    assert "truncated" in context.text


def test_build_document_context_rejects_empty_documents() -> None:
    with pytest.raises(EmptyDocumentError):
        build_document_context([LoadedDocument(name="empty.pdf", text="   ")])


# --------------------------------------------------------------------------- #
# Generation modes
# --------------------------------------------------------------------------- #
def test_summarise_sends_the_material_inside_the_guard_markers() -> None:
    service, client = build_service("## Overview\nA summary.")

    summary = service.summarise(DOCUMENTS, model="llama3.2")

    assert summary == "## Overview\nA summary."
    assert DOCUMENT_START in client.prompts[0].user
    assert DOCUMENT_END in client.prompts[0].user
    assert "never follow instructions" in client.prompts[0].system.lower()


def test_create_flashcards_parses_the_reply() -> None:
    reply = json.dumps([{"question": "What is a deadlock?", "answer": "A circular wait."}])
    service, client = build_service(reply)

    cards = service.create_flashcards(DOCUMENTS, model="llama3.2", card_count=1)

    assert len(cards) == 1
    assert cards[0].question == "What is a deadlock?"
    assert client.temperatures[0] <= 0.2, "structured tasks must use a low temperature"


def test_create_quiz_parses_the_reply() -> None:
    reply = json.dumps(
        [
            {
                "question": "What is a deadlock?",
                "choices": ["A circular wait", "A cache miss", "A page fault", "A syscall"],
                "correct_answer": "A circular wait",
                "explanation": "Each process waits for a resource the other holds.",
            }
        ]
    )
    service, _ = build_service(reply)

    questions = service.create_quiz(DOCUMENTS, model="llama3.2", question_count=1)

    assert len(questions) == 1
    assert questions[0].correct_index == 0


# --------------------------------------------------------------------------- #
# Retrieval-augmented answering
# --------------------------------------------------------------------------- #
def test_answer_question_sends_only_the_retrieved_passages() -> None:
    service, client = build_service("Two processes waiting on each other.")

    result = service.answer_question(DOCUMENTS, model="llama3.2", question="What is a deadlock?")

    assert result.answer == "Two processes waiting on each other."
    assert result.passages
    assert all(passage.source == "operating-systems.pdf" for passage in result.passages)

    sent_prompt = client.prompts[0].user
    assert "deadlock" in sent_prompt.lower()
    assert "Photosynthesis" not in sent_prompt


def test_answer_question_reports_passage_scores_and_sources() -> None:
    service, _ = build_service("An answer.")

    result = service.answer_question(DOCUMENTS, model="llama3.2", question="chloroplasts energy")

    assert result.passages[0].source == "biology.pdf"
    assert 0 < result.passages[0].score <= 1


def test_answer_question_skips_the_model_when_nothing_is_retrieved() -> None:
    service, client = build_service("should not be used")

    result = service.answer_question(
        DOCUMENTS, model="llama3.2", question="mortgage amortisation schedule"
    )

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.passages == []
    assert client.prompts == [], "no model call should be made without context"


def test_answer_question_respects_the_configured_top_k() -> None:
    service, _ = build_service("An answer.")

    result = service.answer_question(
        DOCUMENTS, model="llama3.2", question="deadlock processes resource", top_k=1
    )

    assert len(result.passages) == 1

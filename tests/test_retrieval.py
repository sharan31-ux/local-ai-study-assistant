"""Tests for TF-IDF passage retrieval."""

from __future__ import annotations

import pytest

from local_study_assistant.document_loader import LoadedDocument
from local_study_assistant.exceptions import RetrievalError
from local_study_assistant.retrieval import Passage, TfidfRetriever, build_passages

PASSAGES = [
    Passage(
        text="A binary search tree stores keys so that the left subtree holds smaller keys.",
        source="algorithms.pdf",
        chunk_index=0,
    ),
    Passage(
        text="Photosynthesis converts light energy into chemical energy inside chloroplasts.",
        source="biology.pdf",
        chunk_index=0,
    ),
    Passage(
        text="A deadlock occurs when two processes each wait for a resource the other holds.",
        source="operating-systems.pdf",
        chunk_index=0,
    ),
]


def test_retrieve_ranks_the_relevant_passage_first() -> None:
    retriever = TfidfRetriever(PASSAGES)

    results = retriever.retrieve("What causes a deadlock between processes?", top_k=2)

    assert results
    assert results[0].source == "operating-systems.pdf"
    assert results[0].score > 0


def test_retrieve_respects_top_k() -> None:
    retriever = TfidfRetriever(PASSAGES)

    results = retriever.retrieve("energy light chemical binary deadlock", top_k=2)

    assert len(results) <= 2


def test_retrieve_returns_scores_in_descending_order() -> None:
    retriever = TfidfRetriever(PASSAGES)

    results = retriever.retrieve("binary search tree keys subtree", top_k=3)

    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_drops_passages_with_no_term_overlap() -> None:
    retriever = TfidfRetriever(PASSAGES)

    results = retriever.retrieve("chloroplasts", top_k=3)

    assert len(results) == 1
    assert results[0].source == "biology.pdf"


def test_retrieve_returns_nothing_for_an_unrelated_question() -> None:
    retriever = TfidfRetriever(PASSAGES)

    assert retriever.retrieve("mortgage amortisation schedule", top_k=3) == []


def test_building_a_retriever_without_passages_fails() -> None:
    with pytest.raises(RetrievalError):
        TfidfRetriever([])


def test_retrieve_rejects_an_empty_question() -> None:
    retriever = TfidfRetriever(PASSAGES)

    with pytest.raises(RetrievalError):
        retriever.retrieve("   ")


def test_retrieve_rejects_a_non_positive_top_k() -> None:
    retriever = TfidfRetriever(PASSAGES)

    with pytest.raises(RetrievalError):
        retriever.retrieve("deadlock", top_k=0)


def test_build_passages_keeps_the_source_document_name() -> None:
    documents = [
        LoadedDocument(name="week1.pdf", text="Threads share the memory of their process. " * 40),
        LoadedDocument(name="week2.pdf", text="A semaphore guards a critical section. " * 40),
    ]

    passages = build_passages(documents, chunk_size=300, overlap=50)

    assert len(passages) > 2
    assert {passage.source for passage in passages} == {"week1.pdf", "week2.pdf"}
    assert all(passage.text for passage in passages)


def test_build_passages_ignores_documents_without_text() -> None:
    passages = build_passages([LoadedDocument(name="empty.pdf", text="")])

    assert passages == []

"""Tests for parsing structured model replies."""

from __future__ import annotations

import json

import pytest

from local_study_assistant.exceptions import ResponseParsingError
from local_study_assistant.response_parser import (
    Flashcard,
    QuizQuestion,
    extract_json_payload,
    flashcards_to_markdown,
    parse_flashcards,
    parse_quiz_questions,
    quiz_to_markdown,
    to_json,
)

VALID_FLASHCARDS = json.dumps(
    [
        {"question": "What is a thread?", "answer": "A unit of execution inside a process."},
        {"question": "What is a process?", "answer": "A program in execution with its own memory."},
    ]
)

VALID_QUIZ = json.dumps(
    [
        {
            "question": "Which structure keeps smaller keys in the left subtree?",
            "choices": ["Binary search tree", "Hash table", "Queue", "Stack"],
            "correct_answer": "Binary search tree",
            "explanation": "By definition its left subtree holds smaller keys.",
        }
    ]
)


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #
def test_extract_json_payload_from_plain_json() -> None:
    assert extract_json_payload('[{"a": 1}]') == '[{"a": 1}]'


def test_extract_json_payload_from_markdown_code_fence() -> None:
    raw = 'Here you go:\n```json\n[{"a": 1}]\n```\nHope that helps!'

    assert extract_json_payload(raw) == '[{"a": 1}]'


def test_extract_json_payload_from_unlabelled_code_fence() -> None:
    assert extract_json_payload('```\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_payload_ignores_surrounding_commentary() -> None:
    raw = 'Sure! [{"question": "q", "answer": "a"}] Let me know if you need more.'

    assert extract_json_payload(raw) == '[{"question": "q", "answer": "a"}]'


@pytest.mark.parametrize("raw", ["", "   ", "I cannot help with that request."])
def test_extract_json_payload_rejects_replies_without_json(raw: str) -> None:
    with pytest.raises(ResponseParsingError):
        extract_json_payload(raw)


# --------------------------------------------------------------------------- #
# Flashcards
# --------------------------------------------------------------------------- #
def test_parse_flashcards_from_valid_json() -> None:
    cards = parse_flashcards(VALID_FLASHCARDS)

    assert len(cards) == 2
    assert cards[0] == Flashcard(
        question="What is a thread?", answer="A unit of execution inside a process."
    )


def test_parse_flashcards_inside_code_fences() -> None:
    cards = parse_flashcards(f"```json\n{VALID_FLASHCARDS}\n```")

    assert len(cards) == 2


def test_parse_flashcards_accepts_an_object_wrapper() -> None:
    raw = json.dumps({"flashcards": [{"question": "q", "answer": "a"}]})

    assert parse_flashcards(raw) == [Flashcard(question="q", answer="a")]


def test_parse_flashcards_rejects_malformed_json() -> None:
    with pytest.raises(ResponseParsingError):
        parse_flashcards('[{"question": "missing closing brace"')


def test_parse_flashcards_rejects_cards_without_an_answer() -> None:
    with pytest.raises(ResponseParsingError):
        parse_flashcards('[{"question": "What is a thread?"}]')


def test_parse_flashcards_keeps_the_raw_response_on_failure() -> None:
    raw = "not json at all {"

    with pytest.raises(ResponseParsingError) as error:
        parse_flashcards(raw)

    assert error.value.raw_response == raw


# --------------------------------------------------------------------------- #
# Quiz
# --------------------------------------------------------------------------- #
def test_parse_quiz_questions_from_valid_json() -> None:
    questions = parse_quiz_questions(VALID_QUIZ)

    assert len(questions) == 1
    assert questions[0].choices == ["Binary search tree", "Hash table", "Queue", "Stack"]
    assert questions[0].correct_index == 0


def test_parse_quiz_questions_accepts_a_letter_as_the_correct_answer() -> None:
    raw = json.dumps(
        [
            {
                "question": "Pick the odd one out.",
                "choices": ["Alpha", "Beta", "Gamma", "Delta"],
                "correct_answer": "C)",
                "explanation": "Gamma is the odd one.",
            }
        ]
    )

    assert parse_quiz_questions(raw)[0].correct_answer == "Gamma"


def test_parse_quiz_questions_accepts_an_index_as_the_correct_answer() -> None:
    raw = json.dumps(
        [
            {
                "question": "Pick the odd one out.",
                "options": ["Alpha", "Beta", "Gamma", "Delta"],
                "correct_answer": 1,
                "explanation": "Beta is the odd one.",
            }
        ]
    )

    assert parse_quiz_questions(raw)[0].correct_answer == "Beta"


def test_parse_quiz_questions_rejects_the_wrong_number_of_choices() -> None:
    raw = json.dumps(
        [
            {
                "question": "Too few choices?",
                "choices": ["Yes", "No"],
                "correct_answer": "Yes",
                "explanation": "Only two choices were provided.",
            }
        ]
    )

    with pytest.raises(ResponseParsingError):
        parse_quiz_questions(raw)


def test_parse_quiz_questions_rejects_an_unmatched_correct_answer() -> None:
    raw = json.dumps(
        [
            {
                "question": "Which one?",
                "choices": ["Alpha", "Beta", "Gamma", "Delta"],
                "correct_answer": "Epsilon",
                "explanation": "Epsilon is not one of the choices.",
            }
        ]
    )

    with pytest.raises(ResponseParsingError):
        parse_quiz_questions(raw)


def test_parse_quiz_questions_rejects_malformed_json() -> None:
    with pytest.raises(ResponseParsingError):
        parse_quiz_questions("```json\n[{oops}]\n```")


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def test_to_json_round_trips_flashcards() -> None:
    cards = parse_flashcards(VALID_FLASHCARDS)

    restored = json.loads(to_json(cards))

    assert restored[0]["question"] == "What is a thread?"


def test_flashcards_to_markdown_contains_every_card() -> None:
    markdown = flashcards_to_markdown([Flashcard(question="Q1", answer="A1")])

    assert "Q1" in markdown
    assert "A1" in markdown


def test_quiz_to_markdown_contains_choices_and_an_answer_key() -> None:
    question = QuizQuestion(
        question="Which one?",
        choices=["Alpha", "Beta", "Gamma", "Delta"],
        correct_answer="Beta",
        explanation="Beta is correct.",
    )

    markdown = quiz_to_markdown([question])

    assert "- A) Alpha" in markdown
    assert "Answer key" in markdown
    assert "**B) Beta**" in markdown

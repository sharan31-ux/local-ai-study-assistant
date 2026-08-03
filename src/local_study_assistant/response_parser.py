"""Parsing and validation of structured model replies.

Local models often wrap JSON in Markdown code fences or add a sentence before
it. These helpers recover the JSON payload, validate its shape, and raise a
:class:`ResponseParsingError` that carries the raw reply so the interface can
show it instead of losing the generated content.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

from .exceptions import ResponseParsingError

LOGGER = logging.getLogger(__name__)

#: Number of answer choices every multiple-choice question must have.
CHOICES_PER_QUESTION = 4

_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
#: Display labels for the four answer choices, shared with the interface.
CHOICE_LABELS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class Flashcard:
    """A single question-and-answer study card."""

    question: str
    answer: str


@dataclass(frozen=True)
class QuizQuestion:
    """A multiple-choice question with exactly four choices."""

    question: str
    choices: list[str]
    correct_answer: str
    explanation: str

    @property
    def correct_index(self) -> int:
        """Zero-based position of the correct answer inside ``choices``."""
        return self.choices.index(self.correct_answer)


def extract_json_payload(raw_response: str) -> str:
    """Return the JSON substring of a model reply.

    Handles replies wrapped in ``` fences and replies with commentary around the
    JSON by slicing from the first opening bracket to the last closing one.

    Raises:
        ResponseParsingError: If no JSON-looking payload is present.
    """
    if not raw_response or not raw_response.strip():
        raise ResponseParsingError("The model returned an empty response.", raw_response)

    text = raw_response.strip()

    fenced = _CODE_FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    start_candidates = [position for position in (text.find("["), text.find("{")) if position != -1]
    end_candidates = [text.rfind("]"), text.rfind("}")]
    if not start_candidates or max(end_candidates) == -1:
        raise ResponseParsingError("The model reply did not contain any JSON data.", raw_response)

    start = min(start_candidates)
    end = max(end_candidates)
    if end <= start:
        raise ResponseParsingError("The model reply contained incomplete JSON.", raw_response)
    return text[start : end + 1]


def _load_json_list(raw_response: str, key: str) -> list[Any]:
    """Parse a reply into a list of items, tolerating a ``{key: [...]}`` wrapper."""
    payload = extract_json_payload(raw_response)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ResponseParsingError(
            f"The model reply was not valid JSON ({exc.msg}).", raw_response
        ) from exc

    if isinstance(parsed, dict):
        parsed = parsed.get(key, [])
    if not isinstance(parsed, list) or not parsed:
        raise ResponseParsingError(
            f"The model reply did not contain a list of {key}.", raw_response
        )
    return parsed


def _required_text(item: dict[str, Any], field: str, raw_response: str) -> str:
    """Return a non-empty string field from a parsed item."""
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ResponseParsingError(
            f"A generated item is missing the '{field}' field.", raw_response
        )
    return value.strip()


def parse_flashcards(raw_response: str) -> list[Flashcard]:
    """Parse a model reply into flashcards.

    Raises:
        ResponseParsingError: If the reply is not valid JSON or no card has both
            a question and an answer.
    """
    items = _load_json_list(raw_response, "flashcards")

    cards: list[Flashcard] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cards.append(
            Flashcard(
                question=_required_text(item, "question", raw_response),
                answer=_required_text(item, "answer", raw_response),
            )
        )

    if not cards:
        raise ResponseParsingError("No usable flashcards were found in the reply.", raw_response)
    return cards


def _normalise_correct_answer(item: dict[str, Any], choices: list[str], raw_response: str) -> str:
    """Resolve the declared correct answer to one of the choice strings.

    Accepts the answer as the full choice text, as a letter (``"B"``, ``"B)"``)
    or as a zero-based index, because models are inconsistent about this.
    """
    value = item.get("correct_answer", item.get("answer"))

    if isinstance(value, int) and 0 <= value < len(choices):
        return choices[value]

    if isinstance(value, str):
        candidate = value.strip()
        for choice in choices:
            if candidate.casefold() == choice.casefold():
                return choice
        letter = candidate.rstrip(").:").strip().upper()
        if letter in CHOICE_LABELS[: len(choices)]:
            return choices[CHOICE_LABELS.index(letter)]

    raise ResponseParsingError(
        "A generated question has a correct answer that does not match its choices.",
        raw_response,
    )


def parse_quiz_questions(raw_response: str) -> list[QuizQuestion]:
    """Parse a model reply into validated multiple-choice questions.

    Each question must have text, exactly four distinct choices, a correct
    answer that matches one of them, and an explanation.

    Raises:
        ResponseParsingError: If the reply is not valid JSON or no question
            satisfies the required structure.
    """
    items = _load_json_list(raw_response, "questions")

    questions: list[QuizQuestion] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        raw_choices = item.get("choices", item.get("options"))
        if not isinstance(raw_choices, list) or len(raw_choices) != CHOICES_PER_QUESTION:
            raise ResponseParsingError(
                f"A generated question does not have exactly {CHOICES_PER_QUESTION} choices.",
                raw_response,
            )
        choices = [str(choice).strip() for choice in raw_choices]
        if any(not choice for choice in choices):
            raise ResponseParsingError("A generated question has an empty choice.", raw_response)

        questions.append(
            QuizQuestion(
                question=_required_text(item, "question", raw_response),
                choices=choices,
                correct_answer=_normalise_correct_answer(item, choices, raw_response),
                explanation=_required_text(item, "explanation", raw_response),
            )
        )

    if not questions:
        raise ResponseParsingError("No usable questions were found in the reply.", raw_response)
    return questions


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def to_json(items: list[Flashcard] | list[QuizQuestion]) -> str:
    """Serialise generated study material as indented JSON."""
    return json.dumps([asdict(item) for item in items], indent=2, ensure_ascii=False)


def flashcards_to_markdown(cards: list[Flashcard]) -> str:
    """Render flashcards as a Markdown study sheet."""
    lines = ["# Flashcards", ""]
    for number, card in enumerate(cards, start=1):
        lines += [f"## {number}. {card.question}", "", card.answer, ""]
    return "\n".join(lines)


def quiz_to_markdown(questions: list[QuizQuestion]) -> str:
    """Render a quiz as Markdown with an answer key at the end."""
    lines = ["# Practice quiz", ""]
    for number, question in enumerate(questions, start=1):
        lines.append(f"**{number}. {question.question}**")
        lines.append("")
        for label, choice in zip(CHOICE_LABELS, question.choices, strict=False):
            lines.append(f"- {label}) {choice}")
        lines.append("")

    lines += ["---", "", "## Answer key", ""]
    for number, question in enumerate(questions, start=1):
        label = CHOICE_LABELS[question.correct_index]
        lines.append(f"{number}. **{label}) {question.correct_answer}** — {question.explanation}")
    return "\n".join(lines)

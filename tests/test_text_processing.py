"""Tests for text cleaning and chunking."""

from __future__ import annotations

import pytest

from local_study_assistant.exceptions import ChunkingConfigurationError
from local_study_assistant.text_processing import chunk_text, clean_text

LOREM_SENTENCE = "A process is an independent program in execution with its own memory space. "


def _long_text(sentences: int) -> str:
    return LOREM_SENTENCE * sentences


# --------------------------------------------------------------------------- #
# clean_text
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["", "   ", "\n\n\t  \n"])
def test_clean_text_returns_empty_string_for_blank_input(value: str) -> None:
    assert clean_text(value) == ""


def test_clean_text_collapses_spaces_and_keeps_paragraphs() -> None:
    raw = "Chapter    one\n\n\n\nSecond   paragraph here.  "

    cleaned = clean_text(raw)

    assert cleaned == "Chapter one\n\nSecond paragraph here."
    assert "\n\n" in cleaned, "paragraph break must survive cleaning"


def test_clean_text_removes_control_characters_and_rejoins_hyphenation() -> None:
    raw = "The algo-\nrithm is \x00efficient."

    assert clean_text(raw) == "The algorithm is efficient."


# --------------------------------------------------------------------------- #
# chunk_text
# --------------------------------------------------------------------------- #
def test_chunk_text_returns_empty_list_for_blank_text() -> None:
    assert chunk_text("   ") == []


def test_chunk_text_returns_single_chunk_for_short_text() -> None:
    text = "Short lecture note."

    assert chunk_text(text, chunk_size=500, overlap=50) == [text]


def test_chunk_text_splits_long_text_into_bounded_chunks() -> None:
    chunk_size = 400
    chunks = chunk_text(_long_text(40), chunk_size=chunk_size, overlap=50)

    assert len(chunks) > 1
    assert all(len(chunk) <= chunk_size for chunk in chunks)


def test_chunk_text_overlaps_consecutive_chunks() -> None:
    chunks = chunk_text(_long_text(30), chunk_size=400, overlap=100)

    first_tail = chunks[0][-60:]
    assert first_tail in chunks[1], "the end of a chunk should reappear in the next one"


def test_chunk_text_covers_all_content() -> None:
    text = _long_text(20)

    chunks = chunk_text(text, chunk_size=300, overlap=50)

    # Concatenating the chunks must reproduce every word of the source text.
    joined = " ".join(chunks)
    assert all(word in joined for word in text.split())


def test_chunk_text_terminates_with_zero_overlap() -> None:
    chunks = chunk_text(_long_text(10), chunk_size=200, overlap=0)

    assert len(chunks) > 1


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (50, 10),  # below the minimum chunk size
        (1000, -1),  # negative overlap
        (1000, 1000),  # overlap equal to the chunk size would loop forever
        (1000, 1200),  # overlap larger than the chunk size
    ],
)
def test_chunk_text_rejects_invalid_configuration(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ChunkingConfigurationError):
        chunk_text("some text", chunk_size=chunk_size, overlap=overlap)

"""Tests for in-memory document loading.

PDF bytes used here are generated at runtime with pypdf, so no test depends on
an external file.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter

from local_study_assistant.document_loader import (
    PASTED_TEXT_NAME,
    LoadedDocument,
    load_pasted_text,
    load_pdf_document,
    total_character_count,
)
from local_study_assistant.exceptions import EmptyDocumentError, PdfReadError


def build_blank_pdf(pages: int = 1) -> bytes:
    """Return a valid PDF whose pages contain no extractable text."""
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_load_pdf_document_rejects_a_pdf_without_extractable_text() -> None:
    with pytest.raises(EmptyDocumentError) as error:
        load_pdf_document("scan.pdf", build_blank_pdf())

    assert "OCR" in str(error.value)


def test_load_pdf_document_rejects_bytes_that_are_not_a_pdf() -> None:
    with pytest.raises(PdfReadError):
        load_pdf_document("notes.pdf", b"this is plain text, not a PDF")


def test_load_pdf_document_uses_only_the_file_name_not_the_path() -> None:
    with pytest.raises(EmptyDocumentError) as error:
        load_pdf_document("/Users/someone/private/folder/lecture.pdf", build_blank_pdf())

    message = str(error.value)
    assert "lecture.pdf" in message
    assert "/Users/someone" not in message, "error messages must not leak local paths"


def test_load_pasted_text_cleans_and_labels_the_document() -> None:
    document = load_pasted_text("  A   thread   is   lightweight.  ")

    assert document.name == PASTED_TEXT_NAME
    assert document.text == "A thread is lightweight."
    assert document.character_count == len("A thread is lightweight.")


@pytest.mark.parametrize("value", ["", "     ", "\n\n\n"])
def test_load_pasted_text_rejects_blank_input(value: str) -> None:
    with pytest.raises(EmptyDocumentError):
        load_pasted_text(value)


def test_total_character_count_sums_every_document() -> None:
    documents = [
        LoadedDocument(name="a.pdf", text="12345"),
        LoadedDocument(name="b.pdf", text="123"),
    ]

    assert total_character_count(documents) == 8

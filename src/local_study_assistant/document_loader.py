"""In-memory document loading.

Uploaded PDFs are read from bytes and never written to disk: nothing the user
uploads is persisted by this application.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import DependencyError
from pypdf.errors import PdfReadError as PyPdfReadError

from .exceptions import EmptyDocumentError, EncryptedPdfError, PdfReadError
from .text_processing import clean_text

LOGGER = logging.getLogger(__name__)

#: Display name used for text typed or pasted directly into the interface.
PASTED_TEXT_NAME = "Pasted text"


@dataclass(frozen=True)
class LoadedDocument:
    """A single piece of study material held in memory."""

    name: str
    text: str
    page_count: int = 0

    @property
    def character_count(self) -> int:
        """Number of characters of cleaned text in this document."""
        return len(self.text)


def load_pdf_document(file_name: str, data: bytes) -> LoadedDocument:
    """Extract cleaned text from PDF bytes.

    Args:
        file_name: Original file name, used only as a display label.
        data: Raw PDF bytes.

    Returns:
        The loaded document with cleaned text.

    Raises:
        EncryptedPdfError: If the PDF is password protected.
        PdfReadError: If the file is not a readable PDF.
        EmptyDocumentError: If the PDF contains no extractable text, which is
            typical for scanned pages that would need OCR.
    """
    display_name = Path(file_name).name or "document.pdf"

    try:
        reader = PdfReader(BytesIO(data))
        # An empty password unlocks many "protected" PDFs; anything else needs a
        # password this application deliberately does not ask for.
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise EncryptedPdfError(f"'{display_name}' is password protected and cannot be read.")
        page_texts = [page.extract_text() or "" for page in reader.pages]
    except EncryptedPdfError:
        raise
    except DependencyError as exc:
        raise EncryptedPdfError(
            f"'{display_name}' uses an encryption method that cannot be opened here."
        ) from exc
    except (PyPdfReadError, ValueError, OSError) as exc:
        LOGGER.warning("Failed to read PDF %s: %s", display_name, exc)
        raise PdfReadError(f"'{display_name}' could not be read as a PDF file.") from exc

    text = clean_text("\n\n".join(page_texts))
    if not text:
        raise EmptyDocumentError(
            f"No text could be extracted from '{display_name}'. "
            "Scanned or image-only PDFs need OCR, which this application does not perform."
        )

    LOGGER.info("Loaded %s: %d pages, %d characters", display_name, len(page_texts), len(text))
    return LoadedDocument(name=display_name, text=text, page_count=len(page_texts))


def load_pasted_text(text: str, name: str = PASTED_TEXT_NAME) -> LoadedDocument:
    """Turn pasted text into a document.

    Raises:
        EmptyDocumentError: If the text is empty once cleaned.
    """
    cleaned = clean_text(text)
    if not cleaned:
        raise EmptyDocumentError("The pasted text is empty.")
    return LoadedDocument(name=name, text=cleaned)


def total_character_count(documents: list[LoadedDocument]) -> int:
    """Return the combined character count of all documents."""
    return sum(document.character_count for document in documents)

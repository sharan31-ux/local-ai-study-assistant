"""Custom exceptions raised by the study assistant.

Every exception carries a message that is safe to display directly in the user
interface: it explains what went wrong and what the user can do about it,
without leaking stack traces or internal details.
"""

from __future__ import annotations


class StudyAssistantError(Exception):
    """Base class for every error raised by this application."""


# --------------------------------------------------------------------------- #
# Document handling
# --------------------------------------------------------------------------- #
class DocumentError(StudyAssistantError):
    """Base class for problems with the supplied study material."""


class PdfReadError(DocumentError):
    """Raised when a PDF file cannot be opened or parsed."""


class EncryptedPdfError(DocumentError):
    """Raised when a PDF is password protected and cannot be read."""


class EmptyDocumentError(DocumentError):
    """Raised when a document contains no extractable text."""


# --------------------------------------------------------------------------- #
# Text processing
# --------------------------------------------------------------------------- #
class ChunkingConfigurationError(StudyAssistantError, ValueError):
    """Raised when the chunk size or overlap values are not usable."""


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class RetrievalError(StudyAssistantError):
    """Raised when the TF-IDF index cannot be built or queried."""


# --------------------------------------------------------------------------- #
# Model responses
# --------------------------------------------------------------------------- #
class ResponseParsingError(StudyAssistantError):
    """Raised when the model reply cannot be parsed into the expected shape.

    The raw model reply is kept on the exception so the interface can offer it
    for inspection instead of losing the generated content.
    """

    def __init__(self, message: str, raw_response: str = "") -> None:
        """Store the user-facing message and the unparsed model reply."""
        super().__init__(message)
        self.raw_response = raw_response


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
class OllamaError(StudyAssistantError):
    """Base class for every failure while talking to the local Ollama server."""


class OllamaConnectionError(OllamaError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama does not answer within the configured timeout."""


class OllamaResponseError(OllamaError):
    """Raised on a non-success status code, invalid JSON or an empty reply."""


class NoModelsAvailableError(OllamaError):
    """Raised when Ollama is running but has no model installed."""

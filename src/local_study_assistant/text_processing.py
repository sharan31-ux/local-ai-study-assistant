"""Text cleaning and chunking helpers.

PDF extraction produces ragged text: stray control characters, single words on
their own line, and long runs of blank lines. These helpers normalise that text
without destroying paragraph structure, then split it into overlapping chunks
that are small enough to retrieve and to fit into a local model's context.
"""

from __future__ import annotations

import re

from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, MIN_CHUNK_SIZE
from .exceptions import ChunkingConfigurationError

#: Control characters that carry no meaning in extracted text (tab and newline
#: are deliberately excluded because they carry layout information).
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t ]+")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
#: Word split across a line break by PDF hyphenation, e.g. "algo-\nrithm".
_HYPHENATED_LINE_BREAK = re.compile(r"(\w)-\n(\w)")

#: Separators tried in order when looking for a readable chunk boundary.
_BOUNDARY_SEPARATORS: tuple[str, ...] = ("\n\n", ". ", ".\n", "! ", "? ", "\n", " ")

#: Only the last part of a chunk is searched for a boundary, so a chunk never
#: shrinks to a fraction of the requested size.
_BOUNDARY_SEARCH_RATIO = 0.6


def clean_text(text: str) -> str:
    """Normalise extracted text while preserving paragraph breaks.

    Collapses repeated spaces, removes control characters, rejoins words split
    by hyphenation at a line break, and limits runs of blank lines to one.

    Args:
        text: Raw text, typically from a PDF or a paste area.

    Returns:
        Cleaned text, or an empty string if nothing meaningful is left.
    """
    if not text or not text.strip():
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_CHARACTERS.sub("", cleaned)
    cleaned = _HYPHENATED_LINE_BREAK.sub(r"\1\2", cleaned)
    cleaned = _HORIZONTAL_WHITESPACE.sub(" ", cleaned)
    cleaned = _TRAILING_SPACES.sub("\n", cleaned)
    cleaned = _EXCESS_BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()


def _validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    """Raise :class:`ChunkingConfigurationError` for unusable settings."""
    if chunk_size < MIN_CHUNK_SIZE:
        raise ChunkingConfigurationError(
            f"Chunk size must be at least {MIN_CHUNK_SIZE} characters, got {chunk_size}."
        )
    if overlap < 0:
        raise ChunkingConfigurationError(f"Chunk overlap cannot be negative, got {overlap}.")
    if overlap >= chunk_size:
        raise ChunkingConfigurationError(
            f"Chunk overlap ({overlap}) must be smaller than the chunk size ({chunk_size})."
        )


def _find_boundary(text: str, start: int, hard_end: int) -> int:
    """Return the best readable cut point at or before ``hard_end``.

    Searches the tail of the candidate chunk for a paragraph break, sentence end
    or space. Falls back to ``hard_end`` when no separator is found, which keeps
    the function total and the caller loop-safe.
    """
    earliest = start + int((hard_end - start) * _BOUNDARY_SEARCH_RATIO)
    window = text[earliest:hard_end]
    for separator in _BOUNDARY_SEPARATORS:
        position = window.rfind(separator)
        if position != -1:
            return earliest + position + len(separator)
    return hard_end


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks that end on readable boundaries.

    The overlap keeps a sentence that straddles two chunks retrievable from
    either side.

    Args:
        text: Text to split. Empty or whitespace-only input yields ``[]``.
        chunk_size: Maximum characters per chunk.
        overlap: Characters repeated between consecutive chunks.

    Returns:
        A list of non-empty chunks in document order.

    Raises:
        ChunkingConfigurationError: If ``chunk_size`` is below the minimum, the
            overlap is negative, or the overlap is not smaller than the size.
    """
    _validate_chunk_settings(chunk_size, overlap)

    stripped = text.strip() if text else ""
    if not stripped:
        return []
    if len(stripped) <= chunk_size:
        return [stripped]

    chunks: list[str] = []
    total = len(stripped)
    start = 0

    while start < total:
        hard_end = min(start + chunk_size, total)
        end = hard_end if hard_end >= total else _find_boundary(stripped, start, hard_end)

        chunk = stripped[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= total:
            break

        # ``max`` guarantees forward progress even if a boundary lands early,
        # so the loop can never stall.
        start = max(end - overlap, start + 1)

    return chunks

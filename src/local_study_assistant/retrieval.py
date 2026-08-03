"""Lightweight retrieval over document chunks.

Retrieval uses TF-IDF term weighting with cosine similarity from scikit-learn.
This is keyword-based lexical matching, not semantic embedding search: a
question phrased with different vocabulary than the source text will match less
well. The trade-off is that it needs no embedding model, no vector database and
no network access, which keeps the whole application local and free to run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K
from .document_loader import LoadedDocument
from .exceptions import RetrievalError
from .text_processing import chunk_text

LOGGER = logging.getLogger(__name__)

#: Chunks scoring at or below this value share no meaningful term with the
#: question and are dropped instead of padding the context.
MIN_SIMILARITY_SCORE = 0.0


@dataclass(frozen=True)
class Passage:
    """A chunk of text together with the document it came from."""

    text: str
    source: str
    chunk_index: int


@dataclass(frozen=True)
class RetrievedPassage:
    """A passage ranked against a question."""

    passage: Passage
    score: float

    @property
    def text(self) -> str:
        """Text of the underlying passage."""
        return self.passage.text

    @property
    def source(self) -> str:
        """Name of the document this passage came from."""
        return self.passage.source


def build_passages(
    documents: list[LoadedDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Passage]:
    """Split every document into passages, keeping the source name on each."""
    passages: list[Passage] = []
    for document in documents:
        for index, chunk in enumerate(chunk_text(document.text, chunk_size, overlap)):
            passages.append(Passage(text=chunk, source=document.name, chunk_index=index))
    return passages


class TfidfRetriever:
    """Ranks passages against a question using TF-IDF and cosine similarity."""

    def __init__(self, passages: list[Passage]) -> None:
        """Fit the TF-IDF matrix over the supplied passages.

        Raises:
            RetrievalError: If no passages are supplied, or if the passages
                contain no usable vocabulary (for example only stop words).
        """
        if not passages:
            raise RetrievalError("There is no document text available to search.")

        self._passages = passages
        # ``stop_words`` removes English filler terms; word bigrams let short
        # multi-word concepts ("binary search") outrank the same words apart.
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        try:
            self._matrix = self._vectorizer.fit_transform(p.text for p in passages)
        except ValueError as exc:
            raise RetrievalError(
                "The document text does not contain enough distinct words to search."
            ) from exc

    @property
    def passage_count(self) -> int:
        """Number of indexed passages."""
        return len(self._passages)

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedPassage]:
        """Return the ``top_k`` passages most similar to the question.

        Passages with no term overlap are excluded, so the caller can detect an
        unanswerable question from an empty result.

        Raises:
            RetrievalError: If the question is empty or ``top_k`` is not positive.
        """
        if not question or not question.strip():
            raise RetrievalError("Enter a question before searching the documents.")
        if top_k <= 0:
            raise RetrievalError(
                f"The number of passages to retrieve must be positive, got {top_k}."
            )

        query_vector = self._vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self._matrix)[0]

        ranked_indices = scores.argsort()[::-1][:top_k]
        results = [
            RetrievedPassage(passage=self._passages[index], score=float(scores[index]))
            for index in ranked_indices
            if scores[index] > MIN_SIMILARITY_SCORE
        ]
        LOGGER.info("Retrieved %d of %d passages", len(results), len(self._passages))
        return results

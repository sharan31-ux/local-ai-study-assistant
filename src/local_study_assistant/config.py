"""Application configuration and shared constants.

Configuration comes from environment variables so the application can be tuned
without editing code. Every value has a safe default, and an optional ``.env``
file at the project root is loaded on startup. No secrets are involved: Ollama
runs locally and requires no credentials.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4

#: Smallest chunk size that still produces useful retrieval passages.
MIN_CHUNK_SIZE = 200

#: Upper bound on the characters sent to the model for whole-document tasks
#: (summary, flashcards, quiz). Keeps requests inside a typical local context
#: window; longer material is truncated and the user is told about it.
MAX_CONTEXT_CHARACTERS = 12_000

#: Low temperature keeps JSON-producing tasks predictable.
STRUCTURED_TASK_TEMPERATURE = 0.1
#: Slightly higher temperature for prose, still conservative.
SUMMARY_TEMPERATURE = 0.2

ENV_FILE_NAME = ".env"

ENV_BASE_URL = "OLLAMA_BASE_URL"
ENV_MODEL = "OLLAMA_MODEL"
ENV_TIMEOUT = "OLLAMA_TIMEOUT"


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def load_env_file(env_path: Path | None = None) -> None:
    """Load ``KEY=VALUE`` pairs from a ``.env`` file into the environment.

    Existing environment variables win, so an explicitly exported value is never
    overwritten. A missing file is not an error: every setting has a default.
    """
    path = env_path if env_path is not None else project_root() / ENV_FILE_NAME
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _read_int_env(name: str, default: int) -> int:
    """Read a positive integer environment variable, falling back on default."""
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        LOGGER.warning("Ignoring invalid %s=%r, using %d", name, raw_value, default)
        return default
    if value <= 0:
        LOGGER.warning("Ignoring non-positive %s=%d, using %d", name, value, default)
        return default
    return value


@dataclass(frozen=True)
class AppConfig:
    """Immutable runtime settings for the study assistant."""

    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = ""
    ollama_timeout: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K

    @classmethod
    def from_env(cls, load_dotenv: bool = True) -> AppConfig:
        """Build a configuration from environment variables."""
        if load_dotenv:
            load_env_file()
        return cls(
            ollama_base_url=os.environ.get(ENV_BASE_URL, DEFAULT_OLLAMA_BASE_URL).strip()
            or DEFAULT_OLLAMA_BASE_URL,
            ollama_model=os.environ.get(ENV_MODEL, "").strip(),
            ollama_timeout=_read_int_env(ENV_TIMEOUT, DEFAULT_OLLAMA_TIMEOUT_SECONDS),
        )

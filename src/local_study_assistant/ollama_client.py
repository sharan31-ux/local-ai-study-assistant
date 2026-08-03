"""HTTP client for a locally running Ollama server.

Ollama exposes a plain HTTP API on ``localhost``. This client wraps the two
endpoints the application needs — ``/api/tags`` to list installed models and
``/api/chat`` to generate text — and converts every network, protocol and
payload failure into a :class:`~local_study_assistant.exceptions.OllamaError`
with a message that is safe to show in the interface. Nothing here ever raises a
bare ``requests`` exception, so the Streamlit app cannot crash on a failed call.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from .config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_TIMEOUT_SECONDS
from .exceptions import (
    OllamaConnectionError,
    OllamaError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from .prompts import Prompt

LOGGER = logging.getLogger(__name__)

TAGS_ENDPOINT = "/api/tags"
CHAT_ENDPOINT = "/api/chat"

#: Listing models is a cheap call, so it should fail fast rather than block the
#: interface while the connection status is checked.
_LIST_MODELS_TIMEOUT_SECONDS = 5

_CONNECTION_HINT = (
    "Could not reach Ollama at {base_url}. Make sure Ollama is installed and running "
    "(`ollama serve`), then try again."
)


class OllamaClient:
    """Small, dependency-light client for the local Ollama HTTP API."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        """Create a client.

        Args:
            base_url: Root URL of the Ollama server.
            timeout: Seconds to wait for a generation response.
            session: Optional pre-built session. Injecting one keeps the client
                testable without patching the ``requests`` module.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session if session is not None else requests.Session()

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #
    def list_models(self) -> list[str]:
        """Return the names of the models installed in Ollama, sorted.

        Raises:
            OllamaConnectionError: If the server cannot be reached.
            OllamaTimeoutError: If the server does not answer in time.
            OllamaResponseError: On a non-success status or an unreadable body.
        """
        payload = self._request("GET", TAGS_ENDPOINT, timeout=_LIST_MODELS_TIMEOUT_SECONDS)
        models = payload.get("models")
        if not isinstance(models, list):
            raise OllamaResponseError("Ollama returned an unexpected model list.")

        names = sorted(
            str(entry["name"]) for entry in models if isinstance(entry, dict) and entry.get("name")
        )
        LOGGER.info("Ollama reported %d installed model(s)", len(names))
        return names

    def is_available(self) -> bool:
        """Return ``True`` when the Ollama server answers a model listing."""
        try:
            self.list_models()
        except OllamaError as exc:
            LOGGER.info("Ollama availability check failed: %s", exc)
            return False
        return True

    def chat(self, prompt: Prompt, model: str, temperature: float = 0.2) -> str:
        """Send a prompt to a model and return its text reply.

        Args:
            prompt: System and user messages to send.
            model: Name of an installed Ollama model.
            temperature: Sampling temperature; low values suit structured output.

        Returns:
            The assistant message content, stripped of surrounding whitespace.

        Raises:
            OllamaConnectionError: If the server cannot be reached.
            OllamaTimeoutError: If generation exceeds the configured timeout.
            OllamaResponseError: On a non-success status, invalid JSON, or an
                empty reply.
        """
        if not model:
            raise OllamaResponseError("No Ollama model was selected.")

        body = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "options": {"temperature": temperature},
        }

        payload = self._request("POST", CHAT_ENDPOINT, json_body=body, timeout=self.timeout)
        message = payload.get("message")
        content = message.get("content", "") if isinstance(message, dict) else ""
        if not isinstance(content, str) or not content.strip():
            raise OllamaResponseError(
                f"The model '{model}' returned an empty response. Try again or pick another model."
            )
        return content.strip()

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #
    def _request(
        self,
        method: str,
        endpoint: str,
        timeout: int,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform an HTTP request and return the decoded JSON object."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._session.request(method, url, json=json_body, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            # Checked before ConnectionError: ConnectTimeout subclasses both.
            raise OllamaTimeoutError(
                f"Ollama did not respond within {timeout} seconds. "
                "Large documents and large models take longer; try a shorter document, "
                "a smaller model, or raise OLLAMA_TIMEOUT."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(_CONNECTION_HINT.format(base_url=self.base_url)) from exc
        except requests.exceptions.RequestException as exc:
            LOGGER.warning("Request to %s failed: %s", url, exc)
            raise OllamaResponseError(
                f"The request to Ollama failed: {type(exc).__name__}."
            ) from exc

        if response.status_code != requests.codes.ok:
            raise OllamaResponseError(
                f"Ollama returned HTTP {response.status_code} for {endpoint}. "
                "Check that the selected model is installed (`ollama list`)."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaResponseError("Ollama returned a response that is not valid JSON.") from exc

        if not isinstance(payload, dict):
            raise OllamaResponseError("Ollama returned an unexpected response format.")
        return payload

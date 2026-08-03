"""Tests for the Ollama HTTP client.

No test contacts the network: a fake session stands in for ``requests.Session``
and is injected through the client constructor.
"""

from __future__ import annotations

from typing import Any

import pytest
import requests

from local_study_assistant.exceptions import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from local_study_assistant.ollama_client import OllamaClient
from local_study_assistant.prompts import Prompt

PROMPT = Prompt(system="You are a study assistant.", user="Summarise this.")


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, status_code: int = 200, payload: Any = None, invalid_json: bool = False):
        self.status_code = status_code
        self._payload = payload
        self._invalid_json = invalid_json

    def json(self) -> Any:
        if self._invalid_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


class FakeSession:
    """Session that returns a canned response or raises a canned exception."""

    def __init__(self, response: FakeResponse | None = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def build_client(response: FakeResponse | None = None, error: Exception | None = None):
    session = FakeSession(response=response, error=error)
    return OllamaClient(base_url="http://localhost:11434", timeout=30, session=session), session


# --------------------------------------------------------------------------- #
# list_models
# --------------------------------------------------------------------------- #
def test_list_models_returns_sorted_names() -> None:
    payload = {"models": [{"name": "qwen2.5:7b"}, {"name": "llama3.2:latest"}]}
    client, session = build_client(FakeResponse(payload=payload))

    assert client.list_models() == ["llama3.2:latest", "qwen2.5:7b"]
    assert session.calls[0]["url"] == "http://localhost:11434/api/tags"


def test_list_models_rejects_an_unexpected_payload() -> None:
    client, _ = build_client(FakeResponse(payload={"unexpected": True}))

    with pytest.raises(OllamaResponseError):
        client.list_models()


def test_is_available_is_false_when_ollama_is_down() -> None:
    client, _ = build_client(error=requests.exceptions.ConnectionError("refused"))

    assert client.is_available() is False


def test_is_available_is_true_when_models_are_listed() -> None:
    client, _ = build_client(FakeResponse(payload={"models": [{"name": "llama3.2"}]}))

    assert client.is_available() is True


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #
def test_chat_returns_the_message_content() -> None:
    payload = {"message": {"role": "assistant", "content": "  A short summary.  "}}
    client, session = build_client(FakeResponse(payload=payload))

    assert client.chat(PROMPT, model="llama3.2", temperature=0.1) == "A short summary."

    body = session.calls[0]["json"]
    assert body["model"] == "llama3.2"
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.1
    assert [message["role"] for message in body["messages"]] == ["system", "user"]


def test_chat_uses_the_configured_timeout() -> None:
    client, session = build_client(FakeResponse(payload={"message": {"content": "ok"}}))

    client.chat(PROMPT, model="llama3.2")

    assert session.calls[0]["timeout"] == 30


def test_chat_raises_a_connection_error_when_ollama_is_not_running() -> None:
    client, _ = build_client(error=requests.exceptions.ConnectionError("refused"))

    with pytest.raises(OllamaConnectionError) as error:
        client.chat(PROMPT, model="llama3.2")

    assert "http://localhost:11434" in str(error.value)


def test_chat_raises_a_timeout_error_when_generation_is_too_slow() -> None:
    client, _ = build_client(error=requests.exceptions.ReadTimeout("too slow"))

    with pytest.raises(OllamaTimeoutError):
        client.chat(PROMPT, model="llama3.2")


def test_connect_timeout_is_reported_as_a_timeout_not_a_connection_error() -> None:
    # ConnectTimeout subclasses both ConnectionError and Timeout, so the order
    # of the except blocks in the client matters.
    client, _ = build_client(error=requests.exceptions.ConnectTimeout("slow connect"))

    with pytest.raises(OllamaTimeoutError):
        client.chat(PROMPT, model="llama3.2")


def test_chat_raises_on_a_non_success_status_code() -> None:
    client, _ = build_client(FakeResponse(status_code=404, payload={"error": "model not found"}))

    with pytest.raises(OllamaResponseError) as error:
        client.chat(PROMPT, model="missing-model")

    assert "404" in str(error.value)


def test_chat_raises_on_invalid_json() -> None:
    client, _ = build_client(FakeResponse(invalid_json=True))

    with pytest.raises(OllamaResponseError):
        client.chat(PROMPT, model="llama3.2")


@pytest.mark.parametrize(
    "payload",
    [{"message": {"content": ""}}, {"message": {"content": "   "}}, {"done": True}],
)
def test_chat_raises_on_an_empty_reply(payload: dict[str, Any]) -> None:
    client, _ = build_client(FakeResponse(payload=payload))

    with pytest.raises(OllamaResponseError):
        client.chat(PROMPT, model="llama3.2")


def test_chat_requires_a_model_name() -> None:
    client, _ = build_client(FakeResponse(payload={"message": {"content": "ok"}}))

    with pytest.raises(OllamaResponseError):
        client.chat(PROMPT, model="")


def test_generic_request_failure_is_wrapped() -> None:
    client, _ = build_client(error=requests.exceptions.TooManyRedirects("loop"))

    with pytest.raises(OllamaResponseError):
        client.chat(PROMPT, model="llama3.2")


def test_base_url_trailing_slash_is_normalised() -> None:
    session = FakeSession(FakeResponse(payload={"models": []}))
    client = OllamaClient(base_url="http://localhost:11434/", session=session)

    client.list_models()

    assert session.calls[0]["url"] == "http://localhost:11434/api/tags"

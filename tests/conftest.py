"""Shared fixtures. Every test runs with networking disabled and remote env cleared."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from ragkit import remote

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS = REPO_ROOT / "samples" / "docs"
SAMPLE_QA = REPO_ROOT / "samples" / "qa.json"


class NetworkBlocked(RuntimeError):
    pass


def _blocked(*args, **kwargs):
    raise NetworkBlocked("network access attempted during tests")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        remote.BASE_URL_ENV,
        remote.API_KEY_ENV,
        remote.LLM_MODEL_ENV,
        remote.EMBED_MODEL_ENV,
        remote.EMBEDDINGS_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

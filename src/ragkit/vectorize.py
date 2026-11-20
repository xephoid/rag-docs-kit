"""Turn text into vectors.

The default backend is a small pure-Python TF-IDF vectorizer (sublinear term
frequency, smoothed IDF, L2-normalised sparse vectors). It has no dependencies
and is deterministic. An OpenAI-compatible embedding backend can be selected
with ``RAGKIT_EMBEDDINGS=openai``; it is never used unless asked for.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Protocol

from ragkit.remote import EMBEDDINGS_ENV, RemoteConfig, Transport, remote_config_from_env

SparseVector = dict[str, float]
DenseVector = list[float]
Vector = SparseVector | DenseVector

TOKEN_RE = re.compile(r"[^\W_]+")

STOPWORDS = frozenset(
    """
    a an and are as at be by can do does for from has have how i if in into is it its
    me my of on or our so that the their them then there these they this to was we
    what when where which who why will with you your
    """.split()  # noqa: SIM905
)


def stem(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, strip plural 's'."""
    tokens = TOKEN_RE.findall(text.lower())
    return [stem(t) for t in tokens if t not in STOPWORDS]


class Vectorizer(Protocol):
    """Interface every vector backend implements."""

    name: str

    def fit(self, texts: list[str]) -> None: ...

    def transform(self, texts: list[str]) -> list[Vector]: ...

    def state(self) -> dict[str, Any]: ...


class TfidfVectorizer:
    name = "tfidf"

    def __init__(self, idf: dict[str, float] | None = None, n_docs: int = 0) -> None:
        self.idf: dict[str, float] = dict(idf or {})
        self.n_docs = n_docs

    def fit(self, texts: list[str]) -> None:
        df: Counter[str] = Counter()
        for text in texts:
            df.update(set(tokenize(text)))
        self.n_docs = len(texts)
        self.idf = {
            term: math.log((1 + self.n_docs) / (1 + count)) + 1.0
            for term, count in sorted(df.items())
        }

    def transform(self, texts: list[str]) -> list[Vector]:
        # Terms never seen during fit get the maximum IDF. They cannot match any
        # chunk, but they still count toward the vector norm, so a query made
        # mostly of unknown words scores low instead of being judged only on
        # the few words the corpus happens to share with it.
        unseen_idf = math.log(1 + self.n_docs) + 1.0
        vectors: list[Vector] = []
        for text in texts:
            counts = Counter(tokenize(text))
            weights = {
                term: (1.0 + math.log(count)) * self.idf.get(term, unseen_idf)
                for term, count in counts.items()
            }
            norm = math.sqrt(sum(w * w for w in weights.values()))
            vectors.append({t: w / norm for t, w in sorted(weights.items())} if norm else {})
        return vectors

    def state(self) -> dict[str, Any]:
        return {"idf": self.idf, "n_docs": self.n_docs}


class OpenAIEmbeddingVectorizer:
    """Embeddings from an OpenAI-compatible ``/embeddings`` endpoint (network!)."""

    name = "openai"

    def __init__(self, config: RemoteConfig, transport: Transport | None = None) -> None:
        self.config = config
        self.transport = transport

    def fit(self, texts: list[str]) -> None:
        return None

    def transform(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        transport = self.transport
        if transport is None:
            from ragkit.remote import urllib_transport

            transport = urllib_transport
        response = transport(
            self.config.endpoint("embeddings"),
            {"model": self.config.embed_model, "input": texts},
            self.config.api_key,
        )
        rows = sorted(response["data"], key=lambda row: row["index"])
        return [list(map(float, row["embedding"])) for row in rows]

    def state(self) -> dict[str, Any]:
        return {"model": self.config.embed_model}


def make_vectorizer(name: str | None = None, state: dict[str, Any] | None = None) -> Vectorizer:
    """Build a vectorizer by name (default from ``RAGKIT_EMBEDDINGS``, else tfidf)."""
    name = (name or os.environ.get(EMBEDDINGS_ENV) or "tfidf").lower()
    if name == "tfidf":
        state = state or {}
        return TfidfVectorizer(idf=state.get("idf"), n_docs=state.get("n_docs", 0))
    if name == "openai":
        config = remote_config_from_env()
        if config is None:
            raise RuntimeError(
                "The openai embedding backend needs OPENAI_COMPATIBLE_BASE_URL and API_KEY."
            )
        if state and state.get("model"):
            config = config.model_copy(update={"embed_model": state["model"]})
        return OpenAIEmbeddingVectorizer(config)
    raise ValueError(f"Unknown vectorizer backend: {name!r}")

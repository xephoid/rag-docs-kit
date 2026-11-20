"""In-memory vector store with JSONL persistence and cosine top-k search.

File layout (one JSON object per line)::

    {"type": "meta", "version": 1, "backend": "tfidf", "state": {...}}
    {"type": "chunk", "chunk": {...}, "vector": {...} | [...]}
    ...
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ragkit.chunk import Chunk
from ragkit.vectorize import Vector, Vectorizer, make_vectorizer

STORE_VERSION = 1


class SearchResult(BaseModel):
    chunk: Chunk
    score: float


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity for two sparse (dict) or two dense (list) vectors."""
    if isinstance(a, dict) and isinstance(b, dict):
        if len(a) > len(b):
            a, b = b, a
        dot = sum(w * b.get(t, 0.0) for t, w in a.items())
        norm_a = math.sqrt(sum(w * w for w in a.values()))
        norm_b = math.sqrt(sum(w * w for w in b.values()))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError("Dense vectors must have the same dimension")
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
    else:
        raise TypeError("Cannot compare sparse and dense vectors")
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_index_text(chunk: Chunk) -> str:
    """Text used for vectorizing a chunk: heading path plus body."""
    return f"{chunk.heading}\n{chunk.text}" if chunk.heading else chunk.text


class VectorStore:
    def __init__(self, vectorizer: Vectorizer) -> None:
        self.vectorizer = vectorizer
        self.chunks: list[Chunk] = []
        self.vectors: list[Vector] = []

    @classmethod
    def build(cls, chunks: list[Chunk], vectorizer: Vectorizer) -> VectorStore:
        ids = [c.id for c in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Chunk ids must be unique")
        store = cls(vectorizer)
        texts = [chunk_index_text(c) for c in chunks]
        vectorizer.fit(texts)
        store.chunks = list(chunks)
        store.vectors = vectorizer.transform(texts)
        return store

    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.id == chunk_id), None)

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        """Return up to k chunks by cosine similarity, best first.

        Chunks with zero similarity share nothing with the query and are never
        returned. Ties are broken by chunk order so results are deterministic.
        """
        if k <= 0 or not self.chunks:
            return []
        [query_vector] = self.vectorizer.transform([query])
        scored = [(cosine(query_vector, vector), i) for i, vector in enumerate(self.vectors)]
        scored = [pair for pair in scored if pair[0] > 0.0]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [SearchResult(chunk=self.chunks[i], score=score) for score, i in scored[:k]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {
            "type": "meta",
            "version": STORE_VERSION,
            "backend": self.vectorizer.name,
            "state": self.vectorizer.state(),
        }
        # A unique, exclusively created temp file in the destination directory:
        # concurrent writers never share it, an existing file or symlink is never
        # opened, and the index is replaced only once the write has completed.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
                for chunk, vector in zip(self.chunks, self.vectors, strict=True):
                    row = {"type": "chunk", "chunk": chunk.model_dump(), "vector": vector}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> VectorStore:
        with path.open(encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        if not lines:
            raise ValueError(f"Empty store file: {path}")
        meta = json.loads(lines[0])
        if meta.get("type") != "meta" or meta.get("version") != STORE_VERSION:
            raise ValueError(f"Unrecognised store file: {path}")
        store = cls(make_vectorizer(meta["backend"], meta.get("state")))
        for line in lines[1:]:
            row = json.loads(line)
            store.chunks.append(Chunk.model_validate(row["chunk"]))
            store.vectors.append(row["vector"])
        return store

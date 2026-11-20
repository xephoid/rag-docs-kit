from __future__ import annotations

import math

import pytest

from ragkit.chunk import Chunk, chunk_document
from ragkit.ingest import load_documents
from ragkit.store import VectorStore, cosine
from ragkit.vectorize import TfidfVectorizer, make_vectorizer, tokenize

from .conftest import SAMPLE_DOCS


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, doc_id=cid.split("#")[0], heading="", text=text, ordinal=0, part=0)


@pytest.fixture
def small_store() -> VectorStore:
    chunks = [
        _chunk("a.md#cats/0", "Cats are small furry pets that purr."),
        _chunk("b.md#dogs/0", "Dogs are loyal pets that bark at the mailman."),
        _chunk("c.md#tax/0", "Quarterly tax filings are due in April."),
    ]
    return VectorStore.build(chunks, TfidfVectorizer())


def test_tokenize_lowercases_drops_stopwords_and_plurals():
    assert tokenize("The Invoices are in the listen_port!") == ["invoice", "listen", "port"]
    assert tokenize("naïve 東京") == ["naïve", "東京"]


def test_tfidf_vectors_are_unit_length():
    vec = TfidfVectorizer()
    vec.fit(["alpha beta", "beta gamma"])
    [v] = vec.transform(["alpha beta beta"])
    assert math.isclose(math.sqrt(sum(w * w for w in v.values())), 1.0)


def test_unseen_terms_lower_the_score():
    vec = TfidfVectorizer()
    vec.fit(["refund policy details", "unrelated words here"])
    [doc] = vec.transform(["refund policy details"])
    [known] = vec.transform(["refund policy"])
    [mixed] = vec.transform(["refund policy kubernetes autoscaling"])
    assert cosine(known, doc) > cosine(mixed, doc) > 0


def test_ranking_puts_best_match_first(small_store: VectorStore):
    results = small_store.search("which pets bark?", k=3)
    assert results[0].chunk.id == "b.md#dogs/0"
    assert results[0].score > results[1].score


def test_zero_similarity_chunks_are_not_returned(small_store: VectorStore):
    assert small_store.search("completely unrelated xylophone", k=3) == []
    ids = [r.chunk.id for r in small_store.search("tax", k=3)]
    assert ids == ["c.md#tax/0"]


def test_k_limits_results(small_store: VectorStore):
    assert len(small_store.search("pets", k=1)) == 1
    assert small_store.search("pets", k=0) == []


def test_cosine_handles_sparse_dense_and_zero_vectors():
    assert cosine({"a": 1.0}, {"a": 2.0}) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine({}, {"a": 1.0}) == 0.0
    with pytest.raises(TypeError):
        cosine({"a": 1.0}, [1.0])


def test_duplicate_chunk_ids_rejected():
    chunks = [_chunk("a.md#x/0", "one"), _chunk("a.md#x/0", "two")]
    with pytest.raises(ValueError):
        VectorStore.build(chunks, TfidfVectorizer())


def test_store_roundtrip_preserves_results(tmp_path):
    docs = load_documents(SAMPLE_DOCS)
    chunks = [c for d in docs for c in chunk_document(d)]
    store = VectorStore.build(chunks, make_vectorizer())
    path = tmp_path / "idx" / "store.jsonl"
    store.save(path)
    loaded = VectorStore.load(path)
    assert [c.id for c in loaded.chunks] == [c.id for c in store.chunks]
    query = "how do I request a refund"
    assert [(r.chunk.id, r.score) for r in loaded.search(query)] == [
        (r.chunk.id, r.score) for r in store.search(query)
    ]
    assert loaded.search(query)[0].chunk.id == "billing.md#refunds/0"


def test_load_rejects_bad_file(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"type": "chunk"}\n')
    with pytest.raises(ValueError):
        VectorStore.load(bad)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        make_vectorizer("nope")


def test_openai_backend_requires_env():
    with pytest.raises(RuntimeError):
        make_vectorizer("openai")


# --- issue 9: unique temp file, atomic replace, cleanup on failure ------------


def _saved_store(path):
    docs = load_documents(SAMPLE_DOCS)
    store = VectorStore.build([c for d in docs for c in chunk_document(d)], make_vectorizer())
    store.save(path)
    return store


def test_failed_save_preserves_previous_index_and_cleans_up(tmp_path):
    path = tmp_path / "index.jsonl"
    store = _saved_store(path)
    before = path.read_bytes()
    store.vectors[-1] = {"bad": object()}  # not JSON serialisable: fails mid-write
    with pytest.raises(TypeError):
        store.save(path)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.jsonl"]


def test_save_never_opens_a_planted_symlink_at_a_predictable_temp_path(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    path = tmp_path / "index.jsonl"
    (tmp_path / "index.jsonl.tmp").symlink_to(victim)
    _saved_store(path)
    assert victim.read_text() == "keep me"
    assert VectorStore.load(path).chunks
    assert not path.is_symlink()

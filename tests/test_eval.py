from __future__ import annotations

import pytest

from ragkit.chunk import chunk_document
from ragkit.eval import QAItem, context_precision, load_qa, recall_at_k, run_eval
from ragkit.ingest import load_documents
from ragkit.store import VectorStore
from ragkit.vectorize import make_vectorizer

from .conftest import SAMPLE_DOCS, SAMPLE_QA


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a"], k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["c", "d"], k=3) == 0.5
    assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0
    assert recall_at_k([], ["a"], k=4) == 0.0
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], k=1)


def test_context_precision_is_rank_aware():
    assert context_precision(["g", "x", "y"], ["g"], k=3) == 1.0
    assert context_precision(["x", "g", "y"], ["g"], k=3) == 0.5
    assert context_precision(["x", "y", "z", "g"], ["g"], k=4) == 0.25
    # Gold at ranks 1 and 3: (1/1 + 2/3) / 2
    assert context_precision(["g1", "x", "g2"], ["g1", "g2"], k=3) == pytest.approx(5 / 6)
    assert context_precision(["x", "y"], ["g"], k=2) == 0.0
    assert context_precision(["x", "g"], ["g"], k=1) == 0.0


@pytest.fixture(scope="module")
def store() -> VectorStore:
    chunks = [c for d in load_documents(SAMPLE_DOCS) for c in chunk_document(d)]
    return VectorStore.build(chunks, make_vectorizer("tfidf"))


def test_sample_qa_is_valid_and_references_real_chunks(store):
    qa = load_qa(SAMPLE_QA)
    assert len(qa) == 10
    known = {c.id for c in store.chunks}
    assert all(set(item.gold_chunk_ids) <= known for item in qa)


def test_run_eval_report_shape_and_means(store):
    qa = load_qa(SAMPLE_QA)
    report = run_eval(store, qa, k=4)
    assert report.num_questions == 10 and report.k == 4
    assert len(report.rows) == 10
    assert all(len(r.retrieved_chunk_ids) <= 4 for r in report.rows)
    assert report.recall_at_k == pytest.approx(
        round(sum(r.recall_at_k for r in report.rows) / 10, 4)
    )
    assert 0.0 <= report.context_precision <= 1.0


def test_run_eval_rejects_unknown_gold_ids(store):
    qa = [QAItem(id="x", question="q", answer="a", gold_chunk_ids=["missing#chunk/0"])]
    with pytest.raises(ValueError):
        run_eval(store, qa)


# --- issue 7: answer-context and abstention metrics, threshold + revision -----


def test_answer_context_reflects_threshold(store):
    report = run_eval(store, load_qa(SAMPLE_QA), k=4, threshold=0.2)
    q10 = next(r for r in report.rows if r.id == "q10")
    assert q10.recall_at_k == 1.0  # raw retrieval finds both gold chunks
    assert q10.answer_context_recall == 0.5  # but billing.md#plans/0 is below 0.2
    assert report.recall_at_k == 0.9 and report.context_precision == 0.9
    assert report.answer_context_recall == 0.85
    assert report.threshold == 0.2


def test_abstention_metrics_by_category(store):
    qa = [
        QAItem(
            id="a",
            question="How long before a teammate invitation expires?",
            answer="7 days",
            gold_chunk_ids=["getting-started.md#inviting-teammates/0"],
        ),
        QAItem(id="o", question="Who won the 1998 World Cup?", answer="", category="out_of_domain"),
        QAItem(
            id="u",
            question="Can I extend teammate invitations to 90 days?",
            answer="",
            category="unanswerable",
        ),
    ]
    report = run_eval(store, qa, k=4)
    assert (report.num_answerable, report.num_out_of_domain, report.num_unanswerable) == (1, 1, 1)
    assert report.answer_rate == 1.0
    assert report.out_of_domain_abstention_rate == 1.0
    # Shares vocabulary with the docs, so the threshold does not catch it.
    assert report.unanswerable_abstention_rate == 0.0
    rows = {r.id: r for r in report.rows}
    assert rows["o"].recall_at_k is None and rows["o"].correct_behaviour
    assert not rows["u"].correct_behaviour


def test_qa_item_gold_must_match_category():
    with pytest.raises(ValueError):
        QAItem(id="x", question="q", answer="a")
    with pytest.raises(ValueError):
        QAItem(id="x", question="q", answer="a", category="unanswerable", gold_chunk_ids=["a#b/0"])


def test_report_records_dataset_revision(store, tmp_path):
    from ragkit.eval import dataset_sha256, format_summary

    sha = dataset_sha256(SAMPLE_QA)
    report = run_eval(store, load_qa(SAMPLE_QA), dataset="samples/qa.json", dataset_revision=sha)
    assert report.dataset_sha256 == sha and len(sha) == 64
    summary = format_summary(report)
    assert sha[:12] in summary and "threshold 0.2" in summary


def test_heldout_set_is_frozen_and_mixed(store):
    from .conftest import REPO_ROOT

    items = load_qa(REPO_ROOT / "samples" / "heldout_qa.json")
    cats = [i.category for i in items]
    assert cats.count("answerable") == 5
    assert cats.count("out_of_domain") == 5
    assert cats.count("unanswerable") == 5
    sample_questions = {i.question for i in load_qa(SAMPLE_QA)}
    assert not sample_questions & {i.question for i in items}


# --- issue 8: q07 asks for the fix, matching its reference answer -------------


def test_q07_question_matches_reference_answer():
    q07 = next(i for i in load_qa(SAMPLE_QA) if i.id == "q07")
    assert q07.question.startswith("How do I fix")
    assert q07.answer.startswith("Install a certificate")


def test_readme_eval_table_matches_a_fresh_run(store):
    from ragkit.eval import dataset_sha256, format_summary

    from .conftest import REPO_ROOT

    report = run_eval(
        store,
        load_qa(SAMPLE_QA),
        dataset="samples/qa.json",
        dataset_revision=dataset_sha256(SAMPLE_QA),
    )
    readme = (REPO_ROOT / "README.md").read_text()
    assert format_summary(report) in readme


# --- round 3, issue 6: no answerable rows -> aggregates are null, shown n/a ---


def test_no_answerable_questions_reports_null_aggregates(store):
    import json

    from ragkit.eval import format_summary

    qa = [
        QAItem(id="o", question="Who won the 1998 World Cup?", answer="", category="out_of_domain"),
        QAItem(
            id="u",
            question="Can I extend invitations to 90 days?",
            answer="",
            category="unanswerable",
        ),
    ]
    report = run_eval(store, qa, k=4)
    assert report.num_answerable == 0
    for field in (
        "recall_at_k",
        "context_precision",
        "answer_context_recall",
        "answer_context_precision",
    ):
        assert getattr(report, field) is None
        assert json.loads(report.model_dump_json())[field] is None
    summary = format_summary(report)
    assert "recall@4 n/a, context precision n/a" in summary
    assert "recall n/a, precision n/a" in summary
    assert "0.000" not in summary.splitlines()[-4] + summary.splitlines()[-3]

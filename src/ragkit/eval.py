"""Eval harness.

Each QA item has a category:

* ``answerable`` -- the docs answer it; ``gold_chunk_ids`` lists the chunks that do.
* ``out_of_domain`` -- unrelated to the docs; the right behaviour is to abstain.
* ``unanswerable`` -- on-topic (shares vocabulary with the docs) but the docs do
  not contain the answer; the right behaviour is to abstain.

Three groups of metrics are reported, kept separate on purpose:

1. **Raw retrieval** (answerable items): the top-k chunks from the store, before
   any threshold.

   * recall@k -- fraction of gold chunks that appear in the top k.
   * context precision@k -- rank-aware precision: the mean of precision@i over
     every rank i (1..k) that holds a gold chunk. 1.0 means every gold chunk
     that was retrieved sits above every non-gold chunk; 0.0 means none was.

2. **Answer context** (answerable items): the same two formulas applied to the
   chunks the extractive answer actually cites, i.e. what ``ask`` uses after the
   abstention threshold. A gold chunk retrieved below the threshold counts as a
   miss here.

3. **Abstention**: the share of answerable items that got an answer, and the
   share of out-of-domain and unanswerable items where ``ask`` abstained.
   Answering an unanswerable question is a failure even if the quoted text is
   on-topic.

The report records the threshold, the backend and the SHA-256 of the QA file so
numbers can be traced to an exact dataset revision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from ragkit.generate import DEFAULT_THRESHOLD, generate_extractive
from ragkit.store import VectorStore

Category = Literal["answerable", "out_of_domain", "unanswerable"]
_SHORT = {"answerable": "ans", "out_of_domain": "ood", "unanswerable": "unans"}


class QAItem(BaseModel):
    id: str
    question: str
    answer: str
    category: Category = "answerable"
    gold_chunk_ids: list[str] = []

    @model_validator(mode="after")
    def _gold_matches_category(self) -> QAItem:
        if self.category == "answerable" and not self.gold_chunk_ids:
            raise ValueError(f"answerable QA item {self.id} needs gold_chunk_ids")
        if self.category != "answerable" and self.gold_chunk_ids:
            raise ValueError(f"{self.category} QA item {self.id} must not have gold_chunk_ids")
        return self


class QuestionResult(BaseModel):
    id: str
    question: str
    category: Category
    gold_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    scores: list[float]
    recall_at_k: float | None
    context_precision: float | None
    answer_chunk_ids: list[str]
    answer_context_recall: float | None
    answer_context_precision: float | None
    abstained: bool
    correct_behaviour: bool


class EvalReport(BaseModel):
    dataset: str
    dataset_sha256: str
    backend: str
    k: int
    threshold: float
    num_questions: int
    num_answerable: int
    num_out_of_domain: int
    num_unanswerable: int
    # Raw retrieval (top-k before the threshold), answerable items only.
    # None when the set has no answerable items (nothing was measured).
    recall_at_k: float | None
    context_precision: float | None
    # Chunks the extractive answer cites (after the threshold), answerable items only.
    answer_context_recall: float | None
    answer_context_precision: float | None
    # Abstention behaviour. None when the set has no items of that category.
    answer_rate: float | None
    out_of_domain_abstention_rate: float | None
    unanswerable_abstention_rate: float | None
    rows: list[QuestionResult]


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("gold must not be empty")
    return len(gold_set & set(retrieved[:k])) / len(gold_set)


def context_precision(retrieved: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    hits = 0
    total = 0.0
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in gold_set:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


def parse_qa(raw: bytes) -> list[QAItem]:
    data = json.loads(raw.decode("utf-8"))
    return [QAItem.model_validate(item) for item in data]


def load_qa(path: Path) -> list[QAItem]:
    return parse_qa(path.read_bytes())


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def dataset_sha256(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def run_eval(
    store: VectorStore,
    qa: list[QAItem],
    k: int = 4,
    threshold: float = DEFAULT_THRESHOLD,
    dataset: str = "",
    dataset_revision: str = "",
) -> EvalReport:
    if k <= 0:
        raise ValueError("k must be positive")
    known = {c.id for c in store.chunks}
    rows = []
    for item in qa:
        missing = [g for g in item.gold_chunk_ids if g not in known]
        if missing:
            raise ValueError(f"QA item {item.id} references unknown chunk ids: {missing}")
        results = store.search(item.question, k=k)
        retrieved = [r.chunk.id for r in results]
        # Eval always uses the local extractive generator: offline and deterministic.
        answer = generate_extractive(item.question, results, threshold)
        cited = [c.chunk_id for c in answer.citations]
        gold = item.gold_chunk_ids
        answerable = item.category == "answerable"
        rows.append(
            QuestionResult(
                id=item.id,
                question=item.question,
                category=item.category,
                gold_chunk_ids=gold,
                retrieved_chunk_ids=retrieved,
                scores=[round(r.score, 4) for r in results],
                recall_at_k=recall_at_k(retrieved, gold, k) if answerable else None,
                context_precision=context_precision(retrieved, gold, k) if answerable else None,
                answer_chunk_ids=cited,
                answer_context_recall=recall_at_k(cited, gold, k) if answerable else None,
                answer_context_precision=(
                    context_precision(cited, gold, k) if answerable else None
                ),
                abstained=answer.abstained,
                correct_behaviour=(not answer.abstained) if answerable else answer.abstained,
            )
        )

    by_cat = {cat: [r for r in rows if r.category == cat] for cat in _SHORT}
    ans = by_cat["answerable"]

    def rate(group: list[QuestionResult]) -> float | None:
        return _mean([1.0 if r.correct_behaviour else 0.0 for r in group])

    return EvalReport(
        dataset=dataset,
        dataset_sha256=dataset_revision,
        backend=store.vectorizer.name,
        k=k,
        threshold=threshold,
        num_questions=len(rows),
        num_answerable=len(ans),
        num_out_of_domain=len(by_cat["out_of_domain"]),
        num_unanswerable=len(by_cat["unanswerable"]),
        recall_at_k=_mean([r.recall_at_k for r in ans]),
        context_precision=_mean([r.context_precision for r in ans]),
        answer_context_recall=_mean([r.answer_context_recall for r in ans]),
        answer_context_precision=_mean([r.answer_context_precision for r in ans]),
        answer_rate=rate(ans),
        out_of_domain_abstention_rate=rate(by_cat["out_of_domain"]),
        unanswerable_abstention_rate=rate(by_cat["unanswerable"]),
        rows=rows,
    )


def write_report(report: EvalReport, path: Path) -> None:
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _agg(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _rate(report: EvalReport, value: float | None, n: int) -> str:
    if value is None:
        return "n/a (0 items)"
    return f"{value:.3f} ({round(value * n)}/{n})"


def format_summary(report: EvalReport) -> str:
    rk = f"recall@{report.k}"
    header = (
        f"{'id':<6} {'type':<5} {rk:>9} {'ctx_prec':>9} {'ans_rec':>8} {'ans_prec':>8}"
        f" {'abstain':>7}  question"
    )
    lines = [header, "-" * len(header)]
    for row in report.rows:
        question = row.question if len(row.question) <= 50 else row.question[:47] + "..."
        lines.append(
            f"{row.id:<6} {_SHORT[row.category]:<5} {_num(row.recall_at_k):>9}"
            f" {_num(row.context_precision):>9} {_num(row.answer_context_recall):>8}"
            f" {_num(row.answer_context_precision):>8} {'yes' if row.abstained else 'no':>7}"
            f"  {question}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"retrieval (raw top-{report.k}, {report.num_answerable} answerable):"
        f" recall@{report.k} {_agg(report.recall_at_k)},"
        f" context precision {_agg(report.context_precision)}"
    )
    lines.append(
        f"answer context (cited chunks, threshold {report.threshold:g}):"
        f" recall {_agg(report.answer_context_recall)},"
        f" precision {_agg(report.answer_context_precision)}"
    )
    lines.append(
        f"abstention: answered {_rate(report, report.answer_rate, report.num_answerable)}"
        " answerable; abstained on"
        f" {_rate(report, report.out_of_domain_abstention_rate, report.num_out_of_domain)}"
        " out-of-domain,"
        f" {_rate(report, report.unanswerable_abstention_rate, report.num_unanswerable)}"
        " unanswerable"
    )
    lines.append(
        f"dataset {report.dataset or '?'} sha256 {report.dataset_sha256[:12] or '?'},"
        f" {report.num_questions} questions, backend={report.backend}"
    )
    return "\n".join(lines)

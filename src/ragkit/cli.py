"""Command-line interface: ``ragkit index | ask | chat | eval``.

Everything printed here passes through :func:`ragkit.termsafe.terminal_safe`,
because document text, filenames, QA questions and model output are untrusted
and may contain terminal control sequences.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ragkit.chunk import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, chunk_document
from ragkit.eval import format_summary, parse_qa, run_eval, sha256_hex, write_report
from ragkit.generate import DEFAULT_K, DEFAULT_THRESHOLD, Mode, answer_question, render_answer
from ragkit.ingest import load_documents
from ragkit.remote import RemoteError
from ragkit.store import VectorStore
from ragkit.termsafe import terminal_safe
from ragkit.vectorize import make_vectorizer

DEFAULT_DOCS = Path("samples/docs")
DEFAULT_STORE = Path(".ragkit/index.jsonl")
DEFAULT_QA = Path("samples/qa.json")
DEFAULT_REPORT = Path("eval_report.json")

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

StoreOpt = Annotated[Path, typer.Option("--store", help="Path to the JSONL vector store.")]
KOpt = Annotated[int, typer.Option("--k", "-k", min=1, help="Number of chunks to retrieve.")]
ThresholdOpt = Annotated[
    float, typer.Option("--threshold", help="Abstain if the best similarity is below this.")
]
ExtractiveOpt = Annotated[
    bool,
    typer.Option("--extractive", help="Force the local extractive generator (ignore LLM env)."),
]


def _echo(text: str = "", err: bool = False) -> None:
    typer.echo(terminal_safe(text), err=err)


def _one_line(text: object) -> str:
    """Render untrusted text (e.g. a path) for a one-line diagnostic.

    Line breaks become the visible two-character sequence ``\\n`` so the
    message can never span multiple lines; other control characters and bidi
    overrides are escaped later by :func:`terminal_safe` in :func:`_echo`.
    """
    s = str(text)
    for brk in ("\r\n", "\n", "\r", "\u2028", "\u2029"):
        s = s.replace(brk, "\\n")
    return s


def _load_store(path: Path) -> VectorStore:
    if not path.exists():
        _echo(f"No index at {path}. Run `ragkit index` first.", err=True)
        raise typer.Exit(code=1)
    return VectorStore.load(path)


def _answer(question: str, store: VectorStore, k: int, threshold: float, mode: Mode) -> str:
    try:
        return render_answer(answer_question(question, store, k, threshold, mode))
    except RemoteError as exc:
        _echo(f"Remote backend error: {exc}", err=True)
        raise typer.Exit(code=2) from None


@app.command()
def index(
    docs: Annotated[Path, typer.Option("--docs", help="File or directory to ingest.")] = (
        DEFAULT_DOCS
    ),
    store: StoreOpt = DEFAULT_STORE,
    max_chars: Annotated[int, typer.Option(min=1, help="Maximum chunk size.")] = DEFAULT_MAX_CHARS,
    overlap: Annotated[int, typer.Option(min=0, help="Sliding-window overlap.")] = DEFAULT_OVERLAP,
    backend: Annotated[
        str | None, typer.Option(help="Vector backend: tfidf (default) or openai.")
    ] = None,
) -> None:
    """Ingest documents, chunk them, and persist a vector store."""
    try:
        documents = load_documents(docs)
    except FileNotFoundError:
        _echo(f"No such file or directory: {_one_line(docs)}", err=True)
        raise typer.Exit(code=1) from None
    if not documents:
        _echo(f"No supported documents found in {_one_line(docs)}.", err=True)
        raise typer.Exit(code=1)
    chunks = [c for doc in documents for c in chunk_document(doc, max_chars, overlap)]
    try:
        vector_store = VectorStore.build(chunks, make_vectorizer(backend))
    except RemoteError as exc:
        _echo(f"Remote backend error: {exc}", err=True)
        raise typer.Exit(code=2) from None
    vector_store.save(store)
    _echo(
        f"Indexed {len(documents)} documents into {len(chunks)} chunks "
        f"({vector_store.vectorizer.name}) -> {store}"
    )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to answer.")],
    store: StoreOpt = DEFAULT_STORE,
    k: KOpt = DEFAULT_K,
    threshold: ThresholdOpt = DEFAULT_THRESHOLD,
    extractive: ExtractiveOpt = False,
) -> None:
    """Answer one question with citations, or abstain."""
    vector_store = _load_store(store)
    mode = "extractive" if extractive else "auto"
    _echo(_answer(question, vector_store, k, threshold, mode))


@app.command()
def chat(
    store: StoreOpt = DEFAULT_STORE,
    k: KOpt = DEFAULT_K,
    threshold: ThresholdOpt = DEFAULT_THRESHOLD,
    extractive: ExtractiveOpt = False,
) -> None:
    """Interactive question loop. Empty line, 'exit' or Ctrl-D quits."""
    vector_store = _load_store(store)
    mode = "extractive" if extractive else "auto"
    _echo(f"Loaded {len(vector_store)} chunks. Ask a question (empty line to quit).")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            _echo()
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        _echo(_answer(question, vector_store, k, threshold, mode))
        _echo()


@app.command("eval")
def eval_command(
    store: StoreOpt = DEFAULT_STORE,
    qa: Annotated[Path, typer.Option("--qa", help="QA set JSON file.")] = DEFAULT_QA,
    k: KOpt = DEFAULT_K,
    threshold: ThresholdOpt = DEFAULT_THRESHOLD,
    report: Annotated[Path, typer.Option("--report", help="Where to write the JSON report.")] = (
        DEFAULT_REPORT
    ),
) -> None:
    """Run a QA set: raw retrieval, answer-context and abstention metrics."""
    vector_store = _load_store(store)
    # Read the QA file once so the digest identifies exactly the bytes evaluated.
    raw = qa.read_bytes()
    result = run_eval(
        vector_store,
        parse_qa(raw),
        k=k,
        threshold=threshold,
        dataset=qa.as_posix(),
        dataset_revision=sha256_hex(raw),
    )
    write_report(result, report)
    _echo(format_summary(result))
    _echo(f"\nReport written to {report}")

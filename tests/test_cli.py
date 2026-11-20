"""End-to-end CLI tests. The autouse fixture in conftest blocks all sockets."""

from __future__ import annotations

import json
import socket

import pytest
from typer.testing import CliRunner

from ragkit.cli import app
from ragkit.termsafe import terminal_safe

from .conftest import SAMPLE_DOCS, SAMPLE_QA, NetworkBlocked

runner = CliRunner()


def test_network_is_really_blocked():
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("example.com", 80))


def test_index_ask_eval_offline(tmp_path):
    store = tmp_path / "index.jsonl"
    report = tmp_path / "report.json"

    result = runner.invoke(app, ["index", "--docs", str(SAMPLE_DOCS), "--store", str(store)])
    assert result.exit_code == 0, result.output
    assert store.exists()

    result = runner.invoke(app, ["ask", "How long do invitations last?", "--store", str(store)])
    assert result.exit_code == 0, result.output
    assert "getting-started.md#inviting-teammates/0" in result.output

    result = runner.invoke(app, ["ask", "What is the capital of France?", "--store", str(store)])
    assert result.exit_code == 0
    assert "Not found in these docs." in result.output

    result = runner.invoke(
        app,
        ["eval", "--store", str(store), "--qa", str(SAMPLE_QA), "--report", str(report)],
    )
    assert result.exit_code == 0, result.output
    assert "recall@4" in result.output
    data = json.loads(report.read_text())
    assert data["num_questions"] == 10
    assert {"recall_at_k", "context_precision", "rows"} <= data.keys()


def test_chat_loop(tmp_path):
    store = tmp_path / "index.jsonl"
    runner.invoke(app, ["index", "--docs", str(SAMPLE_DOCS), "--store", str(store)])
    result = runner.invoke(
        app, ["chat", "--store", str(store)], input="Where are server logs written?\nexit\n"
    )
    assert result.exit_code == 0, result.output
    assert "troubleshooting.md#server-logs/0" in result.output


def test_missing_index_is_a_clean_error(tmp_path):
    result = runner.invoke(app, ["ask", "anything", "--store", str(tmp_path / "none.jsonl")])
    assert result.exit_code == 1
    assert "ragkit index" in result.output


# --- issue 10: terminal control characters never reach the terminal raw -------

ESC_DOC = "# Refunds\n\nRefunds \x1b[2J\x1b]0;pwned\x07 take 30 days.\rFAKE‮ line\n"


def test_ask_escapes_control_sequences_from_docs(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "evil\x1b[31m.md").write_text(ESC_DOC)
    store = tmp_path / "index.jsonl"
    result = runner.invoke(app, ["index", "--docs", str(docs), "--store", str(store)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["ask", "refunds days", "--store", str(store)])
    assert result.exit_code == 0, result.output
    for raw in ("\x1b", "\x07", "\r", "‮"):
        assert raw not in result.output
    assert "\\x1b[2J" in result.output and "\\x1b[31m.md" in result.output
    assert "\\u202e" in result.output
    assert "\n" in result.output  # intended newlines survive


def test_eval_summary_and_index_diagnostics_are_escaped(tmp_path, caplog, monkeypatch):
    store = tmp_path / "index.jsonl"
    runner.invoke(app, ["index", "--docs", str(SAMPLE_DOCS), "--store", str(store)])
    qa = tmp_path / "qa.json"
    qa.write_text(
        json.dumps(
            [
                {
                    "id": "e\x1b[1m",
                    "question": "Invitations\x1b[2J expire?",
                    "answer": "7 days",
                    "gold_chunk_ids": ["getting-started.md#inviting-teammates/0"],
                }
            ]
        )
    )
    result = runner.invoke(
        app, ["eval", "--store", str(store), "--qa", str(qa), "--report", str(tmp_path / "r.json")]
    )
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output and "\\x1b[2J" in result.output

    empty = tmp_path / "empty\x1b[2J"
    empty.mkdir()
    result = runner.invoke(app, ["index", "--docs", str(empty), "--store", str(store)])
    assert result.exit_code == 1
    assert "\x1b" not in result.output and "\\x1b[2J" in result.output

    from ragkit import ingest

    monkeypatch.setattr(ingest, "pdf_supported", lambda: False)
    (empty / "x\x1b[2J.pdf").write_bytes(b"%PDF")
    runner.invoke(app, ["index", "--docs", str(empty), "--store", str(store)])
    assert caplog.records
    assert all("\x1b" not in r.getMessage() for r in caplog.records)


def test_remote_misconfiguration_is_a_clean_cli_error(tmp_path, monkeypatch):
    store = tmp_path / "index.jsonl"
    runner.invoke(app, ["index", "--docs", str(SAMPLE_DOCS), "--store", str(store)])
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "http://api.example.com/v1")
    monkeypatch.setenv("API_KEY", "k")
    result = runner.invoke(app, ["ask", "How long do invitations last?", "--store", str(store)])
    assert result.exit_code == 2
    assert "https" in result.output


# --- issue 11 / 12: CI runs offline from a wheelhouse; generated files ignored --

REPO = SAMPLE_DOCS.parent.parent


def test_ci_installs_from_wheelhouse_and_runs_offline_with_assertions():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "pip wheel --wheel-dir wheelhouse" in ci
    assert "--no-index --find-links wheelhouse" in ci
    assert "unshare --user --map-root-user --net" in ci
    assert "scripts/check_demo.sh" in ci
    script = (REPO / "scripts" / "check_demo.sh").read_text()
    for needle in ("inviting-teammates/0", "Not found in these docs.", "recall_at_k", "0.9"):
        assert needle in script


def test_generated_artifacts_are_gitignored():
    ignored = set((REPO / ".gitignore").read_text().split())
    assert {".ragkit/", "eval_report.json", "heldout_report.json", "wheelhouse/"} <= ignored


# --- round 3, issue 7: eval hashes exactly the bytes it parsed ----------------


def test_eval_reads_qa_once_and_hashes_the_parsed_bytes(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path

    store = tmp_path / "index.jsonl"
    report = tmp_path / "report.json"
    qa = tmp_path / "qa.json"
    original = SAMPLE_QA.read_bytes()
    qa.write_bytes(original)
    result = runner.invoke(app, ["index", "--docs", str(SAMPLE_DOCS), "--store", str(store)])
    assert result.exit_code == 0, result.output

    real_read_bytes = Path.read_bytes
    reads: list[Path] = []

    def read_then_swap(self: Path) -> bytes:
        data = real_read_bytes(self)
        if self == qa:
            reads.append(self)
            # Any later read of the QA file would see a different dataset.
            qa.write_bytes(b"[]")
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_swap)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: pytest.fail("re-read via text"))
    result = runner.invoke(
        app, ["eval", "--store", str(store), "--qa", str(qa), "--report", str(report)]
    )
    assert result.exit_code == 0, result.output
    assert len(reads) == 1
    data = json.loads(real_read_bytes(report))
    assert data["num_questions"] == 10
    assert data["dataset_sha256"] == hashlib.sha256(original).hexdigest()


def test_index_missing_docs_path_is_a_clean_error(tmp_path):
    missing = tmp_path / "nope\x1b[2J"
    result = runner.invoke(
        app, ["index", "--docs", str(missing), "--store", str(tmp_path / "s.jsonl")]
    )
    assert result.exit_code == 1
    assert not isinstance(result.exception, FileNotFoundError)
    assert "Traceback" not in result.output
    expected = f"No such file or directory: {terminal_safe(str(missing))}"
    assert result.output.strip().splitlines() == [expected]


@pytest.mark.parametrize("brk", ["\n", "\r\n", "\r", " "])
def test_index_missing_docs_path_with_newline_is_one_line(tmp_path, brk):
    missing = tmp_path / f"evil{brk}second-line\x1b[2J"
    result = runner.invoke(
        app, ["index", "--docs", str(missing), "--store", str(tmp_path / "s.jsonl")]
    )
    assert result.exit_code == 1
    assert not isinstance(result.exception, FileNotFoundError)
    assert "Traceback" not in result.output
    assert result.output.endswith("\n") and result.output.count("\n") == 1
    body = result.output.removesuffix("\n")
    lines = result.output.splitlines()
    assert len(lines) == 1
    assert brk not in body and "\r" not in body
    assert "\x1b" not in result.output
    assert lines[0] == (f"No such file or directory: {tmp_path}/evil\\nsecond-line\\x1b[2J")

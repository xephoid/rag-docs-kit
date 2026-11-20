from __future__ import annotations

from ragkit import ingest
from ragkit.ingest import load_documents

from .conftest import SAMPLE_DOCS


def test_loads_markdown_and_text_and_ignores_other_files(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nalpha")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("beta")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")
    docs = load_documents(tmp_path)
    assert [(d.doc_id, d.kind) for d in docs] == [("a.md", "markdown"), ("sub/b.txt", "text")]


def test_pdf_skipped_gracefully_without_pypdf(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(ingest, "pdf_supported", lambda: False)
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "ok.md").write_text("ok")
    docs = load_documents(tmp_path)
    assert [d.doc_id for d in docs] == ["ok.md"]
    assert "pdf" in caplog.text.lower()


def test_sample_docs_present():
    docs = load_documents(SAMPLE_DOCS)
    assert 4 <= len(docs) <= 6
    assert all(d.kind == "markdown" for d in docs)


# --- issue 4: symlinks outside --docs are never ingested ----------------------


def test_symlinks_to_sibling_directory_are_skipped(tmp_path):
    docs = tmp_path / "docs"
    outside = tmp_path / "outside"
    docs.mkdir()
    outside.mkdir()
    (outside / "secret.md").write_text("# Secret\n\ntop secret credentials")
    (docs / "real.md").write_text("# Real\n\nhello")
    (docs / "link.md").symlink_to(outside / "secret.md")
    (docs / "linkdir").symlink_to(outside, target_is_directory=True)
    (docs / "inner.md").symlink_to(docs / "real.md")  # symlinks are skipped even inside

    loaded = load_documents(docs)
    assert [d.doc_id for d in loaded] == ["real.md"]
    assert all("secret" not in d.text for d in loaded)


# --- round 3, issue 2: swaps to symlinks after discovery cannot escape --docs --


def _swap_on_stat(monkeypatch, trigger: str, swap) -> list[bool]:
    """Run ``swap()`` right after ingest lstat()s the entry named ``trigger``."""
    import os

    real_stat = os.stat
    fired = []

    def stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == trigger and not fired:
            fired.append(True)
            swap()
        return result

    monkeypatch.setattr(os, "stat", stat)
    return fired


def _tree(tmp_path):
    docs, outside = tmp_path / "docs", tmp_path / "outside"
    (docs / "sub").mkdir(parents=True)
    outside.mkdir()
    (docs / "a.md").write_text("# A\n\nsafe a")
    (docs / "sub" / "b.md").write_text("# B\n\nsafe b")
    (outside / "a.md").write_text("# A\n\ntop secret")
    (outside / "b.md").write_text("# B\n\ntop secret")
    return docs, outside


def test_file_swapped_for_symlink_after_discovery_is_not_followed(tmp_path, monkeypatch):
    docs, outside = _tree(tmp_path)

    def swap():
        (docs / "a.md").unlink()
        (docs / "a.md").symlink_to(outside / "a.md")

    fired = _swap_on_stat(monkeypatch, "a.md", swap)
    loaded = load_documents(docs)
    assert fired
    assert [d.doc_id for d in loaded] == ["sub/b.md"]
    assert all("secret" not in d.text for d in loaded)


def test_directory_swapped_for_symlink_after_discovery_is_not_followed(tmp_path, monkeypatch):
    docs, outside = _tree(tmp_path)

    def swap():
        (docs / "sub").rename(tmp_path / "moved")
        (docs / "sub").symlink_to(outside, target_is_directory=True)

    fired = _swap_on_stat(monkeypatch, "sub", swap)
    loaded = load_documents(docs)
    assert fired
    assert [d.doc_id for d in loaded] == ["a.md"]
    assert all("secret" not in d.text for d in loaded)


def test_ancestor_swapped_after_open_still_reads_through_held_descriptor(tmp_path, monkeypatch):
    docs, outside = _tree(tmp_path)

    def swap():  # "sub" is already open; replace its path with a symlink outside
        (docs / "sub").rename(tmp_path / "moved")
        (docs / "sub").symlink_to(outside, target_is_directory=True)

    fired = _swap_on_stat(monkeypatch, "b.md", swap)
    loaded = load_documents(docs)
    assert fired
    assert {d.doc_id: d.text for d in loaded} == {
        "a.md": "# A\n\nsafe a",
        "sub/b.md": "# B\n\nsafe b",
    }


def test_fifo_and_non_regular_files_are_skipped_without_blocking(tmp_path):
    import os

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ok.md").write_text("ok")
    os.mkfifo(docs / "pipe.md")
    assert [d.doc_id for d in load_documents(docs)] == ["ok.md"]


def test_ingest_open_path_on_existing_fifo_does_not_block(tmp_path):
    """Drive a pre-existing FIFO through the walk's own open+read helper.

    Discovery would skip a FIFO before opening it, so call the helper directly:
    without O_NONBLOCK the open blocks forever waiting for a writer.
    """
    import os
    import threading

    os.mkfifo(tmp_path / "pipe.md")
    dir_fd = os.open(tmp_path, os.O_RDONLY)
    result: list[object] = []
    worker = threading.Thread(
        target=lambda: result.append(ingest._read_regular("pipe.md", dir_fd, tmp_path)),
        daemon=True,
    )
    try:
        worker.start()
        worker.join(timeout=5)
        hung = worker.is_alive()
        if hung:  # unblock the stuck open so the thread can exit
            os.close(os.open(tmp_path / "pipe.md", os.O_WRONLY | os.O_NONBLOCK))
            worker.join(timeout=5)
        assert not hung, "opening an existing FIFO blocked the ingest"
        assert result == [None]
    finally:
        os.close(dir_fd)


# --- round 4, issue 1: the root itself is opened once, with no path precheck ---


def test_root_is_not_prechecked_by_path(tmp_path, monkeypatch):
    """Any path-based check before the open is a TOCTOU window on the root."""
    import os.path
    from pathlib import Path

    docs, _ = _tree(tmp_path)

    def guard(original):
        def checked(path, *args, **kwargs):
            if Path(path) == docs:
                raise AssertionError("path-based precheck of the ingest root")
            return original(path, *args, **kwargs)

        return checked

    for name in ("is_dir", "is_file", "exists", "is_symlink", "stat", "lstat", "resolve"):
        monkeypatch.setattr(Path, name, guard(getattr(Path, name)))
    for name in ("isdir", "isfile", "exists", "islink"):
        monkeypatch.setattr(os.path, name, guard(getattr(os.path, name)))

    # The bottom layer: every pathlib / os.path check ends in os.stat or os.lstat.
    # Record (rather than raise, so a broad ``except`` cannot swallow it) any call
    # that resolves the root by path. dir_fd-relative calls on entries are fine.
    root_hits: list[str] = []

    def record(original):
        def checked(path, *args, dir_fd=None, **kwargs):
            by_path = dir_fd is None and not isinstance(path, int)
            if by_path and os.path.abspath(os.fspath(path)) == os.fspath(docs):
                root_hits.append(original.__name__)
            return original(path, *args, dir_fd=dir_fd, **kwargs)

        return checked

    monkeypatch.setattr(os, "stat", record(os.stat))
    monkeypatch.setattr(os, "lstat", record(os.lstat))
    assert [d.doc_id for d in load_documents(docs)] == ["a.md", "sub/b.md"]
    assert root_hits == [], f"path-based precheck of the ingest root via {root_hits}"


def test_root_swapped_for_symlink_during_walk_cannot_escape(tmp_path, monkeypatch):
    docs, outside = _tree(tmp_path)

    def swap():  # the root is already open; replace its path with a symlink outside
        docs.rename(tmp_path / "moved")
        docs.symlink_to(outside, target_is_directory=True)

    fired = _swap_on_stat(monkeypatch, "a.md", swap)
    loaded = load_documents(docs)
    assert fired and docs.is_symlink()
    assert {d.doc_id: d.text for d in loaded} == {
        "a.md": "# A\n\nsafe a",
        "sub/b.md": "# B\n\nsafe b",
    }


def test_user_supplied_symlink_root_is_resolved_once(tmp_path):
    docs, _ = _tree(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(docs, target_is_directory=True)
    assert [d.doc_id for d in load_documents(link)] == ["a.md", "sub/b.md"]


def test_non_directory_root_gives_a_clean_error(tmp_path):
    import os

    import pytest

    (tmp_path / "plain.md").write_text("x")
    os.mkfifo(tmp_path / "pipe")
    for path in (tmp_path / "plain.md", tmp_path / "pipe"):
        with pytest.raises(NotADirectoryError, match="Not a directory"):
            ingest._read_tree(path)
    with pytest.raises(FileNotFoundError, match="No such file"):
        load_documents(tmp_path / "missing")
    # A single file is still accepted by the public entry point; a FIFO is not read.
    assert [d.doc_id for d in load_documents(tmp_path / "plain.md")] == ["plain.md"]
    assert load_documents(tmp_path / "pipe") == []


def test_pdf_is_parsed_from_the_opened_descriptor(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ingest, "pdf_supported", lambda: True)
    monkeypatch.setattr(ingest, "_pdf_text", lambda data: seen.append(data) or "pdf text")
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 bytes")
    [doc] = load_documents(tmp_path)
    assert (doc.doc_id, doc.kind, doc.text) == ("doc.pdf", "pdf", "pdf text")
    assert seen == [b"%PDF-1.4 bytes"]

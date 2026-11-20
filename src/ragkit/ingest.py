"""Load source documents from disk.

Markdown and plain text are always supported. PDF is supported only when the
optional ``pypdf`` package is installed; otherwise PDF files are skipped with a
warning instead of failing the whole ingest.

Directory ingest is POSIX-only. The root is opened once (no path-based
precheck, so it cannot be swapped between a check and the open) and the tree is walked
relative to directory file descriptors: every component (subdirectory or file)
is opened with ``O_NOFOLLOW`` relative to its parent's descriptor, and each
opened file is checked with ``fstat`` to be a regular file before its bytes are
read from that same descriptor. Symlinks are skipped, and swapping a path for a
symlink between discovery and reading cannot make the read escape ``--docs``.
"""

from __future__ import annotations

import errno
import importlib.util
import io
import logging
import os
import stat
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ragkit.termsafe import terminal_safe

logger = logging.getLogger(__name__)

TEXT_SUFFIXES = {".md": "markdown", ".markdown": "markdown", ".txt": "text"}
PDF_SUFFIX = ".pdf"

DocKind = Literal["markdown", "text", "pdf"]

_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
# O_NONBLOCK so a FIFO planted in the tree cannot hang the open; fstat rejects it.
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_FILE_FLAGS = os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC
# The root alone may be a symlink the user passed; it is resolved once at open.
_ROOT_FLAGS = os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC
_DIR_FLAGS = os.O_RDONLY | _O_NOFOLLOW | _O_DIRECTORY | _O_CLOEXEC


class Document(BaseModel):
    doc_id: str
    path: str
    kind: DocKind
    text: str


def pdf_supported() -> bool:
    return importlib.util.find_spec("pypdf") is not None


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _supported(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    return suffix in TEXT_SUFFIXES or suffix == PDF_SUFFIX


def _document_from_bytes(data: bytes, doc_id: str, path: str) -> Document | None:
    """Build a Document from file bytes already read. None if unsupported or unreadable."""
    suffix = Path(doc_id).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = data.decode("utf-8", errors="replace")
        return Document(doc_id=doc_id, path=path, kind=TEXT_SUFFIXES[suffix], text=text)
    if suffix == PDF_SUFFIX:
        if not pdf_supported():
            logger.warning(
                "Skipping %s: install 'ragkit[pdf]' to ingest PDF files.",
                terminal_safe(path),
            )
            return None
        try:
            text = _pdf_text(data)
        except Exception as exc:  # pypdf raises a variety of errors on bad files
            logger.warning(
                "Skipping %s: could not read PDF (%s).",
                terminal_safe(path),
                terminal_safe(str(exc)),
            )
            return None
        return Document(doc_id=doc_id, path=path, kind="pdf", text=text)
    return None


def _read_fd(fd: int) -> bytes:
    with os.fdopen(fd, "rb", closefd=False) as handle:
        return handle.read()


def load_document(path: Path, root: Path | None = None) -> Document | None:
    """Load one explicitly named file. Returns None for unsupported or unreadable files."""
    doc_id = path.relative_to(root).as_posix() if root else path.name
    if not _supported(path.name):
        return None
    fd = os.open(path, os.O_RDONLY | _O_NONBLOCK | _O_CLOEXEC)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        data = _read_fd(fd)
    finally:
        os.close(fd)
    return _document_from_bytes(data, doc_id, str(path))


def _open_nofollow(name: str, dir_fd: int, flags: int) -> int | None:
    """Open ``name`` relative to ``dir_fd`` without following symlinks; None if refused."""
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        # ELOOP: it is a symlink. ENOTDIR: it is (now) not a directory.
        if exc.errno not in (errno.ELOOP, errno.ENOTDIR, errno.ENOENT, errno.EMLINK):
            logger.warning("Skipping %s: %s", terminal_safe(name), terminal_safe(exc.strerror))
        return None


def _read_regular(name: str, dir_fd: int, shown: Path) -> bytes | None:
    """Read ``name`` (relative to ``dir_fd``) only if the opened object is a regular file.

    Opened with ``O_NOFOLLOW | O_NONBLOCK``: a symlink is refused, and a FIFO or
    device swapped in after discovery cannot block the open; ``fstat`` then
    rejects anything that is not a regular file before a byte is read.
    """
    fd = _open_nofollow(name, dir_fd, _FILE_FLAGS)
    if fd is None:
        logger.warning("Skipping %s: changed during ingest", terminal_safe(str(shown)))
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        return _read_fd(fd)
    finally:
        os.close(fd)


def _walk_fd(
    dir_fd: int, rel: tuple[str, ...], display: Path, out: list[tuple[tuple[str, ...], bytes]]
) -> None:
    """Collect (relative parts, bytes) of supported regular files below ``dir_fd``."""
    with os.scandir(dir_fd) as entries:
        names = sorted(entry.name for entry in entries)
    for name in names:
        shown = display / name
        try:
            info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except OSError:
            continue
        if stat.S_ISLNK(info.st_mode):
            logger.warning("Skipping symlink %s", terminal_safe(str(shown)))
            continue
        if stat.S_ISDIR(info.st_mode):
            child = _open_nofollow(name, dir_fd, _DIR_FLAGS)
            if child is None:
                logger.warning("Skipping %s: changed during ingest", terminal_safe(str(shown)))
                continue
            try:
                if stat.S_ISDIR(os.fstat(child).st_mode):
                    _walk_fd(child, (*rel, name), shown, out)
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(info.st_mode) or not _supported(name):
            continue
        data = _read_regular(name, dir_fd, shown)
        if data is not None:
            out.append(((*rel, name), data))


def _open_root(root: Path) -> int:
    """Open the ingest root once and return a descriptor pinned to that directory.

    There is deliberately no path-based precheck (``is_dir()`` and friends): the
    path is resolved exactly once, here, so swapping it between a check and the
    open is impossible. A user-supplied symlink root is followed at this single
    open; the walk then uses only this descriptor and never the path again.
    Raises ``NotADirectoryError`` if the path is not a directory.
    """
    try:
        fd = os.open(root, _ROOT_FLAGS)
    except FileNotFoundError:
        raise FileNotFoundError(f"No such file or directory: {root}") from None
    except NotADirectoryError:
        raise NotADirectoryError(f"Not a directory: {root}") from None
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise NotADirectoryError(f"Not a directory: {root}")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_tree(root: Path) -> list[tuple[tuple[str, ...], bytes]]:
    root_fd = _open_root(root)
    try:
        files: list[tuple[tuple[str, ...], bytes]] = []
        _walk_fd(root_fd, (), root, files)
    finally:
        os.close(root_fd)
    return sorted(files, key=lambda item: item[0])


def load_documents(source: Path) -> list[Document]:
    """Load a single file or every supported file under a directory (sorted, recursive)."""
    try:
        tree = _read_tree(source)
    except NotADirectoryError:
        # Not a directory: treat it as one explicitly named file (fd + fstat checked).
        doc = load_document(source)
        return [doc] if doc else []
    docs = []
    for parts, data in tree:
        doc = _document_from_bytes(data, "/".join(parts), str(source.joinpath(*parts)))
        if doc is not None:
            docs.append(doc)
    return docs

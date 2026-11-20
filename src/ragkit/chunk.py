"""Split documents into retrievable chunks with stable ids.

Markdown is split on h1/h2/h3 headings, ATX (``## Title``) or setext (a line
underlined with ``===`` or ``---``). Headings inside fenced code blocks are
ignored. Any section longer than ``max_chars`` -- and any non-markdown text --
falls back to a sliding window with ``overlap`` characters shared between
consecutive windows. Window edges snap to whitespace when possible so words are
not cut in half; a single unbroken run of text longer than ``max_chars`` is cut
hard.

Chunk ids have the form ``<doc_id>#<heading-slug>/<part>`` and depend only on
the document's content, so re-indexing unchanged docs yields the same ids.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel

from ragkit.ingest import Document

DEFAULT_MAX_CHARS = 800
DEFAULT_OVERLAP = 120

# ATX heading: up to three spaces of indent, 1-6 '#', then whitespace and text
# (or end of line). An optional closing run of '#' preceded by whitespace is
# not part of the title. Only h1-h3 start a new section; h4-h6 stay in the body.
_ATX_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_ATX_CLOSING_RE = re.compile(r"(?:^|[ \t]+)#+$")
# Fence opener: up to three spaces, then three or more backticks or tildes.
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
# Setext underline: up to three spaces, a run of '=' (h1) or '-' (h2), nothing else.
_SETEXT_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
# Lines that cannot be the text of a setext heading (lists, quotes, indented code).
_NOT_PARAGRAPH_RE = re.compile(r"^(?: {4,}|\t| {0,3}(?:>|[-*+](?:[ \t]|$)|\d{1,9}[.)](?:[ \t]|$)))")
MAX_SPLIT_LEVEL = 3


class Chunk(BaseModel):
    id: str
    doc_id: str
    heading: str
    text: str
    ordinal: int
    part: int
    # Indexes of the lines in ``text`` that are Markdown headings. Used to show
    # the body without heading lines; every other line is kept verbatim.
    heading_lines: list[int] = []


class Section(BaseModel):
    heading_path: list[str]
    text: str
    heading_lines: list[int] = []


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "-", value).strip("-")
    return value or "section"


def _atx_heading(line: str) -> tuple[int, str] | None:
    """Return (level, title) for an ATX heading line. The title may be empty
    (``##`` or ``## ##`` are valid empty headings and still start a section)."""
    match = _ATX_RE.match(line)
    if not match:
        return None
    title = _ATX_CLOSING_RE.sub("", match.group(2) or "").strip()
    return len(match.group(1)), title


class _Fence:
    """Track fenced code blocks: a fence closes only on the same marker character,
    at least as long as the opener, with nothing but whitespace after it."""

    def __init__(self) -> None:
        self.marker: str | None = None

    @property
    def open(self) -> bool:
        return self.marker is not None

    def feed(self, line: str) -> bool:
        """Update state for ``line``; return True if the line is a fence line."""
        match = _FENCE_OPEN_RE.match(line)
        if self.marker is None:
            if match and not (match.group(1)[0] == "`" and "`" in match.group(2)):
                self.marker = match.group(1)
                return True
            return False
        if (
            match
            and match.group(1)[0] == self.marker[0]
            and len(match.group(1)) >= len(self.marker)
            and not match.group(2).strip()
        ):
            self.marker = None
        return True


def split_markdown_sections(text: str) -> list[Section]:
    """Split markdown into sections at h1-h3 headings, tracking the heading path.

    Recognises ATX headings (``## Title``, up to three spaces of indent) and
    setext headings (a paragraph underlined with ``===`` or ``---``). Lines
    inside fenced code blocks are never headings.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    current_path: list[str] = []
    buffer: list[str] = []
    heading_idx: set[int] = set()
    # Index in ``buffer`` where the current paragraph started (None outside one).
    paragraph_start: int | None = None
    fence = _Fence()

    def flush(lines: list[str], headings: set[int]) -> None:
        first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
        if first is None:
            return
        last = max(i for i, ln in enumerate(lines) if ln.strip())
        kept = lines[first : last + 1]
        body = "\n".join(kept).strip()
        sections.append(
            Section(
                heading_path=list(current_path),
                text=body,
                heading_lines=sorted(i - first for i in headings if first <= i <= last),
            )
        )

    def start_section(level: int, title: str) -> None:
        nonlocal stack, current_path
        stack = [(lvl, t) for lvl, t in stack if lvl < level]
        stack.append((level, title))
        current_path = [t for _, t in stack]

    for line in text.splitlines():
        was_open = fence.open
        if fence.feed(line) or was_open:
            buffer.append(line)
            paragraph_start = None
            continue

        atx = _atx_heading(line)
        if atx is not None:
            level, title = atx
            if level <= MAX_SPLIT_LEVEL:
                flush(buffer, heading_idx)
                buffer, heading_idx = [], set()
                start_section(level, title)
            heading_idx.add(len(buffer))
            buffer.append(line)
            paragraph_start = None
            continue

        setext = _SETEXT_RE.match(line)
        if setext and paragraph_start is not None:
            level = 1 if setext.group(1)[0] == "=" else 2
            title_lines = buffer[paragraph_start:]
            title = " ".join(ln.strip() for ln in title_lines)
            flush(buffer[:paragraph_start], {i for i in heading_idx if i < paragraph_start})
            buffer = [*title_lines, line]
            heading_idx = set(range(len(buffer)))
            start_section(level, title)
            paragraph_start = None
            continue

        if not line.strip():
            paragraph_start = None
        elif paragraph_start is None and not _NOT_PARAGRAPH_RE.match(line):
            paragraph_start = len(buffer)
        buffer.append(line)
    flush(buffer, heading_idx)
    return sections


def _window_spans(text: str, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) offsets into ``text`` of each window, already whitespace-trimmed."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")
    length = len(text)

    def trimmed(start: int, end: int) -> tuple[int, int] | None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return (start, end) if start < end else None

    whole = trimmed(0, length)
    if whole is None:
        return []
    if whole[1] - whole[0] <= max_chars:
        return [whole]

    spans: list[tuple[int, int]] = []
    start, length = whole
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            # Prefer to end on whitespace in the back half of the window.
            floor = start + max_chars // 2
            cut = max(text.rfind(" ", floor, end + 1), text.rfind("\n", floor, end + 1))
            if cut > start:
                end = cut
        span = trimmed(start, end)
        if span:
            spans.append(span)
        if end >= length:
            break
        next_start = max(end - overlap, start + 1)
        if overlap:
            # Start the next window at a word boundary inside the overlap region.
            for i in range(next_start, min(end, next_start + overlap // 2)):
                if text[i].isspace():
                    next_start = i + 1
                    break
        while next_start < length and text[next_start].isspace():
            next_start += 1
        start = next_start
    return spans


def sliding_window(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split ``text`` into windows of at most ``max_chars`` sharing ``overlap`` chars."""
    return [text[s:e] for s, e in _window_spans(text, max_chars, overlap)]


def _heading_lines_in(section: Section, start: int, end: int) -> list[int]:
    """Line indexes (relative to ``section.text[start:end]``) of whole heading lines."""
    text = section.text
    line_starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]
    result = []
    for idx in section.heading_lines:
        line_start = line_starts[idx]
        line_end = text.find("\n", line_start)
        line_end = len(text) if line_end == -1 else line_end
        if start <= line_start and line_end <= end:
            result.append(text.count("\n", start, line_start))
    return result


def _make_chunks(doc_id: str, sections: list[Section], max_chars: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    used_slugs: set[str] = set()
    for section in sections:
        base = slugify(section.heading_path[-1]) if section.heading_path else "preamble"
        slug, n = base, 1
        # Advance the suffix until it is unused, so "FAQ", "FAQ", "FAQ-2" cannot collide.
        while slug in used_slugs:
            n += 1
            slug = f"{base}-{n}"
        used_slugs.add(slug)
        for part, (start, end) in enumerate(_window_spans(section.text, max_chars, overlap)):
            chunks.append(
                Chunk(
                    id=f"{doc_id}#{slug}/{part}",
                    doc_id=doc_id,
                    heading=" > ".join(section.heading_path),
                    text=section.text[start:end],
                    ordinal=len(chunks),
                    part=part,
                    heading_lines=_heading_lines_in(section, start, end),
                )
            )
    return chunks


def chunk_markdown(
    doc_id: str, text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    return _make_chunks(doc_id, split_markdown_sections(text), max_chars, overlap)


def chunk_text(
    doc_id: str, text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    sections = [Section(heading_path=[], text=text)] if text.strip() else []
    chunks = _make_chunks(doc_id, sections, max_chars, overlap)
    for chunk in chunks:
        chunk.id = f"{doc_id}#text/{chunk.part}"
    return chunks


def chunk_document(
    doc: Document, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    if doc.kind == "markdown":
        return chunk_markdown(doc.doc_id, doc.text, max_chars, overlap)
    return chunk_text(doc.doc_id, doc.text, max_chars, overlap)

from __future__ import annotations

import pytest

from ragkit.chunk import chunk_document, chunk_markdown, chunk_text, sliding_window, slugify
from ragkit.ingest import Document

MD = """\
Intro paragraph before any heading.

# Title

Top level text.

## Alpha

Alpha body.

### Alpha Detail

Detail body.

#### Not A Boundary

Still part of Alpha Detail.

## Beta

```bash
# this is a shell comment, not a heading
echo hi
```
"""


def test_markdown_splits_on_h1_h2_h3_only():
    chunks = chunk_markdown("doc.md", MD)
    assert [c.id for c in chunks] == [
        "doc.md#preamble/0",
        "doc.md#title/0",
        "doc.md#alpha/0",
        "doc.md#alpha-detail/0",
        "doc.md#beta/0",
    ]
    detail = chunks[3]
    assert detail.heading == "Title > Alpha > Alpha Detail"
    assert "#### Not A Boundary" in detail.text
    assert "Still part of Alpha Detail." in detail.text
    assert "Beta" not in detail.text


def test_heading_inside_code_fence_is_not_a_boundary():
    beta = chunk_markdown("doc.md", MD)[-1]
    assert "# this is a shell comment" in beta.text
    assert "echo hi" in beta.text


def test_sibling_heading_resets_path():
    chunks = chunk_markdown("doc.md", MD)
    assert chunks[-1].heading == "Title > Beta"


def test_ids_are_stable_and_unique():
    first = chunk_markdown("doc.md", MD)
    second = chunk_markdown("doc.md", MD)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)


def test_duplicate_headings_get_distinct_ids():
    chunks = chunk_markdown("d.md", "## FAQ\n\none\n\n## FAQ\n\ntwo\n")
    assert [c.id for c in chunks] == ["d.md#faq/0", "d.md#faq-2/0"]


def test_empty_and_whitespace_documents_produce_no_chunks():
    assert chunk_markdown("e.md", "") == []
    assert chunk_markdown("e.md", "   \n\n\t\n") == []
    assert chunk_text("e.txt", "") == []


def test_heading_only_section_is_kept():
    chunks = chunk_markdown("h.md", "# Lonely Heading\n")
    assert [c.text for c in chunks] == ["# Lonely Heading"]


def test_max_chunk_size_enforced_and_overlap_present():
    words = " ".join(f"word{i}" for i in range(400))
    doc = f"# Big\n\n{words}\n"
    chunks = chunk_markdown("big.md", doc, max_chars=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert [c.part for c in chunks] == list(range(len(chunks)))
    assert all(c.id == f"big.md#big/{c.part}" for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        first_word = nxt.text.split()[0]
        assert first_word in prev.text.split(), "consecutive windows must overlap"
    # Windows snap to whitespace, so no word is cut in half.
    for c in chunks:
        assert all(w.startswith("word") or w in {"#", "Big"} for w in c.text.split())


def test_one_giant_line_without_whitespace_is_hard_cut_with_overlap():
    text = "x" * 1000 + "y" * 1000
    windows = sliding_window(text, max_chars=300, overlap=50)
    assert all(len(w) <= 300 for w in windows)
    assert "".join(w[50:] if i else w for i, w in enumerate(windows)) == text
    chunks = chunk_text("giant.txt", text, max_chars=300, overlap=50)
    assert len(chunks) == len(windows)
    assert chunks[0].id == "giant.txt#text/0"


def test_unicode_is_preserved_and_measured_in_characters():
    text = "# Café ☕\n\n" + "naïve façade 東京 🚀 " * 60
    chunks = chunk_markdown("uni.md", text, max_chars=120, overlap=20)
    assert chunks[0].id.startswith("uni.md#café/")
    assert all(len(c.text) <= 120 for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert "東京" in joined and "🚀" in joined and "naïve" in joined


def test_slugify():
    assert slugify("Plans & Billing!") == "plans-billing"
    assert slugify("   ") == "section"


def test_text_documents_use_sliding_window():
    doc = Document(doc_id="n.txt", path="n.txt", kind="text", text="# not a heading\nbody")
    chunks = chunk_document(doc)
    assert [c.id for c in chunks] == ["n.txt#text/0"]
    assert chunks[0].heading == ""


@pytest.mark.parametrize(("max_chars", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_window_parameters(max_chars, overlap):
    with pytest.raises(ValueError):
        sliding_window("abc", max_chars, overlap)


# --- issue 1: slug suffixes never collide with natural headings ---------------


@pytest.mark.parametrize(
    "headings",
    [["FAQ", "FAQ", "FAQ-2"], ["FAQ-2", "FAQ", "FAQ"], ["FAQ", "FAQ-2", "FAQ", "FAQ"]],
)
def test_duplicate_and_suffixed_headings_get_unique_repeatable_ids(headings):
    md = "".join(f"## {h}\n\nbody {i}\n\n" for i, h in enumerate(headings))
    first = [c.id for c in chunk_markdown("d.md", md)]
    assert len(first) == len(headings)
    assert len(set(first)) == len(first)
    assert first == [c.id for c in chunk_markdown("d.md", md)]
    store_ids = {c.id for c in chunk_markdown("d.md", md)}
    assert len(store_ids) == len(headings)


def test_faq_faq_faq2_ids_can_be_indexed():
    from ragkit.store import VectorStore
    from ragkit.vectorize import TfidfVectorizer

    chunks = chunk_markdown("d.md", "## FAQ\n\na\n\n## FAQ\n\nb\n\n## FAQ-2\n\nc\n")
    assert [c.id for c in chunks] == ["d.md#faq/0", "d.md#faq-2/0", "d.md#faq-2-2/0"]
    VectorStore.build(chunks, TfidfVectorizer())  # raised "Chunk ids must be unique" before


# --- issue 2: indented ATX, setext headings, fence matching -------------------


def _ids(md: str) -> list[str]:
    return [c.id for c in chunk_markdown("d.md", md)]


def test_indented_atx_headings_up_to_three_spaces_are_boundaries():
    md = "intro\n\n # One\n\na\n\n  ## Two\n\nb\n\n   ### Three\n\nc\n\n    ## Code\n\nd\n"
    assert _ids(md) == ["d.md#preamble/0", "d.md#one/0", "d.md#two/0", "d.md#three/0"]
    three = chunk_markdown("d.md", md)[-1]
    assert "    ## Code" in three.text  # four spaces is an indented code block


def test_atx_closing_hashes_and_hash_without_space():
    assert _ids("## Title ##\n\nx\n\n#hashtag\n\ny\n") == ["d.md#title/0"]


def test_setext_h1_and_h2_are_boundaries():
    md = "Intro text.\n\nFirst Title\n===========\n\none\n\nSecond\nTitle\n---\n\ntwo\n"
    chunks = chunk_markdown("d.md", md)
    assert [c.id for c in chunks] == [
        "d.md#preamble/0",
        "d.md#first-title/0",
        "d.md#second-title/0",
    ]
    assert chunks[2].heading == "First Title > Second Title"
    assert chunks[2].heading_lines == [0, 1, 2]


def test_thematic_break_and_list_underline_are_not_setext_headings():
    md = "## A\n\ntext\n\n---\n\n- item\n---\n\nmore\n"
    assert _ids(md) == ["d.md#a/0"]


@pytest.mark.parametrize(
    "fence_body",
    [
        "```\n## inside\n~~~\n## still inside\n```",  # different marker does not close
        "````\n## inside\n```\n## still inside\n````",  # shorter run does not close
        "```\n## inside\n``` not a close\n## still inside\n```",  # trailing content
        "~~~\n## inside\n~~~~~ \n",  # longer run with trailing space closes
    ],
)
def test_fences_close_only_on_matching_marker(fence_body):
    md = f"## Real\n\n{fence_body}\n\n## After\n\nx\n"
    assert _ids(md) == ["d.md#real/0", "d.md#after/0"]


def test_heading_lines_tracked_per_window():
    md = "## Big\n\n" + " ".join(f"w{i}" for i in range(200)) + "\n\n#### Sub\n\nend\n"
    chunks = chunk_markdown("b.md", md, max_chars=200, overlap=20)
    assert chunks[0].heading_lines == [0]
    last = chunks[-1]
    assert "#### Sub" in last.text
    assert last.text.splitlines()[last.heading_lines[-1]] == "#### Sub"
    assert all(i < len(c.text.splitlines()) for c in chunks for i in c.heading_lines)


# --- round 3, issue 8: empty ATX headings are still section boundaries --------


@pytest.mark.parametrize("empty", ["##", "## ##", "##   ", "  ##"])
def test_empty_atx_heading_preserves_section_boundary(empty):
    md = f"# Doc\n\n## Section\n\nalpha\n\n## Section\n\nbeta\n\n{empty}\n\ngamma\n\n##\n\ndelta\n"
    chunks = chunk_markdown("d.md", md)
    assert [c.id for c in chunks] == [
        "d.md#doc/0",
        "d.md#section/0",
        "d.md#section-2/0",
        "d.md#section-3/0",  # empty title -> fallback slug, disambiguated
        "d.md#section-4/0",
    ]
    by_id = {c.id: c.text for c in chunks}
    assert "gamma" not in by_id["d.md#section-2/0"]
    assert "gamma" in by_id["d.md#section-3/0"] and "delta" not in by_id["d.md#section-3/0"]
    assert "delta" in by_id["d.md#section-4/0"]
    assert chunks[3].heading_lines == [0]

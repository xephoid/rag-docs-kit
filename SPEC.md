# SPEC — rag-docs-kit (portfolio artifact #3)

Build a **RAG (retrieval-augmented generation) chat-over-documents kit + eval harness** (Python)
in this directory. Portfolio artifact for the "RAG for your docs" upsell.
Hard constraint: **the demo must run fully offline with zero API keys** (deterministic local
retrieval); external LLM/embedding backends are optional plug-ins via env vars.

## Stack
- Python 3.12, Typer CLI, Pydantic v2. No heavy model downloads (this box has ~6 GB RAM).
- Default retrieval: TF-IDF + cosine over chunks (sklearn if available, else a small pure-Python
  vectorizer you write — your call, but it must be fast and dependency-light).
- Optional backends (env-configured, NOT required for demo): OpenAI-compatible embeddings + LLM.
- Tests: pytest, network-free. Lint: ruff clean.

## Required structure
```
src/ragkit/
  __init__.py
  ingest.py         # md/txt (+ optional pdf via pypdf if installed; degrade gracefully)
  chunk.py          # chunking: heading-aware for markdown, sliding window w/ overlap fallback
  vectorize.py      # TF-IDF default backend; pluggable embedding backend interface
  store.py          # in-memory + JSONL-persisted vector store; cosine top-k
  generate.py       # answer assembly: local "extractive" generator (top chunk + highlighted
                    # supporting spans) by default; optional LLM completion via env backend
  eval.py           # eval harness: recall@k + context precision on a QA set; JSON report
  cli.py            # Typer: index, chat, ask, eval
tests/              # unit tests: chunking boundaries, retrieval ranking, eval math
samples/docs/       # 4–6 short markdown docs you write (a fictional product's docs)
samples/qa.json     # 10 question/answer pairs with gold chunk ids, for eval
pyproject.toml
.github/workflows/ci.yml
README.md
```

## Required behavior
1. **`index`**: ingest samples/docs → chunks with stable ids → persisted store.
2. **`ask "question"`**: top-k retrieval, answer with **citations (doc + chunk id)**, and an
   abstention path: if max similarity < threshold, say "not found in these docs" — never hallucinate.
3. **`eval`**: runs the QA set, emits `eval_report.json` (recall@k, context precision, per-question
   rows) + prints a summary table.
4. **Optional LLM backend**: `OPENAI_COMPATIBLE_BASE_URL` + `API_KEY` env → generate via chat
   completions with the retrieved chunks as context. Must be cleanly separable; default path never
   touches the network.
5. **Chunking correctness**: markdown h1/h2/h3 boundaries respected; max chunk size enforced;
   overlap present; edge cases covered by tests (empty doc, one giant line, unicode).

## Definition of done (all must be true)
- [ ] `pip install -e . && pytest` passes offline.
- [ ] `python -m ragkit index && python -m ragkit ask "..." && python -m ragkit eval` all run
      successfully here; README quotes the **real** eval numbers (recall@k etc.).
- [ ] README: architecture diagram (ASCII ok), quickstart, **why extractive-first** (determinism,
      zero cost, auditable), how to plug in an LLM, MIT license.
- [ ] ruff clean. `git status` clean at the end (do NOT commit — orchestrator commits after review).
- [ ] Zero network calls in the default path (proven by tests running with network disabled).

## Style
- Boring, professional code. Do not add features beyond this spec. Do not push to any remote.

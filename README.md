# rag-docs-kit

Chat over your own documents, with citations and an eval harness, that runs
**fully offline with zero API keys**. Retrieval is deterministic TF-IDF over
heading-aware chunks. Answers are extractive: they quote the docs verbatim and
cite the doc and chunk id. When no chunk is similar enough to the question,
the kit says "Not found in these docs." instead of guessing. That check is a
similarity threshold, not a fact check (see [Limits](#limits-of-the-abstention-threshold)).
An OpenAI-compatible LLM or embedding backend can be plugged in with
environment variables, but the default path never touches the network.

The runtime needs only `typer` and `pydantic`. There are no model downloads and
no numpy or sklearn.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # add ",pdf" to ingest PDFs via pypdf

python -m ragkit index           # samples/docs -> .ragkit/index.jsonl
python -m ragkit ask "What happens when a workspace runs out of storage quota?"
python -m ragkit chat            # interactive loop
python -m ragkit eval            # samples/qa.json -> eval_report.json + summary table
python -m ragkit eval --qa samples/heldout_qa.json --report heldout_report.json

pytest && ruff check . && ruff format --check .
bash scripts/check_demo.sh       # runs index/ask/eval and asserts the output
```

Point it at your own docs with `ragkit index --docs path/to/docs`. Every command
takes `--help`.

### Installing with no network (wheelhouse)

`pip install -e .` downloads `typer`, `pydantic` and `setuptools` the first
time. To install on a machine with no network, build a wheelhouse on a
connected machine with the **same OS, CPU architecture and Python version**
(`pydantic-core` ships platform-specific wheels), copy it across with the repo,
and install from it with `--no-index`:

```bash
# on a connected machine, from the repo root
python -m pip download --dest wheelhouse "setuptools>=68"   # build backend
python -m pip wheel --wheel-dir wheelhouse ".[dev]"           # runtime + dev deps

# on the offline machine
python -m pip install --no-index --find-links wheelhouse -e ".[dev]"
```

CI does exactly this (`.github/workflows/ci.yml`). The "Build wheelhouse" step
is the only one that downloads anything. The install step then runs
`pip install --no-index --find-links wheelhouse`, so pip never contacts an
index, but that step and the lint step run on the normal runner, outside any
network namespace. Network isolation is applied only to the final step: it runs
`pytest -q` and `scripts/check_demo.sh` inside
`unshare --user --map-root-user --net`, after first checking that an outbound
connection fails there.

Example output (real run):

```text
$ python -m ragkit ask "What happens when a workspace runs out of storage quota?"
Retrieved evidence (quoted from the docs; it may not fully answer the question):
Each **workspace** has a **quota** measured in gigabytes, set with `default_quota_gb`
(default 100). When a **workspace** reaches 90 percent of its **quota**, admins receive
a warning email. At 100 percent, new uploads are rejected until space is freed.

Supporting:
  - Each **workspace** has its own folders, members, and **storage** **quota**. [getting-started.md#creating-your-first-workspace/0]

Sources:
  [1] configuration.md :: configuration.md#storage-quotas/0 (score 0.320)
  [2] getting-started.md :: getting-started.md#creating-your-first-workspace/0 (score 0.216)

$ python -m ragkit ask "Does Tidepool support Kubernetes autoscaling?"
Not found in these docs.

$ python -m ragkit ask "Can I extend teammate invitations to 90 days?"
Retrieved evidence (quoted from the docs; it may not fully answer the question):
**Invitations** expire after 7 **days**. An admin can resend an expired **invitation** from
the **Members** page. Invited users must install the desktop client and sign in
with the email address that received the **invitation**.

Sources:
  [1] getting-started.md :: getting-started.md#inviting-teammates/0 (score 0.369)
```

The last example is deliberate. The docs only state the 7-day default, so the
question is unanswerable, but it shares enough vocabulary to clear the
threshold. The kit does not abstain. It shows the closest excerpt and labels
it as retrieved evidence, not as an answer. Reading it is up to you.

## Architecture

```text
             samples/docs/*.md|txt|pdf
                        |
                        v
  ingest.py   load files -> Document(doc_id, kind, text)
                        |       (pdf only if pypdf is installed; otherwise skipped + warning)
                        |       (directory ingest skips symlinks and files outside --docs)
                        v
  chunk.py    markdown: split at h1/h2/h3, ATX or setext (code fences respected)
              oversize sections / plain text: sliding window, max_chars + overlap
              stable, unique ids:  <doc>#<heading-slug>/<part>
                        |
                        v
  vectorize.py  TfidfVectorizer (default, pure Python)  |  OpenAIEmbeddingVectorizer (opt-in)
                        |
                        v
  store.py    VectorStore: in-memory vectors, cosine top-k, JSONL persistence
              .ragkit/index.jsonl = 1 meta line (backend + IDF state) + 1 line per chunk
                        |
          +-------------+--------------+
          v                            v
  generate.py                       eval.py
   no chunk >= threshold?            raw retrieval: top-k vs gold ids
     -> "Not found in these docs."     -> recall@k, context precision
   extractive (default):             answer context: cited chunks vs gold ids
     top chunk body + highlighted     abstention: per question category
     supporting spans + citations      -> eval_report.json + summary table
   LLM (if env set):
     JSON in / JSON out; cited ids
     and quotes validated, else
     falls back to extractive
          |
          v
  cli.py      ragkit index | ask | chat | eval     (python -m ragkit ...)
              all output escaped for the terminal
```

## Eval results

There are two QA sets. Both run with k = 4, the TF-IDF backend, threshold 0.2,
and 25 chunks from the five sample docs. The tables below are the actual output
from this repo. `tests/test_eval.py` checks that the first table matches a fresh
run, and `scripts/check_demo.sh` asserts the headline numbers of both.

**`samples/qa.json`**: 10 answerable questions. The chunking and the 0.2
threshold were developed against this set and the off-topic probes in the
tests:

```text
$ python -m ragkit eval
id     type   recall@4  ctx_prec  ans_rec ans_prec abstain  question
--------------------------------------------------------------------
q01    ans       1.000     1.000    1.000    1.000      no  What port does the web console use by default?
q02    ans       0.000     0.000    0.000    0.000     yes  How long do I have to ask for my money back?
q03    ans       1.000     1.000    1.000    1.000      no  How much does the Business plan cost per seat?
q04    ans       1.000     1.000    1.000    1.000      no  How long are old file versions kept and how do ...
q05    ans       1.000     1.000    1.000    1.000      no  What happens when a workspace runs out of stora...
q06    ans       1.000     1.000    1.000    1.000      no  What encryption is used for files at rest?
q07    ans       1.000     1.000    1.000    1.000      no  How do I fix the client saying the certificate ...
q08    ans       1.000     1.000    1.000    1.000      no  Two people edited the same file at once. What d...
q09    ans       1.000     1.000    1.000    1.000      no  How long before a teammate invitation expires?
q10    ans       1.000     1.000    0.500    1.000      no  Can we sign in with SAML single sign-on?
--------------------------------------------------------------------
retrieval (raw top-4, 10 answerable): recall@4 0.900, context precision 0.900
answer context (cited chunks, threshold 0.2): recall 0.850, precision 0.900
abstention: answered 0.900 (9/10) answerable; abstained on n/a (0 items) out-of-domain, n/a (0 items) unanswerable
dataset samples/qa.json sha256 147c5ec69835, 10 questions, backend=tfidf
```

**`samples/heldout_qa.json`**: a frozen set of 15 questions. It has 5
answerable, 5 out-of-domain and 5 in-domain unanswerable questions. It was
written after the threshold was chosen and is not used to tune anything.
Changing it changes its SHA-256 in the report:

```text
$ python -m ragkit eval --qa samples/heldout_qa.json --report heldout_report.json
id     type   recall@4  ctx_prec  ans_rec ans_prec abstain  question
--------------------------------------------------------------------
h01    ans       1.000     1.000    1.000    1.000      no  How often do clients poll the server for changes?
h02    ans       1.000     1.000    1.000    1.000      no  Where do ignore rules for syncing live?
h03    ans       1.000     1.000    1.000    1.000      no  How long do web console sessions last?
h04    ans       1.000     1.000    1.000    1.000      no  Which payment methods are accepted?
h05    ans       1.000     1.000    1.000    1.000      no  What are the minimum server requirements?
h06    ood           -         -        -        -     yes  Who won the 1998 football World Cup?
h07    ood           -         -        -        -     yes  How do I bake sourdough bread at home?
h08    ood           -         -        -        -     yes  What is the boiling point of water in Kelvin?
h09    ood           -         -        -        -     yes  Can you recommend a good science fiction novel?
h10    ood           -         -        -        -     yes  What is the population of Tokyo?
h11    unans         -         -        -        -      no  Can I extend teammate invitations to 90 days?
h12    unans         -         -        -        -     yes  How do I migrate a workspace to another Tidepoo...
h13    unans         -         -        -        -      no  What is the maximum number of workspaces per se...
h14    unans         -         -        -        -      no  How do I rotate the server master key?
h15    unans         -         -        -        -     yes  Does Tidepool offer a discount for nonprofit or...
--------------------------------------------------------------------
retrieval (raw top-4, 5 answerable): recall@4 1.000, context precision 1.000
answer context (cited chunks, threshold 0.2): recall 1.000, precision 1.000
abstention: answered 1.000 (5/5) answerable; abstained on 1.000 (5/5) out-of-domain, 0.400 (2/5) unanswerable
dataset samples/heldout_qa.json sha256 16cf5fa9c0ec, 15 questions, backend=tfidf
```

| metric | qa.json (dev) | heldout_qa.json |
| --- | --- | --- |
| raw retrieval recall@4 | **0.900** | 1.000 |
| raw retrieval context precision@4 | **0.900** | 1.000 |
| answer-context recall (after threshold) | 0.850 | 1.000 |
| answer-context precision (after threshold) | 0.900 | 1.000 |
| answerable questions answered | 9/10 | 5/5 |
| out-of-domain questions abstained | n/a | 5/5 |
| in-domain unanswerable questions abstained | n/a | **2/5** |

What the numbers mean, and their limits:

* **Raw retrieval** metrics score the top-k chunks straight from the store,
  before the abstention threshold.
  **recall@k** is the fraction of a question's gold chunks found in the top k.
  **Context precision@k** is rank-aware: it is the mean of precision@i over
  the ranks i that hold a gold chunk, so it drops when non-gold chunks rank
  *above* gold ones. A question with no gold chunk retrieved scores 0.
* **Answer context** metrics apply the same formulas to the chunks that `ask`
  actually cites, after the threshold. For q10 the second gold chunk
  (`billing.md#plans/0`) is retrieved but scores 0.125, below 0.2, so the
  answer never uses it. Raw recall is 1.0 but answer-context recall is 0.5.
* **Abstention** reports what `ask` does per category. Answering an
  out-of-domain or unanswerable question counts as a failure.
* **q02 is a deliberate paraphrase miss.** "Ask for my money back" shares no
  terms with the refunds section ("refund", "purchase", "30 days"), so lexical
  TF-IDF scores every chunk at 0 and `ask` abstains on a question the docs do
  answer. Closing this kind of vocabulary gap is what the optional embedding
  backend is for.
* These are smoke tests on five short docs, not a benchmark. The per-question
  rows in the JSON report (retrieved ids, scores, cited ids, abstained) are the
  useful part when you tune chunking or thresholds on your own corpus.

### Limits of the abstention threshold

The threshold (`0.2`, chosen on `qa.json`) is a cosine-similarity cut-off. It
reliably catches questions that share no vocabulary with the docs: all five
held-out out-of-domain questions abstained. It does **not** detect questions
that are on-topic but unanswered by the docs. Three of the five held-out
unanswerable questions cleared it, for example "Can I extend teammate
invitations to 90 days?" at 0.369 against a chunk that only states the default.
Above-threshold output is therefore labelled *retrieved evidence*. Treat it as
"the most relevant passage", not "the answer". Re-check the threshold for your
own docs with `--threshold` and a held-out set of your own.

## Why extractive-first

* **Deterministic.** The same docs and the same question always give the same
  chunks, the same scores and the same output. Tests and evals are stable, and a
  regression means something actually changed.
* **Zero cost and zero setup.** No API keys, no GPU, no model download, and it
  runs on a small box or in CI. You can demo and evaluate retrieval quality
  before paying for anything.
* **Auditable.** Every word of the output is a verbatim quote from a cited chunk,
  so there is nothing invented to fact-check. What remains is to check whether
  the quoted passage actually answers your question.
* **Retrieval is the part that has to be right.** An LLM can only rephrase what
  retrieval hands it, so measure and fix retrieval first (that is what `eval`
  does), then add generation on top.

## Plugging in an LLM (optional)

Set these environment variables to use any OpenAI-compatible server (OpenAI,
vLLM, Ollama's `/v1`, LM Studio, and so on):

| variable | meaning |
| --- | --- |
| `OPENAI_COMPATIBLE_BASE_URL` | e.g. `https://api.openai.com/v1`, or `http://localhost:11434/v1` for a local server |
| `API_KEY` | bearer token sent to that server |
| `RAGKIT_LLM_MODEL` | chat model (default `gpt-4o-mini`) |
| `RAGKIT_EMBEDDINGS` | `tfidf` (default) or `openai` to use the `/embeddings` endpoint |
| `RAGKIT_EMBED_MODEL` | embedding model (default `text-embedding-3-small`) |

* **Generation:** when both `OPENAI_COMPATIBLE_BASE_URL` and `API_KEY` are set,
  `ask` and `chat` send the question and the above-threshold chunks to
  `POST {base}/chat/completions`. The user message is a JSON object with
  separate `question` and `excerpts` fields. The system prompt says excerpt
  text is untrusted data and that any instructions in it must be ignored. The
  model must reply with JSON, `{"answer": ..., "evidence": [{"chunk_id": ...,
  "quote": ...}]}`, or the "Not found in these docs." form. Every evidence item
  is validated: the id must be one of the excerpts sent, and the quote must
  appear verbatim (up to whitespace) in that excerpt. Only validated evidence is
  cited. An empty, non-JSON or evidence-less reply, or one with any invalid
  evidence item, is rejected. The kit then shows the local extractive answer
  with a note saying the LLM reply was rejected. If no chunk clears the
  threshold, nothing is sent at all. Pass `--extractive` to force the local
  generator even with the variables set.
* **Embeddings:** `RAGKIT_EMBEDDINGS=openai python -m ragkit index`, or
  `ragkit index --backend openai`, builds the store with remote embeddings.
  The backend name is saved in the store, so later `ask` and `eval` runs use
  the same backend. The TF-IDF threshold won't carry over to embedding scores,
  so re-tune `--threshold`.
* **Isolation:** all network code lives in `src/ragkit/remote.py`
  (`urllib_transport`). It is imported lazily, only on the remote paths. Both
  remote backends take an injectable `transport`, which the tests replace with
  a fake.

### Security notes for the remote backends

* **HTTPS only.** The base URL must be `https://`. Plain `http://` is accepted
  only for loopback hosts (`localhost`, `127.0.0.0/8`, `::1`), and those
  requests ignore `http_proxy`-style environment proxies. URLs with embedded
  credentials (`user:pass@`), a query string, a fragment, or a malformed host
  or port, or a hostname over the DNS limits (63-character labels, 253
  characters total), are rejected. Error messages never echo the URL or a
  redirect target, including a malformed `Location` header. Redirects are never followed, so the bearer token is only sent to
  the configured origin.
  The key is stored as a pydantic `SecretStr` and is redacted from the config's
  `repr` and JSON.
* **What leaves the machine.** With remote embeddings enabled, every indexed
  chunk is sent to the provider at index time, and every question at query
  time. With the LLM enabled, each question and its above-threshold chunks are
  sent. Only enable these for documents you are allowed to share with that
  provider.
* **Prompt injection is mitigated, not solved.** Instructions hidden in your
  documents still reach the model. The prompt tells the model to ignore them,
  and replies are rejected unless their cited ids and quotes check out. But a
  model can still be steered into writing a misleading answer next to a real
  quote. Validation proves the quote exists in the cited chunk, not that it
  supports the answer. If your documents are not trusted, use the extractive
  default.

## Data handling

* **The index contains your source text.** `.ragkit/index.jsonl` stores every
  chunk's text in plain JSON next to its vector. Protect it like the documents
  themselves. It is written to a unique temp file in the same directory,
  created with owner-only permissions, and then atomically renamed over the old
  index. A failed or concurrent `index` run never leaves a half-written index.
* **Directory ingest stays inside `--docs`.** The directory is opened once,
  with no path-based precheck, so the root cannot be swapped between a check
  and the open (a symlink you pass as `--docs` is resolved at that single
  open). The tree is then walked only through file descriptors: every
  subdirectory and file is opened relative to its parent with `O_NOFOLLOW`
  (files also with `O_NONBLOCK`, so a FIFO cannot hang ingest), checked with
  `fstat` to be a regular file, and read from that same descriptor. Symlinked
  files and directories are skipped (with a warning), even if a path, or the
  root itself, is swapped for a symlink mid-ingest. If you pass a single file
  to `--docs`, it is read as-is.
  Directory ingest is POSIX-only: it relies on `O_NOFOLLOW`, `O_DIRECTORY` and
  `dir_fd`-relative `open`/`stat`, and this hardening is verified on Linux
  (locally and in CI).
* **Terminal output is escaped.** Document text, filenames, QA questions and
  model output are untrusted. The CLI replaces control characters (except
  newline and tab) and Unicode bidi overrides with visible escapes such as
  `\x1b`, so they cannot rewrite or spoof the terminal.

## Offline guarantee

`tests/conftest.py` has an autouse fixture that replaces `socket.connect`,
`connect_ex`, `create_connection` and `getaddrinfo` with functions that raise,
and clears all backend environment variables. Every test, including the
end-to-end `index` → `ask` → `chat` → `eval` CLI tests, runs under that block.
`test_network_is_really_blocked` checks that the block is actually in place.

CI goes further. It installs from the wheelhouse with `--no-index` (outside the
namespace), then runs `pytest` and `scripts/check_demo.sh` inside
`unshare --user --map-root-user --net`. It first checks that
an outbound connection fails in that namespace. `check_demo.sh` asserts the
expected answer text, citation, abstention and report values, not just exit
codes. Locally, the same two commands were run inside a `bwrap --unshare-net`
sandbox: All 155 tests passed and `check_demo.sh` passed.

## Layout

```text
src/ragkit/
  ingest.py     md/txt loader, optional pdf via pypdf, symlink-safe directory walk
  chunk.py      heading-aware markdown chunking, sliding-window fallback
  vectorize.py  tokenizer, TF-IDF backend, Vectorizer protocol, OpenAI embeddings
  store.py      VectorStore: cosine top-k, atomic JSONL save/load
  generate.py   extractive generator, validated LLM generator, abstention, rendering
  eval.py       retrieval, answer-context and abstention metrics, JSON report, summary
  remote.py     env config + HTTPS-only, no-redirect transport for the optional backends
  termsafe.py   escaping of terminal control characters for CLI output
  cli.py        Typer app (index, ask, chat, eval)
tests/          chunking boundaries/edge cases, retrieval ranking, eval math, CLI e2e,
                LLM output validation, transport hardening
scripts/check_demo.sh  demo commands with output assertions (used by CI)
samples/docs/   docs for "Tidepool", a fictional self-hosted file-sync product
samples/qa.json 10 answerable questions with gold chunk ids (development set)
samples/heldout_qa.json  15 frozen questions: answerable, out-of-domain, unanswerable
```

## License

MIT. See [LICENSE](LICENSE).

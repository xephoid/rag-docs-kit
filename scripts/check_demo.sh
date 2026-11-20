#!/usr/bin/env bash
# Run the demo commands (index, ask, eval) and assert their output.
# CI runs this inside a network namespace with no network access; it can also be
# run locally, e.g. `bwrap --dev-bind / / --unshare-net bash scripts/check_demo.sh`.
set -euo pipefail

PYTHON="${PYTHON:-python}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
store="$work/index.jsonl"

fail() { echo "check_demo: $*" >&2; exit 1; }
expect() { grep -qF -- "$2" "$1" || { cat "$1" >&2; fail "expected '$2' in $(basename "$1")"; }; }

"$PYTHON" -m ragkit index --store "$store" > "$work/index.txt"
expect "$work/index.txt" "Indexed 5 documents into 25 chunks (tfidf)"

"$PYTHON" -m ragkit ask "How long before a teammate invitation expires?" --store "$store" \
  > "$work/ask.txt"
expect "$work/ask.txt" "Retrieved evidence (quoted from the docs"
expect "$work/ask.txt" "**Invitations** **expire** after 7 days."
expect "$work/ask.txt" "[1] getting-started.md :: getting-started.md#inviting-teammates/0"

"$PYTHON" -m ragkit ask "What is the capital of France?" --store "$store" > "$work/abstain.txt"
[ "$(cat "$work/abstain.txt")" = "Not found in these docs." ] \
  || { cat "$work/abstain.txt" >&2; fail "expected abstention"; }

"$PYTHON" -m ragkit eval --store "$store" --report "$work/report.json" > "$work/eval.txt"
expect "$work/eval.txt" "retrieval (raw top-4, 10 answerable): recall@4 0.900, context precision 0.900"
"$PYTHON" -m ragkit eval --store "$store" --qa samples/heldout_qa.json \
  --report "$work/heldout.json" > "$work/heldout.txt"

"$PYTHON" - "$work/report.json" "$work/heldout.json" <<'PY'
import json
import sys

main, heldout = (json.load(open(path)) for path in sys.argv[1:])
expected_main = {
    "k": 4, "threshold": 0.2, "num_questions": 10, "backend": "tfidf",
    "recall_at_k": 0.9, "context_precision": 0.9,
    "answer_context_recall": 0.85, "answer_context_precision": 0.9, "answer_rate": 0.9,
}
expected_heldout = {
    "num_answerable": 5, "num_out_of_domain": 5, "num_unanswerable": 5,
    "recall_at_k": 1.0, "answer_context_recall": 1.0, "answer_rate": 1.0,
    "out_of_domain_abstention_rate": 1.0, "unanswerable_abstention_rate": 0.4,
}
for name, report, expected in (("qa", main, expected_main), ("heldout", heldout, expected_heldout)):
    for key, value in expected.items():
        if report[key] != value:
            sys.exit(f"check_demo: {name} report {key} = {report[key]!r}, expected {value!r}")
    if len(report["dataset_sha256"]) != 64:
        sys.exit(f"check_demo: {name} report has no dataset revision")
PY
echo "check_demo: all demo assertions passed"

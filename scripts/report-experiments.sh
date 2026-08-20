#!/usr/bin/env bash
# Summarise one or more `experiments.jsonl` journals (Issue #15).
#
# Usage: scripts/report-experiments.sh <experiments.jsonl> [more.jsonl ...]
#
# Prints the `neat_ai_forests report` JSON for each journal, followed by a
# one-line comparison table of the go/no-go metric (scorer-verified
# improvement per wall-clock hour) so strategies/runs can be compared.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <experiments.jsonl> [more.jsonl ...]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${FORESTS_BIN:-$ROOT/target/release/neat_ai_forests}"
if [ ! -x "$BIN" ]; then
  (cd "$ROOT" && cargo build --release -q)
fi

printf '%-40s %10s %8s %14s %16s %12s\n' "journal" "iters" "accepts" "cumulativeΔ" "Δ/wall-hour" "stop"
for journal in "$@"; do
  out="${journal%.jsonl}.report.json"
  "$BIN" report "$journal" > "$out"
  python3 - "$journal" "$out" <<'PY'
import json, sys
with open(sys.argv[2], encoding="utf-8") as fh:
    r = json.load(fh)
def f(v, fmt):
    return "n/a" if v is None else fmt % v
print("%-40s %10d %8d %14s %16s %12s" % (
    sys.argv[1][-40:], r["iterations"], r["acceptances"],
    f(r.get("cumulativeImprovement"), "%.3e"),
    f(r.get("improvementPerWallHour"), "%.3e"),
    r.get("stopReason") or "n/a"))
PY
done
echo "per-journal reports written beside each journal as *.report.json"

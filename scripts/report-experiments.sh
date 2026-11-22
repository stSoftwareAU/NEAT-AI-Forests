#!/usr/bin/env bash
# Summarise one or more `experiments.jsonl` journals (Issue #15).
#
# Usage: scripts/report-experiments.sh <experiments.jsonl> [more.jsonl ...]
#
# Prints the `neat_ai_forests report` JSON for each journal, followed by a
# one-line comparison table of the go/no-go metric (scorer-verified
# improvement per wall-clock hour) so strategies/runs can be compared.
#
# Then, for journals whose run header records `forests-arms` (Issue #147), a
# per-arm table: for each candidate family, the runs where it was drawn on
# against the runs where it was drawn off. Only `drawn` arms count — an arm
# someone set by hand was chosen, not randomised, so it would bias the
# comparison. A verdict needs many runs: the champion and the corpus move
# between them.
#
# FORESTS_BIN overrides the binary (tests point it at a stub).
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

reports=()
printf '%-40s %10s %8s %14s %16s %12s\n' "journal" "iters" "accepts" "cumulativeΔ" "Δ/wall-hour" "stop"
for journal in "$@"; do
  out="${journal%.jsonl}.report.json"
  "$BIN" report "$journal" > "$out"
  reports+=("$out")
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

python3 - "${reports[@]}" <<'PY'
import json, sys

# Strategy-name prefixes/suffixes of each family's accepted patches.
FAMILY = {
    "hidden-features": lambda s: s.endswith("+hidden"),
    "linear-leaf-conditions": lambda s: s.startswith("linear-leaf"),
    "refine-leaves": lambda s: s.startswith("refine-"),
    "refit-every": lambda s: s == "refit",
}

runs = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as fh:
        r = json.load(fh)
    arms = {}
    for item in filter(None, (r.get("arms") or "").split(",")):
        name, _, rest = item.partition("=")
        value, _, source = rest.partition(":")
        arms[name] = (int(value), source)
    if arms:
        runs.append((r, arms))

if not runs:
    sys.exit(0)

def cell(rs):
    n = len(rs)
    acc = sum(r["acceptances"] for r in rs)
    gain = sum(r.get("cumulativeImprovement") or 0.0 for r in rs)
    hours = sum(r.get("wallMs") or 0 for r in rs) / 3_600_000
    calls = sum(r.get("candidatesFullScored") or 0 for r in rs)
    per_hour = "%.3e" % (gain / hours) if hours > 0 else "n/a"
    per_call = "%.3e" % (gain / calls) if calls > 0 else "n/a"
    return n, acc, per_hour, per_call

print()
print("per-arm (drawn runs only, Issue #147)")
print("%-24s %-4s %6s %8s %14s %14s %12s" % (
    "arm", "", "runs", "accepts", "Δ/wall-hour", "Δ/full-call", "family-wins"))
for arm, is_family in FAMILY.items():
    drawn = [(r, a) for r, a in runs if a.get(arm, (0, ""))[1] == "drawn"]
    on = [r for r, a in drawn if a[arm][0] > 0]
    off = [r for r, a in drawn if a[arm][0] == 0]
    for label, rs in (("on", on), ("off", off)):
        n, acc, per_hour, per_call = cell(rs)
        if label == "on":
            wins = sum(1 for r in rs for g in r.get("accepted", []) if is_family(g["strategy"]))
            share = "%d/%d" % (wins, acc)
        else:
            share = "-"
        print("%-24s %-4s %6d %8d %14s %14s %12s" % (
            arm if label == "on" else "", label, n, acc, per_hour, per_call, share))
PY

#!/usr/bin/env bash
# Run the production-shaped histogram search benchmark (Issues #6, #15).
#
# Usage: scripts/run-benchmark.sh [records] [features] [threads]
#
# Runs the CPU-only build first, then the `gpu` feature build when it
# compiles, and prints both JSON reports. Paste the results into
# docs/benchmarks.md together with the hardware description.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORDS="${1:-200000}"
THREADS="${3:-8}"
FEATURES="${2:-2461}"

echo "== CPU build =="
(cd "$ROOT" && cargo run --release -q --example stump_search_bench -- "$RECORDS" "$FEATURES" "$THREADS")

echo "== GPU build (feature gpu) =="
if (cd "$ROOT" && cargo build --release -q --features gpu --example stump_search_bench); then
  (cd "$ROOT" && cargo run --release -q --features gpu --example stump_search_bench -- "$RECORDS" "$FEATURES" "$THREADS")
else
  echo "gpu feature did not build on this host; CPU numbers only"
fi

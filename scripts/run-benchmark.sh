#!/usr/bin/env bash
# Run the production-shaped histogram search benchmark (Issues #6, #15).
#
# Usage: scripts/run-benchmark.sh [records] [features] [threads]
#
# Prints the JSON report. Paste the result into docs/benchmarks.md together
# with the hardware description.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORDS="${1:-200000}"
THREADS="${3:-8}"
FEATURES="${2:-2461}"

(cd "$ROOT" && cargo run --release -q --example stump_search_bench -- "$RECORDS" "$FEATURES" "$THREADS")

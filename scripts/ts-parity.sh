#!/usr/bin/env bash
# rust_scorer vs NEAT-AI TypeScript Creature.scoreDir parity on grafted fixtures
# (Issue #35). Needs deno, a rust_scorer build and a checkout whose Deno import
# map provides @stsoftware/neat-ai (e.g. a GRQ or NEAT-AI checkout).
#
# Usage: NEAT_AI_TS_ROOT=../GRQ [NEAT_SCORER_BIN=...] scripts/ts-parity.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${NEAT_AI_TS_ROOT:?set NEAT_AI_TS_ROOT to a checkout whose import map provides @stsoftware/neat-ai}"
cd "$ROOT" && cargo test --test ts_parity -- --nocapture

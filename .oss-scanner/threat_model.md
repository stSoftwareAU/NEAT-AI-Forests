# Threat model

Disclosure channels and scope are in [SECURITY.md](../SECURITY.md). The
pipeline is described in [docs/architecture.md](../docs/architecture.md), the
on-disk caches in [docs/caches.md](../docs/caches.md), the patch format in
[docs/patch-format.md](../docs/patch-format.md) and the shared learnings in
[docs/learnings.md](../docs/learnings.md). This file is the short brief for
the scanner.

## What this project does and where untrusted input enters

NEAT-AI-Forests is an experimental, offline Rust CLI (`neat_ai_forests`) that
reads an already-fit NEAT-AI creature and its training corpus, searches for
tree-shaped corrections to the residual errors, grafts them onto copies of the
creature as `IF` structure, and keeps one only when the external NEAT-AI-scorer
binary (`rust_scorer`) says it is fitter. It opens no network connections and
holds no credentials. Treat as untrusted:

- creature JSON (the positional creature argument), which in production is
  pulled from a repository shared by many machines;
- the training corpus directory of headerless little-endian `f32` `.bin` files,
  streamed by `src/corpus.rs`;
- the versioned bin and residual caches (`src/bins.rs`, `src/residuals.rs`) in
  `--cache-dir`, which defaults to the corpus directory;
- the fleet-shared learnings cache (`--learnings-dir`, `src/learnings/`), which
  other hosts write;
- `experiments.jsonl` journals read by `report`, XGBoost booster dumps read by
  `import-xgboost` (`src/xgboost.rs`), and the scorer's output.

The scorer path, `--scorer-arg` values and the output directory are chosen by
the operator and are trusted.

## Components that matter most / least

Most important: creature loading (`src/incumbent.rs`), corpus streaming, cache
deserialisation, learnings parsing, the grafting code that rewrites the
creature (`src/graft.rs`, `src/tree.rs`, `src/oblique.rs`) and every file name
built from input data under the output, cache or learnings directory. Lower
priority: `src/report.rs`, `export-matrix`, the benches under
`forests/examples/`, and `scripts/xgboost-control.py`, a research aid.

## How to exercise it

From `/src`: `cargo test --workspace --all-features` runs the suite.
`forests/src/graft.rs` exposes small fixture creatures and
`corpus::write_bin_file` writes a corpus; `target/debug/neat_ai_forests --help`
lists the subcommands. `rust_scorer` is on `PATH` when its optional install
succeeded; `forests/tests/ts_parity.rs` also needs Deno and skips without it.

## How you rate severity

This is a research tool run by its own maintainers on their own hosts. The
crate has no `unsafe` code of its own.

- High: memory unsafety reachable from input, writing outside the output,
  cache or learnings directory (path traversal), running a command other than
  the operator's scorer, or accepting a patch the scorer did not verify.
- Medium: panics, unbounded memory or CPU, or hangs on malformed creature
  JSON, corpus files, caches, journals or learnings entries.
- Low: wrong statistics in reports, or issues that need the operator's own
  arguments to be hostile.

## Anything to leave alone

- Bugs in `neat-core` (NEAT-AI-core), `neat-ai-rebase` or `rust_scorer` belong
  in those repositories.
- A patch that fails to improve the score is the expected outcome the scorer
  gate exists for, not a vulnerability.

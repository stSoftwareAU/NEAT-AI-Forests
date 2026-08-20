# Contributing to NEAT-AI-Forests

## Repository layout

```text
parent/
├── NEAT-AI-core/      # sibling clone; forests/Cargo.toml depends on ../../NEAT-AI-core/neat-core
├── NEAT-AI-scorer/    # build it for integration tests: cargo build --release
└── NEAT-AI-Forests/
```

CI checks NEAT-AI-core out beside the workspace via
`.github/actions/setup-neat-core`. `neat-core.expected-version` records the
last handled neat-core version; `scripts/check-neat-core-version.sh` fails on
an unhandled breaking bump.

## Prerequisites

- Rust pinned by `rust-toolchain.toml` (rustup resolves it automatically).
- `shellcheck`, `codespell` (`pip install --user codespell`),
  `cargo install cargo-deny --locked`; optionally `markdownlint-cli2` and
  `actionlint` (CI runs them regardless).
- For `forests/tests/real_scorer.rs`: a built `rust_scorer` at
  `../NEAT-AI-scorer/target/release/rust_scorer` or `NEAT_SCORER_BIN=…`.
  The tests print a skip notice and pass when it is absent.
- For the `gpu` feature: a native wgpu adapter (Metal on macOS). The parity
  test skips without one.

## Local gate

```bash
./quality.sh < /dev/null
```

mirrors CI: shell syntax + shellcheck, neat-core version gate, codespell,
markdownlint, actionlint, cargo-deny, `cargo fmt --check`, clippy with
`-D warnings -D clippy::filter_next -D clippy::collapsible_if`,
`cargo test --all-features`, rustdoc with `-D warnings`.

## Principles every change must keep

1. The supplied creature is never written to.
2. Only a full-corpus NEAT-AI-scorer result can accept a candidate; proxies,
   samples and GPU gains rank only.
3. `best.json` is never worse than the opening authoritative baseline.
4. Every strategy names itself honestly in provenance.
5. Scorer failure, malformed output or baseline disagreement means no winner.
6. Bump `forests/Cargo.toml` `version` for binary-affecting changes and note
   them under `[Unreleased]` in `CHANGELOG.md`.

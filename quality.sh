#!/usr/bin/env bash
# Local gate — mirrors `.github/workflows/ci.yml` quality checks.
set -euo pipefail

if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

export RUSTFLAGS="-D warnings"
echo "Pre-deployment Quality Check (NEAT-AI-Forests)"
echo "=============================================="

echo "Checking bash script syntax..."
find . -name "*.sh" -type f -not -path "./target/*" -not -path "./.git/*" -exec bash -n {} \;

echo "Running shellcheck on bash scripts..."
if ! command -v shellcheck &>/dev/null; then
  echo "shellcheck is required — install: https://github.com/koalaman/shellcheck#installing"
  exit 1
fi
SHELLCHECK_FAILED=0
while IFS= read -r script; do
  echo "  shellcheck: $script"
  if ! shellcheck -x -s bash "$script"; then
    SHELLCHECK_FAILED=1
  fi
done < <(find . -name "*.sh" -type f -not -path "./target/*" -not -path "./.git/*")
if [[ "$SHELLCHECK_FAILED" -ne 0 ]]; then
  echo "shellcheck: FAILED"
  exit 1
fi
echo "shellcheck: all scripts passed"

if [ -f "./../NEAT-AI-core/Cargo.toml" ]; then
  echo "Gating on unhandled breaking neat-core bump..."
  ./scripts/check-neat-core-version.sh
else
  echo "sibling ../NEAT-AI-core not found — skipping neat-core version gate (CI runs this for real)"
fi

echo "Running codespell preflight..."
if ! ./scripts/spell-check.sh; then
  echo "spell-check: FAILED — fix the typos above or update .codespellrc"
  exit 1
fi

if command -v markdownlint-cli2 &>/dev/null; then
  echo "Running markdownlint-cli2..."
  markdownlint-cli2
else
  echo "markdownlint-cli2 not installed — skipping (CI runs it for real)"
fi

if command -v actionlint &>/dev/null; then
  echo "Running actionlint..."
  actionlint
else
  echo "actionlint not installed — skipping (CI runs it for real)"
fi

echo "Running licence and dependency audit (cargo-deny)..."
if ! command -v cargo-deny &>/dev/null; then
  echo "cargo-deny is required — install: cargo install cargo-deny --locked"
  exit 1
fi
cargo deny check

echo "Checking formatting..."
cargo fmt --all -- --check

echo "Running linter..."
cargo clippy --workspace --all-targets --all-features -- \
  -D warnings \
  -D clippy::filter_next \
  -D clippy::collapsible_if

echo "Running tests..."
cargo test --workspace --all-features -- --test-threads=2

echo "Building documentation..."
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features

echo "All quality checks passed!"

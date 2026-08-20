#!/bin/bash
# spell-check.sh — local codespell preflight that mirrors CI (Issue #22).
#
# Runs codespell against the repository using the shared `.codespellrc`
# configuration so contributors can reproduce the CI spell-check job before
# pushing. The ignore list lives in `.codespellrc` — add curated domain
# terms there rather than silencing files ad hoc.
#
# Usage:
#   scripts/spell-check.sh                 # scan the repo root
#   scripts/spell-check.sh --root <dir>    # scan a different directory
#
# Exit codes:
#   0 — no typos detected
#   1 — typos detected or codespell missing / invalid invocation
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: spell-check.sh [--root <dir>]

Runs codespell against <dir> (default: repository root) using the shared
.codespellrc ignore list. Install codespell with:

  pip install --user codespell

See the "Spell check" section in README.md for the full contributor workflow.
USAGE
}

ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="${2:-}"
      if [[ -z "$ROOT" ]]; then
        echo "--root requires a directory argument" >&2
        usage >&2
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ROOT" ]]; then
  # Default to the repository root relative to this script so the preflight
  # works regardless of the caller's CWD.
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

if [[ ! -d "$ROOT" ]]; then
  echo "spell-check: root directory not found: $ROOT" >&2
  exit 1
fi

# Allow user-site installs (macOS / CI without sudo) to resolve.
for candidate in "$HOME/Library/Python/3.13/bin" "$HOME/.local/bin"; do
  if [[ -d "$candidate" ]]; then
    case ":$PATH:" in
      *":$candidate:"*) ;;
      *) PATH="$candidate:$PATH" ;;
    esac
  fi
done
export PATH

if ! command -v codespell >/dev/null 2>&1; then
  cat >&2 <<'EOF'
spell-check: codespell is not installed.

Install it with one of:

  pip install --user codespell
  pipx install codespell
  brew install codespell

Then re-run: scripts/spell-check.sh
EOF
  exit 1
fi

echo "📝 Running codespell on: $ROOT"
# codespell automatically picks up .codespellrc from its working directory,
# so cd into the target root before invoking it. No explicit flags here —
# keep the single source of truth in `.codespellrc`.
(
  cd "$ROOT"
  if ! codespell; then
    echo "codespell: typos detected — fix the reported words or curate the ignore list in .codespellrc (see README 'Spell check' section)." >&2
    exit 1
  fi
)
echo "codespell: no typos found"

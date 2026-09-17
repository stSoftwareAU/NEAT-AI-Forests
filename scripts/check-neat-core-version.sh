#!/usr/bin/env bash
# Gate against an unhandled breaking neat-core bump.
#
# Forests consumes neat-core through a git dependency pinned to a release tag
# (see `forests/Cargo.toml`), and `scripts/family-pins.sh` moves that pin to
# core's newest release on every PR. The pin therefore still arrives on its own
# — this gate is the safeguard: it fails when the pinned release carries a
# breaking bump Forests has not yet acknowledged, forcing a deliberate upgrade
# instead of adopting a new major (or pre-1.0 minor) blindly.
#
# Mechanism — version-baseline check:
#   * Forests records the last-handled neat-core version in the checked-in
#     `neat-core.expected-version` file.
#   * The pinned version is read from `Cargo.lock` — the resolved `neat-core`
#     package, which must carry a `git+…/NEAT-AI-core?tag=v<semver>` source. No
#     sibling checkout is consulted, so the gate runs anywhere the repository
#     is cloned (Issue #105).
#   * There must be exactly one of them. `neat-ai-rebase` carries its own
#     neat-core release pin, and cargo locks two git sources at different tags
#     without complaint — the build only dies later in rustc, as a type
#     mismatch. This gate names both versions instead.
#   * The "breaking component" is the major for >= 1.0 releases and the minor
#     for pre-1.0 (0.x) releases, per SemVer. The gate FAILS when neat-core's
#     breaking component is greater than the recorded baseline; it PASSES on
#     patch-level drift (policy) and when the two match.
#
# A `neat-core` that is present but not pinned to a release tag — a path
# dependency, a branch pin, a registry version — is a usage error, not a pass:
# the pin this gate reads would otherwise have been silently dropped.
#
# "Handling" a breaking bump = a deliberate Forests PR that makes the
# corresponding code change AND bumps the recorded baseline to the new
# neat-core version.
#
# Usage:
#   check-neat-core-version.sh [--baseline PATH] [--lockfile PATH]
#
# Defaults resolve the repository's own files: the baseline and `Cargo.lock` at
# the repo root.
#
# Exit codes:
#   0  versions are compatible (match, patch drift, or core behind baseline)
#   1  the pin contract is broken — a breaking neat-core bump above the
#      recorded baseline, or more than one neat-core in the lockfile
#   2  usage / parse error (missing file, malformed or missing pin)
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: check-neat-core-version.sh [--baseline PATH] [--lockfile PATH]

Options:
  --baseline PATH        File recording the last-handled neat-core version
                         (default: neat-core.expected-version at the repo root).
  --lockfile PATH        Cargo.lock carrying the resolved neat-core release pin
                         (default: Cargo.lock at the repo root).
  -h, --help             Show this message.

Exits 0 when compatible, 1 on an unhandled breaking bump or a lockfile carrying
more than one neat-core, 2 on a usage error.
EOF
}

BASELINE=""
LOCKFILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      [[ $# -ge 2 ]] || { echo "Missing value for --baseline" >&2; usage >&2; exit 2; }
      BASELINE="$2"
      shift 2
      ;;
    --lockfile)
      [[ $# -ge 2 ]] || { echo "Missing value for --lockfile" >&2; usage >&2; exit 2; }
      LOCKFILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$BASELINE" ]]; then
  BASELINE="$REPO_ROOT/neat-core.expected-version"
fi
if [[ -z "$LOCKFILE" ]]; then
  LOCKFILE="$REPO_ROOT/Cargo.lock"
fi

if [[ ! -f "$BASELINE" ]]; then
  echo "FAIL: baseline file not found: $BASELINE" >&2
  exit 2
fi
if [[ ! -f "$LOCKFILE" ]]; then
  echo "FAIL: lockfile not found: $LOCKFILE" >&2
  echo "      Run 'cargo metadata' to regenerate it, then re-run this gate." >&2
  exit 2
fi

# First non-comment, non-blank line of the baseline file is the version.
read_baseline_version() {
  awk '
    { sub(/#.*/, "") }            # strip inline comments
    { gsub(/[[:space:]]+/, "") }  # trim all whitespace
    NF { print; exit }            # first non-empty line wins
  ' "$BASELINE"
}

# Print one `<version> <source>` line per resolved `neat-core` package in the
# lockfile, or nothing when the lockfile has no such package. Every match is
# printed, not just the first: two of them is the divergence the release pins
# exist to prevent, and reading only the first would measure the baseline
# against whichever entry happened to come first. A package with no `source`
# key (a path or workspace member) prints its version and an empty source, so
# the caller can tell "not pinned" from "not there".
read_core_pins() {
  awk '
    # The source key follows the version key, so a block is only answered once
    # it has ended: an early print would report the pin with no source at all.
    function flush() {
      if (name == "neat-core") print version " " source
      name = ""; version = ""; source = ""
    }
    /^\[/ { flush(); next }
    {
      key = $0
      sub(/=.*$/, "", key)
      gsub(/[[:space:]]/, "", key)
      if (key != "name" && key != "version" && key != "source") next
      if (!match($0, /"[^"]*"/)) next
      value = substr($0, RSTART + 1, RLENGTH - 2)
      if (key == "name") name = value
      else if (key == "version") version = value
      else source = value
    }
    END { flush() }
  ' "$LOCKFILE"
}

# Validate X.Y.Z (optionally with a -prerelease/+build suffix we ignore) and
# echo the bare "major minor patch" triple. Returns non-zero when malformed.
parse_semver() {
  local raw="$1" core
  core="${raw%%[-+]*}"   # drop pre-release / build metadata
  if [[ ! "$core" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    return 1
  fi
  local major minor patch
  # Scope IFS to this single read rather than tampering with it globally.
  IFS='.' read -r major minor patch <<<"$core"
  echo "$major $minor $patch"
}

baseline_raw="$(read_baseline_version)"
if [[ -z "$baseline_raw" ]]; then
  echo "FAIL: baseline file is empty: $BASELINE" >&2
  exit 2
fi

core_pins="$(read_core_pins)"
if [[ -z "$core_pins" ]]; then
  echo "FAIL: no neat-core package in $LOCKFILE" >&2
  echo "      forests/Cargo.toml must declare neat-core as a git dependency" \
    "pinned to a release tag." >&2
  exit 2
fi

# One neat-core for the whole graph. Forests' own pin and the one
# `neat-ai-rebase` carries at its pinned tag must name the same release: cargo
# locks two git sources at different tags perfectly happily, and the build only
# dies afterwards in rustc, where the message is a type mismatch rather than a
# version conflict. Catch it here, where the two versions can be named.
if [[ "$(printf '%s\n' "$core_pins" | wc -l | tr -d ' ')" -gt 1 ]]; then
  echo "FAIL: more than one neat-core in $LOCKFILE:" >&2
  printf '%s\n' "$core_pins" | while read -r dup_version dup_source; do
    echo "        $dup_version  ${dup_source:-<no source>}" >&2
  done
  echo "      neat-ai-rebase pins its own neat-core release; it must equal the" >&2
  echo "      one forests/Cargo.toml pins. Forests stays on the older pair" >&2
  echo "      until NEAT-AI-Rebase's pin PR lands and it cuts a new release." >&2
  exit 1
fi

read -r core_raw core_source <<<"$core_pins"

# The pin itself is part of the contract: a neat-core resolved from anywhere
# other than a NEAT-AI-core release tag means the release pin was dropped, and
# that must fail loud rather than be measured against the baseline.
CORE_TAG_SOURCE_RE='^git\+https://github\.com/stSoftwareAU/NEAT-AI-core\?tag=v[0-9]+\.[0-9]+\.[0-9]+(#.*)?$'
if [[ ! "${core_source:-}" =~ $CORE_TAG_SOURCE_RE ]]; then
  echo "FAIL: neat-core in $LOCKFILE is not pinned to a NEAT-AI-core release tag" >&2
  echo "      resolved source: ${core_source:-<none — a path or workspace member>}" >&2
  echo "      Expected git+https://github.com/stSoftwareAU/NEAT-AI-core?tag=vX.Y.Z" >&2
  exit 2
fi

if ! baseline_parts="$(parse_semver "$baseline_raw")"; then
  echo "FAIL: malformed baseline version '$baseline_raw' in $BASELINE (expected X.Y.Z)" >&2
  exit 2
fi
if ! core_parts="$(parse_semver "$core_raw")"; then
  echo "FAIL: malformed neat-core version '$core_raw' in $LOCKFILE (expected X.Y.Z)" >&2
  exit 2
fi

read -r b_major b_minor b_patch <<<"$baseline_parts"
read -r c_major c_minor c_patch <<<"$core_parts"
: "$b_patch" "$c_patch"  # patch components are informational only

remediation() {
  cat >&2 <<EOF
       neat-core has presented a breaking bump Forests has not handled.
       To clear this gate, in a single deliberate PR:
         1. Update forests for the breaking neat-core change.
         2. Bump the recorded baseline in neat-core.expected-version to $core_raw.
EOF
}

# Breaking-bump decision (SemVer): the major signals breaking changes once a
# crate reaches 1.0; before that, the minor carries that role.
if (( c_major > b_major )); then
  echo "FAIL: breaking neat-core bump: $core_raw exceeds handled baseline $baseline_raw (major increased)" >&2
  remediation
  exit 1
fi

if (( c_major == b_major )); then
  if (( b_major == 0 )); then
    # Pre-1.0: the minor is the breaking component.
    if (( c_minor > b_minor )); then
      echo "FAIL: breaking neat-core bump: $core_raw exceeds handled baseline $baseline_raw (pre-1.0 minor increased)" >&2
      remediation
      exit 1
    fi
    if (( c_minor < b_minor )); then
      echo "OK   neat-core $core_raw is behind handled baseline $baseline_raw (no breaking bump)"
      exit 0
    fi
    echo "OK   neat-core $core_raw matches handled baseline $baseline_raw (patch-level drift allowed)"
    exit 0
  fi
  # >= 1.0: same major — minor/patch drift is additive, never breaking.
  echo "OK   neat-core $core_raw within handled baseline $baseline_raw major line (patch-level drift allowed)"
  exit 0
fi

# c_major < b_major: neat-core sits below the baseline major — not a breaking
# bump scorer needs to act on.
echo "OK   neat-core $core_raw is behind handled baseline $baseline_raw (no breaking bump)"
exit 0

#!/usr/bin/env bash
# hold-core-pin.sh — hold the `neat-core` pin at the release `neat-ai-rebase`
# still carries (PR #111).
#
# `scripts/family-pins.sh` is NEAT-AI-core's canonical pin mover: it moves every
# family pin to that repository's newest release, one pin at a time, and knows
# nothing about the `neat-core` release a *second* family dependency carries.
# NEAT-AI-Rebase pins its own `neat-core`, so the moment core cuts a release
# ahead of Rebase's pin, `Cargo.lock` carries two `neat-core` copies and
# `scripts/check-neat-core-version.sh` reds the `validation` job. Reverting the
# pin by hand does not hold: CI re-runs `family-pins.sh` on the next push and
# moves it forward again.
#
# Only the consumer knows both pins, so the hold belongs here. Run this
# immediately after `family-pins.sh`: it rewrites the `neat-core` tag in
# `forests/Cargo.toml` back to the oldest `neat-core` the lockfile resolved —
# the one Rebase carries — and re-locks, leaving a single `neat-core` in the
# graph. Forests moves forward again once Rebase cuts a release on the newer
# core, which needs no change here: with one `neat-core` locked there is
# nothing to hold back.
#
# It is idempotent: a lockfile already carrying one `neat-core` is left alone
# and the run exits 0 having changed nothing.
#
# Usage:
#   hold-core-pin.sh [--manifest FILE] [--lockfile FILE] [-h|--help]
#
# Exit codes:
#   0  the lockfile carries exactly one neat-core (already, or after the hold)
#   1  the divergence cannot be held from here — forests already pins the older
#      release, so Rebase is ahead and the upgrade is a deliberate one — or the
#      re-lock did not converge on a single neat-core
#   2  usage / parse error (missing file, missing or malformed pin)
set -euo pipefail

CORE_URL="https://github.com/stSoftwareAU/NEAT-AI-core"

die() {
  printf 'hold-core-pin: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: hold-core-pin.sh [--manifest FILE] [--lockfile FILE]

Options:
  --manifest FILE  Manifest carrying the neat-core release pin
                   (default: forests/Cargo.toml at the repo root).
  --lockfile FILE  Cargo.lock to read the resolved neat-core versions from
                   (default: Cargo.lock at the repo root).
  -h, --help       Show this message.

Exits 0 when one neat-core is locked (already, or after the hold), 1 when the
divergence cannot be held from here, 2 on a usage error.
EOF
}

MANIFEST=""
LOCKFILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      [[ $# -ge 2 ]] || { echo "Missing value for --manifest" >&2; usage >&2; exit 2; }
      MANIFEST="$2"
      shift 2
      ;;
    --lockfile)
      [[ $# -ge 2 ]] || { echo "Missing value for --lockfile" >&2; usage >&2; exit 2; }
      LOCKFILE="$2"
      shift 2
      ;;
    -h | --help)
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
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
[[ -n "$MANIFEST" ]] || MANIFEST="${REPO_ROOT}/forests/Cargo.toml"
[[ -n "$LOCKFILE" ]] || LOCKFILE="${REPO_ROOT}/Cargo.lock"

[[ -f "$MANIFEST" ]] || { echo "FAIL: manifest not found: $MANIFEST" >&2; exit 2; }
[[ -f "$LOCKFILE" ]] || { echo "FAIL: lockfile not found: $LOCKFILE" >&2; exit 2; }

# Every resolved `neat-core` version in the lockfile, one per line. Kept
# deliberately narrower than the reader in `check-neat-core-version.sh`: that
# gate reports the source of each pin, this one only has to compare versions.
core_versions() {
  awk '
    function flush() {
      if (name == "neat-core" && version != "") print version
      name = ""; version = ""
    }
    /^\[/ { flush(); next }
    {
      key = $0
      sub(/=.*$/, "", key)
      gsub(/[[:space:]]/, "", key)
      if (key != "name" && key != "version") next
      if (!match($0, /"[^"]*"/)) next
      value = substr($0, RSTART + 1, RLENGTH - 2)
      if (key == "name") name = value; else version = value
    }
    END { flush() }
  ' "$1"
}

# True when version $1 is older than version $2. Both are plain X.Y.Z triples —
# the lockfile never carries anything else for a released git-tag pin.
version_older() {
  local i x y
  local -a a b
  local IFS='.'
  read -r -a a <<<"$1"
  read -r -a b <<<"$2"
  for ((i = 0; i < 3; i++)); do
    x="${a[i]:-0}"
    y="${b[i]:-0}"
    case "$x" in '' | *[!0-9]*) x=0 ;; esac
    case "$y" in '' | *[!0-9]*) y=0 ;; esac
    if ((10#$x < 10#$y)); then return 0; fi
    if ((10#$x > 10#$y)); then return 1; fi
  done
  return 1
}

# The `neat-core` release tag in the manifest, as `<lineno>\t<tag>`. A pin this
# reader cannot find is a usage error, never a silent pass: the hold would
# otherwise report success without having held anything.
manifest_pin() {
  awk -v url="$CORE_URL" '
    /^[[:space:]]*#/ { next }
    index($0, url) == 0 { next }
    {
      if (!match($0, /tag[[:space:]]*=[[:space:]]*"v[0-9]+\.[0-9]+\.[0-9]+"/)) next
      tag = substr($0, RSTART, RLENGTH)
      # Trim to what is inside the quotes — from the *first* quote, never a
      # greedy `^.*"`, which would swallow the tag itself.
      tag = substr(tag, index(tag, "\"") + 1)
      sub(/".*$/, "", tag)
      if (tag == "") next
      printf "%d\t%s\n", NR, tag
      exit
    }
  ' "$1"
}

# Re-resolve the lockfile after the manifest pin moved. Each superseded
# `neat-core` version is named explicitly: with two of them locked, a bare
# `--package neat-core` is ambiguous and cargo refuses it.
relock() {
  # A test seam: the hermetic tests hand this an executable that rewrites the
  # fixture lockfile, so the re-verification below is exercised without cargo
  # and without the network. Unset — every real run — it is `cargo update`.
  if [[ -n "${HOLD_CORE_PIN_RELOCK_CMD:-}" ]]; then
    "$HOLD_CORE_PIN_RELOCK_CMD" "$MANIFEST" "$LOCKFILE" "$@" ||
      die "re-lock command '$HOLD_CORE_PIN_RELOCK_CMD' failed"
    return 0
  fi
  command -v cargo >/dev/null 2>&1 ||
    die "cargo not found — install the Rust toolchain from https://rustup.rs and re-run"
  local spec output
  for spec in "$@"; do
    if ! output="$(cd "$REPO_ROOT" && cargo update --package "neat-core@${spec}" 2>&1)"; then
      printf '%s\n' "$output" >&2
      die "cargo update --package neat-core@${spec} failed — Cargo.lock does not match the held pin"
    fi
  done
}

versions="$(core_versions "$LOCKFILE")" || die "could not read $LOCKFILE"
count="$(printf '%s' "$versions" | grep -c . || true)"
if [[ "$count" -eq 0 ]]; then
  echo "FAIL: no neat-core package in $LOCKFILE" >&2
  exit 2
fi
if [[ "$count" -eq 1 ]]; then
  echo "OK   one neat-core ($versions) in ${LOCKFILE#"$REPO_ROOT"/} — nothing to hold"
  exit 0
fi

oldest=""
while IFS= read -r version; do
  [[ -n "$version" ]] || continue
  if [[ -z "$oldest" ]] || version_older "$version" "$oldest"; then
    oldest="$version"
  fi
done <<EOF
$versions
EOF

# Everything above the oldest is what the re-lock has to drop.
superseded=()
while IFS= read -r version; do
  [[ -n "$version" && "$version" != "$oldest" ]] || continue
  superseded+=("$version")
done <<EOF
$versions
EOF

pin="$(manifest_pin "$MANIFEST")"
if [[ -z "$pin" ]]; then
  echo "FAIL: no neat-core release-tag pin in $MANIFEST" >&2
  echo "      Expected a dependency on $CORE_URL with tag = \"vX.Y.Z\"." >&2
  exit 2
fi
IFS="$(printf '\t')" read -r lineno tag <<<"$pin"

if [[ "$tag" == "v${oldest}" ]]; then
  echo "FAIL: forests already pins the oldest locked neat-core ($tag), yet" \
    "$LOCKFILE carries more than one:" >&2
  printf '%s\n' "$versions" | sed 's/^/        /' >&2
  echo "      neat-ai-rebase is ahead of forests here, so this is a deliberate" >&2
  echo "      upgrade: move the forests pin forward and bump" >&2
  echo "      neat-core.expected-version in one reviewed change." >&2
  exit 1
fi

staged="${MANIFEST}.hold-core-pin.$$"
awk -v n="$lineno" -v old="$tag" -v new="v${oldest}" '
  NR == n {
    needle = "\"" old "\""
    p = index($0, needle)
    if (p == 0) exit 3
    $0 = substr($0, 1, p - 1) "\"" new "\"" substr($0, p + length(needle))
  }
  { print }
' "$MANIFEST" >"$staged" || { rm -f "$staged"; die "$MANIFEST:$lineno no longer carries the pin \"$tag\""; }
mv -f "$staged" "$MANIFEST" || { rm -f "$staged"; die "could not rewrite $MANIFEST"; }

printf '[hold-core-pin] neat-core %s → v%s (%s) to match neat-ai-rebase\n' \
  "$tag" "$oldest" "${MANIFEST#"$REPO_ROOT"/}" >&2
relock ${superseded[@]+"${superseded[@]}"}

# The hold is only done when the graph really carries one neat-core: a re-lock
# that left the divergence in place must fail here, not pass as a success.
versions="$(core_versions "$LOCKFILE")" || die "could not re-read $LOCKFILE"
count="$(printf '%s' "$versions" | grep -c . || true)"
if [[ "$count" -ne 1 || "$versions" != "$oldest" ]]; then
  echo "FAIL: holding neat-core at v${oldest} left $LOCKFILE carrying:" >&2
  printf '%s\n' "$versions" | sed 's/^/        /' >&2
  exit 1
fi
echo "OK   neat-core held at v${oldest} — one neat-core in ${LOCKFILE#"$REPO_ROOT"/}"

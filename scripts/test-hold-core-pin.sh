#!/usr/bin/env bash
# Hermetic tests for scripts/hold-core-pin.sh (PR #111).
#
# Every case drives the real script against a purpose-built manifest and
# lockfile and asserts on its exit code, the rewritten pin and its message. The
# re-lock is supplied through the script's `HOLD_CORE_PIN_RELOCK_CMD` seam, so
# no case needs cargo, a network or a real workspace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=test-lib.sh
. "${SCRIPT_DIR}/test-lib.sh"

HOLD="${SCRIPT_DIR}/hold-core-pin.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

if [[ ! -x "${HOLD}" ]]; then
  echo "FAIL: script not found or not executable: ${HOLD}" >&2
  exit 2
fi

# Git source prefixes, interpolated rather than written literally so the
# repository-wide `source <path>` scan does not mistake a Cargo
# `source = "git+..."` line for a shell `source` of a file.
CORE_SRC="git+https://github.com/stSoftwareAU/NEAT-AI-core"
REBASE_SRC="git+https://github.com/stSoftwareAU/NEAT-AI-Rebase"

# A lockfile carrying one `neat-core` block per version passed.
make_lockfile() {
  local file version
  file="$(mktemp "${WORK_DIR}/Cargo.lock.XXXXXX")"
  cat >"${file}" <<EOF
version = 4

[[package]]
name = "neat-ai-rebase"
version = "0.1.2"
source = "${REBASE_SRC}?tag=v0.1.2#bb67007a"
EOF
  for version in "$@"; do
    cat >>"${file}" <<EOF

[[package]]
name = "neat-core"
version = "${version}"
source = "${CORE_SRC}?tag=v${version}#771ad136"
EOF
  done
  cat >>"${file}" <<'EOF'

[[package]]
name = "serde"
version = "1.0.229"
EOF
  printf '%s' "${file}"
}

# A manifest pinning neat-core at tag v$1, with a decoy comment naming the same
# repository so the reader has to skip commented lines.
make_manifest() {
  local file
  file="$(mktemp "${WORK_DIR}/Cargo.toml.XXXXXX")"
  cat >"${file}" <<EOF
[package]
name = "neat_ai_forests"

[dependencies]
# A commented-out pin on https://github.com/stSoftwareAU/NEAT-AI-core, tag = "v9.9.9"
neat-core = { git = "https://github.com/stSoftwareAU/NEAT-AI-core", tag = "v$1" }
neat-ai-rebase = { git = "https://github.com/stSoftwareAU/NEAT-AI-Rebase", tag = "v0.1.2" }
EOF
  printf '%s' "${file}"
}

# A re-lock stand-in that rewrites the lockfile to carry the versions named in
# RELOCK_VERSIONS, so a converging and a non-converging re-lock are both
# testable. RELOCK_FAIL makes it fail instead.
make_relock() {
  local file
  file="$(mktemp "${WORK_DIR}/relock.XXXXXX")"
  cat >"${file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [ -n "\${RELOCK_FAIL:-}" ]; then exit 1; fi
lock="\$2"
{
  echo 'version = 4'
  # Unquoted on purpose: RELOCK_VERSIONS is a space-separated list.
  for v in \${RELOCK_VERSIONS:-}; do
    printf '\n[[package]]\nname = "neat-core"\nversion = "%s"\nsource = "%s?tag=v%s#771ad136"\n' \\
      "\$v" '${CORE_SRC}' "\$v"
  done
} >"\$lock"
EOF
  chmod +x "${file}"
  printf '%s' "${file}"
}

pin_of() {
  sed -n 's/.*NEAT-AI-core", tag = "\(v[0-9.]*\)".*/\1/p' "$1"
}

RELOCK="$(make_relock)"
export HOLD_CORE_PIN_RELOCK_CMD="${RELOCK}"

echo "=== one neat-core: nothing to hold ==="
manifest="$(make_manifest 0.22.5)"
lock="$(make_lockfile 0.22.5)"
status=0
out="$("${HOLD}" --manifest "${manifest}" --lockfile "${lock}" 2>&1)" || status=$?
assert_eq "exits 0" "0" "${status}"
assert_eq "says nothing to hold" "0" "$(grep -qF 'nothing to hold' <<<"${out}"; echo $?)"
assert_eq "pin untouched" "v0.22.5" "$(pin_of "${manifest}")"

echo "=== two neat-cores: the pin is held at the older one ==="
manifest="$(make_manifest 0.22.7)"
lock="$(make_lockfile 0.22.5 0.22.7)"
status=0
out="$(RELOCK_VERSIONS="0.22.5" "${HOLD}" --manifest "${manifest}" --lockfile "${lock}" 2>&1)" || status=$?
assert_eq "exits 0" "0" "${status}"
assert_eq "pin held at v0.22.5" "v0.22.5" "$(pin_of "${manifest}")"
assert_eq "reports the hold" "0" "$(grep -qF 'neat-core v0.22.7 → v0.22.5' <<<"${out}"; echo $?)"

echo "=== the oldest is chosen, not merely the last read ==="
manifest="$(make_manifest 0.23.1)"
lock="$(make_lockfile 0.23.1 0.22.9 0.22.10)"
status=0
out="$(RELOCK_VERSIONS="0.22.9" "${HOLD}" --manifest "${manifest}" --lockfile "${lock}" 2>&1)" || status=$?
assert_eq "exits 0" "0" "${status}"
assert_eq "pin held at v0.22.9" "v0.22.9" "$(pin_of "${manifest}")"

echo "=== rebase ahead of forests: refused, not silently re-pinned ==="
manifest="$(make_manifest 0.22.5)"
lock="$(make_lockfile 0.22.5 0.22.7)"
status=0
out="$(RELOCK_VERSIONS="0.22.5" "${HOLD}" --manifest "${manifest}" --lockfile "${lock}" 2>&1)" || status=$?
assert_eq "exits 1" "1" "${status}"
assert_eq "pin untouched" "v0.22.5" "$(pin_of "${manifest}")"
assert_eq "names the deliberate upgrade" "0" \
  "$(grep -qF 'deliberate' <<<"${out}"; echo $?)"

echo "=== a re-lock that does not converge fails loud ==="
manifest="$(make_manifest 0.22.7)"
lock="$(make_lockfile 0.22.5 0.22.7)"
status=0
out="$(RELOCK_VERSIONS="0.22.5 0.22.7" "${HOLD}" --manifest "${manifest}" --lockfile "${lock}" 2>&1)" || status=$?
assert_eq "exits 1" "1" "${status}"
assert_eq "names the surviving versions" "0" \
  "$(grep -qF '0.22.7' <<<"${out}"; echo $?)"

echo "=== a failing re-lock fails loud ==="
manifest="$(make_manifest 0.22.7)"
lock="$(make_lockfile 0.22.5 0.22.7)"
status=0
out="$(RELOCK_FAIL=1 RELOCK_VERSIONS="0.22.5" "${HOLD}" --manifest "${manifest}" --lockfile "${lock}" 2>&1)" || status=$?
assert_eq "exits 1" "1" "${status}"
assert_eq "names the re-lock" "0" "$(grep -qF 're-lock command' <<<"${out}"; echo $?)"

echo "=== no neat-core in the lockfile is a usage error ==="
manifest="$(make_manifest 0.22.5)"
lock="$(make_lockfile)"
status=0
out="$("${HOLD}" --manifest "${manifest}" --lockfile "${lock}" 2>&1)" || status=$?
assert_eq "exits 2" "2" "${status}"
assert_eq "names the missing package" "0" \
  "$(grep -qF 'no neat-core package' <<<"${out}"; echo $?)"

echo "=== a manifest with no release-tag pin is a usage error ==="
manifest="$(mktemp "${WORK_DIR}/Cargo.toml.XXXXXX")"
cat >"${manifest}" <<'EOF'
[package]
name = "neat_ai_forests"

[dependencies]
neat-core = { path = "../../NEAT-AI-core" }
EOF
lock="$(make_lockfile 0.22.5 0.22.7)"
status=0
out="$("${HOLD}" --manifest "${manifest}" --lockfile "${lock}" 2>&1)" || status=$?
assert_eq "exits 2" "2" "${status}"
assert_eq "names the missing pin" "0" \
  "$(grep -qF 'no neat-core release-tag pin' <<<"${out}"; echo $?)"

echo "=== a missing file is a usage error ==="
status=0
"${HOLD}" --manifest "${WORK_DIR}/nope.toml" --lockfile "${WORK_DIR}/nope.lock" >/dev/null 2>&1 || status=$?
assert_eq "exits 2" "2" "${status}"

status=0
"${HOLD}" --unknown >/dev/null 2>&1 || status=$?
assert_eq "an unknown argument exits 2" "2" "${status}"

report_summary

#!/usr/bin/env bash
# Hermetic tests for scripts/check-neat-core-version.sh (Issue #105).
#
# The gate now reads the pinned neat-core release out of `Cargo.lock` rather
# than a sibling checkout, so every case below drives the real script with a
# purpose-built lockfile and baseline and asserts on its exit code and message.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=test-lib.sh
. "${SCRIPT_DIR}/test-lib.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GATE="${SCRIPT_DIR}/check-neat-core-version.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

if [[ ! -x "${GATE}" ]]; then
  echo "FAIL: gate script not found or not executable: ${GATE}" >&2
  exit 2
fi

# A lockfile carrying a `neat-core` package at version $1 pinned to tag $2, and
# a second package so the reader has to find the right block.
make_lockfile() {
  local version="$1" tag="$2" file
  file="$(mktemp "${WORK_DIR}/Cargo.lock.XXXXXX")"
  cat >"${file}" <<EOF
version = 4

[[package]]
name = "neat-ai-rebase"
version = "0.1.2"
source = "git+https://github.com/stSoftwareAU/NEAT-AI-Rebase?tag=v0.1.2#bb67007a"

[[package]]
name = "neat-core"
version = "${version}"
source = "git+https://github.com/stSoftwareAU/NEAT-AI-core?tag=${tag}#771ad136"

[[package]]
name = "serde"
version = "1.0.229"
EOF
  printf '%s' "${file}"
}

make_baseline() {
  local version="$1" file
  file="$(mktemp "${WORK_DIR}/baseline.XXXXXX")"
  cat >"${file}" <<EOF
# Last-handled neat-core version.
#
# 0.1.0 -> ${version}: a fixture, not the real history.
${version}
EOF
  printf '%s' "${file}"
}

# Run the gate against baseline $1 and lockfile $2; stdout+stderr land in
# ${WORK_DIR}/out and the exit code is echoed.
run_gate() {
  local rc=0
  "${GATE}" --baseline "$1" --lockfile "$2" >"${WORK_DIR}/out" 2>&1 || rc=$?
  printf '%s' "${rc}"
}

BASELINE="$(make_baseline 0.20.0)"

echo "=== the pinned release matching the baseline passes ==="
assert_eq "exact match exits 0" "0" \
  "$(run_gate "${BASELINE}" "$(make_lockfile 0.20.0 v0.20.0)")"
assert_eq "exact match names the pinned version" "0" \
  "$(grep -q '0.20.0' "${WORK_DIR}/out"; echo $?)"

echo ""
echo "=== patch drift above the baseline passes ==="
assert_eq "patch drift exits 0" "0" \
  "$(run_gate "${BASELINE}" "$(make_lockfile 0.20.7 v0.20.7)")"

echo ""
echo "=== a pin behind the baseline passes ==="
assert_eq "behind the baseline exits 0" "0" \
  "$(run_gate "${BASELINE}" "$(make_lockfile 0.19.0 v0.19.0)")"

echo ""
echo "=== a pre-1.0 minor bump above the baseline fails loud ==="
assert_eq "unhandled minor bump exits 1" "1" \
  "$(run_gate "${BASELINE}" "$(make_lockfile 0.21.0 v0.21.0)")"
assert_eq "unhandled minor bump says it is breaking" "0" \
  "$(grep -q 'breaking neat-core bump' "${WORK_DIR}/out"; echo $?)"
assert_eq "unhandled minor bump names the remedy" "0" \
  "$(grep -q 'neat-core.expected-version' "${WORK_DIR}/out"; echo $?)"

echo ""
echo "=== a major bump above the baseline fails loud ==="
assert_eq "unhandled major bump exits 1" "1" \
  "$(run_gate "${BASELINE}" "$(make_lockfile 1.0.0 v1.0.0)")"

echo ""
echo "=== a missing lockfile is a usage error, never a pass ==="
assert_eq "missing lockfile exits 2" "2" \
  "$(run_gate "${BASELINE}" "${WORK_DIR}/does-not-exist.lock")"

echo ""
echo "=== a lockfile with no neat-core package fails loud ==="
NO_CORE="$(mktemp "${WORK_DIR}/Cargo.lock.XXXXXX")"
cat >"${NO_CORE}" <<'EOF'
version = 4

[[package]]
name = "serde"
version = "1.0.229"
EOF
assert_eq "absent neat-core exits 2" "2" "$(run_gate "${BASELINE}" "${NO_CORE}")"
assert_eq "absent neat-core says so" "0" \
  "$(grep -q 'no neat-core package' "${WORK_DIR}/out"; echo $?)"

echo ""
echo "=== a neat-core that is no longer a release pin fails loud ==="
UNPINNED="$(mktemp "${WORK_DIR}/Cargo.lock.XXXXXX")"
cat >"${UNPINNED}" <<'EOF'
version = 4

[[package]]
name = "neat-core"
version = "0.20.0"
EOF
assert_eq "an unpinned neat-core exits 2" "2" "$(run_gate "${BASELINE}" "${UNPINNED}")"
assert_eq "an unpinned neat-core names the missing tag" "0" \
  "$(grep -q 'release tag' "${WORK_DIR}/out"; echo $?)"

BRANCH_PIN="$(mktemp "${WORK_DIR}/Cargo.lock.XXXXXX")"
cat >"${BRANCH_PIN}" <<'EOF'
version = 4

[[package]]
name = "neat-core"
version = "0.20.0"
source = "git+https://github.com/stSoftwareAU/NEAT-AI-core?branch=Develop#771ad136"
EOF
assert_eq "a branch pin exits 2" "2" "$(run_gate "${BASELINE}" "${BRANCH_PIN}")"

echo ""
echo "=== two neat-core versions in one lockfile fail loud ==="
# The divergence the release pins exist to prevent: Forests' own pin and the
# one neat-ai-rebase carries resolving to different releases. Cargo locks both
# happily — the build only dies later, in rustc — so this gate is what catches
# it while the message still names the two versions.
DIVERGENT="$(mktemp "${WORK_DIR}/Cargo.lock.XXXXXX")"
cat >"${DIVERGENT}" <<'EOF'
version = 4

[[package]]
name = "neat-core"
version = "0.20.0"
source = "git+https://github.com/stSoftwareAU/NEAT-AI-core?tag=v0.20.0#aaaaaaaa"

[[package]]
name = "neat-core"
version = "0.20.1"
source = "git+https://github.com/stSoftwareAU/NEAT-AI-core?tag=v0.20.1#bbbbbbbb"
EOF
assert_eq "a divergent lockfile exits 1" "1" "$(run_gate "${BASELINE}" "${DIVERGENT}")"
assert_eq "a divergent lockfile names both versions" "0" \
  "$(grep -q '0.20.0' "${WORK_DIR}/out" && grep -q '0.20.1' "${WORK_DIR}/out"; echo $?)"
assert_eq "a divergent lockfile says what is wrong" "0" \
  "$(grep -q 'more than one neat-core' "${WORK_DIR}/out"; echo $?)"

echo ""
echo "=== a malformed pinned version is a parse error, never a pass ==="
assert_eq "malformed version exits 2" "2" \
  "$(run_gate "${BASELINE}" "$(make_lockfile 0.20 v0.20)")"

echo ""
echo "=== an empty baseline is a usage error ==="
EMPTY_BASELINE="$(mktemp "${WORK_DIR}/baseline.XXXXXX")"
printf '# only a comment\n' >"${EMPTY_BASELINE}"
assert_eq "empty baseline exits 2" "2" \
  "$(run_gate "${EMPTY_BASELINE}" "$(make_lockfile 0.20.0 v0.20.0)")"

echo ""
echo "=== the committed lockfile clears the committed baseline ==="
assert_eq "the repository's own state passes the gate" "0" \
  "$(run_gate "${REPO_ROOT}/neat-core.expected-version" "${REPO_ROOT}/Cargo.lock")"

report_summary

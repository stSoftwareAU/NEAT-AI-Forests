#!/usr/bin/env bash
# Hermetic tests for scripts/sync-core-helpers.sh (Issues #104, #105).
#
# The real script runs against a throwaway copy of the repository layout with a
# `gh` shim ahead of the real one on PATH, so the production read — the `gh api`
# argv the script builds — is what the tests drive. No network call is made.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=test-lib.sh
. "${SCRIPT_DIR}/test-lib.sh"

SYNC="${SCRIPT_DIR}/sync-core-helpers.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
REAL_PATH="${PATH}"

if [[ ! -x "${SYNC}" ]]; then
  echo "FAIL: sync script not found or not executable: ${SYNC}" >&2
  exit 2
fi

# The copied helpers, and the only endpoints the script is allowed to read.
HELPERS="runlib.sh family-pins.sh"
endpoint_for() {
  printf 'repos/stSoftwareAU/NEAT-AI-core/contents/scripts/%s?ref=Develop' "$1"
}

# One canonical body per helper, so a shim that answered with the wrong file
# would be caught by the byte-for-byte comparison below.
for helper in ${HELPERS}; do
  cat >"${WORK_DIR}/canonical-${helper}" <<EOF
#!/usr/bin/env bash
# canonical ${helper}, owned by NEAT-AI-core
canonical_${helper%.sh}() {
  echo "canonical ${helper}"
}
canonical_${helper%.sh}
EOF
done

STALE_BODY='#!/usr/bin/env bash
# stale downstream edit
echo stale'

# A fresh sandbox repo root holding scripts/sync-core-helpers.sh and a local
# copy of every helper. $1 names the helpers to seed stale; any helper not
# named is seeded with its canonical body. Echoes the sandbox path.
make_sandbox() {
  local stale_list="$1" sandbox helper
  sandbox="$(mktemp -d "${WORK_DIR}/repo.XXXXXX")"
  mkdir -p "${sandbox}/scripts"
  cp "${SYNC}" "${sandbox}/scripts/sync-core-helpers.sh"
  for helper in ${HELPERS}; do
    case " ${stale_list} " in
      *" ${helper} "*) printf '%s\n' "${STALE_BODY}" >"${sandbox}/scripts/${helper}" ;;
      *) cp "${WORK_DIR}/canonical-${helper}" "${sandbox}/scripts/${helper}" ;;
    esac
    chmod +x "${sandbox}/scripts/${helper}"
  done
  printf '%s' "${sandbox}"
}

# A `gh` shim that asserts the argv the script builds, then answers each helper
# from `${WORK_DIR}/answer-<shim>-<helper>`: a file whose first line is `FAIL`
# makes the shim exit non-zero, and anything else is written out verbatim.
make_gh_shim() {
  local name="$1" dir="${WORK_DIR}/shim-${1}" helper
  mkdir -p "${dir}"
  {
    cat <<'SHIM_HEAD'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "api" ]]; then
  echo "UNEXPECTED gh: $*" >&2
  exit 90
fi
case "$*" in
  *"application/vnd.github.raw"*) : ;;
  *) echo "gh called without the raw media type: $*" >&2; exit 91 ;;
esac
case "${2:-}" in
SHIM_HEAD
    for helper in ${HELPERS}; do
      printf '  %q) answer=%q ;;\n' "$(endpoint_for "${helper}")" \
        "${WORK_DIR}/answer-${name}-${helper}"
    done
    cat <<'SHIM_TAIL'
  *) echo "UNEXPECTED endpoint: ${2:-}" >&2; exit 92 ;;
esac
if [[ "$(head -n 1 "${answer}")" == "FAIL" ]]; then
  echo "gh: HTTP 404" >&2
  exit 1
fi
cat "${answer}"
SHIM_TAIL
  } >"${dir}/gh"
  chmod +x "${dir}/gh"
  printf '%s' "${dir}"
}

# Seed shim $1's answer for helper $2 from file $3.
set_answer() {
  cp "$3" "${WORK_DIR}/answer-${1}-${2}"
}

# Seed shim $1's answer for helper $2 with the literal body $3.
set_answer_body() {
  printf '%s' "$3" >"${WORK_DIR}/answer-${1}-${2}"
}

# A shim that answers every helper with its canonical body.
make_ok_shim() {
  local name="$1" dir helper
  dir="$(make_gh_shim "${name}")"
  for helper in ${HELPERS}; do
    set_answer "${name}" "${helper}" "${WORK_DIR}/canonical-${helper}"
  done
  printf '%s' "${dir}"
}

SHIM_OK="$(make_ok_shim ok)"

echo "=== both stale copies are refreshed byte-for-byte ==="
SANDBOX="$(make_sandbox "runlib.sh family-pins.sh")"
OUT="$(PATH="${SHIM_OK}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-core-helpers.sh")" &&
  RC=0 || RC=$?
assert_eq "refresh exits 0" "0" "${RC}"
for helper in ${HELPERS}; do
  assert_eq "refresh says it rewrote scripts/${helper}" "0" \
    "$(printf '%s' "${OUT}" | grep -q "scripts/${helper} refreshed from"; echo $?)"
  assert_eq "scripts/${helper} is byte-identical to the canonical file" "0" \
    "$(cmp -s "${WORK_DIR}/canonical-${helper}" "${SANDBOX}/scripts/${helper}"; echo $?)"
  assert_eq "refreshed scripts/${helper} stays executable" "yes" \
    "$(test -x "${SANDBOX}/scripts/${helper}" && echo yes || echo no)"
done

echo ""
echo "=== a stale family-pins.sh alone is refreshed, runlib.sh left alone ==="
SANDBOX="$(make_sandbox "family-pins.sh")"
OUT="$(PATH="${SHIM_OK}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-core-helpers.sh")" &&
  RC=0 || RC=$?
assert_eq "partial refresh exits 0" "0" "${RC}"
assert_eq "family-pins.sh is refreshed" "0" \
  "$(cmp -s "${WORK_DIR}/canonical-family-pins.sh" "${SANDBOX}/scripts/family-pins.sh"; echo $?)"
assert_eq "runlib.sh reports the match" "0" \
  "$(printf '%s' "${OUT}" | grep -q 'scripts/runlib.sh already matches'; echo $?)"

echo ""
echo "=== matching copies are left alone ==="
SANDBOX="$(make_sandbox "")"
BEFORE="$(cat "${SANDBOX}/scripts/family-pins.sh")"
OUT="$(PATH="${SHIM_OK}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-core-helpers.sh")" &&
  RC=0 || RC=$?
assert_eq "no-op exits 0" "0" "${RC}"
for helper in ${HELPERS}; do
  assert_eq "no-op reports the match for ${helper}" "0" \
    "$(printf '%s' "${OUT}" | grep -q "scripts/${helper} already matches"; echo $?)"
done
assert_eq "no-op leaves the file unchanged" "${BEFORE}" \
  "$(cat "${SANDBOX}/scripts/family-pins.sh")"

# Each bad answer is served for family-pins.sh only: a fault in the *second*
# helper must fail the run just as loudly as one in the first, which is exactly
# what a loop that stopped checking after the first file would miss.
echo ""
echo "=== a failed read fails the step and keeps the local copy ==="
SHIM_FAIL="$(make_ok_shim fail)"
set_answer_body fail family-pins.sh 'FAIL'
SANDBOX="$(make_sandbox "family-pins.sh")"
PATH="${SHIM_FAIL}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-core-helpers.sh" \
  >/dev/null 2>"${WORK_DIR}/fetch.err" && RC=0 || RC=$?
assert_eq "failed read exits non-zero" "1" "${RC}"
assert_eq "failed read names the source" "0" \
  "$(grep -q 'cannot read scripts/family-pins.sh' "${WORK_DIR}/fetch.err"; echo $?)"
assert_eq "failed read does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/family-pins.sh")"

echo ""
echo "=== an empty answer is a failure, not an empty helper ==="
SHIM_EMPTY="$(make_ok_shim empty)"
set_answer_body empty family-pins.sh ''
SANDBOX="$(make_sandbox "family-pins.sh")"
PATH="${SHIM_EMPTY}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-core-helpers.sh" \
  >/dev/null 2>"${WORK_DIR}/empty.err" && RC=0 || RC=$?
assert_eq "empty answer exits non-zero" "1" "${RC}"
assert_eq "empty answer says so" "0" \
  "$(grep -q 'came back empty' "${WORK_DIR}/empty.err"; echo $?)"
assert_eq "empty answer does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/family-pins.sh")"

echo ""
echo "=== a non-script answer is a failure ==="
SHIM_JSON="$(make_ok_shim json)"
set_answer_body json family-pins.sh '{"message":"Not Found"}'
SANDBOX="$(make_sandbox "family-pins.sh")"
PATH="${SHIM_JSON}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-core-helpers.sh" \
  >/dev/null 2>"${WORK_DIR}/json.err" && RC=0 || RC=$?
assert_eq "non-script answer exits non-zero" "1" "${RC}"
assert_eq "non-script answer says so" "0" \
  "$(grep -q 'is not a bash script' "${WORK_DIR}/json.err"; echo $?)"
assert_eq "non-script answer does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/family-pins.sh")"

echo ""
echo "=== an upstream shebang change is still a bash script ==="
ALT_SHEBANG="${WORK_DIR}/alt-shebang-family-pins.sh"
{
  echo '#!/bin/bash'
  tail -n +2 "${WORK_DIR}/canonical-family-pins.sh"
} >"${ALT_SHEBANG}"
SHIM_ALT="$(make_ok_shim alt)"
set_answer alt family-pins.sh "${ALT_SHEBANG}"
SANDBOX="$(make_sandbox "family-pins.sh")"
PATH="${SHIM_ALT}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-core-helpers.sh" \
  >/dev/null 2>"${WORK_DIR}/alt.err" && RC=0 || RC=$?
assert_eq "an alternative bash shebang is accepted" "0" "${RC}"
assert_eq "the alternative-shebang copy lands byte-for-byte" "0" \
  "$(cmp -s "${ALT_SHEBANG}" "${SANDBOX}/scripts/family-pins.sh"; echo $?)"

echo ""
echo "=== a truncated answer keeps its shebang and is still refused ==="
TRUNCATED="${WORK_DIR}/truncated-family-pins.sh"
head -n 4 "${WORK_DIR}/canonical-family-pins.sh" >"${TRUNCATED}"
SHIM_TRUNCATED="$(make_ok_shim truncated)"
set_answer truncated family-pins.sh "${TRUNCATED}"
SANDBOX="$(make_sandbox "family-pins.sh")"
PATH="${SHIM_TRUNCATED}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-core-helpers.sh" \
  >/dev/null 2>"${WORK_DIR}/truncated.err" && RC=0 || RC=$?
assert_eq "truncated answer exits non-zero" "1" "${RC}"
assert_eq "truncated answer says it does not parse" "0" \
  "$(grep -q 'does not parse' "${WORK_DIR}/truncated.err"; echo $?)"
assert_eq "truncated answer does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/family-pins.sh")"

echo ""
echo "=== a host without gh fails loud rather than silently skipping ==="
# A curated PATH rather than a system one: `gh` sits in /usr/bin on a GitHub
# runner and in /usr/local/bin elsewhere, so "a host without gh" has to be
# built out of exactly the tools the script needs, not guessed at.
NO_GH_PATH="${WORK_DIR}/no-gh-bin"
mkdir -p "${NO_GH_PATH}"
for tool in bash mktemp head cat cmp chmod rm dirname; do
  ln -sf "$(command -v "${tool}")" "${NO_GH_PATH}/${tool}"
done
assert_eq "the curated PATH really has no gh" "1" \
  "$(PATH="${NO_GH_PATH}" command -v gh >/dev/null 2>&1; echo $?)"
SANDBOX="$(make_sandbox "runlib.sh family-pins.sh")"
PATH="${NO_GH_PATH}" bash "${SANDBOX}/scripts/sync-core-helpers.sh" \
  >/dev/null 2>"${WORK_DIR}/nogh.err" && RC=0 || RC=$?
assert_eq "a missing gh exits non-zero" "1" "${RC}"
assert_eq "a missing gh names gh" "0" \
  "$(grep -q 'gh is not available' "${WORK_DIR}/nogh.err"; echo $?)"
assert_eq "a missing gh does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/runlib.sh")"

report_summary

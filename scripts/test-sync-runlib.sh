#!/usr/bin/env bash
# Hermetic tests for scripts/sync-runlib.sh (Issue #104).
#
# The real script runs against a throwaway copy of the repository layout with a
# `gh` shim ahead of the real one on PATH, so the production read — the `gh api`
# argv the script builds — is what the tests drive. No network call is made.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=test-lib.sh
. "${SCRIPT_DIR}/test-lib.sh"

SYNC="${SCRIPT_DIR}/sync-runlib.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
REAL_PATH="${PATH}"

if [[ ! -x "${SYNC}" ]]; then
  echo "FAIL: sync script not found or not executable: ${SYNC}" >&2
  exit 2
fi

# The one endpoint the script is allowed to read.
EXPECTED_ENDPOINT="repos/stSoftwareAU/NEAT-AI-core/contents/scripts/runlib.sh?ref=Develop"

CANONICAL="${WORK_DIR}/canonical-runlib.sh"
cat >"${CANONICAL}" <<'EOF'
#!/usr/bin/env bash
# canonical copy, owned by NEAT-AI-core
canonical_install() {
  echo canonical
}
canonical_install
EOF

STALE_BODY='#!/usr/bin/env bash
# stale downstream edit
echo stale'

# A fresh sandbox repo root holding scripts/sync-runlib.sh and a local
# scripts/runlib.sh with the given body. Echoes the sandbox path.
make_sandbox() {
  local body="$1" sandbox
  sandbox="$(mktemp -d "${WORK_DIR}/repo.XXXXXX")"
  mkdir -p "${sandbox}/scripts"
  cp "${SYNC}" "${sandbox}/scripts/sync-runlib.sh"
  printf '%s\n' "${body}" >"${sandbox}/scripts/runlib.sh"
  chmod +x "${sandbox}/scripts/runlib.sh"
  printf '%s' "${sandbox}"
}

# A `gh` shim that asserts the argv the script builds, then runs $2 as its body.
# $2 sees the shim's own arguments and writes the answer on stdout.
make_gh_shim() {
  local body="$2" dir="${WORK_DIR}/shim-${1}"
  mkdir -p "${dir}"
  cat >"${dir}/gh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" != "api" || "\${2:-}" != "${EXPECTED_ENDPOINT}" ]]; then
  echo "UNEXPECTED gh: \$*" >&2
  exit 90
fi
case "\$*" in
  *"application/vnd.github.raw"*) : ;;
  *)
    echo "gh called without the raw media type: \$*" >&2
    exit 91
    ;;
esac
${body}
EOF
  chmod +x "${dir}/gh"
  printf '%s' "${dir}"
}

SHIM_OK="$(make_gh_shim ok "cat \"${CANONICAL}\"")"
SHIM_FAIL="$(make_gh_shim fail 'echo "gh: HTTP 404" >&2; exit 1')"
SHIM_EMPTY="$(make_gh_shim empty 'exit 0')"
SHIM_JSON="$(make_gh_shim json "printf '%s\\n' '{\"message\":\"Not Found\"}'")"
SHIM_TRUNCATED="$(make_gh_shim truncated "head -n 4 \"${CANONICAL}\"")"

echo "=== a stale copy is refreshed byte-for-byte ==="
SANDBOX="$(make_sandbox "${STALE_BODY}")"
OUT="$(PATH="${SHIM_OK}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-runlib.sh")" &&
  RC=0 || RC=$?
assert_eq "refresh exits 0" "0" "${RC}"
assert_eq "refresh says what it did" "0" \
  "$(printf '%s' "${OUT}" | grep -q 'refreshed from'; echo $?)"
assert_eq "local copy is byte-identical to the canonical file" "0" \
  "$(cmp -s "${CANONICAL}" "${SANDBOX}/scripts/runlib.sh"; echo $?)"
assert_eq "refreshed copy stays executable" "yes" \
  "$(test -x "${SANDBOX}/scripts/runlib.sh" && echo yes || echo no)"

echo ""
echo "=== a matching copy is left alone ==="
SANDBOX="$(make_sandbox "unused")"
cp "${CANONICAL}" "${SANDBOX}/scripts/runlib.sh"
BEFORE="$(cat "${SANDBOX}/scripts/runlib.sh")"
OUT="$(PATH="${SHIM_OK}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-runlib.sh")" &&
  RC=0 || RC=$?
assert_eq "no-op exits 0" "0" "${RC}"
assert_eq "no-op reports the match" "0" \
  "$(printf '%s' "${OUT}" | grep -q 'already matches'; echo $?)"
assert_eq "no-op leaves the file unchanged" "${BEFORE}" \
  "$(cat "${SANDBOX}/scripts/runlib.sh")"

echo ""
echo "=== a failed read fails the step and keeps the local copy ==="
SANDBOX="$(make_sandbox "${STALE_BODY}")"
PATH="${SHIM_FAIL}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-runlib.sh" \
  >/dev/null 2>"${WORK_DIR}/fetch.err" && RC=0 || RC=$?
assert_eq "failed read exits non-zero" "1" "${RC}"
assert_eq "failed read names the source" "0" \
  "$(grep -q 'cannot read scripts/runlib.sh' "${WORK_DIR}/fetch.err"; echo $?)"
assert_eq "failed read does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/runlib.sh")"

echo ""
echo "=== an empty answer is a failure, not an empty install script ==="
SANDBOX="$(make_sandbox "${STALE_BODY}")"
PATH="${SHIM_EMPTY}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-runlib.sh" \
  >/dev/null 2>"${WORK_DIR}/empty.err" && RC=0 || RC=$?
assert_eq "empty answer exits non-zero" "1" "${RC}"
assert_eq "empty answer says so" "0" \
  "$(grep -q 'came back empty' "${WORK_DIR}/empty.err"; echo $?)"
assert_eq "empty answer does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/runlib.sh")"

echo ""
echo "=== a non-script answer is a failure ==="
SANDBOX="$(make_sandbox "${STALE_BODY}")"
PATH="${SHIM_JSON}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-runlib.sh" \
  >/dev/null 2>"${WORK_DIR}/json.err" && RC=0 || RC=$?
assert_eq "non-script answer exits non-zero" "1" "${RC}"
assert_eq "non-script answer says so" "0" \
  "$(grep -q 'is not a bash script' "${WORK_DIR}/json.err"; echo $?)"
assert_eq "non-script answer does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/runlib.sh")"

echo ""
echo "=== an upstream shebang change is still a bash script ==="
ALT_SHEBANG="${WORK_DIR}/alt-shebang-runlib.sh"
{
  echo '#!/bin/bash'
  tail -n +2 "${CANONICAL}"
} >"${ALT_SHEBANG}"
SHIM_ALT="$(make_gh_shim alt "cat \"${ALT_SHEBANG}\"")"
SANDBOX="$(make_sandbox "${STALE_BODY}")"
PATH="${SHIM_ALT}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-runlib.sh" \
  >/dev/null 2>"${WORK_DIR}/alt.err" && RC=0 || RC=$?
assert_eq "an alternative bash shebang is accepted" "0" "${RC}"
assert_eq "the alternative-shebang copy lands byte-for-byte" "0" \
  "$(cmp -s "${ALT_SHEBANG}" "${SANDBOX}/scripts/runlib.sh"; echo $?)"

echo ""
echo "=== a truncated answer keeps its shebang and is still refused ==="
SANDBOX="$(make_sandbox "${STALE_BODY}")"
PATH="${SHIM_TRUNCATED}:${REAL_PATH}" bash "${SANDBOX}/scripts/sync-runlib.sh" \
  >/dev/null 2>"${WORK_DIR}/truncated.err" && RC=0 || RC=$?
assert_eq "truncated answer exits non-zero" "1" "${RC}"
assert_eq "truncated answer says it does not parse" "0" \
  "$(grep -q 'does not parse' "${WORK_DIR}/truncated.err"; echo $?)"
assert_eq "truncated answer does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/runlib.sh")"

echo ""
echo "=== a host without gh fails loud rather than silently skipping ==="
SANDBOX="$(make_sandbox "${STALE_BODY}")"
PATH="/usr/bin:/bin" bash "${SANDBOX}/scripts/sync-runlib.sh" \
  >/dev/null 2>"${WORK_DIR}/nogh.err" && RC=0 || RC=$?
assert_eq "a missing gh exits non-zero" "1" "${RC}"
assert_eq "a missing gh names gh" "0" \
  "$(grep -q 'gh is not available' "${WORK_DIR}/nogh.err"; echo $?)"
assert_eq "a missing gh does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/runlib.sh")"

report_summary

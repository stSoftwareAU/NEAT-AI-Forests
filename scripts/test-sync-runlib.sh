#!/usr/bin/env bash
# Hermetic tests for scripts/sync-runlib.sh (Issue #104).
#
# The real script runs against a throwaway copy of the repository layout, with
# the network read replaced by RUNLIB_SYNC_FETCH_HOOK. No `gh` call is made.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC="${SCRIPT_DIR}/sync-runlib.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

PASSED=0
FAILED=0

if [[ ! -x "${SYNC}" ]]; then
  echo "FAIL: sync script not found or not executable: ${SYNC}" >&2
  exit 2
fi

assert_eq() {
  local desc="$1" expected="$2" actual="$3"
  if [[ "${expected}" == "${actual}" ]]; then
    echo "  PASS: ${desc}"
    PASSED=$((PASSED + 1))
  else
    echo "  FAIL: ${desc}"
    echo "    expected: '${expected}'"
    echo "    actual:   '${actual}'"
    FAILED=$((FAILED + 1))
  fi
}

CANONICAL="${WORK_DIR}/canonical-runlib.sh"
cat >"${CANONICAL}" <<'EOF'
#!/usr/bin/env bash
# canonical copy, owned by NEAT-AI-core
echo canonical
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

# Hook printing the canonical file.
HOOK_OK="${WORK_DIR}/hook-ok.sh"
cat >"${HOOK_OK}" <<EOF
#!/usr/bin/env bash
cat "${CANONICAL}"
EOF
chmod +x "${HOOK_OK}"

# Hook failing the way an unreachable or renamed source does.
HOOK_FAIL="${WORK_DIR}/hook-fail.sh"
cat >"${HOOK_FAIL}" <<'EOF'
#!/usr/bin/env bash
echo "gh: HTTP 404" >&2
exit 1
EOF
chmod +x "${HOOK_FAIL}"

# Hook answering successfully with nothing at all.
HOOK_EMPTY="${WORK_DIR}/hook-empty.sh"
cat >"${HOOK_EMPTY}" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "${HOOK_EMPTY}"

# Hook answering with a body that is not the script.
HOOK_JSON="${WORK_DIR}/hook-json.sh"
cat >"${HOOK_JSON}" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' '{"message":"Not Found"}'
EOF
chmod +x "${HOOK_JSON}"

echo "=== a stale copy is refreshed byte-for-byte ==="
SANDBOX="$(make_sandbox "${STALE_BODY}")"
OUT="$(RUNLIB_SYNC_FETCH_HOOK="${HOOK_OK}" bash "${SANDBOX}/scripts/sync-runlib.sh")" &&
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
OUT="$(RUNLIB_SYNC_FETCH_HOOK="${HOOK_OK}" bash "${SANDBOX}/scripts/sync-runlib.sh")" &&
  RC=0 || RC=$?
assert_eq "no-op exits 0" "0" "${RC}"
assert_eq "no-op reports the match" "0" \
  "$(printf '%s' "${OUT}" | grep -q 'already matches'; echo $?)"
assert_eq "no-op leaves the file unchanged" "${BEFORE}" \
  "$(cat "${SANDBOX}/scripts/runlib.sh")"

echo ""
echo "=== a failed fetch fails the step and keeps the local copy ==="
SANDBOX="$(make_sandbox "${STALE_BODY}")"
OUT="$(RUNLIB_SYNC_FETCH_HOOK="${HOOK_FAIL}" bash "${SANDBOX}/scripts/sync-runlib.sh" \
  2>"${WORK_DIR}/fetch.err")" && RC=0 || RC=$?
assert_eq "failed fetch exits non-zero" "1" "${RC}"
assert_eq "failed fetch names the source" "0" \
  "$(grep -q 'cannot read scripts/runlib.sh' "${WORK_DIR}/fetch.err"; echo $?)"
assert_eq "failed fetch does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/runlib.sh")"

echo ""
echo "=== an empty answer is a failure, not an empty install script ==="
SANDBOX="$(make_sandbox "${STALE_BODY}")"
RUNLIB_SYNC_FETCH_HOOK="${HOOK_EMPTY}" bash "${SANDBOX}/scripts/sync-runlib.sh" \
  >/dev/null 2>"${WORK_DIR}/empty.err" && RC=0 || RC=$?
assert_eq "empty answer exits non-zero" "1" "${RC}"
assert_eq "empty answer says so" "0" \
  "$(grep -q 'came back empty' "${WORK_DIR}/empty.err"; echo $?)"
assert_eq "empty answer does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/runlib.sh")"

echo ""
echo "=== a non-script answer is a failure ==="
SANDBOX="$(make_sandbox "${STALE_BODY}")"
RUNLIB_SYNC_FETCH_HOOK="${HOOK_JSON}" bash "${SANDBOX}/scripts/sync-runlib.sh" \
  >/dev/null 2>"${WORK_DIR}/json.err" && RC=0 || RC=$?
assert_eq "non-script answer exits non-zero" "1" "${RC}"
assert_eq "non-script answer says so" "0" \
  "$(grep -q 'is not a bash script' "${WORK_DIR}/json.err"; echo $?)"
assert_eq "non-script answer does not touch the local copy" "${STALE_BODY}" \
  "$(cat "${SANDBOX}/scripts/runlib.sh")"

echo ""
echo "=== summary: ${PASSED} passed, ${FAILED} failed ==="
[[ "${FAILED}" -eq 0 ]]

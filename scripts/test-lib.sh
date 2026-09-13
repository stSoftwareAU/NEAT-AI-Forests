#!/usr/bin/env bash
# Assertion helpers shared by the scripts/test-*.sh harnesses (Issue #104).
#
# Sourced, never run: it defines the counters and `assert_eq`, and leaves the
# `set -euo pipefail` and the temporary-directory trap to the caller.

PASSED=0
FAILED=0

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

# Print the tally and return non-zero when anything failed.
report_summary() {
  echo ""
  echo "=== summary: ${PASSED} passed, ${FAILED} failed ==="
  [[ "${FAILED}" -eq 0 ]]
}

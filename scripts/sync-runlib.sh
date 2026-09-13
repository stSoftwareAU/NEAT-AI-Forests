#!/usr/bin/env bash
# Refresh `scripts/runlib.sh` from its canonical home (Issue #104).
#
# `scripts/runlib.sh` is owned by NEAT-AI-core (core#680): it lives on that
# repository's `Develop` and every Rust sibling carries a byte-identical copy.
# Behaviour changes are made there and re-copied outward, never edited here —
# a downstream edit forks the contract silently and the next refresh discards
# it.
#
# This script is the refresh. CI runs it inside the `version-increment` job, so
# a PR that carries a stale copy has it corrected in the same bump commit.
#
# Fails loud, never silently stale: a fetch that errors, answers empty, or
# answers with something that is not the script exits non-zero and leaves the
# local copy untouched.
#
# Environment (tests only; CI and humans set none of these):
#   RUNLIB_SYNC_SOURCE_REPO   owner/repo holding the canonical file
#   RUNLIB_SYNC_SOURCE_REF    ref to read it from
#   RUNLIB_SYNC_FETCH_HOOK    command printing the canonical file on stdout,
#                             argv: <repo> <ref> <path>; replaces the `gh` read
#
# Cross-platform: macOS bash 3.2, Ubuntu, AWS Linux.
set -euo pipefail

SOURCE_REPO="${RUNLIB_SYNC_SOURCE_REPO:-stSoftwareAU/NEAT-AI-core}"
SOURCE_REF="${RUNLIB_SYNC_SOURCE_REF:-Develop}"
SOURCE_PATH="scripts/runlib.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_COPY="${REPO_ROOT}/${SOURCE_PATH}"

# Print the canonical file on stdout. `gh api` with the raw media type returns
# the file itself and a non-zero status on any HTTP error.
_fetch_canonical() {
  if [[ -n "${RUNLIB_SYNC_FETCH_HOOK:-}" ]]; then
    "${RUNLIB_SYNC_FETCH_HOOK}" "${SOURCE_REPO}" "${SOURCE_REF}" "${SOURCE_PATH}"
    return
  fi
  if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: gh is not available — it is what reads ${SOURCE_REPO}" >&2
    return 1
  fi
  gh api "repos/${SOURCE_REPO}/contents/${SOURCE_PATH}?ref=${SOURCE_REF}" \
    --header 'Accept: application/vnd.github.raw'
}

main() {
  local fetched
  fetched="$(mktemp)"
  # shellcheck disable=SC2064  # expand the path now: it is what must be removed
  trap "rm -f '${fetched}'" EXIT

  if ! _fetch_canonical >"${fetched}"; then
    echo "ERROR: cannot read ${SOURCE_PATH} from ${SOURCE_REPO} ${SOURCE_REF} —" \
      "refusing to ship a possibly stale copy" >&2
    return 1
  fi
  if [[ ! -s "${fetched}" ]]; then
    echo "ERROR: ${SOURCE_REPO} ${SOURCE_REF}:${SOURCE_PATH} came back empty" >&2
    return 1
  fi
  # An error page or a JSON body is a successful HTTP read of the wrong thing.
  if [[ "$(head -n 1 "${fetched}")" != "#!/usr/bin/env bash" ]]; then
    echo "ERROR: ${SOURCE_REPO} ${SOURCE_REF}:${SOURCE_PATH} is not a bash script" >&2
    return 1
  fi

  if cmp -s "${fetched}" "${LOCAL_COPY}"; then
    echo "${SOURCE_PATH} already matches ${SOURCE_REPO} ${SOURCE_REF}"
    return 0
  fi

  # `cat >` rather than `cp`: it rewrites the tracked file in place and keeps
  # its mode, so a refresh never shows up as a permission change.
  cat "${fetched}" >"${LOCAL_COPY}"
  chmod +x "${LOCAL_COPY}"
  echo "${SOURCE_PATH} refreshed from ${SOURCE_REPO} ${SOURCE_REF}"
  return 0
}

main "$@"

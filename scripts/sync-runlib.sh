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
# Fails loud, never silently stale: a read that errors, comes back empty, or
# comes back as something that is not a valid bash script exits non-zero and
# leaves the local copy untouched. `gh` is the only reader — there is no hook
# to substitute another command, so what CI runs is what the tests drive, with
# a `gh` shim ahead of it on PATH.
#
# Cross-platform: macOS bash 3.2, Ubuntu, AWS Linux.
set -euo pipefail

SOURCE_REPO="stSoftwareAU/NEAT-AI-core"
SOURCE_REF="Develop"
SOURCE_PATH="scripts/runlib.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_COPY="${REPO_ROOT}/${SOURCE_PATH}"

# Print the canonical file on stdout. The raw media type returns the file
# itself, and `gh api` exits non-zero on any HTTP error.
_fetch_canonical() {
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
  # An error page or a JSON body is a successful read of the wrong thing. The
  # test is "a bash shebang", not one exact line: the interpreter is upstream's
  # to choose, and pinning the literal would red every Forests PR the day core
  # changed it.
  case "$(head -n 1 "${fetched}")" in
    '#!'*bash*) : ;;
    *)
      echo "ERROR: ${SOURCE_REPO} ${SOURCE_REF}:${SOURCE_PATH} is not a bash script" >&2
      return 1
      ;;
  esac
  # A truncated body keeps its shebang, so the shebang alone proves nothing:
  # parse it before it can be committed and shipped to a fleet host.
  if ! bash -n "${fetched}" 2>/dev/null; then
    echo "ERROR: ${SOURCE_REPO} ${SOURCE_REF}:${SOURCE_PATH} does not parse —" \
      "the read was truncated or corrupted" >&2
    return 1
  fi

  if cmp -s "${fetched}" "${LOCAL_COPY}"; then
    echo "${SOURCE_PATH} already matches ${SOURCE_REPO} ${SOURCE_REF}"
    return 0
  fi

  # `cat >` rather than `cp`: it rewrites the tracked file in place instead of
  # replacing it. The `chmod` then restores the one mode bit that matters —
  # every sibling runs this file directly.
  cat "${fetched}" >"${LOCAL_COPY}"
  chmod +x "${LOCAL_COPY}"
  echo "${SOURCE_PATH} refreshed from ${SOURCE_REPO} ${SOURCE_REF}"
  return 0
}

main "$@"

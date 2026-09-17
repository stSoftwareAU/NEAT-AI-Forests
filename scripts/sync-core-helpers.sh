#!/usr/bin/env bash
# Refresh the copied NEAT-AI-core helpers from their canonical home
# (Issues #104, #105).
#
# Two files are copies, not originals:
#
#   * `scripts/runlib.sh`      — core#680, the build → install → clean helper
#   * `scripts/family-pins.sh` — core#681, the family git-tag pin mover
#
# Both are owned by NEAT-AI-core: they live on that repository's `Develop` and
# every Rust sibling carries a byte-identical copy. Behaviour changes are made
# there and re-copied outward, never edited here — a downstream edit forks the
# contract silently and the next refresh discards it.
#
# This script is the refresh. CI runs it inside the `version-increment` job, so
# a PR that carries a stale copy has it corrected in the same bump commit.
#
# Fails loud, never silently stale: a read that errors, comes back empty, or
# comes back as something that is not a valid bash script exits non-zero and
# leaves every local copy untouched. `gh` is the only reader — there is no hook
# to substitute another command, so what CI runs is what the tests drive, with
# a `gh` shim ahead of it on PATH.
#
# Cross-platform: macOS bash 3.2, Ubuntu, AWS Linux.
set -euo pipefail

SOURCE_REPO="stSoftwareAU/NEAT-AI-core"
SOURCE_REF="Develop"
# The copied helpers, in the order they are refreshed. bash 3.2 has no
# associative arrays, so this is a plain list of repository-relative paths.
SOURCE_PATHS="scripts/runlib.sh scripts/family-pins.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Print the canonical copy of $1 on stdout. The raw media type returns the file
# itself, and `gh api` exits non-zero on any HTTP error.
_fetch_canonical() {
  local source_path="$1"
  if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: gh is not available — it is what reads ${SOURCE_REPO}" >&2
    return 1
  fi
  gh api "repos/${SOURCE_REPO}/contents/${source_path}?ref=${SOURCE_REF}" \
    --header 'Accept: application/vnd.github.raw'
}

# Refresh the single copy $1. Returns non-zero — leaving the local copy exactly
# as it was — on any read that cannot be proven good.
_sync_one() {
  local source_path="$1" local_copy="${REPO_ROOT}/$1" fetched
  fetched="$(mktemp)"

  if ! _fetch_canonical "${source_path}" >"${fetched}"; then
    rm -f "${fetched}"
    echo "ERROR: cannot read ${source_path} from ${SOURCE_REPO} ${SOURCE_REF} —" \
      "refusing to ship a possibly stale copy" >&2
    return 1
  fi
  if [[ ! -s "${fetched}" ]]; then
    rm -f "${fetched}"
    echo "ERROR: ${SOURCE_REPO} ${SOURCE_REF}:${source_path} came back empty" >&2
    return 1
  fi
  # An error page or a JSON body is a successful read of the wrong thing. The
  # test is "a bash shebang", not one exact line: the interpreter is upstream's
  # to choose, and pinning the literal would red every Forests PR the day core
  # changed it.
  case "$(head -n 1 "${fetched}")" in
    '#!'*bash*) : ;;
    *)
      rm -f "${fetched}"
      echo "ERROR: ${SOURCE_REPO} ${SOURCE_REF}:${source_path} is not a bash script" >&2
      return 1
      ;;
  esac
  # A truncated body keeps its shebang, so the shebang alone proves nothing:
  # parse it before it can be committed and shipped to a fleet host.
  if ! bash -n "${fetched}" 2>/dev/null; then
    rm -f "${fetched}"
    echo "ERROR: ${SOURCE_REPO} ${SOURCE_REF}:${source_path} does not parse —" \
      "the read was truncated or corrupted" >&2
    return 1
  fi

  if cmp -s "${fetched}" "${local_copy}"; then
    rm -f "${fetched}"
    echo "${source_path} already matches ${SOURCE_REPO} ${SOURCE_REF}"
    return 0
  fi

  # `cat >` rather than `cp`: it rewrites the tracked file in place instead of
  # replacing it. The `chmod` then restores the one mode bit that matters —
  # every sibling runs these files directly.
  cat "${fetched}" >"${local_copy}"
  chmod +x "${local_copy}"
  rm -f "${fetched}"
  echo "${source_path} refreshed from ${SOURCE_REPO} ${SOURCE_REF}"
  return 0
}

main() {
  local source_path
  for source_path in ${SOURCE_PATHS}; do
    _sync_one "${source_path}" || return 1
  done
  return 0
}

main "$@"

#!/usr/bin/env bash
# Build, install, and clean neat_ai_forests for the fleet worker (#106).
#
# Installs the crate CLI only:
#   ~/.cargo/bin/neat_ai_forests
# stamped with .neat_ai_forests.version beside it.
#
# stdout is the CLI binary path only. Diagnostics go to stderr.
# A second run whose stamp matches prints
#   [neat_ai_forests] already installed v<x>
# and runs no cargo command, then removes nothing (target/ is already gone).
#
# Family-sync / byte-identical copy from NEAT-AI-core waits on
# NEAT-AI-core#680 / this repo #104. This ships the install contract so the
# fleet can run Forests stages.
#
# Cross-platform: macOS bash 3.2, Ubuntu, AWS Linux.
set -euo pipefail

PKG="neat_ai_forests"

_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

_require_tools() {
  export PATH="${HOME}/.cargo/bin:${PATH}"
  if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is not available. Please install jq." >&2
    exit 1
  fi
  if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: cargo is not available. Install Rust (rustup) first." >&2
    exit 1
  fi
}

# Echo "<version>" from the workspace member. Fails loud on zero/many matches.
_crate_version() {
  local root="$1"
  local meta ver count
  meta="$(cargo metadata --no-deps --format-version 1 --manifest-path "${root}/Cargo.toml")"
  count="$(printf '%s\n' "${meta}" | jq --arg n "${PKG}" '[.packages[] | select(.name==$n)] | length')"
  if [[ "${count}" != "1" ]]; then
    echo "ERROR: expected exactly one package named ${PKG}, got ${count}" >&2
    exit 1
  fi
  ver="$(printf '%s\n' "${meta}" | jq -r --arg n "${PKG}" '.packages[] | select(.name==$n) | .version')"
  if [[ -z "${ver}" || "${ver}" == "null" ]]; then
    echo "ERROR: cannot read version for ${PKG}" >&2
    exit 1
  fi
  printf '%s\n' "${ver}"
}

_already_installed() {
  local desired="$1" bin="$2" mark="$3"
  local bv
  [[ -x "${bin}" && -f "${mark}" ]] || return 1
  bv="$(cat "${mark}" 2>/dev/null || echo "")"
  [[ "${bv}" == "${desired}" ]]
}

ensure_installed() {
  _require_tools
  local root desired bin_dir bin mark built_bin
  root="$(_repo_root)"
  desired="$(_crate_version "${root}")"

  bin_dir="${HOME}/.cargo/bin"
  bin="${bin_dir}/${PKG}"
  mark="${bin_dir}/.${PKG}.version"

  if _already_installed "${desired}" "${bin}" "${mark}"; then
    echo "[${PKG}] already installed v${desired}" >&2
    echo "${bin}"
    return 0
  fi

  echo "Building ${PKG} v${desired}" >&2
  (
    cd "${root}"
    cargo build --release -p "${PKG}" --bin "${PKG}"
  ) >&2

  built_bin="${root}/target/release/${PKG}"
  if [[ ! -x "${built_bin}" ]]; then
    echo "ERROR: build produced no executable at ${built_bin}" >&2
    exit 1
  fi

  mkdir -p "${bin_dir}" >&2
  cp "${built_bin}" "${bin}" >&2
  chmod +x "${bin}" >&2
  printf '%s\n' "${desired}" >"${mark}"

  echo "Removing ${root}/target after install" >&2
  rm -rf "${root}/target"

  echo "${bin}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  ensure_installed
fi

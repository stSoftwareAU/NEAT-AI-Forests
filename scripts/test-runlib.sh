#!/usr/bin/env bash
# Hermetic tests for scripts/runlib.sh (Issues #106, #104).
#
# scripts/runlib.sh is NEAT-AI-core's file (core#680) and is never edited here;
# these tests assert the contract *this* repository depends on holds for the
# copy it carries and for this crate's manifest shape:
#
#   * an up-to-date install runs no cargo command at all — a recording shim
#     that fails on every invocation is what makes that an assertion rather
#     than an inference;
#   * a first install writes ~/.cargo/bin/neat_ai_forests and
#     .neat_ai_forests.version, removes target/, and prints the bin path;
#   * a failed build keeps target/ and leaves the installed artefact and its
#     stamp untouched.
#
# No real cargo build ever runs: a shim answers `cargo metadata` and fabricates
# the binary a build would have produced.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNLIB="${SCRIPT_DIR}/runlib.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
REAL_PATH="${PATH}"

PASSED=0
FAILED=0

if [[ ! -x "${RUNLIB}" ]]; then
  echo "FAIL: runlib not found or not executable: ${RUNLIB}" >&2
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

CRATE="neat_ai_forests"

# This crate's declared version, read the way the skip path reads it.
crate_version() {
  awk -F'"' '/^version[[:space:]]*=/ { print $2; exit }' "${REPO_ROOT}/forests/Cargo.toml"
}

# A cargo shim that records every invocation and refuses all of them.
install_refusing_shim() {
  local bin_dir="$1" log="$2"
  mkdir -p "${bin_dir}"
  cat >"${bin_dir}/cargo" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\$*" >>"${log}"
echo "UNEXPECTED cargo: \$*" >&2
exit 99
EOF
  chmod +x "${bin_dir}/cargo"
}

# A cargo shim that answers metadata for the fixture crate and fabricates the
# binary a real build would have produced. \$3 is the build exit code.
install_building_shim() {
  local bin_dir="$1" fixture="$2" build_rc="$3"
  mkdir -p "${bin_dir}"
  cat >"${bin_dir}/cargo" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "metadata" ]]; then
  cat <<'JSON'
{"packages":[{"name":"${CRATE}","version":"1.2.3",
"manifest_path":"${fixture}/forests/Cargo.toml",
"targets":[{"name":"${CRATE}","kind":["bin"]}]}],
"target_directory":"${fixture}/target"}
JSON
  exit 0
fi
if [[ "\${1:-}" == "build" ]]; then
  if [[ "${build_rc}" -ne 0 ]]; then
    echo "error: could not compile ${CRATE}" >&2
    exit ${build_rc}
  fi
  mkdir -p "${fixture}/target/release"
  printf 'fresh build\n' >"${fixture}/target/release/${CRATE}"
  chmod +x "${fixture}/target/release/${CRATE}"
  exit 0
fi
echo "UNEXPECTED cargo: \$*" >&2
exit 99
EOF
  chmod +x "${bin_dir}/cargo"
}

# A throwaway crate with this repository's manifest shape. Echoes its path.
make_fixture() {
  local fixture
  fixture="$(mktemp -d "${WORK_DIR}/fixture.XXXXXX")"
  mkdir -p "${fixture}/forests/src" "${fixture}/target/debug"
  printf '[workspace]\nmembers = ["forests"]\nresolver = "2"\n' >"${fixture}/Cargo.toml"
  cat >"${fixture}/forests/Cargo.toml" <<EOF
[package]
name = "${CRATE}"
version = "1.2.3"
edition = "2024"

[lib]
name = "${CRATE}"
path = "src/lib.rs"
EOF
  printf 'fn main() {}\n' >"${fixture}/forests/src/main.rs"
  printf 'pub fn hi() {}\n' >"${fixture}/forests/src/lib.rs"
  printf 'stale build artefact\n' >"${fixture}/target/debug/scratch"
  printf '%s' "${fixture}"
}

echo "=== already installed: no cargo command at all, bin path on stdout ==="
VERSION="$(crate_version)"
assert_eq "the crate version is readable" "0" \
  "$([[ -n "${VERSION}" ]] && echo 0 || echo 1)"
SANDBOX="${WORK_DIR}/skip"
mkdir -p "${SANDBOX}/.cargo/bin"
printf 'fake\n' >"${SANDBOX}/.cargo/bin/${CRATE}"
chmod +x "${SANDBOX}/.cargo/bin/${CRATE}"
printf '%s\n' "${VERSION}" >"${SANDBOX}/.cargo/bin/.${CRATE}.version"
CARGO_LOG="${WORK_DIR}/cargo-calls.log"
: >"${CARGO_LOG}"
install_refusing_shim "${WORK_DIR}/refusing-shim" "${CARGO_LOG}"

OUT="$(cd "${REPO_ROOT}" && HOME="${SANDBOX}" CARGO_HOME="${SANDBOX}/.cargo" \
  PATH="${WORK_DIR}/refusing-shim:${REAL_PATH}" bash "${RUNLIB}" \
  2>"${WORK_DIR}/skip.err")" && RC=0 || RC=$?
assert_eq "already-installed exits 0" "0" "${RC}"
assert_eq "already-installed stdout is the CLI path" \
  "${SANDBOX}/.cargo/bin/${CRATE}" "${OUT}"
assert_eq "already-installed names the version on stderr" "0" \
  "$(grep -q "\[${CRATE}\] already installed v${VERSION}" "${WORK_DIR}/skip.err"; echo $?)"
assert_eq "already-installed ran no cargo command" "" "$(cat "${CARGO_LOG}")"

echo ""
echo "=== first install: bin, stamp, target/ removed, path on stdout ==="
FIXTURE="$(make_fixture)"
SANDBOX="${WORK_DIR}/install"
mkdir -p "${SANDBOX}/.cargo"
install_building_shim "${WORK_DIR}/building-shim" "${FIXTURE}" 0

OUT="$(cd "${FIXTURE}" && HOME="${SANDBOX}" CARGO_HOME="${SANDBOX}/.cargo" \
  PATH="${WORK_DIR}/building-shim:${REAL_PATH}" bash "${RUNLIB}" \
  2>"${WORK_DIR}/install.err")" && RC=0 || RC=$?
assert_eq "install exits 0" "0" "${RC}"
assert_eq "install stdout is the CLI path" "${SANDBOX}/.cargo/bin/${CRATE}" "${OUT}"
assert_eq "the binary is installed under CARGO_HOME/bin" "fresh build" \
  "$(cat "${SANDBOX}/.cargo/bin/${CRATE}" 2>/dev/null || echo MISSING)"
assert_eq "the stamp records the crate version" "1.2.3" \
  "$(cat "${SANDBOX}/.cargo/bin/.${CRATE}.version" 2>/dev/null || echo MISSING)"
assert_eq "target/ is removed after a successful install" "gone" \
  "$(test -d "${FIXTURE}/target" && echo present || echo gone)"

echo ""
echo "=== a second run over that install builds nothing ==="
: >"${CARGO_LOG}"
install_refusing_shim "${WORK_DIR}/refusing-shim" "${CARGO_LOG}"
OUT="$(cd "${FIXTURE}" && HOME="${SANDBOX}" CARGO_HOME="${SANDBOX}/.cargo" \
  PATH="${WORK_DIR}/refusing-shim:${REAL_PATH}" bash "${RUNLIB}" \
  2>"${WORK_DIR}/second.err")" && RC=0 || RC=$?
assert_eq "second run exits 0" "0" "${RC}"
assert_eq "second run prints the already-installed line" "0" \
  "$(grep -q "\[${CRATE}\] already installed v1.2.3" "${WORK_DIR}/second.err"; echo $?)"
assert_eq "second run ran no cargo command" "" "$(cat "${CARGO_LOG}")"

echo ""
echo "=== a failed build keeps target/ and the installed artefact ==="
FIXTURE="$(make_fixture)"
SANDBOX="${WORK_DIR}/failed"
mkdir -p "${SANDBOX}/.cargo/bin"
printf 'previous\n' >"${SANDBOX}/.cargo/bin/${CRATE}"
chmod +x "${SANDBOX}/.cargo/bin/${CRATE}"
printf '0.0.1\n' >"${SANDBOX}/.cargo/bin/.${CRATE}.version"
install_building_shim "${WORK_DIR}/failing-shim" "${FIXTURE}" 7

(cd "${FIXTURE}" && HOME="${SANDBOX}" CARGO_HOME="${SANDBOX}/.cargo" \
  PATH="${WORK_DIR}/failing-shim:${REAL_PATH}" bash "${RUNLIB}" \
  >/dev/null 2>"${WORK_DIR}/failed.err") && RC=0 || RC=$?
assert_eq "a failed build exits non-zero" "7" "${RC}"
assert_eq "a failed build keeps target/" "present" \
  "$(test -d "${FIXTURE}/target" && echo present || echo gone)"
assert_eq "a failed build leaves the installed binary alone" "previous" \
  "$(cat "${SANDBOX}/.cargo/bin/${CRATE}")"
assert_eq "a failed build leaves the stamp alone" "0.0.1" \
  "$(cat "${SANDBOX}/.cargo/bin/.${CRATE}.version")"
assert_eq "a failed build stages nothing under CARGO_HOME/bin" "0" \
  "$(find "${SANDBOX}/.cargo/bin" -name "*.runlib.*" | wc -l | tr -d ' ')"

echo ""
echo "=== summary: ${PASSED} passed, ${FAILED} failed ==="
[[ "${FAILED}" -eq 0 ]]

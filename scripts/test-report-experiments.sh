#!/usr/bin/env bash
# Hermetic tests for scripts/report-experiments.sh's per-arm table (Issue #147).
#
# The `report` subcommand is replaced through FORESTS_BIN by a stub that prints
# a prepared report for each journal, so no case needs cargo or a real run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=test-lib.sh
. "${SCRIPT_DIR}/test-lib.sh"

REPORT="${SCRIPT_DIR}/report-experiments.sh"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

# Stub binary: `report <journal>` prints <journal minus .jsonl>.fixture.json.
STUB="${WORK_DIR}/neat_ai_forests"
cat >"${STUB}" <<'SH'
#!/usr/bin/env bash
[[ "$1" == "report" ]] || exit 9
cat "${2%.jsonl}.fixture.json"
SH
chmod +x "${STUB}"

# One run: name, arms tag, acceptances, cumulative gain, wall ms, full calls,
# accepted strategies (space separated).
run() {
  local name="$1" arms="$2" acc="$3" gain="$4" wall="$5" calls="$6" strategies="$7"
  local accepted="" s
  for s in ${strategies}; do
    accepted+="${accepted:+,}{\"iteration\":1,\"id\":\"x\",\"strategy\":\"${s}\",\"depth\":1,\"improvement\":0.0,\"affectedFraction\":0.5,\"atWallMs\":1}"
  done
  : >"${WORK_DIR}/${name}.jsonl"
  cat >"${WORK_DIR}/${name}.fixture.json" <<JSON
{"iterations":3,"acceptances":${acc},"cumulativeImprovement":${gain},
 "improvementPerWallHour":null,"wallMs":${wall},"candidatesFullScored":${calls},
 "accepted":[${accepted}],"stopReason":"timeout","arms":"${arms}"}
JSON
}

H=3600000
run a "hidden-features=0:drawn,linear-leaf-conditions=0:set,refine-leaves=8:drawn,refit-every=0:forced" \
  2 0.004 "${H}" 10 "refine-stump histogram-stump"
run b "hidden-features=256:drawn,linear-leaf-conditions=0:set,refine-leaves=0:drawn,refit-every=0:forced" \
  1 0.001 "${H}" 10 "histogram-stump+hidden"
run c "hidden-features=256:drawn,linear-leaf-conditions=16:set,refine-leaves=8:drawn,refit-every=0:forced" \
  1 0.002 "${H}" 20 "refine-tree-depth2"
# A journal from an older build: no arms, so it is left out of the table.
run d "" 5 0.5 "${H}" 10 "histogram-stump"

out="$(FORESTS_BIN="${STUB}" "${REPORT}" "${WORK_DIR}"/{a,b,c,d}.jsonl)"
# The "on" row of an arm in the per-arm table, without its first two columns.
row() { printf '%s\n' "${out}" | awk -v a="$1" -v l="$2" '
  /^per-arm/ { t = 1; next }
  t && $1 == a && $2 == l { $1 = ""; $2 = ""; sub(/^ +/, ""); print; exit }'; }

echo "Test 1: refine-leaves on (a, c) vs off (b)"
assert_eq "refine on" "2 3 3.000e-03 2.000e-04 2/3" "$(row refine-leaves on | tr -s ' ')"
off_after_refine="$(printf '%s\n' "${out}" | awk '/^refine-leaves/ { getline; $1 = ""; sub(/^ +/, ""); print; exit }' | tr -s ' ')"
assert_eq "refine off" "1 1 1.000e-03 1.000e-04 -" "${off_after_refine}"

echo "Test 2: hidden-features on (b, c) counts +hidden wins only"
assert_eq "hidden on" "2 2 1.500e-03 1.000e-04 1/2" "$(row hidden-features on | tr -s ' ')"

echo "Test 3: set and forced arms are not compared"
assert_eq "linear (set) has no drawn runs" "0 0 n/a n/a 0/0" "$(row linear-leaf-conditions on | tr -s ' ')"
assert_eq "refit (forced) has no drawn runs" "0 0 n/a n/a 0/0" "$(row refit-every on | tr -s ' ')"

echo "Test 4: a journal without arms is still summarised, but not tabled"
assert_eq "older journal in the summary" "1" "$(printf '%s\n' "${out}" | grep -c '/d.jsonl')"

echo "Test 5: no arms at all prints no per-arm table"
out_old="$(FORESTS_BIN="${STUB}" "${REPORT}" "${WORK_DIR}/d.jsonl")"
assert_eq "no table" "0" "$(printf '%s\n' "${out_old}" | grep -c '^per-arm' || true)"

report_summary

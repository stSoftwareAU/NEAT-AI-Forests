## Summary

Recorded the scorer identity, successful correctness checks and the missing
production inputs in `docs/benchmarks.md`. Preserved the implementation from
merged PR #150 and its legacy defaults. Part of #136.

## Spec

### Intent and Rationale

- Continue from the merged implementation by verifying its real scorer contract and locating the inputs required for the remaining production experiment.
- Keep correctness evidence separate from production economics; generated fixtures cannot satisfy the production A/B criterion.

### Essential Design Decisions

- Retain legacy ranking, 64 candidates, 8 roots, scales `1.0,0.5,0.25` and 4 combinations while the production 1.5× rule is unevaluated.
- Missing measurements are unmeasured, including failures, fallback execution and baseline vetoes; they are not zero counts or a measured negative result.

### Undiscoverable Facts

- The bounded local filesystem search found no production champion or corpus. This container has the host label `GRQ-23`; no training-host filesystem was inspected.
- GitHub lists no scorer releases or tags. Validation used a local release-profile build of `Develop` commit `a164969ac4d96ff39eace7086ad22c0538d95705`, manifest version `1.1.110`; its CLI rejects `--version`.
- Independent Spec and Standards reviewers were dispatched for this repair. The Spec reviewer received only the branch diff against `origin/Develop` and the issue body; the Standards reviewer received only that diff and the requested `CODING-STANDARDS.md` location. Their verdicts are recorded separately below. The standards file is absent.

## Evidence

- Documentation-only continuation from Forests commit `5df8cb0db7411082ddaf7cd64888b20e2e1b333c`, the merge of PR #150. GitHub confirmed that merge and scorer prerequisite #683's closure and PR #684's merge.
- Scorer binary SHA-256: `8e6e46de1faf4ed46867e55e5ac2422c149932a5124d605c28a6e1099f77e26a`; built with `cargo build --locked --release --bin rust_scorer` and `rustc 1.99.1` on Linux aarch64.
- `forests/tests/real_scorer.rs::real_cohort_matches_standalone_scores_and_ordering_for_generated_grafts` passed against this binary, checking bit-identical scores and ordering for identity, MLP, IF and clamped grafts. The same suite also exercised documented directory fallback, ordinary verification, evolution and import CLI selection.
- `forests/src/report.rs::tests::exact_rank_report_keeps_ranking_verification_and_diagnostics_separate` passed, including rank comparisons, fallback and failed-verification accounting and old-journal compatibility.
- Provenance: #136 — Rank every graft candidate exactly on the full corpus with scorer delta mode, and retire the 5 % screen for them. PR #150 — the merged implementation with that title.
- Backend/CLI validation record; no visual surface or runtime behaviour changed.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **missing** — A/B on the production protocol in `docs/benchmarks.md` (same creature, seed and budget; control is today's defaults). Report Δscore per wall hour, acceptances, candidates scored exactly per iteration, and the wall split. — evidence: `docs/benchmarks.md::Production-input recheck (25 November 2026)` — reviewer: missing — reason: production inputs were not located, both 2700-second trials were not run, and all required production metrics remain unmeasured.
- **partial** — Journal fields for the new path, and a `report` row comparing exact rank with what the 5 % screen would have ranked. — evidence: `docs/benchmarks.md::Production-input recheck (25 November 2026)` — reviewer: partial — reason: this documentation-only diff reports successful reporting tests and preserves the merged implementation, but contains neither journal/report implementation nor actual report output establishing the comparison. This evidence limit does not establish that the preserved implementation is absent. The author's earlier met assessment is replaced with the reviewer's partial verdict.
- **partial** — Predeclared success: Δscore per wall hour at least 1.5× control. Otherwise record the negative result in `docs/benchmarks.md` and keep the old path as the default. — evidence: `docs/benchmarks.md::Production-input recheck (25 November 2026)` — reviewer: partial — reason: the documentation retains the rule and legacy defaults, but no production ratio or measured negative result is supplied; preservation of runtime defaults is asserted rather than demonstrated by implementation changes in this diff.
- **partial** — Widen the cohort (`--candidates`, `--tree-roots`, magnitude variants, combinations) until the call costs about what promotion costs today. — evidence: `docs/benchmarks.md::Production-input recheck (25 November 2026)` — reviewer: partial — reason: production width tuning was not run. This establishes outstanding production tuning, without establishing any defect in the preserved candidate-generation controls.

The permitted diff does not independently establish completion or failure of
the preserved whole-cohort ranking, ordinary winner/parent verification,
`--explore-quota` or legacy fallback implementations. No author status departs
from the independent reviewer's verdicts.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

`CODING-STANDARDS.md` is absent at the requested repository path, so compliance
with that named standard cannot be certified. The independent reviewer found
no actionable violation from the permitted inputs:

- **clean** — Scope: the diff changes only two Markdown documents; no runtime code, configuration, dependencies or tests change.
- **clean** — Evidence boundaries: the benchmark addition distinguishes reported fixture correctness checks from unrun production trials and labels missing production measurements as unmeasured.
- **clean** — Decision reporting: the documentation leaves the production 1.5× criterion unevaluated and does not present absent results as a measured failure, zero counts or grounds to change defaults.
- **clean** — Provenance consistency: stale no-delegation claims are removed; the Standards review marker, recorded findings and stated review inputs agree with the review. Independent documentation review remains distinct from unverified implementation, test-execution and production claims.

These findings concern this documentation-only diff. They do not certify
implementation correctness, reported test execution, production performance
or compliance with the missing standards file.

## Outstanding completion findings

- Required item 1: independent reviewer provenance and verdicts are recorded above, replacing the stale no-delegation statements.
- Required item 2, `docs/benchmarks.md:412`: reproduced and unresolved. The executor rechecked production-input availability and found no usable champion or corpus. The required production width tuning and A/B cannot be performed with the available inputs; generated fixtures cannot close this finding.
- Required item 3 and unmet criteria 1 and 2: production A/B remains missing and the 1.5× success rule remains unevaluated. Legacy defaults remain. These findings are not disputed or claimed closed by this summary repair.

## Test Plan

- No tests were added, modified or removed; no concrete gap was found in the inspected ranking, verification and reporting coverage. These checks are expected green against the merged implementation; no new failing regression is claimed.
- `cargo test --locked --lib exact < /dev/null`: passed all 14 selected tests, including legacy defaults, older-scorer fallback, verification failures and reporting.
- `cargo test --locked --test real_scorer -- --nocapture < /dev/null` with `NEAT_SCORER_BIN` set to the identified binary: passed all 7 integration tests, with no scorer-availability skip.
- Completion-repair rerun: the same 14 selected unit tests and all 7 real-scorer integration tests passed with stdin from `/dev/null`; the scorer binary checksum still matches the identity above. No production benchmark was run.
- Markdownlint, codespell and `git diff --check`: passed for the changed documentation.
- Earlier in this run, `./quality.sh < /dev/null` with `NEAT_SCORER_BIN` set to the identified binary: passed the full gate, including shell contracts, Markdown/spelling checks, actionlint, cargo-deny, formatting, Clippy, all 209 unit tests, integration/contract tests (including the 7 real-scorer tests) and rustdoc.
- TypeScript parity was not exercised: `NEAT_AI_TS_ROOT` is unset, so its harness returns without running the external engine.
- Graft lookup returned no matching nodes; `graft check` failed with `NO GRAPH`. Direct source reads supplied the code evidence; no code graph was rebuilt or claimed fresh.

Delivery checklist:

- [x] Preserve PR #150 and verify the available scorer contract.
- [x] Record source/binary identities and production-input search results.
- [ ] Locate and verify the production champion and corpus.
- [ ] Tune widths, run sequential production A/B trials and apply the 1.5× rule.
- [x] Complete the repository quality gate.

## Merge verification

CI is not yet verified. Independent Spec and Standards reviews are recorded
above; their production-evidence limitations remain open.

Not checked: production cohort economics, production-installed scorer identity,
the production 1.5× threshold and TypeScript cross-engine parity. Risk: the
production cost and gain of wider exact ranking remain unknown.

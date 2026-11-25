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
- The executor instruction explicitly prohibits delegation. No independent reviewers were dispatched, and no reviewer provenance or verdict is claimed here.

## Evidence

- Documentation-only continuation from Forests commit `5df8cb0db7411082ddaf7cd64888b20e2e1b333c`, the merge of PR #150. GitHub confirmed that merge and scorer prerequisite #683's closure and PR #684's merge.
- Scorer binary SHA-256: `8e6e46de1faf4ed46867e55e5ac2422c149932a5124d605c28a6e1099f77e26a`; built with `cargo build --locked --release --bin rust_scorer` and `rustc 1.99.1` on Linux aarch64.
- `forests/tests/real_scorer.rs::real_cohort_matches_standalone_scores_and_ordering_for_generated_grafts` passed against this binary, checking bit-identical scores and ordering for identity, MLP, IF and clamped grafts. The same suite also exercised documented directory fallback, ordinary verification, evolution and import CLI selection.
- `forests/src/report.rs::tests::exact_rank_report_keeps_ranking_verification_and_diagnostics_separate` passed, including rank comparisons, fallback and failed-verification accounting and old-journal compatibility.
- Provenance: #136 — Rank every graft candidate exactly on the full corpus with scorer delta mode, and retire the 5 % screen for them. PR #150 — the merged implementation with that title.
- Backend/CLI validation record; no visual surface or runtime behaviour changed.

## Acceptance Criteria

These are local assessments of the preserved implementation and remaining work,
not independent reviewer verdicts. Independent review was prohibited by the
executor's no-delegation instruction.

- **missing** — A/B on the production protocol in `docs/benchmarks.md` (same creature, seed and budget; control is today's defaults). Report Δscore per wall hour, acceptances, candidates scored exactly per iteration, and the wall split. — evidence: `docs/benchmarks.md::Production-input recheck (25 November 2026)` — reason: no production champion or corpus was located; production width tuning and both 2700-second trials were not run.
- **met** — Journal fields for the new path, and a `report` row comparing exact rank with what the 5 % screen would have ranked. — evidence: existing `forests/src/report.rs::tests::exact_rank_report_keeps_ranking_verification_and_diagnostics_separate`, rerun successfully; implemented in merged PR #150, with no new implementation claimed by this diff.
- **partial** — Predeclared success: Δscore per wall hour at least 1.5× control. Otherwise record the negative result in `docs/benchmarks.md` and keep the old path as the default. — evidence: `docs/benchmarks.md::Production-input recheck (25 November 2026)` and existing `forests/src/config.rs::tests::old_configurations_keep_legacy_ranking_and_exact_mode_requires_an_explicit_choice` — reason: legacy defaults are retained, but production economics and the ratio remain unmeasured; no production negative result can be declared.

## Standards Review

Independent review was not run: delegation is explicitly prohibited for this
executor. Local self-check found that the diff changes only documentation,
retains the existing implementation and defaults, and distinguishes unrun
production measurements from observed correctness results. No independent
review marker is emitted. `CODING-STANDARDS.md` is absent from this repository.

## Test Plan

- No tests were added, modified or removed; no concrete gap was found in the inspected ranking, verification and reporting coverage. These checks are expected green against the merged implementation; no new failing regression is claimed.
- `cargo test --locked --lib exact < /dev/null`: passed all 14 selected tests, including legacy defaults, older-scorer fallback, verification failures and reporting.
- `NEAT_SCORER_BIN=<identified binary> cargo test --locked --test real_scorer -- --nocapture < /dev/null`: passed all 7 integration tests, with no scorer-availability skip.
- Markdownlint, codespell and `git diff --check`: passed for the benchmark documentation.
- Full quality gate has not yet run.

Delivery checklist:

- [x] Preserve PR #150 and verify the available scorer contract.
- [x] Record source/binary identities and production-input search results.
- [ ] Locate and verify the production champion and corpus.
- [ ] Tune widths, run sequential production A/B trials and apply the 1.5× rule.
- [ ] Complete the repository quality gate.

## Merge verification

CI is not yet verified. Independent reviews were not run under the explicit
no-delegation instruction.

Not checked: production cohort economics, production-installed scorer identity,
the production 1.5× threshold and TypeScript cross-engine parity. Risk: the
production cost and gain of wider exact ranking remain unknown.

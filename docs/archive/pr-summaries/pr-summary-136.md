## Summary

Recorded the remaining production prerequisites and fresh real-scorer
validation in `docs/benchmarks.md`. Preserved the implementation from merged
PRs #150 and #151 and the legacy defaults. Part of #136.

## Spec

### Intent and Rationale

- Complete the available correctness checks before attempting production width tuning and the sequential control/treatment experiment.
- Distinguish unavailable production measurements from generated-fixture results and measured negative results.

### Essential Design Decisions

- Retain legacy ranking, 64 candidates, 8 roots, scales `1.0,0.5,0.25` and 4 combinations while the production 1.5× decision remains unevaluated.
- Generated or checked-in test fixtures cannot substitute for the production champion and corpus.
- Preserve diagnostic screening and ordinary winner/parent verification; this delivery changes documentation only.

### Undiscoverable Facts

- The bounded search on Linux aarch64 host `vibe-coder-65924` found only test/generated training files, without a verified production champion or corpus. `/home/agent` and the workspace volume's `lost+found` were unreadable; `/data` is absent.
- GitHub REST reads confirmed scorer prerequisite #683 is closed and Forests PR #151 merged as `229d84f7d56e0717466d57ef770dde9d001a2cf3`. Initial GraphQL reads failed with `API rate limit already exceeded`; REST reads succeeded.
- No production input paths were supplied. The local scorer build does not identify the training host's installed binary.

## Evidence

- Rebuilt a clean scorer checkout at `a164969ac4d96ff39eace7086ad22c0538d95705`, manifest version `1.1.110`, neat-core `v0.23.7`, using `cargo build --locked --release --bin rust_scorer` and `rustc 1.99.1`.
- Scorer binary SHA-256: `8e6e46de1faf4ed46867e55e5ac2422c149932a5124d605c28a6e1099f77e26a`. Set `NEAT_SCORER_BIN` to that binary for integration tests and the quality gate.
- Existing `forests/tests/real_scorer.rs::real_cohort_matches_standalone_scores_and_ordering_for_generated_grafts` passed: bit-identical cohort/standalone scores and matching candidate order for identity, MLP, IF and clamped grafts.
- Existing `forests/tests/real_scorer.rs::real_cohort_directory_fallback_is_reported_and_matches_plain_directory_scoring` and `real_exact_loop_accepts_only_a_standalone_verified_winner` passed, exercising fallback and ordinary verification.
- Existing `forests/src/report.rs::tests::exact_rank_report_keeps_ranking_verification_and_diagnostics_separate` passed, covering rank comparisons, fallback accounting, failed verification and old-journal compatibility. Reporting implementation is preserved, not introduced by this diff.
- Documentation-only backend/CLI evidence; no visual surface changed. No demonstrated implementation defect required a code or test change.
- Provenance: #136 — Rank every graft candidate exactly on the full corpus with scorer delta mode, and retire the 5 % screen for them.

## Acceptance Criteria

These are executor assessments of the current head, not independent reviewer
verdicts. The executor instruction explicitly prohibits delegation; no reviewer
provenance marker is claimed.

- **missing** — A/B on the production protocol in `docs/benchmarks.md` (same creature, seed and budget; control is today's defaults). Report Δscore per wall hour, acceptances, candidates scored exactly per iteration, and the wall split. — reason: no verified production champion/corpus was located; neither 2700-second trial ran and the required production metrics remain unmeasured.
- **met** — Journal fields for the new path, and a `report` row comparing exact rank with what the 5 % screen would have ranked. — evidence: preserved `forests/src/report.rs::tests::exact_rank_report_keeps_ranking_verification_and_diagnostics_separate`, run successfully against the current implementation. This criterion was delivered previously; this diff records validation and does not change reporting code.
- **partial** — Predeclared success: Δscore per wall hour at least 1.5× control. Otherwise record the negative result in `docs/benchmarks.md` and keep the old path as the default. — evidence: `docs/benchmarks.md::Production-input recheck (26 November 2026)` — reason: legacy defaults and the decision rule are retained, but no production ratio or measured negative result exists.

## Standards Review

Independent review was not performed because this executor was instructed
“Do not delegate.” No independent review verdict or provenance is claimed.
The repository has no `CODING-STANDARDS.md`; the supplied engineering standards
were used for local checks of scope, evidence boundaries and commit safety.
Only the benchmark record and this summary differ from the default branch.

## Test Plan

- No tests were added, modified or removed. Existing checks are expected green against the merged implementation; no new failing regression is claimed. No rule or matcher changed, so no twinned rule required updating.
- `cargo test --locked --test real_scorer -- --nocapture < /dev/null` with the identified `NEAT_SCORER_BIN`: passed all 7 tests without scorer-availability skips.
- `cargo test --locked --lib exact < /dev/null`: passed all 14 selected tests, including legacy defaults, older-scorer fallback, verification vetoes and report accounting.
- `./quality.sh < /dev/null` with the identified `NEAT_SCORER_BIN`: passed shell contracts, spelling, Markdown lint, actionlint, cargo-deny, formatting, Clippy, 209 unit tests, integration/contract tests and rustdoc.
- The TypeScript parity harness returned without exercising the external engine because `NEAT_AI_TS_ROOT` is unset; its reported test success is not cross-engine parity evidence.
- Graft MCP tools were not exposed; CLI `graft ask --source` returned no matching nodes. Targeted source reads supplied the implementation evidence.

Delivery checklist:

- [x] Preserve the merged ranking and verification implementation.
- [x] Identify the available scorer and run its real integration tests without availability skips.
- [x] Record the production-input search and unavailable identities.
- [ ] Validate the production champion, corpus checksums, identity and record count.
- [ ] Tune cohort width on production data.
- [ ] Run sequential fresh-output seed-1, 2700-second control and treatment trials.
- [ ] Compare production journals and evaluate the 1.5× decision.
- [x] Run the repository quality gate.

## Merge verification

CI is not yet verified. Independent review was not performed under the
executor's no-delegation constraint.

Not performed: production width tuning, production A/B and journal comparison,
production-installed scorer identification, TypeScript cross-engine parity,
independent reviews and CI verification. Risk: production cost and improvement
from wider exact ranking remain unknown.

## Summary

Corrected the production-input search record and refreshed scorer validation
in `docs/benchmarks.md`. Preserved the implementation delivered by PRs #150
and #151 and its legacy defaults. Production tuning and A/B remain unfinished.
Part of #136.

## Spec

### Intent and Rationale

- Continue from the merged implementation; the branch changes only the benchmark record and this summary.
- Distinguish real-scorer correctness tests from the production measurements required to decide defaults.

### Essential Design Decisions

- Retain legacy ranking, 64 candidates, 8 roots, scales `1.0,0.5,0.25` and 4 combinations while the production 1.5× decision is unevaluated.
- Historical diagnostic creatures and generated fixtures cannot substitute for a verified production champion and matching corpus.
- Missing production metrics are unmeasured, including failures, fallbacks and baseline vetoes; they do not establish a negative result.

### Undiscoverable Facts

- The local Linux aarch64 container has host label `GRQ-23`. The bounded search found test fixtures and two historical GRQ quarantine captures; the only corpus-shaped binary found was a 32-byte scorer fixture. Targeted hidden training/population searches found no matching corpus.
- `/home/agent`, `/root` and the workspace volume's `lost+found` were unreadable; `/data` is absent. No production input paths were supplied. No training-host filesystem was inspected.
- Validation reused the existing scorer release-profile binary, rather than rebuilding it. Its checksum matches the previous correctness build.

## Evidence

- GitHub CLI confirmed scorer prerequisite #683 is closed and Forests PRs #150 and #151 are merged, at `5df8cb0db7411082ddaf7cd64888b20e2e1b333c` and `229d84f7d56e0717466d57ef770dde9d001a2cf3` respectively.
- Local scorer source: clean checkout `a164969ac4d96ff39eace7086ad22c0538d95705`, manifest version `1.1.110`, neat-core `v0.23.7`. Binary SHA-256: `8e6e46de1faf4ed46867e55e5ac2422c149932a5124d605c28a6e1099f77e26a`. Its `--help` confirms `--cohort`; this is not the production-installed scorer's identity.
- Existing `forests/tests/real_scorer.rs::real_cohort_matches_standalone_scores_and_ordering_for_generated_grafts` passed with the binary explicitly configured, checking bit-identical scores and candidate ordering for identity, MLP, IF and clamped grafts. The suite also exercised directory fallback and ordinary winner/parent verification.
- Existing `forests/src/report.rs::tests::exact_rank_report_keeps_ranking_verification_and_diagnostics_separate` passed, covering exact/screen ranks, fallback accounting, failed verification and old-journal compatibility. Reporting is preserved code, not introduced by this diff.
- Source review: `forests/src/promote.rs::screen_and_promote` ranks the full cohort, then invokes ordinary full scoring on the eligible winner with its parent and applies the baseline-drift veto. `forests/src/run.rs::run_forests` resolves unsupported-scorer fallback; `forests/src/config.rs` retains legacy defaults.
- Documentation-only backend/CLI record; no visual surface changed. No demonstrated runtime defect required a code or test change. Provenance: #136 — Rank every graft candidate exactly on the full corpus with scorer delta mode, and retire the 5 % screen for them.

## Acceptance Criteria

These are executor assessments, not independent reviewer verdicts. The explicit
instruction “Do not delegate” prevented independent review; no reviewer marker
or verdict is claimed.

- **missing** — A/B on the production protocol in `docs/benchmarks.md` (same creature, seed and budget; control is today's defaults). Report Δscore per wall hour, acceptances, candidates scored exactly per iteration, and the wall split. — reason: no verified production champion and matching corpus were located; neither 2700-second trial ran and the required production metrics remain unmeasured.
- **met** — Journal fields for the new path, and a `report` row comparing exact rank with what the 5 % screen would have ranked. — evidence: existing `forests/src/report.rs::tests::exact_rank_report_keeps_ranking_verification_and_diagnostics_separate` and `forests/src/run.rs::tests::exact_loop_ranks_every_valid_candidate_but_files_only_verified_verdicts` passed. This criterion was delivered previously; this diff records validation only.
- **partial** — Predeclared success: Δscore per wall hour at least 1.5× control. Otherwise record the negative result in `docs/benchmarks.md` and keep the old path as the default. — evidence: `docs/benchmarks.md::Production-input recheck (26 November 2026)` — reason: the decision rule and legacy defaults remain, but production data is unavailable and no ratio or measured negative result exists. A zero-gain control would leave the ratio undefined.
- **partial** — Widen the cohort (`--candidates`, `--tree-roots`, magnitude variants, combinations) until that call costs about what `promote` costs today. — evidence: `docs/benchmarks.md::Production A/B status and decision rule` — reason: production width tuning was not run; the existing generated-fixture sweep cannot select a production width.

## Standards Review

Independent review was not performed under the explicit no-delegation
constraint. No independent provenance marker is claimed. The repository has no
`CODING-STANDARDS.md`; local review used the supplied standards for scope,
truthful evidence, Australian English and commit safety. Only two Markdown
records differ from `origin/Develop`.

## Test Plan

- No tests were added, modified or removed. Existing tests are expected green against the merged implementation; no failing regression or TDD cycle is claimed. No rule or matcher changed, so no twinned rule required updating.
- `cargo test --locked --test real_scorer -- --nocapture < /dev/null`, with `NEAT_SCORER_BIN` pointing to the identified binary: passed all 7 tests without scorer-availability skips.
- `cargo test --locked --lib exact < /dev/null`: passed all 14 selected tests, including the reporting and loop tests named above, legacy fallback and standalone verification vetoes.
- `./quality.sh < /dev/null`, with the same scorer configured: passed shell contracts, spelling, Markdown lint, actionlint, cargo-deny, formatting, Clippy, 209 unit tests, integration/contract tests and rustdoc.
- Final documentation checks: Markdown lint, codespell and `git diff --check` passed after replacing this summary.
- TypeScript cross-engine parity was not exercised: `NEAT_AI_TS_ROOT` is unset and `forests/tests/ts_parity.rs` returns without invoking that engine. Its green harness result is not parity evidence.

Delivery checklist:

- [x] Preserve merged exact-ranking, diagnostic screening and ordinary verification.
- [x] Identify the available scorer and run real integration tests without availability skips.
- [x] Correct and document the local production-input search.
- [ ] Validate production champion/corpus checksums, counts and resolved settings.
- [ ] Tune production width, including diagnostic costs.
- [ ] Run sequential fresh-output seed-1 CPU control and treatment trials, 2700 seconds per arm, preserving journals.
- [ ] Evaluate production gain per actual wall hour, wall split, acceptances and exact/screen ranks; separate failures, fallbacks and baseline vetoes.
- [x] Run the repository quality gate.

## Merge verification

CI is not yet verified. Independent reviews were not performed under the
no-delegation constraint.

Not performed: production tuning, A/B, journal comparison, production-installed
scorer identification, TypeScript parity, independent review, CI verification
or push. Risk: production cost and improvement from wider exact ranking remain
unknown; #136 is not complete.

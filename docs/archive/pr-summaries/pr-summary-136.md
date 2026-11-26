## Summary

Refreshed production-input availability and real-scorer validation in
`docs/benchmarks.md`. Clarified that the example treatment must be tuned on
production inputs and frozen before the A/B. Preserved the merged implementation
and legacy defaults. Production tuning and A/B remain unfinished. Part of #136.

## Spec

### Intent and Rationale

- Continue the remaining production experiment without redoing the merged ranking implementation.
- Keep generated-fixture correctness results separate from production economics.

### Essential Design Decisions

- Keep legacy ranking and existing cohort defaults until valid production evidence satisfies the predeclared 1.5× criterion.
- Tune width against measured legacy promotion cost, include diagnostics and verification, then freeze treatment settings before either 2700-second trial.
- Missing inputs leave production metrics unmeasured; they establish neither a negative result nor a zero-gain control.

### Undiscoverable Facts

- Local searches on Linux aarch64 host label `GRQ-23` found no verified production champion, matching binary corpus or retained experiment journal. No production paths were supplied; no training-host filesystem was inspected.
- `/data` is absent. `/home/agent`, `/root` and the workspace volume's `lost+found` were unreadable. Located GRQ creatures are fixtures or historical quarantine captures.
- The previous scorer binary was absent. Rebuilding the same source commit reproduced its recorded SHA-256.

## Evidence

- Scorer source: clean commit `a164969ac4d96ff39eace7086ad22c0538d95705`, manifest `1.1.110`, neat-core `v0.23.7`. Built with `cargo build --locked --release --bin rust_scorer`, using `rustc 1.99.1`. Binary SHA-256: `8e6e46de1faf4ed46867e55e5ac2422c149932a5124d605c28a6e1099f77e26a`; help confirms `--cohort`. This identifies the validation binary, not a production installation.
- GitHub REST reads through `gh api` confirmed scorer prerequisite #683 is closed, scorer PR #684 is merged and Forests PR #151 is merged. Initial GraphQL reads failed with `API rate limit already exceeded`; REST succeeded.
- Existing `forests/tests/real_scorer.rs::real_cohort_matches_standalone_scores_and_ordering_for_generated_grafts` passed, checking bit-identical cohort/standalone scores and ordering. The same suite exercised ordinary verification, directory fallback and CLI selection.
- Existing `forests/src/report.rs::tests::exact_rank_report_keeps_ranking_verification_and_diagnostics_separate` and `forests/src/run.rs::tests::exact_loop_ranks_every_valid_candidate_but_files_only_verified_verdicts` passed. Source review confirmed full-cohort ranking and subsequent ordinary winner/parent verification in `forests/src/promote.rs:263` and legacy defaults in `forests/src/config.rs:363`.
- Documentation-only backend/CLI record; no visual surface or runtime behaviour changed. Provenance: #136 — Rank every graft candidate exactly on the full corpus with scorer delta mode, and retire the 5 % screen for them.

## Acceptance Criteria

These are executor assessments. Independent review was not performed because
the task explicitly prohibits delegation; no reviewer provenance is claimed.

- **missing** — A/B on the production protocol in `docs/benchmarks.md` (same creature, seed and budget; control is today's defaults). Report Δscore per wall hour, acceptances, candidates scored exactly per iteration, and the wall split. — reason: no verified production champion and matching corpus were located; neither 2700-second trial ran and production metrics remain unmeasured.
- **met** — Journal fields for the new path, and a `report` row comparing exact rank with what the 5 % screen would have ranked. — evidence: existing `forests/src/report.rs::tests::exact_rank_report_keeps_ranking_verification_and_diagnostics_separate` and `forests/src/run.rs::tests::exact_loop_ranks_every_valid_candidate_but_files_only_verified_verdicts` passed. Delivered previously; this diff records validation.
- **partial** — Predeclared success: Δscore per wall hour at least 1.5× control. Otherwise record the negative result in `docs/benchmarks.md` and keep the old path as the default. — evidence: `docs/benchmarks.md` — reason: legacy defaults and the decision rule remain, but no production ratio or measured negative result exists. A zero-gain control would leave the ratio undefined.
- **partial** — Widen the cohort (`--candidates`, `--tree-roots`, magnitude variants, combinations) until that call costs about what `promote` costs today. — evidence: `docs/benchmarks.md` — reason: production width tuning remains unmeasured; generated fixtures cannot select production settings.

## Standards Review

Independent review was not performed under the no-delegation constraint. Local
review checked scope, truthful evidence, Australian English and commit safety.
Only the benchmark record and this summary differ from `origin/Develop`; no
production code, tests, dependencies or workflow files are changed by this diff.

## Test Plan

- No tests were added, edited or removed; no demonstrated defect warranted a fix or new failing regression. Existing tests are expected green on the merged implementation. No twinned rule or matcher changed.
- `cargo test --locked --test real_scorer -- --nocapture < /dev/null`: all 7 tests passed with `NEAT_SCORER_BIN` explicitly configured; no scorer-availability skips.
- `cargo test --locked --lib exact < /dev/null`: all 14 selected tests passed, including diagnostic bypasses, zero-quota diagnostics, verification vetoes, older-scorer fallback and reporting compatibility.
- `timeout 1080 ./quality.sh < /dev/null`, with the same scorer configured: passed shell contracts, codespell, Markdown lint, actionlint, cargo-deny, formatting, Clippy, 209 unit tests, integration/contract tests and rustdoc.
- TypeScript cross-engine parity was not exercised: `NEAT_AI_TS_ROOT` is unset, so its harness returned without invoking that engine.
- Graft lookup returned no matching nodes; `graft check` failed with `NO GRAPH`. Direct source reads supplied code evidence; no graph freshness is claimed.

Delivery checklist:

- [x] Preserve merged implementation and legacy defaults.
- [x] Recheck production-input availability and identify the rebuilt scorer.
- [x] Run real-scorer tests, focused exact-ranking tests and the quality gate.
- [ ] Verify production champion/corpus checksums, counts and resolved settings.
- [ ] Tune and freeze production cohort width, including diagnostic and verification costs.
- [ ] Run sequential seed-1 CPU trials, 2700 seconds per arm, retaining separate journals.
- [ ] Compare improvement per actual wall hour, acceptances, exact candidates, complete wall split and rank comparisons; separate failures, fallbacks and vetoes.

## Merge verification

CI and independent reviews are not verified. No push was assigned to this pass.

Not performed: production tuning, A/B and journal comparison, production-installed
scorer identification, TypeScript parity, independent review, CI verification
or push. Risk: issue #136 remains incomplete; wider-cohort production cost and
improvement remain unknown.

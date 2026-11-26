## Summary

Refreshed production-input availability and real-scorer validation in
`docs/benchmarks.md`. Clarified that the example treatment must be tuned on
production inputs and frozen before the A/B. Preserved the merged implementation
and legacy defaults. Production tuning and A/B remain unfinished. Part of #136.
The completion retry records another bounded input search and independent Spec
and Standards reviews; it does not supply the missing production measurements.

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
- The completion retry at `a4a7497` again found no verified production inputs or retained journals. No production paths were supplied in the issue body or comments. The executor's additional availability record is in `docs/benchmarks.md`.
- Separate independent reviewers received only the proposed branch diff and issue body (Spec), or that diff and the requested repository `CODING-STANDARDS.md` path (Standards). The standards file is absent; unrelated sibling standards were not substituted.

## Evidence

- Scorer source: clean commit `a164969ac4d96ff39eace7086ad22c0538d95705`, manifest `1.1.110`, neat-core `v0.23.7`. Built with `cargo build --locked --release --bin rust_scorer`, using `rustc 1.99.1`. Binary SHA-256: `8e6e46de1faf4ed46867e55e5ac2422c149932a5124d605c28a6e1099f77e26a`; help confirms `--cohort`. This identifies the validation binary, not a production installation.
- GitHub REST reads through `gh api` confirmed scorer prerequisite #683 is closed, scorer PR #684 is merged and Forests PR #151 is merged. Initial GraphQL reads failed with `API rate limit already exceeded`; REST succeeded.
- Existing `forests/tests/real_scorer.rs::real_cohort_matches_standalone_scores_and_ordering_for_generated_grafts` passed, checking bit-identical cohort/standalone scores and ordering. The same suite exercised ordinary verification, directory fallback and CLI selection.
- Existing `forests/src/report.rs::tests::exact_rank_report_keeps_ranking_verification_and_diagnostics_separate` and `forests/src/run.rs::tests::exact_loop_ranks_every_valid_candidate_but_files_only_verified_verdicts` passed. Source review confirmed full-cohort ranking and subsequent ordinary winner/parent verification in `forests/src/promote.rs:263` and legacy defaults in `forests/src/config.rs:363`.
- Documentation-only backend/CLI record; no visual surface or runtime behaviour changed. Provenance: #136 — Rank every graft candidate exactly on the full corpus with scorer delta mode, and retire the 5 % screen for them.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **missing** — A/B on the production protocol in `docs/benchmarks.md` (same creature, seed and budget; control is today's defaults). Report Δscore per wall hour, acceptances, candidates scored exactly per iteration, and the wall split. — evidence: `docs/benchmarks.md::Production-input recheck (26 November 2026)` — reviewer: missing — reason: production inputs are unavailable, neither sequential 2700-second trial ran and all required production metrics remain unmeasured. Correctness checks do not satisfy this criterion.
- **partial** — Journal fields for the new path, and a `report` row comparing exact rank with what the 5 % screen would have ranked. — evidence: `docs/benchmarks.md::Production-input recheck (26 November 2026)` — reviewer: partial — reason: this documentation-only diff reports existing compatibility tests but contains neither the journal/report implementation nor report output demonstrating the comparison. The earlier author assessment of met is replaced with the reviewer's partial verdict; this evidence limit does not establish that the merged implementation is absent.
- **partial** — Predeclared success: Δscore per wall hour at least 1.5× control. Otherwise record the negative result in `docs/benchmarks.md` and keep the old path as the default. — evidence: `docs/benchmarks.md::Production A/B status and decision rule` — reviewer: partial — reason: the rule and legacy defaults are documented, but no measured production success or negative result exists. Runtime default preservation cannot be independently verified from this diff.
- **partial** — Widen the cohort (`--candidates`, `--tree-roots`, magnitude variants, combinations) until that call costs about what `promote` costs today. — evidence: `docs/benchmarks.md::Production A/B status and decision rule` and `docs/benchmarks.md::Production-input recheck (26 November 2026)` — reviewer: partial — reason: measuring promotion cost, tuning the controls and freezing settings are documented, but production cost and treatment width remain unmeasured; the example is explicitly not production-tuned.

No author status departs from the independent Spec verdict. The reviewer found
no unrelated functional changes. Its permitted inputs do not independently
establish the preserved ranking, winner/parent verification, explore-quota or
fallback implementation, or verify reported test execution.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

The independent reviewer checked the proposed branch diff and the requested
repository standards path. `CODING-STANDARDS.md` is absent, so compliance with
that named document cannot be assessed or certified. Its absence alone does
not establish an implementation violation.

- **clean** — Scope: only the benchmark record and PR summary change; no runtime, configuration, dependency or test changes.
- **clean** — Evidence boundaries: generated-fixture correctness claims are separate from unavailable production inputs and unrun production trials.
- **clean** — Decision reporting: production tuning, metrics and the 1.5× rule remain unmeasured or unevaluated; missing evidence is not a measured failure or zero-gain control.
- **violation** — Review provenance: the previous summary removed the review markers and attributed missing independent review to a delegation prohibition — evidence: `docs/archive/pr-summaries/pr-summary-136.md` — reason: fixed in this diff by recording the actual separate reviews, their permitted inputs and the missing-standard limitation.

The reviewer identified no other actionable documentation violation within
these evidence limits. These findings do not independently certify source
implementation, test execution, production performance or compliance with the
missing standards document. Acceptance status follows the separate Spec review.

## Outstanding completion findings

- Required item 1: separate independent reviews, provenance markers and reviewer verdicts are recorded above; stale no-delegation claims are removed.
- Required item 2, `docs/benchmarks.md:476`: reproduced and unresolved. The executor's completion-retry search again found no verified champion or matching corpus. No production tuning, A/B or performance measurements ran. This finding is not disputed or claimed closed by the documentation update.
- Required item 3: the independent Spec reviewer still finds the production criteria incomplete; the issue remains open under Part of #136.
- Unmet criterion 1: no matched production A/B or retained trial journals exist for this change; all required production metrics remain unmeasured.
- Unmet criterion 2: the production 1.5× rule remains unevaluated. No measured negative result is claimed; legacy defaults remain documented.
- Unmet criterion 3: production promotion cost was not measured, so no production cohort width was tuned or frozen. Generated-fixture profiles are not a substitute.

## Test Plan

- Completion retry: `markdownlint-cli2 docs/benchmarks.md docs/archive/pr-summaries/pr-summary-136.md < /dev/null` passed (27 configured Markdown files); `codespell docs/benchmarks.md docs/archive/pr-summaries/pr-summary-136.md < /dev/null` and `git diff --check` passed. No code or tests changed.

The following runtime results were recorded by earlier branch validation;
this documentation retry does not independently re-run or certify them.

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

Independent Spec and Standards reviews are recorded above. CI is not verified.
No push was assigned to this pass.

Not performed: production tuning, A/B and journal comparison, production-installed
scorer identification, TypeScript parity, CI verification
or push. Risk: issue #136 remains incomplete; wider-cohort production cost and
improvement remain unknown.

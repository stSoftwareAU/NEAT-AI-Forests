# Changelog

All notable changes to NEAT-AI-Forests are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- **`affectedFraction` no longer exceeds 1 under `--row-sampling
  residual-weighted` (#64).** In one production run 898 of 1,024 candidates
  reported a share above 1, the largest being 11.67 — a patch supposedly
  correcting 1,167 % of the search set.

  The numerator was not the problem. `strategies::plan` gives each kept row an
  importance weight (stratum population over stratum sample), so the histogram
  counts — and `affectedRecords` with them — estimate a count over the *whole
  corpus*. The journal then divided that by the number of rows searched, mixing
  a corpus-scale numerator with a sample-scale denominator. It now divides by
  the search set's weighted total, which is the like-for-like comparison and
  equals the row count whenever every weight is 1 (every stride-sampled set, and
  every run before weighting existed). The new `journal::affected_fraction`
  clamps to `[0, 1]` and falls back to the row count when no weighted total is
  recorded.

  Nothing about scoring or acceptance was affected — the scorer never sees these
  fields — but they are what `neat_ai_forests report` aggregates, and a share
  above 1 quietly poisoned any comparison between a weighted run and a stride
  run. Both fields now say in their documentation which scale they are on.

### Removed

- **The GPU histogram path and the `--gpu` switch (#67).** The kernel was built,
  measured and kept behind a cargo feature that nothing ever compiled, which
  left a flag whose only correct setting was `off`: `auto` meant "an adapter
  exists" rather than "the GPU is faster", so on any host with an adapter it
  selected a path measured at **0.17×** the 8-thread CPU one — the 490 MB upload
  of the `u8` matrix dominates, and the kernel cannot beat threads writing into
  cache-resident histograms.

  The ceiling was small even if it had worked: across every production run to
  date the histogram search is **19.4 % of wall clock** (14,140 s of 72,967 s),
  the rest being full-corpus scoring. Gone with it: `forests/src/gpu.rs`, the
  WGSL kernel, `docs/gpu.md`, `GpuMode`, the `gpu` cargo feature and the `wgpu`
  / `pollster` / `bytemuck` dependencies. The journal still records a search
  backend, so a path added later can be told apart from this one in journals
  written before it existed.

  `--scorer-arg=--gpu=off` is unaffected — that is NEAT-AI-scorer's own flag for
  a different code path. The real search-time lever is reusing accumulations
  across tree levels (#69), which depth-3-by-default (#63) has made worth doing.

### Changed

- **The candidate cohort is spent where the journal says it pays (#63), and the
  defaults are now the measured-best configuration.** Full scoring is about half
  of wall-clock, so the resource being allocated is the full-corpus scorer call.
  Measured over 23 production runs and ~3,600 calls, a depth-3 tree returned
  `3.50e-5` of score per call and a one-sided stump `2.48e-7` — 141× less — yet
  one-sided stumps took 24 % of the budget and trees 8 %. Two accidents caused
  that: the tree-root count was hard-coded at 3, and `expand_discoveries` emitted
  one-sided variants before magnitude variants, so the `--candidates` cap cut
  exactly where the returns began.

  Magnitude variants are now emitted before one-sided ones, and `--tree-roots`
  controls how many distinct stump features are grown into trees. Defaults move
  with the evidence: `--max-depth` 1 → **3** (stumps alone measured zero
  improvement per hour across a whole run), `--growth` level-wise →
  **best-first** (1.94e-3 per hour against 6.5e-4 for depth-2), `--tree-roots`
  **8**, and `--magnitude-scales` `1,0.5,1.5,-1` → **`1,0.5,0.25`** (a shrunk
  leaf wins 15.9 % of the time against 6.2 % for the analytical optimum, z = 9.3;
  1.5 won nothing in 56 attempts and -1.0 was never reached before the cap).

  A/B on a production creature — same source, same seed, same wall clock —
  gave `+4.92e-4` against the control's `+2.45e-4`: **1.9× the improvement per
  wall hour on fewer scorer calls** (63 against 72). Both verified by
  `rust_scorer` and NEAT-AI's TypeScript `ConfirmScore`. See
  `docs/benchmarks.md`.

### Added

- A **shared learnings cache** (#60), off unless `--learnings-dir` is given. A
  patch names feature indices, never neuron uuids, so it can be grafted onto a
  different creature on a different host — even an island whose neurons share no
  uuid with anything we have seen — provided the widths and the corpus match.
  The cache exploits that in both directions: a win some host got past the full
  scorer is replayed onto creatures that do not already carry it, so it is not
  lost when the fittest creature moves on; and a candidate the fleet has already
  scored and turned down is dropped from the cohort, so the slot goes to
  something else instead of to a scorer call whose answer is on file. Both are
  shortcuts, not assumptions: a replayed win clears the same full-corpus gate as
  any other candidate, and a failure becomes worth retrying once
  `--learnings-retry-after-hours` (default 168) has passed.

  Records are `<--learnings-dir>/corpus-<identity>/<host>.jsonl`, one
  append-only file per host — so machines sharing the directory through a git
  repository never conflict — named by `--learnings-host` (the hostname by
  default). `--learnings-replay` (default 8) caps how many are replayed per
  iteration. Only full-corpus verdicts are cached: a graft refusal belongs to
  the creature rather than the patch, and the sampled screen ranks rather than
  judges. An unreadable or unwritable cache is logged and the run continues.
  See `docs/learnings.md`.

### Fixed

- A creature whose output is a `MINIMUM`/`MAXIMUM` clamp can be improved again
  (#58). The fleet wrapped the production champion's `IF` body — the neuron
  every earlier graft attached to — in two `MINIMUM` clamps, and because an
  extra synapse competes with a clamp's value rather than adding to it, every
  candidate was refused with "output neuron squash `MINIMUM` is an aggregate
  whose value is not additive in a new synapse" and the run made no progress.
  A clamp is linear in the source it selects, so the graft now walks past it
  onto that source, keeping a gain of the weights it passes, and attaches to
  the first neuron a correction can be added to; the root's outward edge
  carries `1 / gain`. Nothing pre-existing is rewritten — where a clamp binds
  the correction is capped, which the scorer judges. On the production creature
  the anchor is the `IF` body and 97.8 % of records reach it. The walk fails
  closed (`GraftError::NoGraftAnchor`) where the source a clamp selects is
  ambiguous, where the neuron behind it is point-wise, or after eight
  aggregates; `MEAN`/`HYPOT` outputs are still refused outright. New
  `graft::graft_anchor` reports the anchor and gain, which `run` logs each
  iteration when the anchor is not the output itself. The `rust_scorer` vs
  TypeScript parity test covers the clamped shape, and its Deno probe is granted
  `--allow-net` — NEAT-AI fetches its WASM activation bundle at import time, so
  without it the test failed locally before it scored anything.

### Changed

- Every graft shape is now emitted by NEAT-AI-core (#48): the post-order batch
  goes through `neat_core::graft_if_nodes`, which lets a child leave its
  outward edge to the parent that reads it, and an `IF` output takes a typed
  `positive` edge from the root plus `neat_core::graft_relay_node` for the
  IDENTITY relay on the `negative` branch. `graft::write_spec` and the local
  neuron/synapse writers are gone, as is the "reused constants do not reach the
  creature's last constant" fallback — NEAT-AI-core now lists a grafted node
  after every constant the creature carries.

- Grafts are described with NEAT-AI-core's canonical `IfNodeSpec` and built by
  `neat_core::graft_if_node` wherever that helper covers the shape (#42,
  NEAT-AI-core #555). The per-split IDENTITY threshold neuron is gone: the split
  point now rides as a **weight** on a shared bias-1 constant, so thresholds and
  leaves are all trainable and a stump costs one neuron fewer. Grafted creatures
  therefore carry three shared bias-1 constants (one per synapse role) instead
  of two. The two shapes the helper could not express at the time — a child node
  feeding its parent's branch, and the typed pair an `IF` output needs — were
  written out locally until #48 finished the adoption (above).

- `check_no_duplicate_synapses` is now a wrapper over NEAT-AI-core's
  `validate_no_duplicate_synapses` (NEAT-AI-core #556) rather than a second
  implementation of the same rule.

- The duplicate-pair rule is part of the graft's validation gate itself
  (#50, upstream NEAT-AI-core #572) rather than a separate call beside it, and
  the gate now runs ahead of `compile_creature`, so every path that validates a
  grafted creature enforces "at most one synapse per ordered `(from, to)` pair"
  and a repeat is still reported as `GraftError::DuplicateSynapse`.

### Added

- `--graft-constants shared|per-patch` (#56). The default stays `shared`, so
  nothing changes unless the flag is passed. Under `per-patch` every patch
  creates its own three bias-1 constants, named for it
  (`forest-<patch id>-one-c` / `-one-p` / `-one-n`), instead of reusing the
  creature's. Shared constants concentrate the blast radius: on a mature
  creature every grafted `IF` node hangs off the same three neurons, so one
  external prune of a single constant silently re-routes hundreds of nodes at
  once. Per-patch constants bound that to the patch that made them. The two are
  numerically identical — every constant holds the same `1.0` and every
  threshold and leaf is the same synapse weight — which is asserted record for
  record against the abstract evaluator. Measured on a six-patch depth-2 graft:
  deleting the worst single constant costs 12 `IF` nodes across all six patches
  under `shared` and 2 nodes in one patch under `per-patch`, for three extra
  constant neurons per patch (+66 neurons on a 23-patch graft) and no extra
  synapses.

- Every new creature is validated with `neat_core::creature_validate` before
  the graft returns it (#39), so a structurally invalid candidate is caught at
  the graft that produced it instead of surfacing downstream. The failure
  policy is reject-and-journal: the candidate is discarded and the
  `ValidationFailure` class, `reason`, `message` and offending
  neuron/synapse index are written to `experiments.jsonl` as a `discarded`
  entry. See [docs/architecture.md](docs/architecture.md#creature-validation).

- CI auto-increments the crate version on every PR (#45). The unattended
  machines rebuild only when `forests/Cargo.toml`'s version changes, so a PR
  that lands code at an unchanged version ships nothing. The new
  `version-increment` job runs `scripts/auto-version.sh`, which bumps the patch
  (in the manifest and `Cargo.lock`) when the PR is level with the base branch,
  leaves a version the author already bumped alone, and fails loud on a
  downgrade.

### Fixed

- A graft onto a creature that already carries the name `forest-one-a/b/c` on a
  neuron the graft cannot reuse — a constant of another bias, a hidden neuron —
  no longer fails with `UuidCollision`, which would have made *every* later
  graft on that creature fail too. The shared bias-1 constants take the next
  free name (`forest-one-a2`, …) instead (#50).

- A graft onto a creature with a constant listed after the three bias-1
  constants it reuses no longer produces a `constant, hidden, constant`
  listing, which `creature_validate` rejects under `NEURON_ORDER`.
  NEAT-AI-core's `graft_if_node` listed the new node immediately after its
  latest source, so the node was written out locally — ahead of the outputs, and
  therefore after every constant — whenever the reused constants did not reach
  the creature's last one (#50). NEAT-AI-core now applies that rule itself and
  the local fallback is gone (#48).

- Grafted creatures are emitted in NEAT-AI's canonical order (#39). New
  constants are listed ahead of the first hidden neuron and the assembled
  synapse list is sorted ascending by `(from, to)` index — the two ordering
  rules `creature_validate` enforces, which appending blindly broke on every
  graft. Incumbent neurons and synapses keep their content and relative order;
  only their position in the list can move.

- `graft.rs` compiles against neat-core 0.9.9, which added the optional
  `NeuronExport::id` and `CreatureExport::memetic` fields (NEAT-AI-core #559).
  Grafted neurons carry no runtime id and no memetic record, so both are
  `None`.

- Grafts no longer create a constant per leaf (#43): leaves are synapse weights
  from two shared bias-1 constants, reused from the creature when it already
  has them (constants are never mutated by evolution; weights are), else
  created once per creature. A stump now adds 2 neurons instead of 4.

- **Grafts no longer repeat a (from, to) synapse pair.** NEAT-AI's TypeScript
  loader collapses duplicates while `rust_scorer` sums them, so the first
  production creature scored +1.8e-3 under `rust_scorer` but ≈ −1.2e-5 under
  the fleet's TypeScript re-score. Thresholds now live in a per-split IDENTITY
  neuron, each leaf in its own constant, and an `IF` output's negative branch
  is fed through an identity relay; `check_no_duplicate_synapses` guards every
  candidate. Verified against `Creature.scoreDir` on fixtures (1e-7).

- One-sided stump variants now report the records of the kept side as
  `affectedRecords` instead of inheriting the two-leaf parent's count.

- Creatures whose output neuron is an `IF` aggregate (the production
  champion): correction-space residuals are now the output-space residuals
  (no `unsquash`), and grafts feed the output through both a `positive` and a
  `negative` synapse so the correction reaches every record. Previously such
  creatures produced zero stumps.

### Added

- Frugal verification (#40): `--boost-rounds K` re-searches the sample after
  subtracting the chosen patch (XGBoost-style rounds) and verifies the
  bundle's prefixes in one full-corpus call; the sample screen is skipped when
  the cohort already fits `--promote-count`. The check-in contract is
  unchanged: only same-call full-scorer winners become `best.json`.

- `tests/ts_parity.rs` + `scripts/ts-parity.sh`: optional rust_scorer vs
  NEAT-AI TypeScript `Creature.scoreDir` parity check on grafted fixtures,
  including a synapse-count-after-`fromJSON` assertion (#35).
- Weekly `cargo-upgrade.yml` dependency-update workflow (#16); CI pins
  `markdownlint-cli2@0.22.1` (#36).

- Combination candidates (`--combo-candidates`): the top-k distinct
  discoveries stacked on one clone, and the previous iteration's positive
  full-scored non-winners carried forward (alone, together, and with the new
  best). Journal records `combo` members; strategy `combo/<k>:<primary>`.
- `best.json` preserves per-neuron `tags` by uuid, tags every grafted neuron
  with `forests` / `forests-patch` provenance, and the creature-level
  `forests` tag is a Lamarck-style run summary usable as a commit subject
  (`🌳 Forests · N accepts / M iters · last: … · 🎯 output · score: … improved by …`).

- Rust workspace, `forests` crate, quality gate and CI hygiene (#1).
- Immutable incumbent workspace, checksum and authoritative scorer baseline
  with local-MSE parity gate (#2).
- Versioned quantile-bin cache for training observations (#3).
- Incumbent residual extraction, sidecar cache and regional diagnostics (#4).
- CPU reference histogram search for depth-1 residual stumps (#5).
- Optional `gpu` feature: wgpu/WGSL histogram accumulation with CPU oracle
  parity tests (#6). Measured slower than the feature-split multi-threaded CPU
  path on unified-memory hosts, so `--gpu off` is the default
  (docs/benchmarks.md).
- Portable Forest patch format and conservative `IF` grafts (#7).
- Depth-1 stump candidate population generation (#8).
- Two-phase screening + authoritative full-corpus promotion (#9).
- 45-minute iterative evolution loop, `experiments.jsonl` journal and
  `report` subcommand (#10, #15).
- Depth-2/3 residual trees and sequential boosting (#11).
- Sampling / jitter / diversity / random "dirty trick" strategies (#12).
- XGBoost external-control export/import (#13).
- Oblique multi-feature `IF` split exploration (#14).

# Changelog

All notable changes to NEAT-AI-Forests are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

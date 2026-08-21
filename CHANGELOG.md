# Changelog

All notable changes to NEAT-AI-Forests are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- CI auto-increments the crate version on every PR (#45). The unattended
  machines rebuild only when `forests/Cargo.toml`'s version changes, so a PR
  that lands code at an unchanged version ships nothing. The new
  `version-increment` job runs `scripts/auto-version.sh`, which bumps the patch
  (in the manifest and `Cargo.lock`) when the PR is level with the base branch,
  leaves a version the author already bumped alone, and fails loud on a
  downgrade.

### Fixed

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

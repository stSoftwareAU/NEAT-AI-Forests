# Changelog

All notable changes to NEAT-AI-Forests are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- Creatures whose output neuron is an `IF` aggregate (the production
  champion): correction-space residuals are now the output-space residuals
  (no `unsquash`), and grafts feed the output through both a `positive` and a
  `negative` synapse so the correction reaches every record. Previously such
  creatures produced zero stumps.

### Added

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

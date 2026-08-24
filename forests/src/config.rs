//! Run configuration — every knob the CLI exposes, with defaults.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::histogram::StumpKind;

/// Default wall-clock budget (45 minutes, matching Lamarck's stale-champion economics).
pub const DEFAULT_TIMEOUT_SECONDS: u64 = 45 * 60;
/// Default strict minimum authoritative improvement (NEAT-AI / Lamarck convention).
pub const DEFAULT_MIN_IMPROVEMENT: f64 = 1e-6;
/// Default screen sample rate.
pub const DEFAULT_SCREEN_SAMPLE_RATE: f64 = 0.05;
/// Default records streamed per chunk.
pub const DEFAULT_CHUNK_RECORDS: usize = 4096;
/// Default in-memory search sample (records). `0` streams the full corpus.
pub const DEFAULT_SEARCH_RECORDS: u64 = 200_000;

/// GPU preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum GpuMode {
    /// Use the GPU when the `gpu` feature is compiled in and an adapter exists.
    Auto,
    /// Require the GPU; fail if unavailable.
    On,
    /// CPU only (default: measured faster on unified-memory hosts, see docs/benchmarks.md).
    #[default]
    Off,
}

/// Tree growth policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum GrowthPolicy {
    /// Expand every leaf at the current depth in one pass.
    #[default]
    LevelWise,
    /// Expand only the leaf with the best split each pass.
    BestFirst,
}

/// Row sampling for the in-memory search set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum RowSampling {
    /// Deterministic stride (every k-th record).
    #[default]
    Stride,
    /// Uniform random subset (seeded).
    Uniform,
    /// Stratified by residual sign/magnitude quantile (seeded).
    Stratified,
    /// Probability ∝ |residual| (seeded) — records carry importance weights.
    ResidualWeighted,
}

/// Which bias-1 constants a graft's `IF` nodes hang off (Issue #56).
///
/// A graft needs three distinct bias-1 constants — one per synapse role — so a
/// node's condition, positive and negative edges read three different neurons.
/// The question is who owns them. The two policies are numerically identical:
/// every constant holds the same `1.0` and every threshold and leaf is the same
/// synapse weight either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum GraftConstants {
    /// One set of three shared by every patch on the creature — reused from the
    /// incumbent's own bias-1 constants where it has them, else created once as
    /// `forest-one-a/b/c` (#43). **The default**: it adds no neurons after the
    /// first graft, at the price of putting every grafted node on the creature
    /// behind the same three neurons.
    #[default]
    Shared,
    /// Three constants **per patch**, named for it (`forest-<patch id>-one-c`
    /// / `-one-p` / `-one-n`). A patch's `IF` nodes then only depend on
    /// constants that patch introduced, so an external pruner that removes one
    /// damages that patch and nothing else. Costs three extra constant neurons
    /// per patch. Opt-in, for measuring whether a creature built this way
    /// survives the fleet better (Issue #56).
    PerPatch,
}

/// Feature subset selection per search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum FeatureSelection {
    /// All features.
    #[default]
    All,
    /// Random subset (seeded).
    Random,
    /// Top features by |correlation(feature bin, residual)| on the search set.
    ErrorRanked,
}

/// Complete configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ForestsConfig {
    /// Source creature path (read-only).
    pub creature: PathBuf,
    /// Training corpus directory.
    pub training_data: PathBuf,
    /// Output directory (`best.json`, `experiments.jsonl`, `winners/`, `workspace/`).
    pub output_dir: PathBuf,
    /// Where caches live (defaults to the training directory).
    pub cache_dir: Option<PathBuf>,
    /// Scorer binary.
    pub scorer_path: PathBuf,
    /// Extra scorer arguments.
    pub scorer_args: Vec<String>,
    /// Wall-clock budget.
    pub timeout: Duration,
    /// Maximum iterations (`None` = until timeout).
    pub max_iterations: Option<u64>,
    /// RNG seed (`None` = drawn).
    pub seed: Option<u64>,
    /// Strict minimum authoritative improvement.
    pub min_improvement: f64,
    /// Bins per observation.
    pub bins: usize,
    /// Bin-cache sample records.
    pub bin_sample_records: u64,
    /// Bin-cache memory budget (bytes).
    pub bin_memory_budget_bytes: usize,
    /// Records per streaming chunk.
    pub chunk_records: usize,
    /// Threads for residual extraction and CPU histogram accumulation.
    pub analysis_threads: usize,
    /// In-memory search records (`0` = stream the full corpus each pass).
    pub search_records: u64,
    /// Row sampling scheme for the search set.
    pub row_sampling: RowSampling,
    /// Feature selection scheme.
    pub feature_selection: FeatureSelection,
    /// Fraction of features kept under `random` / `error-ranked` selection.
    pub feature_fraction: f64,
    /// Minimum records per corrected leaf.
    pub min_leaf_records: f64,
    /// Clamp on |leaf correction|.
    pub max_correction: f64,
    /// Minimum proxy gain.
    pub min_gain: f64,
    /// Stump kinds searched.
    pub stump_kinds: Vec<StumpKind>,
    /// Top-K stumps taken from each search.
    pub top_k: usize,
    /// Diversity cap per feature (0 = unlimited).
    pub max_per_feature: usize,
    /// Maximum tree depth (1 = stumps only).
    pub max_depth: usize,
    /// Growth policy for depth > 1.
    pub growth: GrowthPolicy,
    /// Distinct stump features grown into trees each iteration, on top of the
    /// unconstrained best-first tree (Issue #63).
    ///
    /// Trees are the most valuable thing a full-corpus scorer call can be spent
    /// on — `3.5e-5` of score per call at depth 3 against `2.1e-6` for a stump,
    /// measured over 23 production runs — so the supply of them is worth
    /// tuning. More roots means more trees competing for the same capped
    /// cohort.
    pub tree_roots: usize,
    /// Leaf magnitude scales tried around the analytical optimum.
    pub magnitude_scales: Vec<f32>,
    /// Threshold jitter: neighbouring bin offsets tried around each top stump.
    pub threshold_jitter: usize,
    /// Random stump candidates per iteration (first-class controls).
    pub random_candidates: usize,
    /// Oblique (2–3 feature) candidates per iteration (0 = off).
    pub oblique_candidates: usize,
    /// Boosting rounds on the in-memory sample: after the best patch is chosen
    /// its correction is subtracted from the sample residuals and the search
    /// repeats, producing a bundle whose prefixes are verified in one scorer
    /// call (1 = off).
    pub boost_rounds: usize,
    /// Combination candidates: stack the top-2…top-N distinct-feature
    /// discoveries on one clone, and carry forward the previous iteration's
    /// near-winners (0 = off).
    pub combo_candidates: usize,
    /// Maximum candidates grafted per iteration.
    pub candidates: usize,
    /// Who owns a graft's three bias-1 constants (Issue #56).
    pub graft_constants: GraftConstants,
    /// Screen sample rate (`None` = no screen; every candidate is fully scored).
    pub screen_sample_rate: Option<f64>,
    /// Screen promotion threshold on sampled Δscore.
    pub screen_threshold: f64,
    /// Maximum candidates promoted to full scoring.
    pub promote_count: usize,
    /// Exploratory bypass quota: screen-rejected candidates fully scored to measure false negatives.
    pub explore_quota: usize,
    /// Tolerated |same-call full baseline − authoritative baseline|.
    pub baseline_drift_epsilon: f64,
    /// Parity abs tolerance (`None` = skip parity).
    pub parity_abs: Option<f64>,
    /// Parity rel tolerance.
    pub parity_rel: f64,
    /// GPU preference.
    pub gpu: GpuMode,
    /// Keep per-iteration candidate directories.
    pub preserve_candidates: bool,
    /// Consecutive scorer failures tolerated before aborting.
    pub max_consecutive_scorer_failures: u32,
    /// Shared learnings cache root (`None` = the cache is off, Issue #60).
    ///
    /// What worked and what failed is written here as one append-only file per
    /// host, and every host's file is read back, so a fleet that shares the
    /// directory — through a git repository, say — re-applies each other's wins
    /// even after the fittest creature has moved on.
    pub learnings_dir: Option<PathBuf>,
    /// Name this machine files its learnings under (`None` = the hostname).
    pub learnings_host: Option<String>,
    /// Cached candidates replayed per iteration (0 = write only, never read).
    pub learnings_replay: usize,
    /// How long a candidate that only ever failed is left alone before it is
    /// offered again.
    pub learnings_retry_after_secs: u64,
}

impl Default for ForestsConfig {
    fn default() -> Self {
        Self {
            creature: "creature.json".into(),
            training_data: "training".into(),
            output_dir: ".".into(),
            cache_dir: None,
            scorer_path: "rust_scorer".into(),
            scorer_args: Vec::new(),
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECONDS),
            max_iterations: None,
            seed: None,
            min_improvement: DEFAULT_MIN_IMPROVEMENT,
            bins: crate::bins::DEFAULT_BINS,
            bin_sample_records: crate::bins::DEFAULT_BIN_SAMPLE_RECORDS,
            bin_memory_budget_bytes: crate::bins::DEFAULT_BIN_MEMORY_BUDGET_BYTES,
            chunk_records: DEFAULT_CHUNK_RECORDS,
            analysis_threads: 4,
            search_records: DEFAULT_SEARCH_RECORDS,
            row_sampling: RowSampling::Stride,
            feature_selection: FeatureSelection::All,
            feature_fraction: 0.25,
            min_leaf_records: 50.0,
            max_correction: 1.0,
            min_gain: 0.0,
            stump_kinds: StumpKind::ALL.to_vec(),
            top_k: 16,
            max_per_feature: 2,
            max_depth: 3,
            growth: GrowthPolicy::BestFirst,
            tree_roots: 8,
            magnitude_scales: vec![1.0, 0.5, 0.25],
            threshold_jitter: 0,
            random_candidates: 4,
            oblique_candidates: 0,
            combo_candidates: 4,
            boost_rounds: 1,
            candidates: 64,
            graft_constants: GraftConstants::Shared,
            screen_sample_rate: Some(DEFAULT_SCREEN_SAMPLE_RATE),
            screen_threshold: 0.0,
            promote_count: 8,
            explore_quota: 1,
            baseline_drift_epsilon: 1e-6,
            parity_abs: Some(1e-7),
            parity_rel: 1e-4,
            gpu: GpuMode::Off,
            preserve_candidates: false,
            max_consecutive_scorer_failures: 3,
            learnings_dir: None,
            learnings_host: None,
            learnings_replay: 8,
            learnings_retry_after_secs: crate::learnings::DEFAULT_RETRY_AFTER_SECS,
        }
    }
}

impl ForestsConfig {
    /// Validate cross-field constraints; error messages name the flag.
    pub fn validate(&self) -> Result<(), String> {
        if self.bins < 2 || self.bins > crate::bins::MAX_BINS {
            return Err(format!("--bins must be in 2..={}", crate::bins::MAX_BINS));
        }
        if let Some(r) = self.screen_sample_rate
            && !(r > 0.0 && r < 1.0)
        {
            return Err("--screen-sample-rate must be in (0, 1); use 0 to disable".into());
        }
        if self.min_improvement <= 0.0 || self.min_improvement.is_nan() {
            return Err("--min-improvement must be > 0".into());
        }
        if self.max_depth == 0 || self.max_depth > 3 {
            return Err("--max-depth must be 1, 2 or 3".into());
        }
        if self.boost_rounds == 0 || self.boost_rounds > 8 {
            return Err("--boost-rounds must be in 1..=8".into());
        }
        if self.candidates == 0 {
            return Err("--candidates must be > 0".into());
        }
        if self.analysis_threads == 0 {
            return Err("--analysis-threads must be > 0".into());
        }
        if !(self.feature_fraction > 0.0 && self.feature_fraction <= 1.0) {
            return Err("--feature-fraction must be in (0, 1]".into());
        }
        if self.stump_kinds.is_empty() {
            return Err("--stump-kinds must name at least one kind".into());
        }
        if self.magnitude_scales.is_empty() || self.magnitude_scales.iter().any(|s| !s.is_finite())
        {
            return Err("--magnitude-scales must be finite and non-empty".into());
        }
        if self.max_correction <= 0.0 || self.max_correction.is_nan() {
            return Err("--max-correction must be > 0".into());
        }
        Ok(())
    }

    /// Cache directory (defaults to the training directory).
    pub fn cache_dir(&self) -> PathBuf {
        self.cache_dir
            .clone()
            .unwrap_or_else(|| self.training_data.clone())
    }

    /// Search controls derived from the config.
    pub fn search_controls(&self) -> crate::histogram::SearchControls {
        crate::histogram::SearchControls {
            min_leaf_records: self.min_leaf_records,
            max_correction: self.max_correction,
            min_gain: self.min_gain,
            kinds: self.stump_kinds.clone(),
            top_k: self.top_k,
            max_per_feature: self.max_per_feature,
        }
    }

    /// Parity policy derived from the config.
    pub fn parity_policy(&self) -> crate::baseline::ParityPolicy {
        match self.parity_abs {
            Some(abs) => crate::baseline::ParityPolicy::Abort {
                abs,
                rel: self.parity_rel,
            },
            None => crate::baseline::ParityPolicy::Skip,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate_and_bad_values_name_the_flag() {
        let c = ForestsConfig::default();
        c.validate().unwrap();
        let bad = ForestsConfig {
            screen_sample_rate: Some(1.5),
            ..c.clone()
        };
        assert!(bad.validate().unwrap_err().contains("--screen-sample-rate"));
        let bad = ForestsConfig {
            max_depth: 4,
            ..c.clone()
        };
        assert!(bad.validate().unwrap_err().contains("--max-depth"));
        let bad = ForestsConfig { bins: 1, ..c };
        assert!(bad.validate().unwrap_err().contains("--bins"));
    }
}

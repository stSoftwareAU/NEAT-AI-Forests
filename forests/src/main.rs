//! `neat_ai_forests` command-line interface.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};
use neat_ai_forests::config::{
    FeatureSelection, ForestsConfig, GpuMode, GrowthPolicy, RowSampling,
};
use neat_ai_forests::histogram::StumpKind;
use neat_ai_forests::{CancelToken, ExternalScorer, log};

#[derive(Parser, Debug)]
#[command(name = "neat_ai_forests")]
#[command(
    about = "Experimental residual decision-tree optimiser for already-fit NEAT-AI creatures"
)]
#[command(version)]
struct Cli {
    /// Source creature JSON (never modified).
    #[arg(global = true)]
    creature: Option<PathBuf>,
    /// Training corpus directory of `.bin` files.
    #[arg(global = true)]
    training_data: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,

    /// Output directory for `best.json`, `experiments.jsonl`, `winners/`, `workspace/`.
    #[arg(long, default_value = ".")]
    output_dir: PathBuf,
    /// Cache directory for the bin and residual caches (default: the training directory).
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// NEAT-AI-scorer binary.
    #[arg(long, default_value = "rust_scorer")]
    scorer: PathBuf,
    /// Extra argument passed verbatim to the scorer (repeatable), e.g. `--scorer-arg=--gpu=off`.
    #[arg(long = "scorer-arg")]
    scorer_args: Vec<String>,
    /// Wall-clock budget in seconds.
    #[arg(long, default_value_t = neat_ai_forests::config::DEFAULT_TIMEOUT_SECONDS)]
    timeout_seconds: u64,
    /// Stop after this many iterations.
    #[arg(long)]
    max_iterations: Option<u64>,
    /// RNG seed (drawn from OS entropy when absent; printed for replay).
    #[arg(long)]
    seed: Option<u64>,
    /// Strict minimum authoritative score improvement.
    #[arg(long, default_value_t = neat_ai_forests::config::DEFAULT_MIN_IMPROVEMENT)]
    min_improvement: f64,
    /// Quantile bins per observation (2..=256).
    #[arg(long, default_value_t = neat_ai_forests::bins::DEFAULT_BINS)]
    bins: usize,
    /// Records sampled per feature when building the bin cache.
    #[arg(long, default_value_t = neat_ai_forests::bins::DEFAULT_BIN_SAMPLE_RECORDS)]
    bin_sample_records: u64,
    /// Memory budget (MiB) per bin-cache pass.
    #[arg(long, default_value_t = 256)]
    bin_memory_budget_mib: usize,
    /// Records per streaming chunk.
    #[arg(long, default_value_t = neat_ai_forests::config::DEFAULT_CHUNK_RECORDS)]
    chunk_records: usize,
    /// Threads for residual extraction and CPU histogram accumulation.
    #[arg(long, default_value_t = 4)]
    analysis_threads: usize,
    /// In-memory search records per iteration (0 = whole corpus).
    #[arg(long, default_value_t = neat_ai_forests::config::DEFAULT_SEARCH_RECORDS)]
    search_records: u64,
    /// Row sampling scheme for the search set.
    #[arg(long, value_enum, default_value_t = RowSampling::Stride)]
    row_sampling: RowSampling,
    /// Feature subset scheme.
    #[arg(long, value_enum, default_value_t = FeatureSelection::All)]
    feature_selection: FeatureSelection,
    /// Fraction of features kept under random / error-ranked selection.
    #[arg(long, default_value_t = 0.25)]
    feature_fraction: f64,
    /// Minimum records in any corrected leaf.
    #[arg(long, default_value_t = 50.0)]
    min_leaf_records: f64,
    /// Clamp on |leaf correction| (pre-squash units).
    #[arg(long, default_value_t = 1.0)]
    max_correction: f64,
    /// Minimum proxy gain for a stump to be reported.
    #[arg(long, default_value_t = 0.0)]
    min_gain: f64,
    /// Stump kinds searched (comma separated: left-only,right-only,two-leaf).
    #[arg(long, value_delimiter = ',', default_values_t = [StumpKind::LeftOnly, StumpKind::RightOnly, StumpKind::TwoLeaf], value_parser = parse_kind)]
    stump_kinds: Vec<StumpKind>,
    /// Top-K stumps per search.
    #[arg(long, default_value_t = 16)]
    top_k: usize,
    /// Diversity cap: stumps per feature in the top-K (0 = unlimited).
    #[arg(long, default_value_t = 2)]
    max_per_feature: usize,
    /// Maximum tree depth (1 = stumps only, up to 3).
    #[arg(long, default_value_t = 1)]
    max_depth: usize,
    /// Tree growth policy for depth > 1.
    #[arg(long, value_enum, default_value_t = GrowthPolicy::LevelWise)]
    growth: GrowthPolicy,
    /// Leaf magnitude scales tried around the analytical optimum (comma separated).
    #[arg(long, value_delimiter = ',', default_values_t = [1.0f32, 0.5, 1.5, -1.0])]
    magnitude_scales: Vec<f32>,
    /// Neighbouring bins tried around each top threshold.
    #[arg(long, default_value_t = 0)]
    threshold_jitter: usize,
    /// Deliberately random stump candidates per iteration.
    #[arg(long, default_value_t = 4)]
    random_candidates: usize,
    /// Oblique (multi-feature) split candidates per iteration (0 = off).
    #[arg(long, default_value_t = 0)]
    oblique_candidates: usize,
    /// Maximum candidates grafted per iteration.
    #[arg(long, default_value_t = 64)]
    candidates: usize,
    /// Screen sample rate in (0,1); 0 disables the screen (every candidate is fully scored).
    #[arg(long, default_value_t = neat_ai_forests::config::DEFAULT_SCREEN_SAMPLE_RATE)]
    screen_sample_rate: f64,
    /// Sampled Δscore a candidate must exceed to be promoted.
    #[arg(long, default_value_t = 0.0)]
    screen_threshold: f64,
    /// Maximum candidates promoted to full scoring per iteration.
    #[arg(long, default_value_t = 8)]
    promote_count: usize,
    /// Screen-rejected candidates fully scored anyway, to measure false negatives.
    #[arg(long, default_value_t = 1)]
    explore_quota: usize,
    /// Tolerated |same-call full baseline − authoritative baseline|.
    #[arg(long, default_value_t = 1e-6)]
    baseline_drift_epsilon: f64,
    /// Skip the local-MSE vs scorer parity gate (e.g. non-MSE cost); local proxies become unverified.
    #[arg(long)]
    skip_parity: bool,
    /// Parity absolute tolerance.
    #[arg(long, default_value_t = 1e-7)]
    parity_abs: f64,
    /// Parity relative tolerance.
    #[arg(long, default_value_t = 1e-4)]
    parity_rel: f64,
    /// GPU histogram search preference (needs the `gpu` cargo feature; CPU is faster on unified-memory hosts).
    #[arg(long, value_enum, default_value_t = GpuMode::Off)]
    gpu: GpuMode,
    /// Keep per-iteration candidate directories under `workspace/`.
    #[arg(long)]
    preserve_candidates: bool,
    /// Consecutive scorer failures tolerated before stopping.
    #[arg(long, default_value_t = 3)]
    max_consecutive_scorer_failures: u32,
}

fn parse_kind(s: &str) -> Result<StumpKind, String> {
    match s {
        "left-only" => Ok(StumpKind::LeftOnly),
        "right-only" => Ok(StumpKind::RightOnly),
        "two-leaf" => Ok(StumpKind::TwoLeaf),
        other => Err(format!(
            "unknown stump kind `{other}` (left-only|right-only|two-leaf)"
        )),
    }
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Summarise an `experiments.jsonl` journal as JSON on stdout.
    Report {
        /// Journal path.
        journal: PathBuf,
    },
    /// Export `f0..fN,residual,correction` CSV for the XGBoost control experiment.
    ExportMatrix {
        /// Output CSV path (a `.meta.json` sidecar is written beside it).
        #[arg(long, default_value = "forests-matrix.csv")]
        out: PathBuf,
        /// Maximum records (deterministic stride; 0 = all).
        #[arg(long, default_value_t = 200_000)]
        max_records: u64,
        /// Output index whose residual is exported.
        #[arg(long, default_value_t = 0)]
        output: usize,
    },
    /// Convert an XGBoost JSON dump into Forest patches and judge them with the scorer.
    ImportXgboost {
        /// `booster.get_dump(dump_format="json")` written as a JSON array.
        #[arg(long)]
        dump: PathBuf,
        /// Output index the patches correct.
        #[arg(long, default_value_t = 0)]
        output: usize,
        /// Accept nodes whose `missing` branch differs from `yes` (documented divergence).
        #[arg(long)]
        allow_missing_divergence: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Some(Command::Report { journal }) = &cli.command {
        return match neat_ai_forests::report_from_journal(journal) {
            Ok(r) => {
                println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        };
    }
    let (Some(creature), Some(training)) = (cli.creature.clone(), cli.training_data.clone()) else {
        eprintln!(
            "usage: neat_ai_forests <creature.json> <training-data-dir> [OPTIONS]\n       neat_ai_forests report <experiments.jsonl>\n       neat_ai_forests <creature.json> <training-data-dir> export-matrix|import-xgboost [OPTIONS]"
        );
        return ExitCode::FAILURE;
    };
    let config = ForestsConfig {
        creature,
        training_data: training,
        output_dir: cli.output_dir.clone(),
        cache_dir: cli.cache_dir.clone(),
        scorer_path: cli.scorer.clone(),
        scorer_args: cli.scorer_args.clone(),
        timeout: Duration::from_secs(cli.timeout_seconds),
        max_iterations: cli.max_iterations,
        seed: cli.seed,
        min_improvement: cli.min_improvement,
        bins: cli.bins,
        bin_sample_records: cli.bin_sample_records,
        bin_memory_budget_bytes: cli.bin_memory_budget_mib.saturating_mul(1024 * 1024),
        chunk_records: cli.chunk_records,
        analysis_threads: cli.analysis_threads,
        search_records: cli.search_records,
        row_sampling: cli.row_sampling,
        feature_selection: cli.feature_selection,
        feature_fraction: cli.feature_fraction,
        min_leaf_records: cli.min_leaf_records,
        max_correction: cli.max_correction,
        min_gain: cli.min_gain,
        stump_kinds: cli.stump_kinds.clone(),
        top_k: cli.top_k,
        max_per_feature: cli.max_per_feature,
        max_depth: cli.max_depth,
        growth: cli.growth,
        magnitude_scales: cli.magnitude_scales.clone(),
        threshold_jitter: cli.threshold_jitter,
        random_candidates: cli.random_candidates,
        oblique_candidates: cli.oblique_candidates,
        candidates: cli.candidates,
        screen_sample_rate: if cli.screen_sample_rate > 0.0 && cli.screen_sample_rate < 1.0 {
            Some(cli.screen_sample_rate)
        } else {
            None
        },
        screen_threshold: cli.screen_threshold,
        promote_count: cli.promote_count,
        explore_quota: cli.explore_quota,
        baseline_drift_epsilon: cli.baseline_drift_epsilon,
        parity_abs: if cli.skip_parity {
            None
        } else {
            Some(cli.parity_abs)
        },
        parity_rel: cli.parity_rel,
        gpu: cli.gpu,
        preserve_candidates: cli.preserve_candidates,
        max_consecutive_scorer_failures: cli.max_consecutive_scorer_failures,
    };
    if let Err(e) = config.validate() {
        eprintln!("{e}");
        return ExitCode::from(2);
    }
    let scorer = ExternalScorer {
        binary: config.scorer_path.clone(),
        extra_args: config.scorer_args.clone(),
    };
    match &cli.command {
        Some(Command::ExportMatrix {
            out,
            max_records,
            output,
        }) => {
            return run_or_fail(neat_ai_forests::tools::export_matrix(
                &config,
                *output,
                *max_records,
                out,
            ));
        }
        Some(Command::ImportXgboost {
            dump,
            output,
            allow_missing_divergence,
        }) => {
            return run_or_fail(neat_ai_forests::tools::import_xgboost(
                &config,
                &scorer,
                dump,
                *output,
                *allow_missing_divergence,
            ));
        }
        Some(Command::Report { .. }) => unreachable!(),
        None => {}
    }
    let cancel = CancelToken::new();
    #[cfg(unix)]
    if let Err(e) = neat_ai_forests::cancel::install_cancel_signals(&cancel) {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    match neat_ai_forests::run_forests(&config, &scorer, &cancel) {
        Ok(result) => {
            neat_ai_forests::run::print_run_summary(&result);
            ExitCode::SUCCESS
        }
        Err(e) => {
            log::warn(&format!("run aborted: {e}"));
            ExitCode::FAILURE
        }
    }
}

fn run_or_fail(r: Result<(), String>) -> ExitCode {
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

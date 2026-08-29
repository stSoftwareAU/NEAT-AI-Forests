//! `neat_ai_forests` command-line interface.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};
use neat_ai_forests::config::{
    FeatureSelection, ForestsConfig, GraftConstants, GrowthPolicy, IfCorrection, RowSampling,
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
    /// Maximum tree depth (1 = stumps only, up to 3). Depth is what pays:
    /// stumps alone returned no improvement at all in the measured rotation.
    #[arg(long, default_value_t = 3)]
    max_depth: usize,
    /// Tree growth policy for depth > 1.
    #[arg(long, value_enum, default_value_t = GrowthPolicy::BestFirst)]
    growth: GrowthPolicy,
    /// How a correction reaches both branches of an `IF` anchor: `typed-pair`
    /// (one source feeding both roles — a neuron cheaper, and what every engine
    /// agrees on from @stsoftware/neat-ai 6.6.40) or `relay` (an IDENTITY
    /// neuron per graft, for creatures that must load under an older one).
    #[arg(long, value_enum, default_value_t = IfCorrection::TypedPair)]
    if_correction: IfCorrection,
    /// Distinct stump features grown into trees each iteration (on top of the
    /// unconstrained best-first tree). Trees are the most valuable candidates
    /// per scorer call; more roots means more of them competing for the cohort.
    #[arg(long, default_value_t = 8)]
    tree_roots: usize,
    /// Leaf magnitude scales tried around the analytical optimum (comma separated).
    #[arg(long, value_delimiter = ',', default_values_t = [1.0f32, 0.5, 0.25])]
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
    /// Boosting rounds on the sample: re-search after subtracting the best patch's correction; prefixes of the bundle are verified in one scorer call (1 = off).
    #[arg(long, default_value_t = 1)]
    boost_rounds: usize,
    /// Combination candidates per iteration: top-2…top-N discoveries stacked, plus carried-forward near-winners (0 = off).
    #[arg(long, default_value_t = 4)]
    combo_candidates: usize,
    /// Maximum candidates grafted per iteration.
    #[arg(long, default_value_t = 64)]
    candidates: usize,
    /// Who owns a graft's three bias-1 constants: `shared` (one set for the
    /// whole creature — the default, no extra neurons) or `per-patch` (each
    /// patch gets its own, bounding an external pruner's blast radius to one
    /// patch at three extra constant neurons per patch).
    #[arg(long, value_enum, default_value_t = GraftConstants::Shared)]
    graft_constants: GraftConstants,
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
    /// Keep per-iteration candidate directories under `workspace/`.
    #[arg(long)]
    preserve_candidates: bool,
    /// Consecutive scorer failures tolerated before stopping.
    #[arg(long, default_value_t = 3)]
    max_consecutive_scorer_failures: u32,
    /// File every accepted patch as a Rebase enhancement bundle
    /// (`enhancements.json`, beside `best.json`), so re-entry grafts the run's
    /// discoveries onto a freshly fetched champion instead of publishing this
    /// run's own descendant. Off by default; it changes how a run publishes,
    /// never what it accepts.
    #[arg(long)]
    enhancements: bool,
    /// Shared learnings cache root (off when absent): what worked and what
    /// failed, written per host and read back from every host, so the fleet
    /// re-applies each other's wins even after the fittest creature moved on.
    #[arg(long)]
    learnings_dir: Option<PathBuf>,
    /// Name this machine files its learnings under (default: the hostname).
    #[arg(long)]
    learnings_host: Option<String>,
    /// Cached candidates replayed per iteration (0 = write only, never read).
    #[arg(long, default_value_t = 8)]
    learnings_replay: usize,
    /// Hours before a candidate that only ever failed is offered again.
    ///
    /// Global so `prune-learnings` can check its own retention against it: a
    /// rejection dropped before it was ever retried is an experiment silently
    /// skipped rather than freed (Issue #61).
    #[arg(long, global = true, default_value_t = neat_ai_forests::learnings::DEFAULT_RETRY_AFTER_SECS / 3600)]
    learnings_retry_after_hours: u64,
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
    /// Prune this host's file in the shared learnings cache (Issue #61).
    ///
    /// Safe to run from cron, when the host is idle. Only ever touches the file
    /// this host writes: the rule that keeps a shared directory conflict-free
    /// on write keeps it conflict-free on prune.
    PruneLearnings {
        /// Shared learnings cache root (the same `--learnings-dir` runs use).
        #[arg(long)]
        dir: PathBuf,
        /// Corpus identity to prune, as it appears in the directory name.
        /// Omit to prune this host's file in every corpus the directory holds,
        /// which is what a cron job wants.
        #[arg(long)]
        corpus: Option<String>,
        /// Host whose file to prune (default: this machine's hostname).
        #[arg(long)]
        host: Option<String>,
        /// Drop rejections older than this. Dropping one puts that experiment
        /// back on the table, which is the point.
        #[arg(long, default_value_t = 720)]
        rejected_after_hours: u64,
        /// Drop acceptances older than this. Far longer: wins are what the
        /// cache is for and a small fraction of the volume.
        #[arg(long, default_value_t = 4320)]
        accepted_after_hours: u64,
        /// Keep at most this many records, newest first (0 = uncapped).
        #[arg(long, default_value_t = 0)]
        max_records: usize,
        /// Report what would go without writing anything.
        #[arg(long)]
        dry_run: bool,
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
    if let Some(Command::PruneLearnings {
        dir,
        corpus,
        host,
        rejected_after_hours,
        accepted_after_hours,
        max_records,
        dry_run,
    }) = &cli.command
    {
        // Pruning must never race the retry window: a rejection dropped before
        // it was ever offered again is an experiment silently skipped rather
        // than freed (Issue #61).
        let retry_hours = cli.learnings_retry_after_hours;
        if *rejected_after_hours <= retry_hours {
            eprintln!(
                "--rejected-after-hours {rejected_after_hours} must exceed --learnings-retry-after-hours {retry_hours}, or a rejection is dropped before it is ever retried"
            );
            return ExitCode::FAILURE;
        }
        let host = host
            .clone()
            .unwrap_or_else(neat_ai_forests::learnings::default_host);
        let corpora = match corpus {
            Some(c) => vec![c.clone()],
            None => match neat_ai_forests::learnings::corpora(dir) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
            },
        };
        let policy = neat_ai_forests::learnings::PrunePolicy {
            rejected_after_secs: rejected_after_hours.saturating_mul(3600),
            accepted_after_secs: accepted_after_hours.saturating_mul(3600),
            max_records: *max_records,
            now_unix: neat_ai_forests::incumbent::now_unix(),
        };
        let mut total = neat_ai_forests::learnings::PruneOutcome::default();
        for corpus in &corpora {
            let store = neat_ai_forests::learnings::LearningsStore::new(
                dir.clone(),
                corpus.clone(),
                host.clone(),
            );
            match store.prune(&policy, *dry_run) {
                Ok(outcome) => total.add(&outcome),
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&total).unwrap_or_default()
        );
        return ExitCode::SUCCESS;
    }
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
        if_correction: cli.if_correction,
        tree_roots: cli.tree_roots,
        magnitude_scales: cli.magnitude_scales.clone(),
        threshold_jitter: cli.threshold_jitter,
        random_candidates: cli.random_candidates,
        oblique_candidates: cli.oblique_candidates,
        combo_candidates: cli.combo_candidates,
        boost_rounds: cli.boost_rounds,
        candidates: cli.candidates,
        graft_constants: cli.graft_constants,
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
        preserve_candidates: cli.preserve_candidates,
        max_consecutive_scorer_failures: cli.max_consecutive_scorer_failures,
        enhancements: cli.enhancements,
        learnings_dir: cli.learnings_dir.clone(),
        learnings_host: cli.learnings_host.clone(),
        learnings_replay: cli.learnings_replay,
        learnings_retry_after_secs: cli.learnings_retry_after_hours.saturating_mul(3600),
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
        // Both are handled before the creature and corpus are required.
        Some(Command::Report { .. }) | Some(Command::PruneLearnings { .. }) => unreachable!(),
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

//! Synthetic cohort-width sweep through the real evolution loop.
//!
//! Timings are indicative on shared hosts. This cannot satisfy the production A/B.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use neat_ai_forests::arms::{Arm, ArmPolicy};
use neat_ai_forests::config::{ForestsConfig, RankingMode};
use neat_ai_forests::corpus::write_bin_file;
use neat_ai_forests::graft::fixtures::small_mlp;
use neat_ai_forests::report::report_from_journal;
use neat_ai_forests::scorer::{DirectoryScorer, ExternalScorer};
use neat_ai_forests::{CancelToken, run_forests};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Parser)]
struct Args {
    /// A scorer binary with --cohort support.
    #[arg(long)]
    scorer: PathBuf,
    /// Generated corpus records.
    #[arg(long, default_value_t = 20_000)]
    records: usize,
    /// Repeats, alternating profile order.
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    /// Iterations per profile and repeat, each starting from the same creature.
    #[arg(long, default_value_t = 1)]
    iterations: u64,
}

fn run(args: Args) -> Result<(), String> {
    if args.records < 400 || args.repeats == 0 || args.iterations == 0 {
        return Err("records must be >= 400; repeats and iterations must be positive".into());
    }
    let scorer = ExternalScorer {
        binary: args.scorer,
        extra_args: vec!["--gpu=off".into()],
    };
    if !scorer.supports_cohort().map_err(|e| e.to_string())? {
        return Err("the width sweep requires --cohort support".into());
    }
    let root = tempfile::tempdir().map_err(|e| e.to_string())?;
    let train = root.path().join("training");
    std::fs::create_dir(&train).map_err(|e| e.to_string())?;
    let creature = small_mlp(32);
    let mut network = neat_core::compile_creature(&creature).map_err(|e| e.to_string())?;
    let mut rng = StdRng::seed_from_u64(136);
    let mut records = Vec::with_capacity(args.records);
    for _ in 0..args.records {
        let inputs: Vec<f32> = (0..32).map(|_| rng.random_range(-1.0..1.0)).collect();
        let residual = if inputs[1] > 0.0 { 0.03 } else { -0.02 }
            + if inputs[2] > 0.2 && inputs[3] < 0.1 {
                0.02
            } else {
                0.0
            };
        let target = network.activate(&inputs, 1)[0] + residual;
        records.push((inputs, vec![target]));
    }
    write_bin_file(&train.join("0.bin"), &records).map_err(|e| e.to_string())?;
    let source = root.path().join("creature.json");
    std::fs::write(
        &source,
        neat_core::creature_to_json(&creature).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let profiles = [
        (
            "legacy-64",
            RankingMode::Legacy,
            64,
            8,
            16,
            4,
            vec![1.0, 0.5, 0.25],
        ),
        (
            "exact-64",
            RankingMode::Exact,
            64,
            8,
            16,
            4,
            vec![1.0, 0.5, 0.25],
        ),
        (
            "exact-128",
            RankingMode::Exact,
            128,
            16,
            32,
            8,
            vec![1.0, 0.75, 0.5, 0.25, 0.125],
        ),
        (
            "exact-256",
            RankingMode::Exact,
            256,
            32,
            64,
            16,
            vec![1.0, 0.875, 0.75, 0.5, 0.25, 0.125],
        ),
    ];
    for repeat in 0..args.repeats {
        let mut order: Vec<usize> = (0..profiles.len()).collect();
        if repeat % 2 == 1 {
            order.reverse();
        }
        for index in order {
            let (name, mode, candidates, roots, top_k, combos, scales) = &profiles[index];
            let cfg = ForestsConfig {
                creature: source.clone(),
                training_data: train.clone(),
                output_dir: root.path().join(format!("{repeat}-{name}")),
                cache_dir: Some(root.path().join(format!("{repeat}-{name}-cache"))),
                ranking_mode: *mode,
                candidates: *candidates,
                tree_roots: *roots,
                top_k: *top_k,
                combo_candidates: *combos,
                magnitude_scales: scales.clone(),
                seed: Some(1),
                max_iterations: Some(args.iterations),
                timeout: Duration::from_secs(600),
                arm_policy: Arm::ALL
                    .into_iter()
                    .map(|arm| (arm, ArmPolicy::Default))
                    .collect(),
                ..Default::default()
            };
            let result = run_forests(&cfg, &scorer, &CancelToken::new())?;
            let report = report_from_journal(&result.journal_path)?;
            if report.scorer_failures > 0 || report.baseline_vetoes > 0 {
                return Err(format!(
                    "{name}: scorer failure or baseline veto; comparison is invalid"
                ));
            }
            println!(
                "{}",
                serde_json::json!({
                    "profile": name, "repeat": repeat + 1, "synthetic": true,
                    "records": args.records, "seed": 1,
                    "openingScore": report.opening_score, "finalScore": report.final_score,
                    "improvementPerWallHour": report.improvement_per_wall_hour,
                    "acceptances": report.acceptances, "exactCandidatesPerIteration": report.exact_candidates_per_iteration,
                    "cohortFallbacks": report.cohort_fallbacks,
                    "wallMs": report.wall_ms, "searchMs": report.search_ms, "graftMs": report.graft_ms,
                    "openingBaselineMs": report.opening_baseline_ms,
                    "screenScorerMs": report.screen_scorer_ms, "rankingScorerMs": report.ranking_scorer_ms,
                    "fullScorerMs": report.full_scorer_ms, "verificationScorerMs": report.verification_scorer_ms,
                    "promotionOverheadMs": report.promotion_overhead_ms, "setupAndOtherMs": report.setup_and_other_ms,
                    "scorerFailures": report.scorer_failures,
                })
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cohort ranking benchmark: {error}");
            ExitCode::FAILURE
        }
    }
}

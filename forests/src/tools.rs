//! Auxiliary subcommands: matrix export and XGBoost import (Issue #13).

use std::path::Path;

use neat_core::training_data::TrainingDataConfig;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::baseline::establish_baseline;
use crate::candidates::{CandidateConfig, generate_candidates};
use crate::config::ForestsConfig;
use crate::corpus::corpus_info;
use crate::incumbent::{load_incumbent, now_unix};
use crate::journal::{
    CandidateRecord, ExperimentRecord, FullSummary, JournalLine, ScreenSummary, append_journal_line,
};
use crate::log;
use crate::promote::{PromoteConfig, screen_and_promote};
use crate::residuals::ensure_residual_cache;
use crate::scorer::DirectoryScorer;
use crate::xgboost::{convert_dump, export_matrix as export_csv, parse_dump};

/// `export-matrix`: residuals for the current incumbent as CSV.
pub fn export_matrix(
    cfg: &ForestsConfig,
    output: usize,
    max_records: u64,
    out: &Path,
) -> Result<(), String> {
    let incumbent = load_incumbent(&cfg.creature).map_err(|e| e.to_string())?;
    if output >= incumbent.creature.output {
        return Err(format!(
            "--output {output} >= creature output width {}",
            incumbent.creature.output
        ));
    }
    let tcfg = TrainingDataConfig::new(incumbent.creature.input, incumbent.creature.output);
    let corpus = corpus_info(&cfg.training_data, &tcfg)?;
    let residuals = ensure_residual_cache(
        &incumbent,
        &cfg.training_data,
        &cfg.cache_dir(),
        &corpus,
        cfg.chunk_records,
        cfg.analysis_threads,
    )
    .map_err(|e| e.to_string())?;
    let n = export_csv(
        &cfg.training_data,
        &residuals,
        incumbent.creature.input,
        output,
        max_records,
        out,
    )?;
    log::ok(&format!(
        "exported {n} records × {} features to {}",
        incumbent.creature.input,
        out.display()
    ));
    Ok(())
}

/// `import-xgboost`: convert dumped trees and judge them with the scorer.
///
/// Writes `experiments.jsonl` (one experiment line) and, when a converted tree
/// wins, `best.json` — through exactly the same screen → full path as native
/// candidates.
pub fn import_xgboost(
    cfg: &ForestsConfig,
    scorer: &dyn DirectoryScorer,
    dump_path: &Path,
    output: usize,
    allow_missing_divergence: bool,
) -> Result<(), String> {
    let incumbent = load_incumbent(&cfg.creature).map_err(|e| e.to_string())?;
    let tcfg = TrainingDataConfig::new(incumbent.creature.input, incumbent.creature.output);
    let corpus = corpus_info(&cfg.training_data, &tcfg)?;
    let text =
        std::fs::read_to_string(dump_path).map_err(|e| format!("{}: {e}", dump_path.display()))?;
    let dump = parse_dump(&text)?;
    let (patches, rejected) = convert_dump(
        &dump,
        incumbent.creature.input,
        output,
        &incumbent.checksum,
        allow_missing_divergence,
        vec![format!("dump={}", dump_path.display())],
    );
    for r in &rejected {
        log::warn(&format!("xgboost tree {} rejected: {}", r.tree, r.reason));
    }
    if patches.is_empty() {
        return Err("no XGBoost tree could be converted".into());
    }
    std::fs::create_dir_all(&cfg.output_dir).map_err(|e| e.to_string())?;
    let workspace = cfg.output_dir.join("workspace");
    incumbent
        .write_workspace(&workspace)
        .map_err(|e| e.to_string())?;
    let journal = cfg.output_dir.join("experiments.jsonl");
    let baseline = establish_baseline(
        &incumbent,
        &cfg.training_data,
        &corpus,
        scorer,
        &workspace,
        cfg.parity_policy(),
        None,
    )?;
    append_journal_line(&journal, &JournalLine::Baseline(baseline.clone()))?;
    let cand_cfg = CandidateConfig {
        magnitude_scales: vec![1.0],
        random_candidates: 0,
        max_candidates: cfg.candidates,
        backend: "external:xgboost".into(),
        search_records: 0,
        max_correction: cfg.max_correction as f32,
        random_scale: 0.0,
        threshold_jitter: 0,
        notes: vec![],
        graft_constants: cfg.graft_constants,
        if_correction: cfg.if_correction,
    };
    let (candidates, discarded) = generate_candidates(&incumbent, patches, &cand_cfg);
    for (id, why) in &discarded {
        log::warn(&format!("discarded {id}: {why}"));
    }
    let mut rng = StdRng::seed_from_u64(cfg.seed.unwrap_or(0));
    let promote = PromoteConfig {
        screen_sample_rate: cfg.screen_sample_rate,
        screen_phase: 0,
        screen_threshold: cfg.screen_threshold,
        promote_count: cfg.promote_count,
        explore_quota: cfg.explore_quota,
        min_improvement: cfg.min_improvement,
        authoritative_baseline: baseline.score,
        baseline_drift_epsilon: cfg.baseline_drift_epsilon,
        preserve_dirs: cfg.preserve_candidates,
    };
    let outcome = screen_and_promote(
        &candidates,
        &incumbent.creature,
        &cfg.training_data,
        scorer,
        &workspace.join("xgboost"),
        &promote,
        &mut rng,
    )
    .map_err(|e| e.to_string())?;
    let mut record = ExperimentRecord {
        iteration: 1,
        timestamp_unix: now_unix(),
        incumbent_checksum: incumbent.checksum.clone(),
        baseline_score: baseline.score,
        search_backend: "external:xgboost".into(),
        search_set: "xgboost-dump".into(),
        search_records: 0,
        search_features: incumbent.creature.input as u64,
        strategies: vec!["xgboost-import".into()],
        search_ms: 0,
        graft_ms: 0,
        candidates_generated: (candidates.len() + discarded.len()) as u64,
        candidates_discarded: discarded.len() as u64,
        discarded: discarded
            .iter()
            .map(|(id, why)| crate::journal::DiscardRecord {
                id: id.clone(),
                reason: why.clone(),
            })
            .collect(),
        candidates: candidates
            .iter()
            .map(|c| CandidateRecord {
                id: c.id.clone(),
                strategy: c.patch.provenance.strategy.clone(),
                backend: c.patch.provenance.backend.clone(),
                depth: c.patch.root.depth(),
                features: c.patch.root.features(),
                predicted_gain: 0.0,
                affected_records: 0,
                affected_fraction: 0.0,
                screen_score: outcome
                    .screen
                    .as_ref()
                    .and_then(|s| s.scores.get(&c.id))
                    .map(|r| r.score),
                screen_delta: outcome
                    .screen
                    .as_ref()
                    .and_then(|s| s.scores.get(&c.id).map(|r| r.score - s.baseline_score)),
                promoted: outcome
                    .screen
                    .as_ref()
                    .is_none_or(|s| s.promoted.contains(&c.id)),
                bypass: outcome
                    .screen
                    .as_ref()
                    .is_some_and(|s| s.bypass.contains(&c.id)),
                full_score: outcome
                    .full
                    .as_ref()
                    .and_then(|f| f.scores.get(&c.id))
                    .map(|r| r.score),
                full_delta: outcome
                    .full
                    .as_ref()
                    .and_then(|f| f.scores.get(&c.id).map(|r| r.score - f.baseline_score)),
                patch: c.patch.clone(),
                combo: c.combo.clone(),
            })
            .collect(),
        screen: outcome.screen.as_ref().map(|s| ScreenSummary {
            sample_rate: s.rate,
            sample_phase: s.phase,
            baseline_score: s.baseline_score,
            screened: (s.scores.len() - 1) as u64,
            promoted: s.promoted.len() as u64,
            bypass: s.bypass.len() as u64,
            scorer_ms: s.scorer_ms,
        }),
        full: outcome.full.as_ref().map(|f| FullSummary {
            baseline_score: f.baseline_score,
            baseline_drift: f.baseline_drift,
            scored: (f.scores.len() - 1) as u64,
            scorer_ms: f.scorer_ms,
        }),
        winner: outcome.winner.as_ref().map(|w| w.0.clone()),
        improvement: outcome.winner.as_ref().map(|w| w.1),
        accepted: outcome.winner.is_some(),
        new_incumbent_checksum: None,
        screen_false_positives: outcome.false_positives,
        screen_false_negatives: outcome.false_negatives,
        scorer_error: outcome.veto.clone(),
    };
    if let Some((id, delta)) = &outcome.winner {
        let winner = candidates.iter().find(|c| &c.id == id).unwrap();
        let text =
            neat_core::creature_to_json_pretty(&winner.creature).map_err(|e| e.to_string())?;
        let best = cfg.output_dir.join("best.json");
        std::fs::write(&best, text).map_err(|e| e.to_string())?;
        record.new_incumbent_checksum = Some(crate::incumbent::sha256_hex(
            std::fs::read(&best).map_err(|e| e.to_string())?.as_slice(),
        ));
        log::ok(&format!(
            "xgboost tree {id} verified by the scorer: Δscore {delta:+.3e}; wrote {}",
            best.display()
        ));
    } else {
        log::info("no converted XGBoost tree beat the incumbent under the authoritative scorer");
    }
    append_journal_line(&journal, &JournalLine::Experiment(Box::new(record)))?;
    Ok(())
}

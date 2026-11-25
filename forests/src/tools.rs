//! Auxiliary subcommands: matrix export and XGBoost import (Issue #13).

use std::path::Path;
use std::time::Instant;

use neat_core::training_data::TrainingDataConfig;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::baseline::establish_baseline;
use crate::candidates::{CandidateConfig, generate_candidates};
use crate::config::ForestsConfig;
use crate::corpus::corpus_info;
use crate::incumbent::{load_incumbent, now_unix};
use crate::journal::{
    CandidateRecord, ExperimentRecord, JournalLine, append_journal_line,
};
use crate::log;
use crate::promote::{PromoteConfig, resolve_exact_ranking, screen_and_promote};
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
/// wins, `best.json` — through the same ranking and ordinary full verification
/// policies as native candidates.
pub fn import_xgboost(
    cfg: &ForestsConfig,
    scorer: &dyn DirectoryScorer,
    dump_path: &Path,
    output: usize,
    allow_missing_divergence: bool,
) -> Result<(), String> {
    cfg.validate()?;
    let exact_ranking = resolve_exact_ranking(cfg.ranking_mode, scorer).map_err(|e| e.to_string())?;
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
            exact_ranking,
            corpus_records: corpus.record_count,
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
    let scoring_started = Instant::now();
    let outcome_result = screen_and_promote(
        &candidates,
        &incumbent.creature,
        &cfg.training_data,
        scorer,
        &workspace.join("xgboost"),
        &promote,
        &mut rng,
    );
    let scoring_ms = scoring_started.elapsed().as_millis() as u64;
    let mut record = ExperimentRecord {
            ranking_mode: Some(if exact_ranking { "exact" } else { "legacy" }.into()),
            ranking: None,
            verification_ms: None,
            scoring_ms: Some(scoring_ms),
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
                hidden_sources: c.patch.root.hidden_sources(),
                predicted_gain: 0.0,
                affected_records: 0,
                affected_fraction: 0.0,
                screen_score: None,
                screen_delta: None,
                screen_rank: None,
                screen_promoted: None,
                screen_bypass: None,
                exact_score: None,
                exact_delta: None,
                exact_rank: None,
                promoted: false,
                bypass: false,
                full_score: None,
                full_delta: None,
                patch: c.patch.clone(),
                combo: c.combo.clone(),
            })
            .collect(),
        screen: None,
        full: None,
        winner: None,
        improvement: None,
        accepted: false,
        new_incumbent_checksum: None,
        screen_false_positives: 0,
        screen_false_negatives: 0,
        scorer_error: None,
    };
    let outcome = match outcome_result {
        Ok(outcome) => outcome,
        Err(e) => {
            record.scorer_error = Some(e.to_string());
            append_journal_line(&journal, &JournalLine::Experiment(Box::new(record)))?;
            return Err(e.to_string());
        }
    };
    record.record_scoring(&outcome);
    if let Some(e) = &outcome.verification_error {
        append_journal_line(&journal, &JournalLine::Experiment(Box::new(record)))?;
        return Err(e.to_string());
    }
    record.winner = outcome.winner.as_ref().map(|w| w.0.clone());
    record.improvement = outcome.winner.as_ref().map(|w| w.1);
    record.accepted = outcome.winner.is_some();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::fake::{CohortScorer, Verification};
    use crate::config::RankingMode;
    use crate::corpus::write_bin_file;
    use crate::graft::fixtures::identity_creature_json;
    use crate::journal::read_journal;

    fn fixture() -> (tempfile::TempDir, ForestsConfig, std::path::PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let train = tmp.path().join("train");
        std::fs::create_dir_all(&train).unwrap();
        let records: Vec<_> = (0..400).map(|i| {
            let x = if i % 2 == 0 { 1.0 } else { -1.0 };
            (vec![0.2, x], vec![0.2 + if x > 0.0 { 0.3 } else { 0.0 }])
        }).collect();
        write_bin_file(&train.join("0.bin"), &records).unwrap();
        let creature = tmp.path().join("creature.json");
        std::fs::write(&creature, identity_creature_json(2, 1)).unwrap();
        let dump = tmp.path().join("dump.json");
        std::fs::write(&dump, r#"[
            {"nodeid":0,"split":"f1","split_condition":0.0,"yes":1,"no":2,"missing":1,
             "children":[{"nodeid":1,"leaf":0.0},{"nodeid":2,"leaf":0.3}]},
            {"nodeid":0,"split":"f1","split_condition":0.0,"yes":1,"no":2,"missing":1,
             "children":[{"nodeid":1,"leaf":0.0},{"nodeid":2,"leaf":-0.3}]}
        ]"#).unwrap();
        let cfg = ForestsConfig {
            creature, training_data: train, output_dir: tmp.path().join("out"),
            ranking_mode: RankingMode::Exact, seed: Some(1), screen_sample_rate: Some(0.5),
            promote_count: 1, ..Default::default()
        };
        (tmp, cfg, dump)
    }

    #[test]
    fn xgboost_import_ranks_every_valid_tree_and_verifies_only_the_exact_winner() {
        let (_tmp, cfg, dump) = fixture();
        let scorer = CohortScorer::default();
        import_xgboost(&cfg, &scorer, &dump, 0, false).unwrap();
        assert_eq!(scorer.probes.get(), 1);
        assert_eq!(scorer.cohorts.get(), 1);
        let lines = read_journal(&cfg.output_dir.join("experiments.jsonl")).unwrap();
        let JournalLine::Experiment(e) = lines.last().unwrap() else { panic!("experiment") };
        assert_eq!(e.ranking.as_ref().unwrap().scored, 2);
        assert_eq!(e.ranking.as_ref().unwrap().cohort_size, 3);
        assert_eq!(e.full.as_ref().unwrap().scored, 1);
        assert!(e.screen.as_ref().unwrap().diagnostic_only);
        assert!(e.accepted);
        let winner = e.candidates.iter().find(|c| Some(&c.id) == e.winner.as_ref()).unwrap();
        assert_eq!(winner.exact_rank, Some(1));
        assert_eq!(winner.full_delta, e.improvement);
        let best = crate::incumbent::load_incumbent(&cfg.output_dir.join("best.json")).unwrap();
        let original = crate::incumbent::load_incumbent(&cfg.creature).unwrap();
        assert!(best.creature.neurons.len() > original.creature.neurons.len());
    }

    #[test]
    fn xgboost_import_never_publishes_a_ranking_after_bad_verification() {
        for verification in [Verification::Failed, Verification::Drift, Verification::Regression] {
            let (_tmp, cfg, dump) = fixture();
            let scorer = CohortScorer { verification, ..Default::default() };
            let result = import_xgboost(&cfg, &scorer, &dump, 0, false);
            if matches!(verification, Verification::Failed) {
                assert!(result.unwrap_err().contains("verification fixture failure"));
            } else {
                result.unwrap();
            }
            assert!(!cfg.output_dir.join("best.json").exists());
            let lines = read_journal(&cfg.output_dir.join("experiments.jsonl")).unwrap();
            let JournalLine::Experiment(e) = lines.last().unwrap() else { panic!("experiment") };
            assert!(!e.accepted);
            assert!(e.winner.is_none());
            assert_eq!(e.ranking.as_ref().unwrap().scored, 2);
            assert_eq!(e.candidates.iter().filter(|c| c.full_score.is_some()).count(),
                usize::from(!matches!(verification, Verification::Failed)));
            if matches!(verification, Verification::Drift) {
                assert!(e.scorer_error.as_ref().unwrap().contains("disagrees"));
            }
        }
    }

    #[test]
    fn xgboost_import_with_an_older_scorer_preserves_the_legacy_path() {
        let (_tmp, cfg, dump) = fixture();
        let scorer = CohortScorer { supported: false, ..Default::default() };
        import_xgboost(&cfg, &scorer, &dump, 0, false).unwrap();
        assert_eq!(scorer.probes.get(), 1);
        assert_eq!(scorer.cohorts.get(), 0);
        let lines = read_journal(&cfg.output_dir.join("experiments.jsonl")).unwrap();
        let JournalLine::Experiment(e) = lines.last().unwrap() else { panic!("experiment") };
        assert!(e.ranking.is_none());
        assert_eq!(e.ranking_mode.as_deref(), Some("legacy"));
        assert!(e.accepted);
        assert!(!e.screen.as_ref().unwrap().diagnostic_only);
    }
}

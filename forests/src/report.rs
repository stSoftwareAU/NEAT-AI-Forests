//! Economics report from `experiments.jsonl` (Issue #15).
//!
//! The go/no-go metric is **scorer-verified improvement per wall-clock hour**.
//! Everything else is context: where the time went (search vs scorer), which
//! strategies/backends/depths actually produced authoritative winners, how
//! concentrated accepted gains were, and what the screen's false-positive /
//! false-negative evidence says.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::journal::{JournalLine, read_journal};

/// Per-strategy (or backend / depth) aggregate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct StrategyStats {
    /// Candidates generated.
    pub generated: u64,
    /// Promoted to full scoring.
    pub promoted: u64,
    /// Fully scored (promoted + bypass).
    pub full_scored: u64,
    /// Candidates ranked by the exact cohort call, independently of verification.
    #[serde(default)]
    pub exact_ranked: u64,
    /// Authoritative winners accepted.
    pub winners: u64,
    /// Sum of accepted authoritative improvements.
    pub accepted_gain: f64,
    /// Mean full Δscore over fully scored candidates.
    pub mean_full_delta: Option<f64>,
}

/// Concentration of an accepted gain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcceptedGain {
    /// Iteration.
    pub iteration: u64,
    /// Winner id.
    pub id: String,
    /// Strategy.
    pub strategy: String,
    /// Depth.
    pub depth: usize,
    /// Authoritative improvement.
    pub improvement: f64,
    /// Affected fraction of the search set.
    pub affected_fraction: f64,
    /// Wall ms since run start.
    pub at_wall_ms: u64,
}

/// Candidate-level comparison of exact and hypothetical sampled ranks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RankComparison {
    /// Iteration that generated this candidate.
    pub iteration: u64,
    /// Candidate id.
    pub id: String,
    /// Exact full-corpus rank.
    pub exact_rank: u64,
    /// Sampled rank, absent when diagnostics were disabled.
    pub screen_rank: Option<u64>,
    /// Whether the diagnostic screen would promote the candidate.
    pub screen_promoted: Option<bool>,
    /// Whether the seeded diagnostic bypass would select the candidate.
    pub screen_bypass: Option<bool>,
}

/// Number of candidates ranked exactly in one iteration (excludes the parent).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExactIteration {
    /// Iteration number.
    pub iteration: u64,
    /// Exact candidates scored.
    pub scored: u64,
}

/// The report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct JournalReport {
    /// Exact candidates ranked, separate from authoritative full scores.
    #[serde(default)]
    pub candidates_exact_ranked: u64,
    /// Exact candidates scored in each iteration that ran cohort ranking.
    #[serde(default)]
    pub exact_candidates_per_iteration: Vec<ExactIteration>,
    /// Rows comparing exact rank with the screen's hypothetical rank.
    #[serde(default)]
    pub rank_comparisons: Vec<RankComparison>,
    /// Cohort ranking scorer time.
    #[serde(default)]
    pub ranking_scorer_ms: u64,
    /// Ordinary winner/parent verification time, including failures (subset of full scorer time).
    #[serde(default)]
    pub verification_scorer_ms: u64,
    /// Whole promotion phase, including failed calls and cohort I/O.
    #[serde(default)]
    pub promotion_wall_ms: u64,
    /// Cohort I/O and failed scoring time without a separately recorded result.
    #[serde(default)]
    pub promotion_overhead_ms: u64,
    /// Opening authoritative baseline call (accepted baseline records reuse verification).
    #[serde(default)]
    pub opening_baseline_ms: u64,
    /// Setup, residual extraction, accepted-creature I/O and other unallocated wall time.
    #[serde(default)]
    pub setup_and_other_ms: u64,
    /// Diagnostic screen promotes measured against exact scores.
    #[serde(default)]
    pub diagnostic_screen_false_positives: u64,
    /// Seeded diagnostic bypasses clearing the exact threshold.
    #[serde(default)]
    pub diagnostic_screen_false_negatives: u64,
    /// Iterations.
    pub iterations: u64,
    /// Accepted winners.
    pub acceptances: u64,
    /// Opening authoritative score.
    pub opening_score: Option<f64>,
    /// Final authoritative score.
    pub final_score: Option<f64>,
    /// `final − opening`.
    pub cumulative_improvement: Option<f64>,
    /// Wall time covered by the journal (ms).
    pub wall_ms: u64,
    /// Improvement per wall-clock hour (`None` when unmeasurable — never 0.0).
    pub improvement_per_wall_hour: Option<f64>,
    /// Wall ms to the first acceptance.
    pub time_to_first_acceptance_ms: Option<u64>,
    /// Candidates generated (pre-graft).
    pub candidates_generated: u64,
    /// Candidates discarded by graft validation.
    pub candidates_discarded: u64,
    /// Candidates screened.
    pub candidates_screened: u64,
    /// Candidates promoted.
    pub candidates_promoted: u64,
    /// Candidates fully scored.
    pub candidates_full_scored: u64,
    /// Candidate searches (iterations) per minute.
    pub searches_per_minute: Option<f64>,
    /// Candidates generated per minute.
    pub candidates_per_minute: Option<f64>,
    /// Total search ms.
    pub search_ms: u64,
    /// Total graft ms.
    pub graft_ms: u64,
    /// Total screen scorer ms.
    pub screen_scorer_ms: u64,
    /// Total full scorer ms.
    pub full_scorer_ms: u64,
    /// `search / (search + scorer)`.
    pub search_time_fraction: Option<f64>,
    /// Screen false positives.
    pub screen_false_positives: u64,
    /// Screen false negatives (from exploratory bypasses).
    pub screen_false_negatives: u64,
    /// False-positive rate among promoted.
    pub screen_false_positive_rate: Option<f64>,
    /// False-negative rate among bypasses.
    pub screen_false_negative_rate: Option<f64>,
    /// Bypass candidates fully scored.
    pub bypass_scored: u64,
    /// Iterations that failed on the scorer.
    pub scorer_failures: u64,
    /// Iterations vetoed by baseline disagreement.
    pub baseline_vetoes: u64,
    /// By strategy.
    pub by_strategy: BTreeMap<String, StrategyStats>,
    /// By search backend.
    pub by_backend: BTreeMap<String, StrategyStats>,
    /// By depth.
    pub by_depth: BTreeMap<String, StrategyStats>,
    /// Accepted gains in order.
    pub accepted: Vec<AcceptedGain>,
    /// Score after each acceptance (saturation curve).
    pub score_trajectory: Vec<f64>,
    /// Stop reason.
    pub stop_reason: Option<String>,
    /// The run's candidate-family arms (`forests-arms`, Issue #147), from
    /// the run header. Empty for journals from older builds.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub arms: String,
}

fn bump(map: &mut BTreeMap<String, StrategyStats>, key: &str, f: impl FnOnce(&mut StrategyStats)) {
    f(map.entry(key.to_string()).or_default());
}

/// Build a report from journal lines.
pub fn report_from_lines(lines: &[JournalLine]) -> JournalReport {
    let mut r = JournalReport::default();
    let mut start: Option<u64> = None;
    let mut end: Option<u64> = None;
    let mut deltas: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut deltas_backend: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut deltas_depth: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for line in lines {
        match line {
            JournalLine::RunHeader {
                timestamp_unix,
                arms,
                ..
            } => {
                start.get_or_insert(*timestamp_unix);
                if r.arms.is_empty() {
                    r.arms.clone_from(arms);
                }
            }
            JournalLine::Baseline(b) => {
                if r.opening_score.is_none() {
                    r.opening_baseline_ms = b.scorer_ms;
                }
                r.opening_score.get_or_insert(b.score);
                r.final_score = Some(b.score);
            }
            JournalLine::Experiment(e) => {
                r.iterations += 1;
                end = Some(e.timestamp_unix);
                r.search_ms += e.search_ms;
                r.graft_ms += e.graft_ms;
                r.candidates_generated += e.candidates_generated;
                r.candidates_discarded += e.candidates_discarded;
                r.screen_false_positives += e.screen_false_positives;
                r.screen_false_negatives += e.screen_false_negatives;
                if let Some(s) = &e.screen {
                    r.candidates_screened += s.screened;
                    if !s.diagnostic_only {
                        r.candidates_promoted += s.promoted;
                        r.bypass_scored += s.bypass;
                    }
                    r.screen_scorer_ms += s.scorer_ms;
                }
                if let Some(ranking) = &e.ranking {
                    r.candidates_exact_ranked += ranking.scored;
                    r.exact_candidates_per_iteration.push(ExactIteration {
                        iteration: e.iteration, scored: ranking.scored,
                    });
                    r.ranking_scorer_ms += ranking.scorer_ms;
                    r.diagnostic_screen_false_positives += ranking.screen_false_positives;
                    r.diagnostic_screen_false_negatives += ranking.screen_false_negatives;
                }
                if let Some(f) = &e.full {
                    r.candidates_full_scored += f.scored;
                    r.full_scorer_ms += f.scorer_ms;
                } else {
                    r.full_scorer_ms += e.verification_ms.unwrap_or(0);
                }
                r.verification_scorer_ms += e.verification_ms.unwrap_or(0);
                let measured_calls = e.screen.as_ref().map_or(0, |s| s.scorer_ms)
                    + e.ranking.as_ref().map_or(0, |r| r.scorer_ms)
                    + e.verification_ms.unwrap_or_else(|| e.full.as_ref().map_or(0, |f| f.scorer_ms));
                let promotion_wall = e.scoring_ms.unwrap_or(measured_calls);
                r.promotion_wall_ms += promotion_wall;
                r.promotion_overhead_ms += promotion_wall.saturating_sub(measured_calls);
                if e.scorer_error.is_some() {
                    r.scorer_failures += 1;
                }
                if e.scorer_error
                    .as_deref()
                    .is_some_and(|m| m.contains("disagrees"))
                {
                    r.baseline_vetoes += 1;
                }
                for c in &e.candidates {
                    if e.ranking.is_some() {
                        r.candidates_promoted += u64::from(c.promoted);
                    }
                    if let Some(exact_rank) = c.exact_rank {
                        r.rank_comparisons.push(RankComparison {
                            iteration: e.iteration, id: c.id.clone(), exact_rank,
                            screen_rank: c.screen_rank,
                            screen_promoted: c.screen_promoted, screen_bypass: c.screen_bypass,
                        });
                    }
                    let depth_key = format!("depth{}", c.depth);
                    for (map, key) in [
                        (&mut r.by_strategy, c.strategy.as_str()),
                        (&mut r.by_backend, c.backend.as_str()),
                        (&mut r.by_depth, depth_key.as_str()),
                    ] {
                        bump(map, key, |s| {
                            s.generated += 1;
                            s.promoted += u64::from(c.promoted);
                            s.full_scored += u64::from(c.full_score.is_some());
                            s.exact_ranked += u64::from(c.exact_score.is_some());
                        });
                    }
                    if let Some(d) = c.full_delta {
                        deltas.entry(c.strategy.clone()).or_default().push(d);
                        deltas_backend.entry(c.backend.clone()).or_default().push(d);
                        deltas_depth.entry(depth_key.clone()).or_default().push(d);
                    }
                }
                if e.accepted
                    && let (Some(id), Some(imp)) = (&e.winner, e.improvement)
                {
                    r.acceptances += 1;
                    let at = start.map_or(0, |s| e.timestamp_unix.saturating_sub(s) * 1000);
                    r.time_to_first_acceptance_ms.get_or_insert(at);
                    if let Some(c) = e.candidates.iter().find(|c| &c.id == id) {
                        let depth_key = format!("depth{}", c.depth);
                        for (map, key) in [
                            (&mut r.by_strategy, c.strategy.as_str()),
                            (&mut r.by_backend, c.backend.as_str()),
                            (&mut r.by_depth, depth_key.as_str()),
                        ] {
                            bump(map, key, |s| {
                                s.winners += 1;
                                s.accepted_gain += imp;
                            });
                        }
                        r.accepted.push(AcceptedGain {
                            iteration: e.iteration,
                            id: id.clone(),
                            strategy: c.strategy.clone(),
                            depth: c.depth,
                            improvement: imp,
                            affected_fraction: c.affected_fraction,
                            at_wall_ms: at,
                        });
                    }
                    r.score_trajectory.push(e.baseline_score + imp);
                }
            }
            JournalLine::Summary(s) => {
                r.stop_reason = Some(s.stop_reason.clone());
                r.wall_ms = s.wall_ms;
                r.final_score = Some(s.final_score);
                r.opening_score.get_or_insert(s.opening_score);
            }
        }
    }
    if r.wall_ms == 0
        && let (Some(s), Some(e)) = (start, end)
    {
        r.wall_ms = e.saturating_sub(s) * 1000;
    }
    for (map, d) in [
        (&mut r.by_strategy, &deltas),
        (&mut r.by_backend, &deltas_backend),
        (&mut r.by_depth, &deltas_depth),
    ] {
        for (k, v) in d {
            if let Some(s) = map.get_mut(k)
                && !v.is_empty()
            {
                s.mean_full_delta = Some(v.iter().sum::<f64>() / v.len() as f64);
            }
        }
    }
    if let (Some(o), Some(f)) = (r.opening_score, r.final_score) {
        r.cumulative_improvement = Some(f - o);
        if r.wall_ms > 0 {
            r.improvement_per_wall_hour = Some((f - o) / (r.wall_ms as f64 / 3_600_000.0));
        }
    }
    if r.wall_ms > 0 {
        let minutes = r.wall_ms as f64 / 60_000.0;
        r.searches_per_minute = Some(r.iterations as f64 / minutes);
        r.candidates_per_minute = Some(r.candidates_generated as f64 / minutes);
    }
    let scorer = r.screen_scorer_ms + r.full_scorer_ms + r.ranking_scorer_ms;
    r.setup_and_other_ms = r.wall_ms.saturating_sub(
        r.search_ms + r.graft_ms + r.promotion_wall_ms + r.opening_baseline_ms,
    );
    if r.search_ms + scorer > 0 {
        r.search_time_fraction = Some(r.search_ms as f64 / (r.search_ms + scorer) as f64);
    }
    let screen_promoted: u64 = lines.iter().filter_map(|line| match line {
        JournalLine::Experiment(e) => e.screen.as_ref().filter(|s| !s.diagnostic_only).map(|s| s.promoted),
        _ => None,
    }).sum();
    if screen_promoted > 0 {
        r.screen_false_positive_rate =
            Some(r.screen_false_positives as f64 / screen_promoted as f64);
    }
    if r.bypass_scored > 0 {
        r.screen_false_negative_rate =
            Some(r.screen_false_negatives as f64 / r.bypass_scored as f64);
    }
    r
}

/// Read and report a journal file.
pub fn report_from_journal(path: &Path) -> Result<JournalReport, String> {
    Ok(report_from_lines(&read_journal(path)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::AuthoritativeBaseline;
    use crate::journal::*;
    use crate::patch::{Node, Patch, Provenance};

    fn cand(id: &str, strategy: &str, promoted: bool, full: Option<f64>) -> CandidateRecord {
        CandidateRecord {
            id: id.into(),
            strategy: strategy.into(),
            backend: "cpu".into(),
            depth: 1,
            features: vec![0],
            hidden_sources: Vec::new(),
            predicted_gain: 1.0,
            affected_records: 10,
            affected_fraction: 0.1,
            screen_score: Some(0.5),
            screen_delta: Some(0.01),
            screen_rank: None,
            screen_promoted: None,
            screen_bypass: None,
            exact_score: None,
            exact_delta: None,
            exact_rank: None,
            promoted,
            bypass: !promoted,
            full_score: full.map(|d| 0.5 + d),
            full_delta: full,
            patch: Patch::new(0, Node::stump(0, 0.0, 0.0, 1.0), Provenance::default()),
            combo: Vec::new(),
        }
    }

    #[test]
    fn exact_rank_report_keeps_ranking_verification_and_diagnostics_separate() {
        let mut winner = cand("winner", "histogram-stump", true, Some(0.02));
        winner.exact_score = Some(0.55);
        winner.exact_delta = Some(0.05);
        winner.exact_rank = Some(1);
        winner.screen_rank = Some(2);
        winner.screen_promoted = Some(false);
        winner.screen_bypass = Some(true);
        winner.bypass = false;
        let mut loser = cand("loser", "histogram-stump", false, None);
        loser.exact_score = Some(0.49);
        loser.exact_delta = Some(-0.01);
        loser.exact_rank = Some(2);
        loser.screen_rank = Some(1);
        loser.screen_promoted = Some(true);
        loser.screen_bypass = Some(false);
        loser.bypass = false;
        let experiment = ExperimentRecord {
            ranking_mode: Some("exact".into()),
            ranking: Some(RankingSummary {
                baseline_score: 0.5, scored: 2, cohort_size: 3,
                execution: crate::scorer::CohortExecution::Hosted,
                scorer_ms: 100, screen_false_positives: 1, screen_false_negatives: 1,
            }),
            verification_ms: Some(40), scoring_ms: Some(200),
            iteration: 1, timestamp_unix: 1, incumbent_checksum: "a".into(),
            baseline_score: 0.5, search_backend: "cpu".into(), search_set: "memory-full".into(),
            search_records: 400, search_features: 2, strategies: vec![],
            search_ms: 20, graft_ms: 10, candidates_generated: 2, candidates_discarded: 0,
            discarded: vec![], candidates: vec![winner, loser],
            screen: Some(ScreenSummary {
                diagnostic_only: true, sample_rate: 0.05, sample_phase: 0,
                baseline_score: 0.5, screened: 2, promoted: 1, bypass: 1, scorer_ms: 30,
            }),
            full: Some(FullSummary { baseline_score: 0.5, baseline_drift: 0.0, scored: 1, scorer_ms: 40 }),
            winner: Some("winner".into()), improvement: Some(0.02), accepted: true,
            new_incumbent_checksum: Some("b".into()), screen_false_positives: 0,
            screen_false_negatives: 0, scorer_error: None,
        };
        let line = JournalLine::Experiment(Box::new(experiment));
        let report = report_from_lines(std::slice::from_ref(&line));
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["candidatesExactRanked"], 2);
        assert_eq!(json["exactCandidatesPerIteration"], serde_json::json!([{ "iteration": 1, "scored": 2 }]));
        assert_eq!(json["rankComparisons"][0]["exactRank"], 1);
        assert_eq!(json["rankComparisons"][0]["screenRank"], 2);
        assert_eq!(json["rankComparisons"][1]["screenRank"], 1);
        assert_eq!(json["diagnosticScreenFalseNegatives"], 1);
        assert_eq!(json["rankingScorerMs"], 100);
        assert_eq!(json["verificationScorerMs"], 40);
        assert_eq!(json["promotionOverheadMs"], 30);
        assert_eq!(report.candidates_full_scored, 1);
        assert_eq!(report.bypass_scored, 0);
        assert_eq!(report.screen_false_negative_rate, None);
        assert_eq!(report.by_strategy["histogram-stump"].mean_full_delta, Some(0.02));

        // A body written by the previous release has none of the new fields.
        let mut older = serde_json::to_value(line).unwrap();
        for key in ["rankingMode", "ranking", "verificationMs", "scoringMs"] {
            older.as_object_mut().unwrap().remove(key);
        }
        older["screen"].as_object_mut().unwrap().remove("diagnosticOnly");
        for c in older["candidates"].as_array_mut().unwrap() {
            for key in ["exactScore", "exactDelta", "exactRank", "screenRank", "screenPromoted", "screenBypass"] {
                c.as_object_mut().unwrap().remove(key);
            }
        }
        let old_line: JournalLine = serde_json::from_value(older).unwrap();
        let old_report = report_from_lines(&[old_line]);
        assert_eq!(old_report.candidates_full_scored, 1);
        assert_eq!(old_report.screen_scorer_ms, 30);
        assert_eq!(old_report.full_scorer_ms, 40);
        assert_eq!(old_report.by_strategy["histogram-stump"].mean_full_delta, Some(0.02));
    }

    #[test]
    fn aggregates_strategies_and_economics() {
        let baseline = AuthoritativeBaseline {
            incumbent_checksum: "a".into(),
            score: 0.5,
            error: 0.5,
            complexity_penalty: 0.0,
            record_count: 10,
            scorer_identity: "t".into(),
            cost_name: None,
            scorer_backend: None,
            corpus_identity: "c".into(),
            corpus_record_count: 10,
            local_mse: None,
            parity: "skipped".into(),
            scorer_ms: 1,
            created_at_unix: 0,
        };
        let exp = ExperimentRecord {
            ranking_mode: None,
            ranking: None,
            verification_ms: None,
            scoring_ms: None,
            iteration: 1,
            timestamp_unix: 1060,
            incumbent_checksum: "a".into(),
            baseline_score: 0.5,
            search_backend: "cpu".into(),
            search_set: "memory-full".into(),
            search_records: 10,
            search_features: 2,
            strategies: vec!["histogram-stump".into()],
            search_ms: 100,
            graft_ms: 10,
            candidates_generated: 3,
            candidates_discarded: 0,
            discarded: Vec::new(),
            candidates: vec![
                cand("w", "histogram-stump", true, Some(0.02)),
                cand("l", "histogram-stump", true, Some(-0.01)),
                cand("b", "random-stump", false, Some(0.005)),
            ],
            screen: Some(ScreenSummary {
                diagnostic_only: false,
                sample_rate: 0.1,
                sample_phase: 0,
                baseline_score: 0.5,
                screened: 3,
                promoted: 2,
                bypass: 1,
                scorer_ms: 50,
            }),
            full: Some(FullSummary {
                baseline_score: 0.5,
                baseline_drift: 0.0,
                scored: 3,
                scorer_ms: 250,
            }),
            winner: Some("w".into()),
            improvement: Some(0.02),
            accepted: true,
            new_incumbent_checksum: Some("b".into()),
            screen_false_positives: 1,
            screen_false_negatives: 1,
            scorer_error: None,
        };
        let lines = vec![
            JournalLine::RunHeader {
                timestamp_unix: 1000,
                seed: 1,
                seed_source: "supplied".into(),
                version: "v".into(),
                source_creature: "c".into(),
                incumbent_checksum: "a".into(),
                corpus_identity: "c".into(),
                bin_cache: "c:256".into(),
                config: serde_json::json!({}),
                arms: "refine-leaves=8:drawn,refit-every=0:set".into(),
            },
            JournalLine::Baseline(baseline.clone()),
            JournalLine::Experiment(Box::new(exp)),
            JournalLine::Baseline(AuthoritativeBaseline {
                score: 0.52,
                incumbent_checksum: "b".into(),
                ..baseline
            }),
            JournalLine::Summary(SummaryRecord {
                timestamp_unix: 1120,
                stop_reason: "timeout".into(),
                iterations: 1,
                acceptances: 1,
                opening_score: 0.5,
                final_score: 0.52,
                wall_ms: 120_000,
                final_checksum: "b".into(),
            }),
        ];
        let r = report_from_lines(&lines);
        assert_eq!(r.iterations, 1);
        assert_eq!(r.arms, "refine-leaves=8:drawn,refit-every=0:set");
        assert_eq!(r.acceptances, 1);
        assert!((r.cumulative_improvement.unwrap() - 0.02).abs() < 1e-12);
        assert!((r.improvement_per_wall_hour.unwrap() - 0.6).abs() < 1e-9);
        assert_eq!(r.time_to_first_acceptance_ms, Some(60_000));
        assert_eq!(r.by_strategy["histogram-stump"].winners, 1);
        assert_eq!(r.by_strategy["random-stump"].winners, 0);
        assert_eq!(r.by_strategy["random-stump"].full_scored, 1);
        assert_eq!(r.screen_false_positive_rate, Some(0.5));
        assert_eq!(r.screen_false_negative_rate, Some(1.0));
        assert!((r.search_time_fraction.unwrap() - 0.25).abs() < 1e-9);
        assert_eq!(r.accepted[0].affected_fraction, 0.1);
        assert_eq!(r.stop_reason.as_deref(), Some("timeout"));
        assert_eq!(r.score_trajectory, vec![0.52]);
        let json = serde_json::to_string(&r).unwrap();
        let back: JournalReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.by_strategy, r.by_strategy);
        assert_eq!(back.accepted, r.accepted);
        assert_eq!(back.iterations, r.iterations);
    }

    #[test]
    fn empty_journal_reports_none_not_zero() {
        let r = report_from_lines(&[]);
        assert_eq!(r.improvement_per_wall_hour, None);
        assert_eq!(r.searches_per_minute, None);
    }
}

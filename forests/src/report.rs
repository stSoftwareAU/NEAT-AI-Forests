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

/// The report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct JournalReport {
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
            JournalLine::RunHeader { timestamp_unix, .. } => {
                start.get_or_insert(*timestamp_unix);
            }
            JournalLine::Baseline(b) => {
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
                    r.candidates_promoted += s.promoted;
                    r.bypass_scored += s.bypass;
                    r.screen_scorer_ms += s.scorer_ms;
                }
                if let Some(f) = &e.full {
                    r.candidates_full_scored += f.scored;
                    r.full_scorer_ms += f.scorer_ms;
                }
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
    let scorer = r.screen_scorer_ms + r.full_scorer_ms;
    if r.search_ms + scorer > 0 {
        r.search_time_fraction = Some(r.search_ms as f64 / (r.search_ms + scorer) as f64);
    }
    if r.candidates_promoted > 0 {
        r.screen_false_positive_rate =
            Some(r.screen_false_positives as f64 / r.candidates_promoted as f64);
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
            predicted_gain: 1.0,
            affected_records: 10,
            affected_fraction: 0.1,
            screen_score: Some(0.5),
            screen_delta: Some(0.01),
            promoted,
            bypass: !promoted,
            full_score: full.map(|d| 0.5 + d),
            full_delta: full,
            patch: Patch::new(0, Node::stump(0, 0.0, 0.0, 1.0), Provenance::default()),
            combo: Vec::new(),
        }
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

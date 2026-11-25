//! Candidate ranking and authoritative promotion.
//!
//! Legacy mode screens a sample and fully scores the survivors plus seeded
//! bypasses together with the incumbent. Exact mode ranks every valid candidate
//! in one full-corpus `--cohort` call, then verifies the top eligible candidate
//! with its parent in a separate ordinary full-corpus call. An optional sample
//! comparison in exact mode records hypothetical decisions and excludes nobody.
//!
//! Only ordinary full results can accept: the same-call delta must exceed the
//! improvement threshold, and the parent must agree with the stored baseline.
//! Cohort ranking scores are never acceptance results or shared-cache verdicts.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::Rng;
use rand::rngs::StdRng;

use crate::candidates::Candidate;
use crate::scorer::{
    CohortExecution, DirectoryScorer, ScoreResult, ScorerError, ScorerMode, validate_scores,
};

/// Promotion controls.
#[derive(Debug, Clone, PartialEq)]
pub struct PromoteConfig {
    /// Use exact cohort ranking (capability resolved once by the caller).
    pub exact_ranking: bool,
    /// Number of records in the canonical corpus, checked on full results.
    pub corpus_records: u64,
    /// Screen sample rate (`None` = no screen); diagnostic only under exact ranking.
    pub screen_sample_rate: Option<f64>,
    /// Phase for the screen stride (vary per iteration).
    pub screen_phase: u64,
    /// Promote when `sampled Δ > screen_threshold`.
    pub screen_threshold: f64,
    /// Max promoted.
    pub promote_count: usize,
    /// Bypass quota of screen rejects.
    pub explore_quota: usize,
    /// Strict authoritative threshold.
    pub min_improvement: f64,
    /// Authoritative baseline score the same-call baseline must match.
    pub authoritative_baseline: f64,
    /// Tolerated |same-call − authoritative|.
    pub baseline_drift_epsilon: f64,
    /// Keep cohort directories.
    pub preserve_dirs: bool,
}

/// Screen results.
#[derive(Debug, Clone, PartialEq)]
pub struct ScreenOutcome {
    /// One-based hypothetical screen ranks, with candidate-id tie breaking.
    pub ranks: BTreeMap<String, u64>,
    /// Sample rate.
    pub rate: f64,
    /// Phase.
    pub phase: u64,
    /// Sampled baseline score.
    pub baseline_score: f64,
    /// Sampled scores by candidate id.
    pub scores: BTreeMap<String, ScoreResult>,
    /// Promoted ids (in rank order).
    pub promoted: Vec<String>,
    /// Bypass ids.
    pub bypass: Vec<String>,
    /// Scorer ms.
    pub scorer_ms: u64,
}

/// Full-corpus cohort ranking; separate from authoritative verification.
#[derive(Debug, Clone, PartialEq)]
pub struct RankingOutcome {
    /// Same-call incumbent score used for ranking deltas.
    pub baseline_score: f64,
    /// Exact scores, including the incumbent.
    pub scores: BTreeMap<String, ScoreResult>,
    /// One-based candidate ranks, with candidate-id tie breaking.
    pub ranks: BTreeMap<String, u64>,
    /// Hosted cohort or documented directory fallback.
    pub execution: CohortExecution,
    /// Ranking scorer wall time.
    pub scorer_ms: u64,
    /// Hypothetical screen promotes that did not clear the exact threshold.
    pub screen_false_positives: u64,
    /// Seeded bypass candidates above the exact threshold.
    pub screen_false_negatives: u64,
}

/// Full-corpus results.
#[derive(Debug, Clone, PartialEq)]
pub struct FullOutcome {
    /// Same-call baseline.
    pub baseline_score: f64,
    /// `baseline_score − authoritative`.
    pub baseline_drift: f64,
    /// Full scores by id.
    pub scores: BTreeMap<String, ScoreResult>,
    /// Scorer ms.
    pub scorer_ms: u64,
}

/// Complete outcome of one cohort.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PromotionOutcome {
    /// Exact ranking, when enabled.
    pub ranking: Option<RankingOutcome>,
    /// Standalone verification time, including failed attempts.
    pub verification_ms: Option<u64>,
    /// Standalone verification failure; ranking scores must not become verdicts.
    pub verification_error: Option<ScorerError>,
    /// Screen stage (if run).
    pub screen: Option<ScreenOutcome>,
    /// Full stage (if any candidate reached it).
    pub full: Option<FullOutcome>,
    /// Winner id and its authoritative improvement.
    pub winner: Option<(String, f64)>,
    /// Promoted candidates that failed the authoritative threshold.
    pub false_positives: u64,
    /// Bypass candidates that cleared the authoritative threshold.
    pub false_negatives: u64,
    /// Why no acceptance happened despite a full run (baseline disagreement).
    pub veto: Option<String>,
}

/// Resolve capability once per run. A probe error fails closed; an older
/// binary explicitly falls back to the legacy path.
pub(crate) fn resolve_exact_ranking(
    requested: crate::config::RankingMode,
    scorer: &dyn DirectoryScorer,
) -> Result<bool, ScorerError> {
    if requested == crate::config::RankingMode::Legacy {
        return Ok(false);
    }
    let supported = scorer.supports_cohort()?;
    if !supported {
        crate::log::warn("scorer lacks --cohort; using legacy screen and promotion");
    }
    Ok(supported)
}

fn ranked_ids(scores: &BTreeMap<String, ScoreResult>) -> Vec<String> {
    let mut ids: Vec<_> = scores
        .keys()
        .filter(|id| id.as_str() != "baseline")
        .cloned()
        .collect();
    ids.sort_by(|a, b| scores[b].score.total_cmp(&scores[a].score).then(a.cmp(b)));
    ids
}

fn ranks(scores: &BTreeMap<String, ScoreResult>) -> BTreeMap<String, u64> {
    ranked_ids(scores)
        .into_iter()
        .enumerate()
        .map(|(i, id)| (id, i as u64 + 1))
        .collect()
}

fn write_cohort(
    dir: &Path,
    incumbent: &neat_core::CreatureExport,
    candidates: &[&Candidate],
) -> Result<(), String> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let write = |name: &str, c: &neat_core::CreatureExport| -> Result<(), String> {
        let text = neat_core::creature_to_json(c).map_err(|e| e.to_string())?;
        std::fs::write(dir.join(format!("{name}.json")), text).map_err(|e| e.to_string())
    };
    write("baseline", incumbent)?;
    for c in candidates {
        if c.id == "baseline" {
            return Err("candidate id `baseline` is reserved".into());
        }
        write(&c.id, &c.creature)?;
    }
    Ok(())
}

fn cleanup(dir: &Path, preserve: bool) {
    if !preserve {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Rank and verify one cohort under the selected promotion policy.
pub fn screen_and_promote(
    candidates: &[Candidate],
    incumbent: &neat_core::CreatureExport,
    training_dir: &Path,
    scorer: &dyn DirectoryScorer,
    work_dir: &Path,
    cfg: &PromoteConfig,
    rng: &mut StdRng,
) -> Result<PromotionOutcome, ScorerError> {
    let mut outcome = PromotionOutcome::default();
    if candidates.is_empty() {
        return Ok(outcome);
    }
    let mut to_full: Vec<&Candidate> = Vec::new();
    let mut bypass_ids: Vec<String> = Vec::new();
    if let Some(rate) = cfg.screen_sample_rate
        && (!cfg.exact_ranking || cfg.explore_quota > 0)
    {
        let dir: PathBuf = work_dir.join("screen");
        let refs: Vec<&Candidate> = candidates.iter().collect();
        write_cohort(&dir, incumbent, &refs).map_err(ScorerError::Spawn)?;
        let started = Instant::now();
        let result = scorer.score_directory(
            &dir,
            training_dir,
            ScorerMode::Sample {
                rate,
                phase: cfg.screen_phase,
            },
        );
        cleanup(&dir, cfg.preserve_dirs);
        let scores = result?;
        validate_scores(&scores, candidates.iter().map(|c| c.id.as_str()), None)?;
        let base = scores["baseline"].score;
        let mut ranked: Vec<(&Candidate, f64)> = candidates
            .iter()
            .filter_map(|c| scores.get(&c.id).map(|s| (c, s.score - base)))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.id.cmp(&b.0.id)));
        let promoted: Vec<&Candidate> = ranked
            .iter()
            .filter(|(_, d)| *d > cfg.screen_threshold)
            .take(cfg.promote_count)
            .map(|(c, _)| *c)
            .collect();
        let promoted_ids: Vec<String> = promoted.iter().map(|c| c.id.clone()).collect();
        let mut rejects: Vec<&Candidate> = ranked
            .iter()
            .map(|(c, _)| *c)
            .filter(|c| !promoted_ids.contains(&c.id))
            .collect();
        // Exploratory bypass: random screen rejects go to full scoring anyway.
        let mut bypass: Vec<&Candidate> = Vec::new();
        for _ in 0..cfg.explore_quota.min(rejects.len()) {
            let i = rng.random_range(0..rejects.len());
            bypass.push(rejects.swap_remove(i));
        }
        bypass_ids = bypass.iter().map(|c| c.id.clone()).collect();
        to_full.extend(promoted);
        to_full.extend(bypass);
        outcome.screen = Some(ScreenOutcome {
            ranks: ranks(&scores),
            rate,
            phase: cfg.screen_phase,
            baseline_score: base,
            scores,
            promoted: promoted_ids,
            bypass: bypass_ids.clone(),
            scorer_ms: started.elapsed().as_millis() as u64,
        });
    } else {
        to_full.extend(candidates.iter());
    }
    if cfg.exact_ranking {
        let dir = work_dir.join("ranking");
        let refs: Vec<_> = candidates.iter().collect();
        write_cohort(&dir, incumbent, &refs).map_err(ScorerError::Spawn)?;
        let started = Instant::now();
        let result = scorer.score_cohort(&dir, training_dir);
        let scorer_ms = started.elapsed().as_millis() as u64;
        cleanup(&dir, cfg.preserve_dirs);
        let result = result?;
        let scores = result.scores;
        validate_scores(
            &scores,
            candidates.iter().map(|c| c.id.as_str()),
            Some(cfg.corpus_records),
        )?;
        let base = scores["baseline"].score;
        let ordered = ranked_ids(&scores);
        let eligible = ordered
            .first()
            .filter(|id| scores[*id].score - base > cfg.min_improvement);
        to_full = eligible
            .into_iter()
            .map(|id| candidates.iter().find(|c| &c.id == id).unwrap())
            .collect();
        let mut ranking = RankingOutcome {
            baseline_score: base,
            ranks: ranks(&scores),
            scores,
            execution: result.execution,
            scorer_ms,
            screen_false_positives: 0,
            screen_false_negatives: 0,
        };
        if let Some(screen) = &outcome.screen {
            ranking.screen_false_positives = screen
                .promoted
                .iter()
                .filter(|id| ranking.scores[*id].score - base <= cfg.min_improvement)
                .count() as u64;
            ranking.screen_false_negatives = screen
                .bypass
                .iter()
                .filter(|id| ranking.scores[*id].score - base > cfg.min_improvement)
                .count() as u64;
        }
        outcome.ranking = Some(ranking);
    }
    if to_full.is_empty() {
        return Ok(outcome);
    }
    let dir = work_dir.join("full");
    write_cohort(&dir, incumbent, &to_full).map_err(ScorerError::Spawn)?;
    let started = Instant::now();
    let result = scorer.score_directory(&dir, training_dir, ScorerMode::Full);
    let scorer_ms = started.elapsed().as_millis() as u64;
    cleanup(&dir, cfg.preserve_dirs);
    let result = result.and_then(|scores| {
        validate_scores(
            &scores,
            to_full.iter().map(|c| c.id.as_str()),
            Some(cfg.corpus_records),
        )?;
        Ok(scores)
    });
    if cfg.exact_ranking {
        outcome.verification_ms = Some(scorer_ms);
    }
    let scores = match result {
        Ok(scores) => scores,
        Err(e) if cfg.exact_ranking => {
            outcome.verification_error = Some(e);
            return Ok(outcome);
        }
        Err(e) => return Err(e),
    };
    let base = scores["baseline"].score;
    let drift = base - cfg.authoritative_baseline;
    let mut best: Option<(String, f64)> = None;
    for c in &to_full {
        let Some(s) = scores.get(&c.id) else { continue };
        let delta = s.score - base;
        let cleared = delta > cfg.min_improvement;
        if !cfg.exact_ranking {
            if bypass_ids.contains(&c.id) {
                if cleared {
                    outcome.false_negatives += 1;
                }
            } else if !cleared {
                outcome.false_positives += 1;
            }
        }
        if cleared
            && best
                .as_ref()
                .is_none_or(|(bid, bd)| delta > *bd || (delta == *bd && c.id < *bid))
        {
            best = Some((c.id.clone(), delta));
        }
    }
    outcome.full = Some(FullOutcome {
        baseline_score: base,
        baseline_drift: drift,
        scores,
        scorer_ms,
    });
    if drift.abs() > cfg.baseline_drift_epsilon {
        outcome.veto = Some(format!(
            "same-call full baseline {base} disagrees with authoritative baseline {} by {drift:e} (> {}); no acceptance",
            cfg.authoritative_baseline, cfg.baseline_drift_epsilon
        ));
        return Ok(outcome);
    }
    outcome.winner = best;
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::fake::{CohortScorer, LocalMseScorer, Verification};
    use crate::corpus::write_bin_file;
    use crate::graft::fixtures::identity_creature;
    use crate::incumbent::Incumbent;
    use crate::patch::{Node, Patch, Provenance};
    use rand::SeedableRng;

    /// Corpus where target = x0 + 0.3·[x1 > 0]; identity incumbent has residual 0.3 on the right.
    fn setup() -> (tempfile::TempDir, Incumbent, Vec<Candidate>) {
        let tmp = tempfile::tempdir().unwrap();
        let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..400)
            .map(|i| {
                let x0 = (i % 13) as f32 / 13.0;
                let x1 = if (i * 7) % 5 < 2 { 1.0 } else { -1.0 };
                (vec![x0, x1], vec![x0 + if x1 > 0.0 { 0.3 } else { 0.0 }])
            })
            .collect();
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let inc = Incumbent::from_creature(identity_creature(2, 1), "t").unwrap();
        let mk = |root: Node, tag: &str| {
            let p = Patch::new(
                0,
                root,
                Provenance {
                    strategy: tag.into(),
                    ..Default::default()
                },
            );
            let id = p.id();
            let creature = crate::graft::graft_patch(&inc.creature, &p)
                .unwrap()
                .creature;
            Candidate {
                id,
                patch: p,
                combo: Vec::new(),
                creature,
                added_uuids: Vec::new(),
            }
        };
        let cands = vec![
            mk(Node::stump(1, 0.0, 0.0, 0.3), "winner"),
            mk(Node::stump(1, 0.0, 0.0, 0.1), "partial"),
            mk(Node::stump(1, 0.0, 0.0, -0.3), "loser"),
            mk(Node::stump(0, 0.5, -0.3, 0.0), "noise"),
        ];
        (tmp, inc, cands)
    }

    fn cfg(base: f64) -> PromoteConfig {
        PromoteConfig {
            exact_ranking: false,
            corpus_records: 400,
            screen_sample_rate: Some(0.5),
            screen_phase: 0,
            screen_threshold: 0.0,
            promote_count: 2,
            explore_quota: 1,
            min_improvement: 1e-6,
            authoritative_baseline: base,
            baseline_drift_epsilon: 1e-9,
            preserve_dirs: false,
        }
    }

    fn baseline_score(tmp: &Path, inc: &Incumbent) -> f64 {
        let scorer = LocalMseScorer::new();
        let dir = tmp.join("b");
        write_cohort(&dir, &inc.creature, &[]).unwrap();
        scorer.score_directory(&dir, tmp, ScorerMode::Full).unwrap()["baseline"].score
    }

    #[test]
    fn winner_is_accepted_only_on_full_corpus_and_journal_separates_stages() {
        let (tmp, inc, cands) = setup();
        let base = baseline_score(tmp.path(), &inc);
        let scorer = LocalMseScorer::new();
        let mut rng = StdRng::seed_from_u64(1);
        let out = screen_and_promote(
            &cands,
            &inc.creature,
            tmp.path(),
            &scorer,
            &tmp.path().join("w"),
            &cfg(base),
            &mut rng,
        )
        .unwrap();
        let screen = out.screen.as_ref().unwrap();
        assert_eq!(
            screen.promoted,
            vec![cands[0].id.clone(), cands[1].id.clone()]
        );
        assert_eq!(screen.bypass.len(), 1);
        let full = out.full.as_ref().unwrap();
        assert!((full.baseline_drift).abs() < 1e-12);
        assert_eq!(full.scores.len(), 4); // baseline + 2 promoted + 1 bypass
        let (id, delta) = out.winner.clone().unwrap();
        assert_eq!(id, cands[0].id);
        assert!((delta - 0.036).abs() < 1e-3, "{delta}"); // 0.3² × 0.4 of records
        let calls = scorer.calls.borrow();
        assert_eq!(calls[0].0, "sample");
        assert_eq!(calls[1].0, "full");
        assert!(!tmp.path().join("w/full").exists());
    }

    #[test]
    fn screen_false_positive_is_caught_and_losers_rejected() {
        let (tmp, inc, cands) = setup();
        let base = baseline_score(tmp.path(), &inc);
        // Optimistic screen: every candidate looks +0.5 better than it is.
        let scorer = LocalMseScorer {
            sample_bias: 0.5,
            ..Default::default()
        };
        let mut rng = StdRng::seed_from_u64(1);
        let losers = vec![cands[2].clone(), cands[3].clone()];
        let out = screen_and_promote(
            &losers,
            &inc.creature,
            tmp.path(),
            &scorer,
            &tmp.path().join("w"),
            &cfg(base),
            &mut rng,
        )
        .unwrap();
        assert_eq!(out.screen.as_ref().unwrap().promoted.len(), 2);
        assert!(out.winner.is_none());
        assert_eq!(out.false_positives, 2);
    }

    #[test]
    fn scorer_failures_and_baseline_disagreement_block_acceptance() {
        let (tmp, inc, cands) = setup();
        let mut rng = StdRng::seed_from_u64(1);
        let failing = LocalMseScorer {
            fail_with: Some("crash".into()),
            ..Default::default()
        };
        assert!(
            screen_and_promote(
                &cands,
                &inc.creature,
                tmp.path(),
                &failing,
                &tmp.path().join("w"),
                &cfg(0.9),
                &mut rng
            )
            .is_err()
        );
        let malformed = LocalMseScorer {
            raw_output: Some(r#"{"nope":{"score":1,"error":0}}"#.into()),
            ..Default::default()
        };
        assert_eq!(
            screen_and_promote(
                &cands,
                &inc.creature,
                tmp.path(),
                &malformed,
                &tmp.path().join("w"),
                &cfg(0.9),
                &mut rng
            )
            .unwrap_err(),
            ScorerError::MissingBaseline
        );
        // Authoritative baseline that the same-call baseline cannot match → veto.
        let scorer = LocalMseScorer::new();
        let out = screen_and_promote(
            &cands,
            &inc.creature,
            tmp.path(),
            &scorer,
            &tmp.path().join("w"),
            &cfg(0.123),
            &mut rng,
        )
        .unwrap();
        assert!(out.winner.is_none());
        assert!(out.veto.unwrap().contains("disagrees"));
    }

    #[test]
    fn no_screen_scores_everything_fully() {
        let (tmp, inc, cands) = setup();
        let base = baseline_score(tmp.path(), &inc);
        let scorer = LocalMseScorer::new();
        let mut rng = StdRng::seed_from_u64(1);
        let c = PromoteConfig {
            screen_sample_rate: None,
            ..cfg(base)
        };
        let out = screen_and_promote(
            &cands,
            &inc.creature,
            tmp.path(),
            &scorer,
            &tmp.path().join("w"),
            &c,
            &mut rng,
        )
        .unwrap();
        assert!(out.screen.is_none());
        assert_eq!(out.full.unwrap().scores.len(), 5);
        assert_eq!(out.winner.unwrap().0, cands[0].id);
    }

    #[test]
    fn an_incomplete_cohort_is_an_error_instead_of_silently_dropping_candidates() {
        let (tmp, inc, cands) = setup();
        let scorer = LocalMseScorer {
            raw_output: Some(r#"{"baseline":{"score":0.9,"error":0.1,"recordCount":400}}"#.into()),
            ..Default::default()
        };
        let result = screen_and_promote(
            &cands,
            &inc.creature,
            tmp.path(),
            &scorer,
            &tmp.path().join("w"),
            &PromoteConfig {
                screen_sample_rate: None,
                ..cfg(0.9)
            },
            &mut StdRng::seed_from_u64(1),
        );
        assert!(
            matches!(result, Err(ScorerError::Malformed(ref message)) if message.contains(cands.iter().map(|c| c.id.as_str()).min().unwrap())),
            "{result:?}"
        );
    }

    #[test]
    fn exact_ranking_finds_the_winner_even_when_the_screen_rejects_every_candidate() {
        let (tmp, inc, cands) = setup();
        let base = baseline_score(tmp.path(), &inc);
        let scorer = LocalMseScorer {
            cohort_supported: true,
            sample_bias: -0.5,
            ..Default::default()
        };
        let config = PromoteConfig {
            exact_ranking: true,
            explore_quota: 1,
            ..cfg(base)
        };
        let out = screen_and_promote(
            &cands,
            &inc.creature,
            tmp.path(),
            &scorer,
            &tmp.path().join("w"),
            &config,
            &mut StdRng::seed_from_u64(1),
        )
        .unwrap();
        assert!(out.screen.as_ref().unwrap().promoted.is_empty());
        assert!(!out.screen.as_ref().unwrap().bypass.contains(&cands[0].id));
        assert_eq!(out.winner.as_ref().map(|w| &w.0), Some(&cands[0].id));
        let calls = scorer.calls.borrow();
        let exact = calls
            .iter()
            .find(|(mode, _)| mode == "cohort")
            .expect("an exact cohort call");
        let mut expected: Vec<String> = cands.iter().map(|c| c.id.clone()).collect();
        expected.push("baseline".into());
        expected.sort();
        assert_eq!(exact.1, expected);
        let verification = calls.last().unwrap();
        assert_eq!(verification.0, "full");
        let mut pair = vec!["baseline".to_string(), cands[0].id.clone()];
        pair.sort();
        assert_eq!(verification.1, pair);
        assert_eq!(out.full.unwrap().scores.len(), 2);
    }

    #[test]
    fn tied_exact_scores_are_ranked_by_candidate_id_and_zero_quota_skips_diagnostics() {
        let (tmp, inc, mut cands) = setup();
        let mut twin = cands[0].clone();
        twin.id = "000-tied-winner".into();
        cands.push(twin);
        cands.reverse();
        let base = baseline_score(tmp.path(), &inc);
        let scorer = CohortScorer::default();
        let out = screen_and_promote(
            &cands,
            &inc.creature,
            tmp.path(),
            &scorer,
            &tmp.path().join("w"),
            &PromoteConfig {
                exact_ranking: true,
                explore_quota: 0,
                ..cfg(base)
            },
            &mut StdRng::seed_from_u64(1),
        )
        .unwrap();
        assert_eq!(out.winner.as_ref().unwrap().0, "000-tied-winner");
        let ranking = out.ranking.unwrap();
        assert_eq!(ranking.ranks["000-tied-winner"], 1);
        assert_eq!(ranking.ranks[&cands.last().unwrap().id], 2);
        assert_eq!(ranking.scores.len(), cands.len() + 1);
        assert!(out.screen.is_none());
        assert_eq!(scorer.local.calls.borrow().len(), 2);
    }

    #[test]
    fn standalone_verification_is_required_even_after_a_good_exact_ranking() {
        let (tmp, inc, cands) = setup();
        let base = baseline_score(tmp.path(), &inc);
        for verification in [
            Verification::Normal,
            Verification::Failed,
            Verification::Drift,
            Verification::Regression,
            Verification::MissingCandidate,
            Verification::Sampled,
            Verification::WrongCount,
        ] {
            let scorer = CohortScorer {
                verification,
                ..Default::default()
            };
            let out = screen_and_promote(
                &cands,
                &inc.creature,
                tmp.path(),
                &scorer,
                &tmp.path().join("w"),
                &PromoteConfig {
                    exact_ranking: true,
                    explore_quota: 0,
                    ..cfg(base)
                },
                &mut StdRng::seed_from_u64(1),
            )
            .unwrap();
            assert_eq!(out.ranking.as_ref().unwrap().ranks[&cands[0].id], 1);
            assert!(out.verification_ms.is_some());
            match verification {
                Verification::Normal => assert_eq!(out.winner.unwrap().0, cands[0].id),
                Verification::Drift => {
                    assert!(out.winner.is_none());
                    assert!(out.veto.unwrap().contains("disagrees"));
                    assert!(out.full.as_ref().unwrap().baseline_drift.abs() > 0.09);
                }
                Verification::Regression => {
                    assert!(out.winner.is_none());
                    assert!(out.full.unwrap().scores[&cands[0].id].score < base);
                }
                Verification::Failed => assert!(matches!(out.verification_error,
                    Some(ScorerError::Failed { ref stderr, .. }) if stderr.contains("verification fixture failure"))),
                Verification::MissingCandidate => assert!(matches!(out.verification_error,
                    Some(ScorerError::Malformed(ref m)) if m.contains(&cands[0].id))),
                Verification::Sampled | Verification::WrongCount => {
                    assert!(matches!(out.verification_error,
                    Some(ScorerError::Malformed(ref m)) if m.contains("did not score the full corpus")))
                }
            }
        }
    }

    #[test]
    fn exact_ranking_with_no_eligible_candidate_does_not_spend_a_verification_call() {
        let (tmp, inc, cands) = setup();
        let base = baseline_score(tmp.path(), &inc);
        let scorer = CohortScorer::default();
        let out = screen_and_promote(
            &cands[2..],
            &inc.creature,
            tmp.path(),
            &scorer,
            &tmp.path().join("w"),
            &PromoteConfig {
                exact_ranking: true,
                explore_quota: 0,
                ..cfg(base)
            },
            &mut StdRng::seed_from_u64(1),
        )
        .unwrap();
        assert_eq!(out.ranking.unwrap().scores.len(), 3);
        assert!(out.winner.is_none());
        assert!(out.full.is_none());
        assert!(out.verification_ms.is_none());
        assert_eq!(scorer.local.calls.borrow().len(), 1);
    }

    #[test]
    fn malformed_or_incomplete_exact_results_fail_closed_before_verification() {
        let (tmp, inc, cands) = setup();
        let base = baseline_score(tmp.path(), &inc);
        for raw in [
            "not json",
            r#"{"baseline":{"score":0.9,"error":0.1,"recordCount":400}}"#,
        ] {
            let scorer = LocalMseScorer {
                cohort_supported: true,
                raw_output: Some(raw.into()),
                ..Default::default()
            };
            let result = screen_and_promote(
                &cands,
                &inc.creature,
                tmp.path(),
                &scorer,
                &tmp.path().join("w"),
                &PromoteConfig {
                    exact_ranking: true,
                    explore_quota: 0,
                    ..cfg(base)
                },
                &mut StdRng::seed_from_u64(1),
            );
            assert!(
                matches!(result, Err(ScorerError::Malformed(_))),
                "{raw}: {result:?}"
            );
            assert!(scorer.calls.borrow().iter().all(|(mode, _)| mode != "full"));
        }
        let supported = CohortScorer::default();
        let good = screen_and_promote(
            &cands,
            &inc.creature,
            tmp.path(),
            &supported,
            &tmp.path().join("good"),
            &PromoteConfig {
                exact_ranking: true,
                explore_quota: 0,
                ..cfg(base)
            },
            &mut StdRng::seed_from_u64(1),
        )
        .unwrap();
        assert_eq!(good.winner.unwrap().0, cands[0].id);
    }

    #[test]
    fn documented_cohort_fallback_still_requires_ordinary_verification() {
        let (tmp, inc, cands) = setup();
        let base = baseline_score(tmp.path(), &inc);
        for verification in [Verification::Normal, Verification::Failed] {
            let scorer = CohortScorer {
                verification,
                fallback_reason: Some("cost MAE is not hosted".into()),
                ..Default::default()
            };
            let out = screen_and_promote(
                &cands,
                &inc.creature,
                tmp.path(),
                &scorer,
                &tmp.path().join("w"),
                &PromoteConfig {
                    exact_ranking: true,
                    explore_quota: 0,
                    ..cfg(base)
                },
                &mut StdRng::seed_from_u64(1),
            )
            .unwrap();
            assert!(
                matches!(out.ranking.unwrap().execution, CohortExecution::DirectoryFallback { ref reason } if reason.contains("MAE"))
            );
            match verification {
                Verification::Normal => assert_eq!(out.winner.unwrap().0, cands[0].id),
                Verification::Failed => {
                    assert!(out.winner.is_none());
                    assert!(out.full.is_none());
                    assert!(matches!(
                        out.verification_error,
                        Some(ScorerError::Failed { .. })
                    ));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn capability_resolution_preserves_legacy_and_fails_closed_on_probe_errors() {
        use crate::config::RankingMode;
        let scorer = CohortScorer {
            supported: false,
            ..Default::default()
        };
        assert!(!resolve_exact_ranking(RankingMode::Legacy, &scorer).unwrap());
        assert_eq!(scorer.probes.get(), 0);
        assert!(!resolve_exact_ranking(RankingMode::Exact, &scorer).unwrap());
        assert_eq!(scorer.probes.get(), 1);
        assert!(matches!(
            resolve_exact_ranking(
                RankingMode::Exact,
                &crate::scorer::ExternalScorer::new("/missing/scorer")
            ),
            Err(ScorerError::Spawn(_))
        ));
        let supported = CohortScorer::default();
        assert!(resolve_exact_ranking(RankingMode::Exact, &supported).unwrap());
    }
}

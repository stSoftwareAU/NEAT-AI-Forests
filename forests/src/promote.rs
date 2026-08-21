//! Two-phase screening and authoritative promotion (Issue #9).
//!
//! 1. Every cohort includes the current incumbent as `baseline.json`.
//! 2. **Screen** (optional): the scorer's record-sampling mode ranks
//!    candidates cheaply. This is explicitly non-authoritative.
//! 3. The best `promote_count` screen survivors, plus an exploratory bypass
//!    quota of screen rejects (to measure false negatives), are **fully
//!    scored** on the canonical corpus in one call together with the baseline.
//! 4. A candidate is accepted only when its full `score` beats the *same-call*
//!    full baseline by more than `min_improvement`, **and** that same-call
//!    baseline agrees with the stored authoritative baseline within
//!    `baseline_drift_epsilon`. Any scorer failure means no acceptance.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rand::Rng;
use rand::rngs::StdRng;

use crate::candidates::Candidate;
use crate::scorer::{DirectoryScorer, ScoreResult, ScorerError, ScorerMode};

/// Promotion controls.
#[derive(Debug, Clone, PartialEq)]
pub struct PromoteConfig {
    /// Screen sample rate (`None` = no screen).
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

/// Run the screen + promote pipeline for one cohort.
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
    if let Some(rate) = cfg.screen_sample_rate {
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
    if to_full.is_empty() {
        return Ok(outcome);
    }
    let dir = work_dir.join("full");
    write_cohort(&dir, incumbent, &to_full).map_err(ScorerError::Spawn)?;
    let started = Instant::now();
    let result = scorer.score_directory(&dir, training_dir, ScorerMode::Full);
    cleanup(&dir, cfg.preserve_dirs);
    let scores = result?;
    let base = scores["baseline"].score;
    let drift = base - cfg.authoritative_baseline;
    let mut best: Option<(String, f64)> = None;
    for c in &to_full {
        let Some(s) = scores.get(&c.id) else { continue };
        let delta = s.score - base;
        let cleared = delta > cfg.min_improvement;
        if bypass_ids.contains(&c.id) {
            if cleared {
                outcome.false_negatives += 1;
            }
        } else if !cleared {
            outcome.false_positives += 1;
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
        scorer_ms: started.elapsed().as_millis() as u64,
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
    use crate::baseline::fake::LocalMseScorer;
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
}

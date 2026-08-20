//! Authoritative baseline and scorer parity gate (Issue #2).
//!
//! Before any search, the incumbent is scored by NEAT-AI-scorer on the full
//! corpus. The resulting `score` is the number every candidate must beat. A
//! local forward-pass MSE is compared against the scorer's `error` so that
//! Forests' proxies are known to agree with the judge; disagreement beyond the
//! documented tolerance aborts the run (fail closed) unless the operator
//! explicitly asks to skip the check (e.g. for a non-MSE cost function, where
//! the proxy is then disabled rather than trusted).

use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::corpus::CorpusInfo;
use crate::incumbent::Incumbent;
use crate::scorer::{DirectoryScorer, ScorerMode};

/// How to treat a local-MSE vs scorer-error disagreement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParityPolicy {
    /// Abort when `|local − scorer| > abs + rel·|scorer|`.
    Abort {
        /// Absolute tolerance.
        abs: f64,
        /// Relative tolerance.
        rel: f64,
    },
    /// Do not compare (non-MSE cost); local proxies are flagged as unverified.
    Skip,
}

impl Default for ParityPolicy {
    fn default() -> Self {
        Self::Abort {
            abs: 1e-7,
            rel: 1e-4,
        }
    }
}

/// Scorer-verified baseline for one incumbent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthoritativeBaseline {
    /// Incumbent checksum.
    pub incumbent_checksum: String,
    /// Authoritative score (larger is better).
    pub score: f64,
    /// Authoritative error.
    pub error: f64,
    /// Complexity penalty.
    pub complexity_penalty: f64,
    /// Records the scorer saw.
    pub record_count: u64,
    /// Scorer identity (binary digest).
    pub scorer_identity: String,
    /// Cost function the scorer reported.
    pub cost_name: Option<String>,
    /// Scorer backend label.
    pub scorer_backend: Option<String>,
    /// Corpus identity.
    pub corpus_identity: String,
    /// Corpus record count (must equal `record_count`).
    pub corpus_record_count: u64,
    /// Local forward-pass MSE, when computed.
    pub local_mse: Option<f64>,
    /// `verified`, `skipped`, or the reason local proxies were disabled.
    pub parity: String,
    /// Scorer wall time (ms).
    pub scorer_ms: u64,
    /// Unix seconds.
    pub created_at_unix: u64,
}

/// Score the incumbent alone and record the authoritative baseline.
///
/// Writes `workspace/baseline-score/baseline.json` (the creature) for the
/// scorer, then `workspace/baseline.json` (this record).
pub fn establish_baseline(
    incumbent: &Incumbent,
    training_dir: &Path,
    corpus: &CorpusInfo,
    scorer: &dyn DirectoryScorer,
    workspace: &Path,
    parity: ParityPolicy,
    local_mse: Option<f64>,
) -> Result<AuthoritativeBaseline, String> {
    let dir = workspace.join("baseline-score");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let compact = neat_core::creature_to_json(&incumbent.creature).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("baseline.json"), compact).map_err(|e| e.to_string())?;
    let started = Instant::now();
    let results = scorer
        .score_directory(&dir, training_dir, ScorerMode::Full)
        .map_err(|e| format!("baseline: {e}"))?;
    let scorer_ms = started.elapsed().as_millis() as u64;
    let _ = std::fs::remove_dir_all(&dir);
    let r = results
        .get("baseline")
        .ok_or("baseline: scorer returned no `baseline` entry")?;
    if results.len() != 1 {
        return Err(format!(
            "baseline: scorer returned {} entries for a single creature",
            results.len()
        ));
    }
    if r.record_count != 0 && r.record_count != corpus.record_count {
        return Err(format!(
            "baseline: scorer saw {} records but the corpus has {} — refusing to continue",
            r.record_count, corpus.record_count
        ));
    }
    let parity_label = match (parity, local_mse) {
        (ParityPolicy::Skip, _) => "skipped: local proxies unverified against scorer".to_string(),
        (ParityPolicy::Abort { abs, rel }, Some(local)) => {
            let is_mse = r
                .cost_name
                .as_deref()
                .is_none_or(|c| c.eq_ignore_ascii_case("MSE"));
            if !is_mse {
                format!(
                    "disabled: scorer cost {} is not MSE; local proxies unverified",
                    r.cost_name.clone().unwrap_or_default()
                )
            } else {
                let diff = (local - r.error).abs();
                if diff > abs + rel * r.error.abs() {
                    return Err(format!(
                        "baseline parity mismatch: local MSE {local} vs scorer error {} (|Δ|={diff} > {abs}+{rel}·|error|); aborting before search",
                        r.error
                    ));
                }
                format!("verified: |Δ|={diff:.3e} within {abs}+{rel}·|error|")
            }
        }
        (ParityPolicy::Abort { .. }, None) => "unavailable: no local MSE supplied".to_string(),
    };
    let baseline = AuthoritativeBaseline {
        incumbent_checksum: incumbent.checksum.clone(),
        score: r.score,
        error: r.error,
        complexity_penalty: r.complexity_penalty,
        record_count: r.record_count,
        scorer_identity: scorer.identity(),
        cost_name: r.cost_name.clone(),
        scorer_backend: r.gpu_backend.clone(),
        corpus_identity: corpus.identity.clone(),
        corpus_record_count: corpus.record_count,
        local_mse,
        parity: parity_label,
        scorer_ms,
        created_at_unix: crate::incumbent::now_unix(),
    };
    let json = serde_json::to_string_pretty(&baseline).map_err(|e| e.to_string())?;
    std::fs::write(workspace.join("baseline.json"), json).map_err(|e| e.to_string())?;
    Ok(baseline)
}

/// In-process fake scorers for tests.
pub mod fake {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::path::Path;

    use neat_core::training_data::TrainingDataConfig;

    use crate::scorer::{DirectoryScorer, ScoreResult, ScorerError, ScorerMode};

    /// Scores every creature with a real forward pass (MSE over the corpus),
    /// `score = 1 − error − 1e-7·hidden`, optionally applying a sample stride.
    /// Records every call so tests can inspect modes.
    #[derive(Default)]
    pub struct LocalMseScorer {
        /// Calls made: (mode label, creature stems).
        pub calls: RefCell<Vec<(String, Vec<String>)>>,
        /// When set, every call fails with this message.
        pub fail_with: Option<String>,
        /// When set, the output is this raw string (to simulate malformed output).
        pub raw_output: Option<String>,
        /// Added to every candidate's (non-baseline) score — simulates screen optimism.
        pub sample_bias: f64,
    }

    impl LocalMseScorer {
        /// Fresh scorer.
        pub fn new() -> Self {
            Self::default()
        }
    }

    impl DirectoryScorer for LocalMseScorer {
        fn score_directory(
            &self,
            creature_dir: &Path,
            training_dir: &Path,
            mode: ScorerMode,
        ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
            if let Some(m) = &self.fail_with {
                return Err(ScorerError::Failed {
                    status: "exit 1".into(),
                    stderr: m.clone(),
                });
            }
            if let Some(raw) = &self.raw_output {
                return crate::scorer::parse_scorer_output(raw);
            }
            let mut paths: Vec<_> = std::fs::read_dir(creature_dir)
                .map_err(|e| ScorerError::Spawn(e.to_string()))?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "json"))
                .collect();
            paths.sort();
            let mut out = BTreeMap::new();
            let mut stems = Vec::new();
            for p in paths {
                let stem = p.file_stem().unwrap().to_string_lossy().to_string();
                let text =
                    std::fs::read_to_string(&p).map_err(|e| ScorerError::Spawn(e.to_string()))?;
                let creature = neat_core::parse_creature_json(&text)
                    .map_err(|e| ScorerError::Malformed(e.to_string()))?;
                let mut net = neat_core::compile_creature(&creature)
                    .map_err(|e| ScorerError::Malformed(e.to_string()))?;
                let cfg = TrainingDataConfig::new(creature.input, creature.output);
                let (stride, phase) = match mode {
                    ScorerMode::Full => (1u64, 0u64),
                    ScorerMode::Sample { rate, phase } => {
                        ((1.0 / rate).round().max(1.0) as u64, phase)
                    }
                };
                let mut sum = 0.0;
                let mut n = 0u64;
                crate::corpus::for_each_chunk(training_dir, &cfg, 512, |c| {
                    for r in 0..c.records {
                        if !(c.first_index + r as u64 + phase).is_multiple_of(stride) {
                            continue;
                        }
                        let out = net.activate(
                            &c.inputs[r * creature.input..(r + 1) * creature.input],
                            creature.output,
                        );
                        sum += neat_core::loss::mse_record(
                            &c.targets[r * creature.output..(r + 1) * creature.output],
                            &out,
                        );
                        n += 1;
                    }
                    Ok(())
                })
                .map_err(ScorerError::Spawn)?;
                let error = if n == 0 { 0.0 } else { sum / n as f64 };
                let hidden = creature
                    .neurons
                    .iter()
                    .filter(|x| x.neuron_type == "hidden")
                    .count() as f64;
                let bias = if stem == "baseline" || mode.is_authoritative() {
                    0.0
                } else {
                    self.sample_bias
                };
                out.insert(
                    stem.clone(),
                    ScoreResult {
                        score: 1.0 - error - 1e-7 * hidden + bias,
                        error,
                        complexity_penalty: 1e-7 * hidden,
                        record_count: n,
                        sample_rate: match mode {
                            ScorerMode::Sample { rate, .. } => Some(rate),
                            ScorerMode::Full => None,
                        },
                        gpu_backend: Some("fake".into()),
                        cost_name: Some("MSE".into()),
                        time_taken: 0.0,
                    },
                );
                stems.push(stem);
            }
            self.calls
                .borrow_mut()
                .push((mode.label().to_string(), stems));
            if !out.contains_key("baseline") {
                return Err(ScorerError::MissingBaseline);
            }
            Ok(out)
        }

        fn identity(&self) -> String {
            "fake:local-mse".into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::LocalMseScorer;
    use super::*;
    use crate::corpus::{corpus_info, write_bin_file};
    use crate::graft::fixtures::identity_creature;
    use neat_core::training_data::TrainingDataConfig;

    fn setup() -> (tempfile::TempDir, Incumbent, CorpusInfo) {
        let tmp = tempfile::tempdir().unwrap();
        let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..8)
            .map(|i| (vec![i as f32], vec![i as f32 + 0.5]))
            .collect();
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let inc = Incumbent::from_creature(identity_creature(1, 1), "t").unwrap();
        let corpus = corpus_info(tmp.path(), &TrainingDataConfig::new(1, 1)).unwrap();
        (tmp, inc, corpus)
    }

    #[test]
    fn baseline_is_recorded_and_parity_verified() {
        let (tmp, inc, corpus) = setup();
        let scorer = LocalMseScorer::new();
        let ws = tmp.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let b = establish_baseline(
            &inc,
            tmp.path(),
            &corpus,
            &scorer,
            &ws,
            ParityPolicy::default(),
            Some(0.25),
        )
        .unwrap();
        assert!((b.error - 0.25).abs() < 1e-9);
        assert!(b.parity.starts_with("verified"));
        assert_eq!(b.record_count, 8);
        assert!(ws.join("baseline.json").exists());
        assert!(!ws.join("baseline-score").exists());
        let back: AuthoritativeBaseline =
            serde_json::from_str(&std::fs::read_to_string(ws.join("baseline.json")).unwrap())
                .unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn parity_mismatch_failure_and_malformed_abort() {
        let (tmp, inc, corpus) = setup();
        let ws = tmp.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let scorer = LocalMseScorer::new();
        let err = establish_baseline(
            &inc,
            tmp.path(),
            &corpus,
            &scorer,
            &ws,
            ParityPolicy::default(),
            Some(0.9),
        )
        .unwrap_err();
        assert!(err.contains("parity mismatch"), "{err}");
        let skip = establish_baseline(
            &inc,
            tmp.path(),
            &corpus,
            &scorer,
            &ws,
            ParityPolicy::Skip,
            Some(0.9),
        )
        .unwrap();
        assert!(skip.parity.starts_with("skipped"));
        let failing = LocalMseScorer {
            fail_with: Some("boom".into()),
            ..Default::default()
        };
        assert!(
            establish_baseline(
                &inc,
                tmp.path(),
                &corpus,
                &failing,
                &ws,
                ParityPolicy::Skip,
                None
            )
            .unwrap_err()
            .contains("boom")
        );
        let malformed = LocalMseScorer {
            raw_output: Some("{not json".into()),
            ..Default::default()
        };
        assert!(
            establish_baseline(
                &inc,
                tmp.path(),
                &corpus,
                &malformed,
                &ws,
                ParityPolicy::Skip,
                None
            )
            .unwrap_err()
            .contains("malformed")
        );
        let wrong_count = LocalMseScorer {
            raw_output: Some(r#"{"baseline":{"score":0.5,"error":0.5,"recordCount":3}}"#.into()),
            ..Default::default()
        };
        assert!(
            establish_baseline(
                &inc,
                tmp.path(),
                &corpus,
                &wrong_count,
                &ws,
                ParityPolicy::Skip,
                None
            )
            .unwrap_err()
            .contains("records")
        );
    }
}

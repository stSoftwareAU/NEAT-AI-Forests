//! NEAT-AI-scorer integration — the final judge.
//!
//! The external `rust_scorer` binary is invoked in directory mode:
//! `rust_scorer [--cohort | --sample-rate R --sample-phase P] <creatures> <training-dir>`
//! and prints one JSON object keyed by creature file stem. Any failure —
//! non-zero exit, malformed JSON, missing `baseline` entry, non-finite score —
//! is an error. Nothing here ever guesses a score.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde::{Deserialize, Serialize};

/// How the scorer executed an exact cohort request (its documented stderr contract).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "camelCase")]
pub enum CohortExecution {
    /// The scorer shared computation across the cohort.
    Hosted,
    /// The scorer could not host the cohort and scored the directory normally.
    DirectoryFallback {
        /// Scorer-reported reason for falling back.
        reason: String,
    },
}

/// Exact ranking scores and their execution provenance; these never authorise acceptance.
#[derive(Debug, Clone, PartialEq)]
pub struct CohortScores {
    /// Full-corpus scores keyed by creature stem.
    pub scores: BTreeMap<String, ScoreResult>,
    /// Hosted or documented fallback.
    pub execution: CohortExecution,
}

/// One scorer result. `score` is the acceptance metric — **larger is better**.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ScoreResult {
    /// `1 - error - complexityPenalty - versionPenalty`.
    pub score: f64,
    /// Mean cost over scored records.
    pub error: f64,
    /// Structural penalty (includes `3e-9` per `IF` neuron).
    #[serde(default)]
    pub complexity_penalty: f64,
    /// Records scored (the sampled count in sample mode).
    #[serde(default)]
    pub record_count: u64,
    /// Sample rate when the scorer ran in sample mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<f64>,
    /// Scorer-reported backend label (`cpu-fallback`, `metal`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_backend: Option<String>,
    /// Scorer-reported cost function name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_name: Option<String>,
    /// Scorer wall time in seconds.
    #[serde(default, rename = "timeTaken")]
    pub time_taken: f64,
}

/// How the scorer is asked to run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScorerMode {
    /// Full canonical corpus — the only authoritative mode.
    Full,
    /// Record sub-sampling — a cheap, explicitly non-authoritative screen.
    Sample {
        /// Rate in `(0, 1)`.
        rate: f64,
        /// Stride phase so successive screens see different records.
        phase: u64,
    },
}

impl ScorerMode {
    /// `true` only for [`ScorerMode::Full`].
    pub fn is_authoritative(self) -> bool {
        matches!(self, Self::Full)
    }

    /// Label used in the journal.
    pub fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Sample { .. } => "sample",
        }
    }
}

/// Scorer failure — always fail closed.
#[derive(Debug, Clone, PartialEq)]
pub enum ScorerError {
    /// Cohort scoring was requested from an implementation that lacks it.
    UnsupportedCohort,
    /// Could not spawn the binary.
    Spawn(String),
    /// Non-zero exit.
    Failed {
        /// Exit status description.
        status: String,
        /// Tail of stderr.
        stderr: String,
    },
    /// Output was not the expected JSON.
    Malformed(String),
    /// Reserved `baseline` stem missing from output.
    MissingBaseline,
    /// A score/error was NaN/∞.
    NonFinite(String),
}

impl fmt::Display for ScorerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedCohort => write!(f, "scorer does not support --cohort"),
            Self::Spawn(m) => write!(f, "cannot run scorer: {m}"),
            Self::Failed { status, stderr } => write!(f, "scorer failed ({status}): {stderr}"),
            Self::Malformed(m) => write!(f, "scorer output malformed: {m}"),
            Self::MissingBaseline => write!(f, "scorer output has no `baseline` entry"),
            Self::NonFinite(k) => write!(f, "scorer returned a non-finite result for `{k}`"),
        }
    }
}

impl std::error::Error for ScorerError {}

/// Scores a directory of creature JSON files in one corpus pass.
pub trait DirectoryScorer {
    /// Probe once per exact run; older implementations retain the legacy pipeline.
    fn supports_cohort(&self) -> Result<bool, ScorerError> {
        Ok(false)
    }

    /// Rank a directory on the full corpus using the scorer's cohort mode.
    fn score_cohort(
        &self,
        _creature_dir: &Path,
        _training_dir: &Path,
    ) -> Result<CohortScores, ScorerError> {
        Err(ScorerError::UnsupportedCohort)
    }

    /// Score every `*.json` in `creature_dir` against `training_dir`.
    fn score_directory(
        &self,
        creature_dir: &Path,
        training_dir: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError>;

    /// Stable identity of the scorer (binary digest or test label) for the
    /// baseline record.
    fn identity(&self) -> String;
}

/// The real `rust_scorer` binary.
#[derive(Debug, Clone)]
pub struct ExternalScorer {
    /// Path (or `$PATH` name) of the scorer binary.
    pub binary: PathBuf,
    /// Extra arguments appended verbatim (e.g. `--cost MSE`, `--gpu off`).
    pub extra_args: Vec<String>,
}

impl ExternalScorer {
    /// Scorer at `binary` with no extra arguments.
    pub fn new(binary: impl Into<PathBuf>) -> Self {
        Self {
            binary: binary.into(),
            extra_args: Vec::new(),
        }
    }

    fn run(&self, cmd: &mut Command) -> Result<Output, ScorerError> {
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        crate::log::flush();
        let out = cmd.output()
            .map_err(|e| ScorerError::Spawn(format!("{}: {e}", self.binary.display())))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let tail: String = stderr.chars().rev().take(2000).collect::<Vec<_>>()
                .into_iter().rev().collect();
            return Err(ScorerError::Failed {
                status: out.status.to_string(),
                stderr: tail.trim().to_string(),
            });
        }
        Ok(out)
    }

    fn invoke(&self, creatures: &Path, training: &Path, mode: ScorerMode, cohort: bool)
        -> Result<Output, ScorerError>
    {
        // These flags belong to this adapter: extra arguments must not turn a
        // verification into a sample, cohort, or a different scorer protocol.
        for arg in &self.extra_args {
            let flag = arg.split('=').next().unwrap_or(arg);
            if matches!(flag, "--cohort" | "--sample-rate" | "--sample-phase" | "--race-stdio" | "--creature-stdin" | "--ensemble" | "--perturb-sigma") {
                return Err(ScorerError::Malformed(format!("scorer argument {flag} is controlled by Forests")));
            }
        }
        let mut cmd = Command::new(&self.binary);
        if cohort {
            cmd.arg("--cohort");
        }
        if let ScorerMode::Sample { rate, phase } = mode {
            cmd.arg("--sample-rate").arg(rate.to_string())
                .arg("--sample-phase").arg(phase.to_string());
        }
        cmd.args(&self.extra_args).arg(creatures).arg(training);
        self.run(&mut cmd)
    }
}

/// Reject incomplete, unexpected, non-finite or sampled full-corpus results.
pub(crate) fn validate_scores<'a>(
    scores: &BTreeMap<String, ScoreResult>,
    candidate_ids: impl IntoIterator<Item = &'a str>,
    full_records: Option<u64>,
) -> Result<(), ScorerError> {
    if !scores.contains_key("baseline") {
        return Err(ScorerError::MissingBaseline);
    }
    let mut expected: BTreeSet<&str> = candidate_ids.into_iter().collect();
    expected.insert("baseline");
    for id in &expected {
        if !scores.contains_key(*id) {
            return Err(ScorerError::Malformed(format!("missing result for `{id}`")));
        }
    }
    for id in scores.keys() {
        if !expected.contains(id.as_str()) {
            return Err(ScorerError::Malformed(format!("unexpected result for `{id}`")));
        }
    }
    check_finite(scores)?;
    if let Some(records) = full_records {
        for (id, result) in scores {
            if result.sample_rate.is_some() || records == 0 || result.record_count != records {
                return Err(ScorerError::Malformed(format!(
                    "`{id}` did not score the full corpus: expected {records} records without sampling, got {} (sample rate {:?})",
                    result.record_count, result.sample_rate
                )));
            }
        }
    }
    Ok(())
}

fn cohort_execution(stderr: &str) -> Result<CohortExecution, ScorerError> {
    let mut notes = stderr.lines().filter(|line| line.starts_with("[cohort] "));
    let note = notes.next().ok_or_else(|| ScorerError::Malformed("missing cohort execution note".into()))?;
    if notes.next().is_some() {
        return Err(ScorerError::Malformed("multiple cohort execution notes".into()));
    }
    if let Some(reason) = note.strip_prefix("[cohort] fallback to directory mode: ")
        && !reason.trim().is_empty()
    {
        return Ok(CohortExecution::DirectoryFallback { reason: reason.to_string() });
    }
    if note.starts_with("[cohort] hosted ") {
        return Ok(CohortExecution::Hosted);
    }
    Err(ScorerError::Malformed("unrecognised cohort execution note".into()))
}

/// Parse scorer stdout; every entry must be finite and `baseline` must exist.
pub fn parse_scorer_output(stdout: &str) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
    struct UniqueScores;
    impl<'de> serde::de::Visitor<'de> for UniqueScores {
        type Value = BTreeMap<String, ScoreResult>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a map of unique creature stems to scorer results")
        }

        fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut scores = BTreeMap::new();
            while let Some((id, result)) = map.next_entry::<String, ScoreResult>()? {
                if scores.insert(id.clone(), result).is_some() {
                    return Err(serde::de::Error::custom(format!("duplicate scorer result for `{id}`")));
                }
            }
            Ok(scores)
        }
    }
    let mut de = serde_json::Deserializer::from_str(stdout.trim());
    let parsed = serde::Deserializer::deserialize_map(&mut de, UniqueScores)
        .map_err(|e| ScorerError::Malformed(e.to_string()))?;
    de.end().map_err(|e| ScorerError::Malformed(e.to_string()))?;
    if !parsed.contains_key("baseline") {
        return Err(ScorerError::MissingBaseline);
    }
    check_finite(&parsed)?;
    Ok(parsed)
}

/// Reject any non-finite score or error.
pub fn check_finite(results: &BTreeMap<String, ScoreResult>) -> Result<(), ScorerError> {
    for (k, v) in results {
        if !v.score.is_finite() || !v.error.is_finite() {
            return Err(ScorerError::NonFinite(k.clone()));
        }
    }
    Ok(())
}

impl DirectoryScorer for ExternalScorer {
    fn supports_cohort(&self) -> Result<bool, ScorerError> {
        let mut cmd = Command::new(&self.binary);
        cmd.arg("--help");
        let out = self.run(&mut cmd)?;
        Ok(String::from_utf8_lossy(&out.stdout).split_whitespace().any(|word| word == "--cohort"))
    }

    fn score_cohort(&self, creature_dir: &Path, training_dir: &Path) -> Result<CohortScores, ScorerError> {
        let out = self.invoke(creature_dir, training_dir, ScorerMode::Full, true)?;
        Ok(CohortScores {
            scores: parse_scorer_output(&String::from_utf8_lossy(&out.stdout))?,
            execution: cohort_execution(&String::from_utf8_lossy(&out.stderr))?,
        })
    }

    fn score_directory(
        &self,
        creature_dir: &Path,
        training_dir: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let out = self.invoke(creature_dir, training_dir, mode, false)?;
        parse_scorer_output(&String::from_utf8_lossy(&out.stdout))
    }

    fn identity(&self) -> String {
        match std::fs::read(&self.binary) {
            Ok(bytes) => format!("sha256:{}", &crate::incumbent::sha256_hex(&bytes)[..16]),
            Err(_) => format!("path:{}", self.binary.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_directory_output() {
        let s = r#"{"baseline":{"score":0.9,"error":0.1,"complexityPenalty":1e-8,"recordCount":4,"timeTaken":0.1,"costName":"MSE"},
                    "candidate-000":{"score":0.95,"error":0.05}}"#;
        let m = parse_scorer_output(s).unwrap();
        assert_eq!(m["baseline"].record_count, 4);
        assert_eq!(m["candidate-000"].score, 0.95);
        assert_eq!(m["baseline"].cost_name.as_deref(), Some("MSE"));
    }

    #[test]
    fn missing_baseline_and_malformed_fail_closed() {
        assert_eq!(
            parse_scorer_output(r#"{"x":{"score":1,"error":0}}"#),
            Err(ScorerError::MissingBaseline)
        );
        assert!(matches!(
            parse_scorer_output("not json"),
            Err(ScorerError::Malformed(_))
        ));
        assert!(matches!(
            parse_scorer_output("{}"),
            Err(ScorerError::MissingBaseline)
        ));
    }

    #[test]
    fn non_finite_is_rejected() {
        // serde_json refuses NaN/∞ literals and out-of-range exponents at parse
        // time (Malformed); a hand-built map exercises the explicit guard.
        let s = r#"{"baseline":{"score":1e999,"error":0}}"#;
        assert!(parse_scorer_output(s).is_err());
        let mut m = BTreeMap::new();
        m.insert(
            "baseline".to_string(),
            ScoreResult {
                score: f64::NAN,
                error: 0.0,
                complexity_penalty: 0.0,
                record_count: 0,
                sample_rate: None,
                gpu_backend: None,
                cost_name: None,
                time_taken: 0.0,
            },
        );
        assert!(matches!(check_finite(&m), Err(ScorerError::NonFinite(_))));
    }

    #[test]
    fn spawn_failure_is_reported() {
        let sc = ExternalScorer::new("/definitely/not/a/scorer");
        let tmp = tempfile::tempdir().unwrap();
        assert!(matches!(
            sc.score_directory(tmp.path(), tmp.path(), ScorerMode::Full),
            Err(ScorerError::Spawn(_))
        ));
    }

    #[test]
    fn duplicate_results_are_malformed_instead_of_overwriting_a_score() {
        let raw = r#"{"baseline":{"score":0.9,"error":0.1},"baseline":{"score":0.1,"error":0.9}}"#;
        assert!(matches!(parse_scorer_output(raw), Err(ScorerError::Malformed(ref m)) if m.contains("duplicate")));
        assert_eq!(parse_scorer_output(r#"{"baseline":{"score":0.9,"error":0.1}}"#).unwrap()["baseline"].score, 0.9);
    }

    #[test]
    fn cohort_execution_notes_distinguish_hosting_fallback_and_unknown_output() {
        assert_eq!(cohort_execution("noise\n[cohort] hosted 5 creatures: 9 neuron activations per record\n").unwrap(), CohortExecution::Hosted);
        assert_eq!(cohort_execution("[cohort] fallback to directory mode: cost MAE is not hosted\n").unwrap(),
            CohortExecution::DirectoryFallback { reason: "cost MAE is not hosted".into() });
        for (stderr, cause) in [
            ("no cohort note", "missing"),
            ("[cohort] fallback to directory mode: ", "unrecognised"),
            ("[cohort] unknown result", "unrecognised"),
            ("[cohort] hosted 5 creatures\n[cohort] fallback to directory mode: MAE", "multiple"),
            ("prefix [cohort] hosted 5 creatures", "missing"),
        ] {
            assert!(matches!(cohort_execution(stderr), Err(ScorerError::Malformed(ref m)) if m.contains(cause)), "{stderr}");
        }
    }

    #[test]
    fn extra_arguments_cannot_change_the_scoring_protocol() {
        let tmp = tempfile::tempdir().unwrap();
        for flag in ["--sample-rate", "--sample-phase", "--cohort", "--race-stdio", "--creature-stdin", "--ensemble", "--perturb-sigma"] {
            for arg in [flag.to_string(), format!("{flag}=0.05")] {
                let scorer = ExternalScorer { binary: "/missing/scorer".into(), extra_args: vec![arg] };
                assert!(matches!(scorer.score_directory(tmp.path(), tmp.path(), ScorerMode::Full),
                    Err(ScorerError::Malformed(ref m)) if m.contains(flag) && m.contains("controlled by Forests")));
            }
        }
        let allowed = ExternalScorer { binary: "/missing/scorer".into(), extra_args: vec!["--gpu=off".into()] };
        assert!(matches!(allowed.score_directory(tmp.path(), tmp.path(), ScorerMode::Full), Err(ScorerError::Spawn(_))));
    }
}

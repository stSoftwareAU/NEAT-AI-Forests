//! NEAT-AI-scorer integration — the final judge.
//!
//! The external `rust_scorer` binary is invoked in directory mode:
//! `rust_scorer [--sample-rate R --sample-phase P] <dir-of-creatures> <training-dir>`
//! and prints one JSON object keyed by creature file stem. Any failure —
//! non-zero exit, malformed JSON, missing `baseline` entry, non-finite score —
//! is an error. Nothing here ever guesses a score.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::{Deserialize, Serialize};

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
}

/// Parse scorer stdout; every entry must be finite and `baseline` must exist.
pub fn parse_scorer_output(stdout: &str) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
    let parsed: BTreeMap<String, ScoreResult> =
        serde_json::from_str(stdout.trim()).map_err(|e| ScorerError::Malformed(e.to_string()))?;
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
    fn score_directory(
        &self,
        creature_dir: &Path,
        training_dir: &Path,
        mode: ScorerMode,
    ) -> Result<BTreeMap<String, ScoreResult>, ScorerError> {
        let mut cmd = Command::new(&self.binary);
        if let ScorerMode::Sample { rate, phase } = mode {
            cmd.arg("--sample-rate")
                .arg(format!("{rate}"))
                .arg("--sample-phase")
                .arg(phase.to_string());
        }
        cmd.args(&self.extra_args);
        cmd.arg(creature_dir).arg(training_dir);
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        crate::log::flush();
        let out = cmd
            .output()
            .map_err(|e| ScorerError::Spawn(format!("{}: {e}", self.binary.display())))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let tail: String = stderr
                .chars()
                .rev()
                .take(2000)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            return Err(ScorerError::Failed {
                status: out.status.to_string(),
                stderr: tail.trim().to_string(),
            });
        }
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
}

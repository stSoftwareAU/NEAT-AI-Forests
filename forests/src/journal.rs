//! Append-only `experiments.jsonl` journal (Issue #10).
//!
//! Every line is one JSON object with a `record` discriminator. Lines are
//! written with a single `write_all` of the complete line so an interrupted
//! run leaves a valid prefix. A malformed line is an error on read — a broken
//! journal must never be read as an empty run.

use std::io::{BufRead, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::baseline::AuthoritativeBaseline;
use crate::patch::Patch;

/// Per-candidate journal entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateRecord {
    /// Candidate id (patch id).
    pub id: String,
    /// Strategy label.
    pub strategy: String,
    /// Search backend.
    pub backend: String,
    /// Tree depth.
    pub depth: usize,
    /// Features referenced.
    pub features: Vec<usize>,
    /// Predicted proxy gain.
    pub predicted_gain: f64,
    /// Affected records on the search set.
    pub affected_records: u64,
    /// Affected fraction on the search set.
    pub affected_fraction: f64,
    /// Screen (sampled, non-authoritative) score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screen_score: Option<f64>,
    /// Screen score minus the screen baseline score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screen_delta: Option<f64>,
    /// Promoted to full scoring by the screen.
    pub promoted: bool,
    /// Sent to full scoring by the exploratory bypass quota.
    pub bypass: bool,
    /// Authoritative full-corpus score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_score: Option<f64>,
    /// Full score minus the same-call full baseline score.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full_delta: Option<f64>,
    /// The patch itself (complete provenance).
    pub patch: Patch,
    /// Further patches stacked on `patch` for a combination candidate.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub combo: Vec<Patch>,
}

/// Screen stage summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScreenSummary {
    /// Sample rate.
    pub sample_rate: f64,
    /// Sample phase.
    pub sample_phase: u64,
    /// Baseline score under the sample.
    pub baseline_score: f64,
    /// Candidates screened.
    pub screened: u64,
    /// Candidates promoted.
    pub promoted: u64,
    /// Bypass candidates.
    pub bypass: u64,
    /// Scorer wall time (ms).
    pub scorer_ms: u64,
}

/// Full-corpus stage summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FullSummary {
    /// Same-call baseline score.
    pub baseline_score: f64,
    /// `baseline_score − authoritative baseline` (must be within drift epsilon).
    pub baseline_drift: f64,
    /// Candidates scored.
    pub scored: u64,
    /// Scorer wall time (ms).
    pub scorer_ms: u64,
}

/// One evolution iteration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExperimentRecord {
    /// 1-based iteration.
    pub iteration: u64,
    /// Unix seconds.
    pub timestamp_unix: u64,
    /// Incumbent searched.
    pub incumbent_checksum: String,
    /// Authoritative baseline score of that incumbent.
    pub baseline_score: f64,
    /// Backend that produced split statistics (`cpu` / `gpu`).
    pub search_backend: String,
    /// Search set label (`memory-sample`, `streaming-full`).
    pub search_set: String,
    /// Records searched.
    pub search_records: u64,
    /// Features searched.
    pub search_features: u64,
    /// Strategy labels used this iteration.
    pub strategies: Vec<String>,
    /// Search wall time (ms).
    pub search_ms: u64,
    /// Graft wall time (ms).
    pub graft_ms: u64,
    /// Candidates generated (before graft validation).
    pub candidates_generated: u64,
    /// Candidates discarded by graft validation.
    pub candidates_discarded: u64,
    /// Candidate details.
    pub candidates: Vec<CandidateRecord>,
    /// Screen stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub screen: Option<ScreenSummary>,
    /// Full stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub full: Option<FullSummary>,
    /// Winner id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<String>,
    /// Authoritative improvement of the winner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub improvement: Option<f64>,
    /// Whether a winner was accepted.
    pub accepted: bool,
    /// Checksum of the new incumbent when accepted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub new_incumbent_checksum: Option<String>,
    /// Promoted candidates whose full delta did not clear the threshold.
    pub screen_false_positives: u64,
    /// Bypass candidates (rejected by the screen) whose full delta cleared the threshold.
    pub screen_false_negatives: u64,
    /// Scorer error, if the iteration failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scorer_error: Option<String>,
}

/// Run summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryRecord {
    /// Unix seconds.
    pub timestamp_unix: u64,
    /// Why the run stopped.
    pub stop_reason: String,
    /// Iterations run.
    pub iterations: u64,
    /// Accepted winners.
    pub acceptances: u64,
    /// Opening authoritative score.
    pub opening_score: f64,
    /// Final authoritative score.
    pub final_score: f64,
    /// Wall time (ms).
    pub wall_ms: u64,
    /// Final incumbent checksum.
    pub final_checksum: String,
}

/// Journal line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "camelCase")]
pub enum JournalLine {
    /// First line of a run.
    RunHeader {
        /// Unix seconds.
        timestamp_unix: u64,
        /// Seed used.
        seed: u64,
        /// `supplied` or `drawn`.
        seed_source: String,
        /// Forests version.
        version: String,
        /// Source creature path.
        source_creature: String,
        /// Incumbent checksum.
        incumbent_checksum: String,
        /// Corpus identity.
        corpus_identity: String,
        /// Bin cache identity (`corpus identity:bins`).
        bin_cache: String,
        /// Effective configuration.
        config: serde_json::Value,
    },
    /// Authoritative baseline (also written after each acceptance).
    Baseline(AuthoritativeBaseline),
    /// One iteration.
    Experiment(Box<ExperimentRecord>),
    /// Final summary.
    Summary(SummaryRecord),
}

/// Append one line.
pub fn append_journal_line(path: &Path, line: &JournalLine) -> Result<(), String> {
    let mut text = serde_json::to_string(line).map_err(|e| e.to_string())?;
    text.push('\n');
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    f.write_all(text.as_bytes())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    f.flush().map_err(|e| e.to_string())
}

/// Read every line; any malformed line is an error.
pub fn read_journal(path: &Path) -> Result<Vec<JournalLine>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).map_err(|e| format!("journal line {}: {e}", i + 1))?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_round_trip_and_malformed_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("experiments.jsonl");
        let header = JournalLine::RunHeader {
            timestamp_unix: 1,
            seed: 2,
            seed_source: "supplied".into(),
            version: "x".into(),
            source_creature: "c.json".into(),
            incumbent_checksum: "abc".into(),
            corpus_identity: "def".into(),
            bin_cache: "def:256".into(),
            config: serde_json::json!({"a": 1}),
        };
        append_journal_line(&p, &header).unwrap();
        let summary = JournalLine::Summary(SummaryRecord {
            timestamp_unix: 3,
            stop_reason: "timeout".into(),
            iterations: 1,
            acceptances: 0,
            opening_score: 0.5,
            final_score: 0.5,
            wall_ms: 10,
            final_checksum: "abc".into(),
        });
        append_journal_line(&p, &summary).unwrap();
        assert_eq!(read_journal(&p).unwrap(), vec![header, summary]);
        std::fs::OpenOptions::new()
            .append(true)
            .open(&p)
            .unwrap()
            .write_all(b"{\"record\":\"exp")
            .unwrap();
        assert!(read_journal(&p).unwrap_err().contains("journal line 3"));
    }
}

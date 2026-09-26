//! The learnings domain model: what one record says, and filing verdicts as
//! records (Issue #60).

use serde::{Deserialize, Serialize};

use crate::patch::{Patch, Provenance};

/// Current learnings format version.
pub const LEARNINGS_FORMAT_VERSION: u32 = 1;

/// How a candidate ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Outcome {
    /// The authoritative full-corpus scorer accepted it: a known win.
    Accepted,
    /// Fully scored and not good enough.
    Rejected,
}

/// One thing the fleet has learned: a portable candidate and how it ended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Learning {
    /// Format version.
    pub version: u32,
    /// Candidate identity — the patch id, or the joined ids of a combination.
    pub id: String,
    /// The patch itself, replayable as-is onto any creature of the same width.
    pub patch: Patch,
    /// The remaining patches of a combination candidate, in graft order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combo: Option<Vec<Patch>>,
    /// What the full-corpus scorer decided.
    pub outcome: Outcome,
    /// Δscore against the incumbent it was tried on.
    pub delta: f64,
    /// Corpus identity — a patch is only meaningful against the data it was
    /// measured on.
    pub corpus: String,
    /// Creature input width.
    pub inputs: usize,
    /// Creature output width.
    pub outputs: usize,
    /// Checksum of the creature it was tried on.
    pub incumbent: String,
    /// That creature's authoritative score.
    pub incumbent_score: f64,
    /// Host that tried it.
    pub host: String,
    /// When, in Unix seconds.
    pub at_unix: u64,
    /// `neat_ai_forests` version that recorded it.
    pub tool_version: String,
    /// Free text — a graft's refusal, a strategy label, the island name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

impl Learning {
    /// Every patch of the candidate, in graft order.
    pub fn patches(&self) -> impl Iterator<Item = &Patch> {
        std::iter::once(&self.patch).chain(self.combo.iter().flatten())
    }

    /// The candidate as patches to graft again, with provenance rewritten to
    /// say honestly that it came from the cache and not from this run's search.
    ///
    /// The tree itself is untouched: the same thresholds and corrections that
    /// were measured on this corpus, whichever creature they were measured
    /// against.
    pub fn replay(&self, incumbent: &str) -> Vec<Patch> {
        self.patches()
            .map(|p| {
                let mut notes = vec![format!(
                    "replayed from {} on {} ({}{})",
                    self.host,
                    &self.incumbent[..self.incumbent.len().min(12)],
                    match self.outcome {
                        Outcome::Accepted => "accepted",
                        Outcome::Rejected => "rejected",
                    },
                    format!(" {:+.3e}", self.delta)
                )];
                notes.push(format!("origin strategy {}", p.provenance.strategy));
                notes.extend(p.provenance.notes.iter().cloned());
                Patch::new(
                    p.output,
                    p.root.clone(),
                    Provenance {
                        strategy: "replay".into(),
                        backend: "learnings".into(),
                        predicted_gain: p.provenance.predicted_gain,
                        affected_records: p.provenance.affected_records,
                        search_records: p.provenance.search_records,
                        incumbent_checksum: incumbent.to_string(),
                        seed: p.provenance.seed,
                        notes,
                    },
                )
            })
            .collect()
    }
}

/// What one full-corpus verdict looked like, before it is filed.
#[derive(Debug, Clone)]
pub struct Verdict<'a> {
    /// Candidate id (patch id, or the combination's joined id).
    pub id: &'a str,
    /// The candidate's patches, in graft order.
    pub patches: Vec<Patch>,
    /// Accepted or rejected.
    pub outcome: Outcome,
    /// Δscore against the incumbent.
    pub delta: f64,
}

/// Where and when a batch of verdicts was reached.
#[derive(Debug, Clone)]
pub struct Context {
    /// Corpus identity.
    pub corpus: String,
    /// Creature input width.
    pub inputs: usize,
    /// Creature output width.
    pub outputs: usize,
    /// Checksum of the creature the verdicts were reached against.
    pub incumbent: String,
    /// That creature's authoritative score.
    pub incumbent_score: f64,
    /// Host recording them.
    pub host: String,
    /// Now, in Unix seconds.
    pub at_unix: u64,
}

/// File `verdicts` as learnings, dropping anything `known` already records for
/// the same candidate on the same creature.
///
/// The dedupe is what keeps a shared directory from growing by a copy of the
/// same line every time a cycle restarts from the creature it started from
/// last time. Run-local provenance notes (sampling rates, jitter offsets, the
/// search set's shape) are dropped too: they describe the run, not the patch,
/// and they are the bulk of a line.
pub fn file_verdicts(verdicts: &[Verdict<'_>], ctx: &Context, known: &[Learning]) -> Vec<Learning> {
    let seen: std::collections::HashSet<(&str, &str)> = known
        .iter()
        .map(|l| (l.id.as_str(), l.incumbent.as_str()))
        .collect();
    verdicts
        .iter()
        .filter(|v| !seen.contains(&(v.id, ctx.incumbent.as_str())))
        .filter(|v| !v.patches.is_empty())
        .map(|v| {
            let mut patches = v.patches.iter().map(trimmed);
            let patch = patches.next().expect("checked non-empty");
            let combo: Vec<Patch> = patches.collect();
            Learning {
                version: LEARNINGS_FORMAT_VERSION,
                id: v.id.to_string(),
                patch,
                combo: (!combo.is_empty()).then_some(combo),
                outcome: v.outcome,
                delta: v.delta,
                corpus: ctx.corpus.clone(),
                inputs: ctx.inputs,
                outputs: ctx.outputs,
                incumbent: ctx.incumbent.clone(),
                incumbent_score: ctx.incumbent_score,
                host: ctx.host.clone(),
                at_unix: ctx.at_unix,
                tool_version: env!("CARGO_PKG_VERSION").to_string(),
                notes: Vec::new(),
            }
        })
        .collect()
}

/// A patch with its run-local provenance notes dropped.
fn trimmed(p: &Patch) -> Patch {
    let mut out = p.clone();
    out.provenance.notes = Vec::new();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learnings::test_support::*;

    #[test]
    fn filing_drops_what_is_already_known_and_the_run_local_notes() {
        let mut p = patch(1, 0.1);
        p.provenance.notes = vec!["stride 12".into(), "jitter 2".into()];
        let ctx = Context {
            corpus: "abc123".into(),
            inputs: 8,
            outputs: 1,
            incumbent: "creature-a".into(),
            incumbent_score: 0.5,
            host: "host-a".into(),
            at_unix: 100,
        };
        let verdict = Verdict {
            id: &p.id(),
            patches: vec![p.clone()],
            outcome: Outcome::Accepted,
            delta: 1e-5,
        };
        let filed = file_verdicts(std::slice::from_ref(&verdict), &ctx, &[]);
        assert_eq!(filed.len(), 1);
        assert!(
            filed[0].patch.provenance.notes.is_empty(),
            "run-local notes are not the fleet's business"
        );
        assert_eq!(filed[0].patch.root, p.root, "the tree is kept verbatim");
        assert_eq!(filed[0].incumbent, "creature-a");
        // The same candidate on the same creature is not filed twice, however
        // many cycles start from that creature.
        assert!(file_verdicts(&[verdict], &ctx, &filed).is_empty());
    }

    #[test]
    fn a_combination_is_filed_and_replayed_whole() {
        let (a, b) = (patch(1, 0.1), patch(2, 0.2));
        let id = crate::candidates::combo_id(&[a.clone(), b.clone()]);
        let ctx = Context {
            corpus: "abc123".into(),
            inputs: 8,
            outputs: 1,
            incumbent: "creature-a".into(),
            incumbent_score: 0.5,
            host: "host-a".into(),
            at_unix: 100,
        };
        let filed = file_verdicts(
            &[Verdict {
                id: &id,
                patches: vec![a.clone(), b.clone()],
                outcome: Outcome::Accepted,
                delta: 4e-5,
            }],
            &ctx,
            &[],
        );
        assert_eq!(filed.len(), 1);
        assert_eq!(filed[0].patches().count(), 2);
        let replayed = filed[0].replay("creature-b");
        assert_eq!(replayed.len(), 2, "both halves come back");
        assert_eq!(replayed[0].root, a.root);
        assert_eq!(replayed[1].root, b.root);
    }

    #[test]
    fn replay_keeps_the_tree_and_rewrites_the_provenance() {
        let original = patch(4, 0.25);
        let l = learning(original.clone(), Outcome::Accepted, 3e-5, 10, "host-c");
        let replayed = l.replay("new-incumbent");
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].root, original.root, "the tree is untouched");
        assert_eq!(replayed[0].id(), original.id(), "so is its identity");
        assert_eq!(replayed[0].provenance.strategy, "replay");
        assert_eq!(replayed[0].provenance.backend, "learnings");
        assert_eq!(replayed[0].provenance.incumbent_checksum, "new-incumbent");
        assert!(
            replayed[0].provenance.notes[0].contains("host-c"),
            "says where it came from: {:?}",
            replayed[0].provenance.notes
        );
        assert!(
            replayed[0]
                .provenance
                .notes
                .iter()
                .any(|n| n.contains("histogram-stump")),
            "and how it was found first time"
        );
    }
}

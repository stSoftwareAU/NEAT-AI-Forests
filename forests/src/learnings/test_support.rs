//! Fixtures shared by the learnings tests.

use super::{LEARNINGS_FORMAT_VERSION, Learning, Outcome, PrunePolicy};
use crate::patch::{Node, Patch, Provenance};

pub(super) fn patch(feature: usize, correction: f32) -> Patch {
    Patch::new(
        0,
        Node::stump(feature, 0.5, 0.0, correction),
        Provenance {
            strategy: "histogram-stump".into(),
            backend: "cpu".into(),
            ..Provenance::default()
        },
    )
}

pub(super) fn learning(p: Patch, outcome: Outcome, delta: f64, at: u64, host: &str) -> Learning {
    Learning {
        version: LEARNINGS_FORMAT_VERSION,
        id: p.id(),
        patch: p,
        combo: None,
        outcome,
        delta,
        corpus: "abc123".into(),
        inputs: 8,
        outputs: 1,
        incumbent: format!("creature-{at}"),
        incumbent_score: 0.5,
        host: host.into(),
        at_unix: at,
        tool_version: "test".into(),
        notes: Vec::new(),
    }
}

pub(super) fn policy(now: u64) -> PrunePolicy {
    PrunePolicy {
        rejected_after_secs: 100,
        accepted_after_secs: 1000,
        max_records: 0,
        now_unix: now,
    }
}

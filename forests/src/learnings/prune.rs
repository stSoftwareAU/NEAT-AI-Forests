//! Pure retention planning for a host's own learnings (Issue #61).

use serde::{Deserialize, Serialize};

use super::model::{Learning, Outcome};

/// How long a host keeps its own learnings before dropping them (Issue #61).
#[derive(Debug, Clone)]
pub struct PrunePolicy {
    /// Rejections older than this are dropped. Dropping one is not only
    /// housekeeping: it puts that experiment back on the table, which is the
    /// point — the creature it failed against is long gone.
    pub rejected_after_secs: u64,
    /// Acceptances older than this are dropped. Far longer than the rejection
    /// window: wins are what the cache is for, and they are a small fraction of
    /// the volume.
    pub accepted_after_secs: u64,
    /// Cap on the records a host keeps, newest first (0 = uncapped).
    pub max_records: usize,
    /// Now, in Unix seconds.
    pub now_unix: u64,
}

/// What a prune would do, before anything is written.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PruneOutcome {
    /// Records read.
    pub read: usize,
    /// Records that survive.
    pub kept: usize,
    /// Rejections dropped for age — each one an experiment made available again.
    pub dropped_stale_rejected: usize,
    /// Acceptances dropped for age.
    pub dropped_stale_accepted: usize,
    /// Repeats of a candidate already filed against the same creature.
    pub dropped_duplicate: usize,
    /// Records dropped because the file was over `max_records`.
    pub dropped_over_cap: usize,
}

impl PruneOutcome {
    /// Fold another corpus's outcome into this one.
    pub fn add(&mut self, other: &Self) {
        self.read += other.read;
        self.kept += other.kept;
        self.dropped_stale_rejected += other.dropped_stale_rejected;
        self.dropped_stale_accepted += other.dropped_stale_accepted;
        self.dropped_duplicate += other.dropped_duplicate;
        self.dropped_over_cap += other.dropped_over_cap;
    }
}

/// Decide what a host's file should keep, without touching it.
///
/// Ordering of the survivors is preserved so a pruned file reads like the one
/// it replaced, minus what went. Duplicates keep the **newest** record for a
/// `(candidate, creature)` pair: the later verdict was reached against the same
/// creature, so it is the one worth keeping, and a repeat only exists because a
/// run filed without having pulled what another run had already pushed.
pub fn plan_prune(all: &[Learning], policy: &PrunePolicy) -> (Vec<Learning>, PruneOutcome) {
    let mut out = PruneOutcome {
        read: all.len(),
        ..PruneOutcome::default()
    };
    // Newest wins a duplicate, so walk newest-first and keep the first sighting.
    let mut order: Vec<usize> = (0..all.len()).collect();
    order.sort_by(|&a, &b| all[b].at_unix.cmp(&all[a].at_unix).then(b.cmp(&a)));
    let mut seen: std::collections::HashSet<(&str, &str)> = std::collections::HashSet::new();
    let mut keep = vec![false; all.len()];
    let mut kept_so_far = 0usize;
    for i in order {
        let l = &all[i];
        if !seen.insert((l.id.as_str(), l.incumbent.as_str())) {
            out.dropped_duplicate += 1;
            continue;
        }
        let age = policy.now_unix.saturating_sub(l.at_unix);
        let stale = match l.outcome {
            Outcome::Accepted => age >= policy.accepted_after_secs,
            Outcome::Rejected => age >= policy.rejected_after_secs,
        };
        if stale {
            match l.outcome {
                Outcome::Accepted => out.dropped_stale_accepted += 1,
                Outcome::Rejected => out.dropped_stale_rejected += 1,
            }
            continue;
        }
        if policy.max_records > 0 && kept_so_far >= policy.max_records {
            out.dropped_over_cap += 1;
            continue;
        }
        keep[i] = true;
        kept_so_far += 1;
    }
    let kept: Vec<Learning> = all
        .iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(l, _)| l.clone())
        .collect();
    out.kept = kept.len();
    (kept, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learnings::test_support::*;

    /// Issue #61 — dropping an old rejection is the point, not a side effect:
    /// it puts that experiment back on the table.
    #[test]
    fn an_old_rejection_is_dropped_and_a_win_of_the_same_age_is_not() {
        let all = vec![
            learning(patch(1, 0.1), Outcome::Rejected, -1e-5, 10, "host-a"),
            learning(patch(2, 0.2), Outcome::Accepted, 2e-5, 10, "host-a"),
            learning(patch(3, 0.3), Outcome::Rejected, -1e-5, 950, "host-a"),
        ];
        let (kept, out) = plan_prune(&all, &policy(1000));
        assert_eq!(out.dropped_stale_rejected, 1);
        assert_eq!(out.dropped_stale_accepted, 0);
        assert_eq!(
            kept.iter().map(|l| l.id.clone()).collect::<Vec<_>>(),
            vec![patch(2, 0.2).id(), patch(3, 0.3).id()],
            "the win stays, the recent rejection stays, the old rejection goes"
        );
        // A win outlives its window too, eventually.
        let (kept, out) = plan_prune(&all, &policy(2000));
        assert_eq!(out.dropped_stale_accepted, 1);
        assert!(kept.is_empty());
    }

    #[test]
    fn a_repeat_of_the_same_candidate_on_the_same_creature_keeps_the_newest() {
        // Two runs filed the same verdict because neither had pulled the
        // other's push yet.
        let mut first = learning(patch(1, 0.1), Outcome::Rejected, -1e-5, 900, "host-a");
        first.incumbent = "creature-x".into();
        let mut second = first.clone();
        second.at_unix = 950;
        second.delta = -2e-5;
        // A different creature is not a duplicate (and is recent enough that
        // only duplication could drop it).
        let mut elsewhere = second.clone();
        elsewhere.incumbent = "creature-y".into();
        let (kept, out) = plan_prune(&[first, second.clone(), elsewhere], &policy(1000));
        assert_eq!(out.dropped_duplicate, 1);
        assert_eq!(kept.len(), 2);
        assert!(
            kept.iter().any(|l| l.at_unix == 950 && l.delta == -2e-5),
            "the newer verdict is the one kept"
        );
    }

    #[test]
    fn a_capped_file_keeps_the_newest_records() {
        let all: Vec<Learning> = (0..5)
            .map(|i| {
                learning(
                    patch(i, 0.1 * (i + 1) as f32),
                    Outcome::Accepted,
                    1e-5,
                    900 + i as u64,
                    "host-a",
                )
            })
            .collect();
        let (kept, out) = plan_prune(
            &all,
            &PrunePolicy {
                max_records: 2,
                ..policy(1000)
            },
        );
        assert_eq!(out.dropped_over_cap, 3);
        assert_eq!(
            kept.iter().map(|l| l.at_unix).collect::<Vec<_>>(),
            vec![903, 904],
            "newest two, still in file order"
        );
    }

    #[test]
    fn planning_never_reorders_what_it_keeps() {
        let all: Vec<Learning> = (0..4)
            .map(|i| {
                learning(
                    patch(i, 0.1 * (i + 1) as f32),
                    Outcome::Accepted,
                    1e-5,
                    // Deliberately out of order in the file.
                    [950u64, 910, 940, 920][i],
                    "host-a",
                )
            })
            .collect();
        let (kept, _) = plan_prune(&all, &policy(1000));
        assert_eq!(
            kept.iter().map(|l| l.at_unix).collect::<Vec<_>>(),
            vec![950, 910, 940, 920],
            "a pruned file reads like the one it replaced, minus what went"
        );
    }
}

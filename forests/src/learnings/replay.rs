//! Replay policy: which learnings a run offers back to a creature, and which
//! failures it keeps out of the cohort (Issue #60).

use std::collections::HashMap;

use super::model::{Learning, Outcome};

/// Failures older than this may be tried again (7 days).
pub const DEFAULT_RETRY_AFTER_SECS: u64 = 7 * 24 * 3600;

/// Which learnings a run is willing to replay.
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Most candidates replayed in one iteration.
    pub max: usize,
    /// A candidate that only ever failed is tried again once its most recent
    /// failure is this old.
    pub retry_after_secs: u64,
    /// Now, in Unix seconds.
    pub now_unix: u64,
}

/// Candidate ids the fleet has proved do not work, and has not yet waited long
/// enough to try again.
///
/// This is the other half of the cheat. Replaying a win saves the fleet from
/// losing it; this saves the fleet from making the same mistake repeatedly.
/// The search rediscovers the same splits run after run — the residual surface
/// barely moves between creatures — and each rediscovery costs a full-corpus
/// scorer call to reach the verdict some other host already reached. Dropping
/// them from the cohort spends that slot on the next discovery instead.
///
/// A candidate that ever cleared the full scorer is never in this set, however
/// often it has been turned down since: it is worth trying on a new creature,
/// and [`choose`] handles how eagerly. Once `retry_after_secs` has passed the
/// mistake is worth making again — the creature it failed against is long gone.
pub fn known_failures(all: &[Learning], cfg: &ReplayConfig) -> std::collections::HashSet<String> {
    let mut newest_failure: HashMap<&str, u64> = HashMap::new();
    let mut ever_worked: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for l in all {
        match l.outcome {
            Outcome::Accepted => {
                ever_worked.insert(l.id.as_str());
            }
            Outcome::Rejected => {
                let slot = newest_failure.entry(l.id.as_str()).or_default();
                *slot = (*slot).max(l.at_unix);
            }
        }
    }
    newest_failure
        .into_iter()
        .filter(|(id, _)| !ever_worked.contains(id))
        .filter(|(_, at)| cfg.now_unix.saturating_sub(*at) < cfg.retry_after_secs)
        .map(|(id, _)| id.to_string())
        .collect()
}

/// Patch ids a creature already carries, read off the uuids a graft leaves
/// behind (`forest-<patch id>-if0`, `forest-<patch id>-relay1`).
///
/// Replaying a patch a creature already contains is not wrong — the graft
/// refuses it on the uuid collision — but it is a wasted replay slot and a
/// confusing line in the log, and the case is common: the fittest creature is
/// very often a descendant of the one the win was filed against.
pub fn grafted_patch_ids(
    creature: &neat_core::CreatureExport,
) -> std::collections::HashSet<String> {
    creature
        .neurons
        .iter()
        .filter_map(|n| {
            let rest = n.uuid.strip_prefix("forest-")?;
            let (id, tail) = rest.split_once('-')?;
            // `forest-one-a` and friends are the shared constants, not a patch.
            (!id.is_empty() && !tail.is_empty() && id != "one").then(|| id.to_string())
        })
        .collect()
}

/// Choose what to replay onto `incumbent`.
///
/// Ordering is by what the fleet knows, best evidence first:
///
/// 1. candidates some host got past the full-corpus scorer, best Δscore first;
/// 2. candidates whose only records are failures old enough to retry, the
///    longest-untried first — so the retry queue drains evenly rather than
///    re-offering the same patch every run.
///
/// Skipped entirely:
///
/// * anything already tried against *this* creature — the journal covers it,
///   and replaying it would spend a scorer call to learn what the file says;
/// * anything `carried`, meaning the creature already contains that patch;
/// * anything measured on a creature of a different width, where a feature
///   index means something else.
///
/// A win the fleet has since rejected more often than it has accepted drops to
/// the retry queue. A patch that helped one creature is worth trying on the
/// next, but not worth a replay slot every run forever once the evidence has
/// turned against it.
pub fn choose(
    all: &[Learning],
    incumbent: &str,
    inputs: usize,
    outputs: usize,
    carried: &std::collections::HashSet<String>,
    cfg: &ReplayConfig,
) -> Vec<Learning> {
    if cfg.max == 0 {
        return Vec::new();
    }
    let mut by_id: HashMap<&str, Vec<&Learning>> = HashMap::new();
    for l in all
        .iter()
        .filter(|l| l.inputs == inputs && l.outputs == outputs)
    {
        by_id.entry(l.id.as_str()).or_default().push(l);
    }
    let mut wins: Vec<(f64, &Learning)> = Vec::new();
    let mut retries: Vec<(u64, &Learning)> = Vec::new();
    for records in by_id.values() {
        if records.iter().any(|l| l.incumbent == incumbent) {
            continue;
        }
        if records
            .iter()
            .all(|l| l.patches().all(|p| carried.contains(&p.id())))
        {
            continue;
        }
        let accepted = records
            .iter()
            .filter(|l| l.outcome == Outcome::Accepted)
            .count();
        let best_win = records
            .iter()
            .filter(|l| l.outcome == Outcome::Accepted)
            .max_by(|a, b| {
                a.delta
                    .total_cmp(&b.delta)
                    .then_with(|| a.at_unix.cmp(&b.at_unix))
            });
        if let Some(win) = best_win
            && records.len() - accepted <= accepted
        {
            wins.push((win.delta, win));
            continue;
        }
        // Only failures. The most recent one decides whether the wait is over.
        let Some(last) = records.iter().max_by_key(|l| l.at_unix) else {
            continue;
        };
        if cfg.now_unix.saturating_sub(last.at_unix) >= cfg.retry_after_secs {
            retries.push((last.at_unix, last));
        }
    }
    wins.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));
    retries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.id.cmp(&b.1.id)));
    wins.into_iter()
        .map(|(_, l)| l)
        .chain(retries.into_iter().map(|(_, l)| l))
        .take(cfg.max)
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learnings::test_support::*;
    use crate::patch::{Node, Patch, Provenance};

    fn nothing_carried() -> std::collections::HashSet<String> {
        std::collections::HashSet::new()
    }

    fn config(max: usize, now: u64) -> ReplayConfig {
        ReplayConfig {
            max,
            retry_after_secs: 100,
            now_unix: now,
        }
    }

    #[test]
    fn a_win_on_another_creature_is_replayed_before_a_retry() {
        let all = vec![
            learning(patch(1, 0.1), Outcome::Accepted, 2e-5, 10, "host-a"),
            learning(patch(2, 0.2), Outcome::Accepted, 9e-5, 20, "host-b"),
            // Old enough to retry.
            learning(patch(3, 0.3), Outcome::Rejected, -1e-5, 30, "host-a"),
        ];
        let picked = choose(
            &all,
            "somewhere-else",
            8,
            1,
            &nothing_carried(),
            &config(8, 1000),
        );
        assert_eq!(
            picked.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(),
            vec![
                patch(2, 0.2).id().as_str(),
                patch(1, 0.1).id().as_str(),
                patch(3, 0.3).id().as_str()
            ],
            "wins by Δscore, then retries"
        );
    }

    #[test]
    fn a_recent_failure_is_left_alone_until_the_wait_is_over() {
        let all = vec![learning(
            patch(1, 0.1),
            Outcome::Rejected,
            -1e-5,
            950,
            "host-a",
        )];
        assert!(
            choose(
                &all,
                "elsewhere",
                8,
                1,
                &nothing_carried(),
                &config(8, 1000)
            )
            .is_empty()
        );
        // …and offered again once `retry_after` has passed.
        assert_eq!(
            choose(
                &all,
                "elsewhere",
                8,
                1,
                &nothing_carried(),
                &config(8, 1100)
            )
            .len(),
            1
        );
    }

    #[test]
    fn a_patch_the_creature_already_carries_is_not_offered_back_to_it() {
        // The fittest creature is usually a descendant of the one a win was
        // filed against, and already contains it.
        let p = patch(1, 0.1);
        let all = vec![learning(p.clone(), Outcome::Accepted, 2e-5, 10, "host-a")];
        let carried: std::collections::HashSet<String> = [p.id()].into_iter().collect();
        assert!(choose(&all, "elsewhere", 8, 1, &carried, &config(8, 1000)).is_empty());
        assert_eq!(
            choose(
                &all,
                "elsewhere",
                8,
                1,
                &nothing_carried(),
                &config(8, 1000)
            )
            .len(),
            1
        );
    }

    #[test]
    fn the_uuids_a_graft_leaves_behind_say_which_patches_a_creature_carries() {
        let mut c = crate::graft::fixtures::identity_creature(3, 1);
        let grafted = crate::graft::graft_patch(
            &c,
            &Patch::new(0, Node::stump(1, 0.0, 0.0, 0.2), Provenance::default()),
        )
        .unwrap();
        c = grafted.creature;
        let carried = grafted_patch_ids(&c);
        let expected = Patch::new(0, Node::stump(1, 0.0, 0.0, 0.2), Provenance::default()).id();
        assert!(carried.contains(&expected), "{carried:?}");
        // The shared bias-1 constants are not patches.
        assert!(!carried.iter().any(|id| id == "one"), "{carried:?}");
    }

    #[test]
    fn a_win_the_fleet_keeps_rejecting_stops_taking_a_replay_slot() {
        let p = patch(1, 0.1);
        let mut all = vec![learning(p.clone(), Outcome::Accepted, 2e-5, 10, "host-a")];
        // One win, one later rejection: still worth another creature.
        all.push(learning(p.clone(), Outcome::Rejected, -1e-6, 20, "host-b"));
        let picked = choose(&all, "elsewhere", 8, 1, &nothing_carried(), &config(8, 30));
        assert_eq!(picked.len(), 1, "still offered while the evidence is even");
        assert_eq!(picked[0].outcome, Outcome::Accepted);
        // A second rejection tips it: it now waits its turn in the retry queue.
        all.push(learning(p, Outcome::Rejected, -2e-6, 25, "host-c"));
        assert!(
            choose(&all, "elsewhere", 8, 1, &nothing_carried(), &config(8, 30)).is_empty(),
            "not offered again until the retry window has passed"
        );
        assert_eq!(
            choose(&all, "elsewhere", 8, 1, &nothing_carried(), &config(8, 200)).len(),
            1
        );
    }

    #[test]
    fn a_known_failure_is_kept_out_of_this_iteration_s_cohort() {
        // The search rediscovers the same split all the time. Spending a
        // full-corpus scorer call to re-prove what the fleet already proved is
        // the mistake this set exists to stop: the slot goes to the next
        // discovery instead.
        let (bad, good) = (patch(1, 0.1), patch(2, 0.2));
        let all = vec![
            learning(bad.clone(), Outcome::Rejected, -1e-5, 990, "host-a"),
            learning(good.clone(), Outcome::Accepted, 2e-5, 990, "host-b"),
        ];
        let avoid = known_failures(&all, &config(8, 1000));
        assert!(avoid.contains(&bad.id()), "the rejected split is avoided");
        assert!(
            !avoid.contains(&good.id()),
            "a patch that worked is never avoided"
        );
        // Once the wait is over the mistake is worth making again — the
        // creature it failed against is long gone.
        assert!(known_failures(&all, &config(8, 2000)).is_empty());
    }

    #[test]
    fn a_patch_that_worked_somewhere_is_never_avoided_even_after_a_failure() {
        let p = patch(1, 0.1);
        let all = vec![
            learning(p.clone(), Outcome::Accepted, 2e-5, 900, "host-a"),
            learning(p.clone(), Outcome::Rejected, -1e-6, 990, "host-b"),
        ];
        assert!(!known_failures(&all, &config(8, 1000)).contains(&p.id()));
    }

    #[test]
    fn what_this_creature_has_already_tried_is_not_offered_again() {
        let mut win = learning(patch(1, 0.1), Outcome::Accepted, 2e-5, 10, "host-a");
        win.incumbent = "here".into();
        assert!(choose(&[win], "here", 8, 1, &nothing_carried(), &config(8, 1000)).is_empty());
    }

    #[test]
    fn a_creature_of_another_width_is_never_replayed_onto() {
        // Islands run their own creatures; a feature index only means the same
        // thing where the widths agree.
        let all = vec![learning(
            patch(1, 0.1),
            Outcome::Accepted,
            2e-5,
            10,
            "host-a",
        )];
        assert!(
            choose(
                &all,
                "elsewhere",
                9,
                1,
                &nothing_carried(),
                &config(8, 1000)
            )
            .is_empty()
        );
        assert!(
            choose(
                &all,
                "elsewhere",
                8,
                2,
                &nothing_carried(),
                &config(8, 1000)
            )
            .is_empty()
        );
        assert_eq!(
            choose(
                &all,
                "elsewhere",
                8,
                1,
                &nothing_carried(),
                &config(8, 1000)
            )
            .len(),
            1
        );
    }
}

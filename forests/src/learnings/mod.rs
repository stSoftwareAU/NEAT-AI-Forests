//! Fleet-shared cache of what has already been tried (Issue #60).
//!
//! A [`crate::patch::Patch`] names **feature indices, thresholds and leaf
//! corrections** — never a neuron uuid. That is what makes this cache possible:
//! the same patch can be grafted onto a different creature, on a different
//! host, in a different island whose neurons share no uuid with ours. What a
//! patch *is* tied to is the corpus it was measured against, so records are
//! filed under the corpus identity and never replayed across corpora.
//!
//! Two things are worth caching, and the fleet learns from both:
//!
//! * **what worked** — a patch some host got past the full-corpus scorer. When
//!   the fittest creature moves on before we can re-apply it, the win is
//!   otherwise lost; replayed onto the new incumbent it usually still helps,
//!   because it corrects the corpus, not the creature.
//! * **what failed** — a patch the full-corpus scorer rejected. Re-deriving it
//!   costs a scorer call every time. Skipping it is the cheapest speed-up there
//!   is, and after `retry_after` has passed it is offered again: a patch that
//!   failed against one creature may well fit the next.
//!
//! Only **full-corpus verdicts** are cached. Two kinds of failure are
//! deliberately left out, because sharing them would cost the fleet more than
//! it saves:
//!
//! * a candidate the **graft refused** — that is a property of the creature it
//!   was tried on (an output squash it cannot enter, a uuid already taken), not
//!   of the patch. Suppressing it fleet-wide would hide a patch that grafts
//!   perfectly onto the next creature, and re-deriving it costs no scorer call.
//! * a candidate the **sampled screen** dropped — the screen ranks, it does not
//!   judge, and it is wrong often enough (Issue #17 measured 52 % false
//!   negatives at a 0.05 sample rate) that recording its opinion as a fleet-wide
//!   failure would bury good patches for a week at a time.
//!
//! ## Layout
//!
//! ```text
//! <root>/corpus-<identity>/<host>.jsonl
//! ```
//!
//! One file per host, so the machines of a fleet that share the directory
//! through a git repository never touch each other's lines and never conflict.
//! Every host reads all of them. A line is one JSON [`Learning`]; the file is
//! append-only, which is also what makes a `git pull --rebase` cheap.
//!
//! Nothing here talks to git: the caller pulls before a run and pushes after
//! it. Nothing here deletes, either — pruning is deliberately somebody else's
//! job (Issue #61), because "very much later" is a fleet-wide policy decision,
//! not something one run should take into its own hands.

mod model;
mod prune;
mod replay;
mod store;
#[cfg(test)]
mod test_support;

pub use model::{Context, LEARNINGS_FORMAT_VERSION, Learning, Outcome, Verdict, file_verdicts};
pub use prune::{PruneOutcome, PrunePolicy, plan_prune};
pub use replay::{
    DEFAULT_RETRY_AFTER_SECS, ReplayConfig, choose, grafted_patch_ids, known_failures,
};
pub use store::{LearningsStore, corpora, default_host, usable_root};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learnings::test_support::*;

    fn nothing_carried() -> std::collections::HashSet<String> {
        std::collections::HashSet::new()
    }

    /// Issue #117 — each concern lives in its own module, and every item is
    /// still reachable at its old `learnings::*` path as the same type.
    #[test]
    fn every_concern_has_its_own_module_and_the_old_paths_still_reach_it() {
        let l: Learning = learning(patch(1, 0.1), model::Outcome::Accepted, 1e-5, 10, "host-a");
        let (kept, out): (Vec<model::Learning>, PruneOutcome) =
            prune::plan_prune(std::slice::from_ref(&l), &policy(20));
        assert_eq!(out.kept, 1);
        let tmp = tempfile::tempdir().unwrap();
        let s: LearningsStore = store::LearningsStore::new(tmp.path(), "c", "host-a");
        s.append(&kept).unwrap();
        assert_eq!(store::corpora(tmp.path()).unwrap(), vec!["c".to_string()]);
        assert_eq!(corpora(tmp.path()).unwrap(), vec!["c".to_string()]);
        let cfg = replay::ReplayConfig {
            max: 1,
            retry_after_secs: replay::DEFAULT_RETRY_AFTER_SECS,
            now_unix: 20,
        };
        let picked = choose(
            &s.load().unwrap(),
            "elsewhere",
            8,
            1,
            &nothing_carried(),
            &cfg,
        );
        assert_eq!(picked, vec![l]);
    }
}

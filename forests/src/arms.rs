//! Randomised candidate-family arms (Issue #147).
//!
//! Four candidate families are off by default until production evidence says
//! whether they pay: refining existing leaves (#140), refitting every leaf
//! (#137), splitting on hidden neurons (#139) and linear leaves (#138).
//! Rather than hand-run an A/B per family, every ordinary run is an
//! experiment: a family flag the caller left **unset** is drawn on or off at
//! random, and the choice is stamped on the creature the run produces as the
//! creature-level tag [`ARMS_TAG`], beside `score`, `forests` and
//! `forests-detail`. Across the champion pool the tags then show which
//! families were on when each gain was made.
//!
//! * An explicit value — `0` included — is always used as given.
//! * The draw is deterministic in the run seed (salted, so it does not share a
//!   stream with candidate generation): `--seed` reproduces it.
//! * Under `--enhancements` every arm Rebase cannot replay is forced off and
//!   never drawn.
//! * `--no-random-arms` leaves every unset arm at its default, off.
//!
//! The tag is written only on a creature this run changed (`write_best` runs
//! on acceptance only), so an inherited tag always describes the run that
//! produced that creature.

use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::config::ForestsConfig;

/// Creature-level tag that records how each arm was decided.
pub const ARMS_TAG: &str = "forests-arms";

/// Salt on the run seed for the arm draw.
const ARM_SEED_SALT: u64 = 0xa5e5_147a_57a6_1e5d;

/// A candidate family whose flag may be drawn at random.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Arm {
    /// `--hidden-features` (Issue #139).
    HiddenFeatures,
    /// `--linear-leaf-conditions` (Issue #138).
    LinearLeafConditions,
    /// `--refine-leaves` (Issue #140).
    RefineLeaves,
    /// `--refit-every` (Issue #137).
    RefitEvery,
}

impl Arm {
    /// Every arm, in tag order (sorted by flag name).
    pub const ALL: [Arm; 4] = [
        Arm::HiddenFeatures,
        Arm::LinearLeafConditions,
        Arm::RefineLeaves,
        Arm::RefitEvery,
    ];

    /// The flag's name without the leading dashes.
    pub fn flag(self) -> &'static str {
        match self {
            Arm::HiddenFeatures => "hidden-features",
            Arm::LinearLeafConditions => "linear-leaf-conditions",
            Arm::RefineLeaves => "refine-leaves",
            Arm::RefitEvery => "refit-every",
        }
    }

    /// The value an arm takes when drawn on: the production protocol lines
    /// G–J in `docs/benchmarks.md`. Refit's very large cadence means "only
    /// the opening refit".
    pub fn on_value(self) -> u64 {
        match self {
            Arm::HiddenFeatures => 256,
            Arm::LinearLeafConditions => 16,
            Arm::RefineLeaves => 8,
            Arm::RefitEvery => 100_000,
        }
    }

    /// Whether NEAT-AI-Rebase can carry what this arm produces. None of the
    /// four can yet: three produce format-v2 patches, and a refit is not a
    /// patch at all, so `--enhancements` refuses every one of them.
    pub fn rebase_can_carry(self) -> bool {
        false
    }

    /// The arm's current value in `cfg`.
    pub fn value(self, cfg: &ForestsConfig) -> u64 {
        match self {
            Arm::HiddenFeatures => cfg.hidden_features as u64,
            Arm::LinearLeafConditions => cfg.linear_leaf_conditions as u64,
            Arm::RefineLeaves => cfg.refine_leaves as u64,
            Arm::RefitEvery => cfg.refit_every,
        }
    }

    fn set(self, cfg: &mut ForestsConfig, value: u64) {
        match self {
            Arm::HiddenFeatures => cfg.hidden_features = value as usize,
            Arm::LinearLeafConditions => cfg.linear_leaf_conditions = value as usize,
            Arm::RefineLeaves => cfg.refine_leaves = value as usize,
            Arm::RefitEvery => cfg.refit_every = value,
        }
    }
}

/// What the caller said about an arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArmPolicy {
    /// Unset: draw it at random from the run seed.
    Draw,
    /// Unset under `--no-random-arms`: leave it at its default, off.
    Default,
}

/// How an arm's value was decided for this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArmSource {
    /// Given explicitly on the command line or in the config.
    Set,
    /// Drawn at random from the run seed.
    Drawn,
    /// Unset, and forced off because `--enhancements` cannot carry it.
    Forced,
    /// Unset under `--no-random-arms`: left at its default, off.
    Default,
}

impl ArmSource {
    fn label(self) -> &'static str {
        match self {
            ArmSource::Set => "set",
            ArmSource::Drawn => "drawn",
            ArmSource::Forced => "forced",
            ArmSource::Default => "default",
        }
    }
}

/// One arm as the run resolved it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmChoice {
    /// The arm.
    pub arm: Arm,
    /// Its value for this run.
    pub value: u64,
    /// How the value was decided.
    pub source: ArmSource,
}

/// Every arm as the run resolved it, in tag order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmsRecord(pub Vec<ArmChoice>);

impl ArmsRecord {
    /// The `forests-arms` tag value: sorted, comma separated
    /// `flag=value:source`, e.g.
    /// `hidden-features=256:drawn,linear-leaf-conditions=0:set,refine-leaves=8:drawn,refit-every=0:forced`.
    pub fn tag(&self) -> String {
        self.0
            .iter()
            .map(|c| format!("{}={}:{}", c.arm.flag(), c.value, c.source.label()))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Resolve every arm of `cfg` for a run seeded with `seed`: draw the ones the
/// caller left to chance, force off what `--enhancements` cannot carry, and
/// return the resolved config with the record of how each arm was decided.
///
/// An arm missing from `cfg.arm_policy` was given explicitly and is used as
/// is, so a programmatic config with no policy draws nothing.
pub fn resolve(cfg: &ForestsConfig, seed: u64) -> (ForestsConfig, ArmsRecord) {
    let mut out = cfg.clone();
    let mut rng = StdRng::seed_from_u64(seed ^ ARM_SEED_SALT);
    let mut choices = Vec::with_capacity(Arm::ALL.len());
    for arm in Arm::ALL {
        // One draw per arm whatever the policy, so one arm's policy never
        // shifts another's draw.
        let coin: bool = rng.random_bool(0.5);
        let (value, source) = match cfg.arm_policy.get(&arm) {
            None => (arm.value(cfg), ArmSource::Set),
            Some(ArmPolicy::Default) => (0, ArmSource::Default),
            Some(ArmPolicy::Draw) if cfg.enhancements && !arm.rebase_can_carry() => {
                (0, ArmSource::Forced)
            }
            Some(ArmPolicy::Draw) => (if coin { arm.on_value() } else { 0 }, ArmSource::Drawn),
        };
        arm.set(&mut out, value);
        choices.push(ArmChoice { arm, value, source });
    }
    // Resolved: nothing is left to draw if this config is resolved again.
    out.arm_policy = BTreeMap::new();
    (out, ArmsRecord(choices))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(policy: ArmPolicy) -> BTreeMap<Arm, ArmPolicy> {
        Arm::ALL.iter().map(|a| (*a, policy)).collect()
    }

    #[test]
    fn an_unset_arm_is_drawn_and_reproducible_from_the_seed() {
        let cfg = ForestsConfig {
            arm_policy: all(ArmPolicy::Draw),
            ..Default::default()
        };
        let (a, ra) = resolve(&cfg, 42);
        let (b, rb) = resolve(&cfg, 42);
        assert_eq!(ra, rb, "same seed, same draw");
        assert_eq!(a.refine_leaves, b.refine_leaves);
        assert!(ra.0.iter().all(|c| c.source == ArmSource::Drawn));
        for c in &ra.0 {
            assert!(c.value == 0 || c.value == c.arm.on_value());
            assert_eq!(c.arm.value(&a), c.value, "the config carries the draw");
        }
        assert!(a.arm_policy.is_empty(), "nothing left to draw");
        // Over many seeds every arm is drawn both ways.
        for arm in Arm::ALL {
            let values: std::collections::HashSet<u64> =
                (0..64).map(|s| arm.value(&resolve(&cfg, s).0)).collect();
            assert_eq!(values.len(), 2, "{arm:?} drawn both on and off");
        }
        // The resolved config is valid.
        a.validate().unwrap();
    }

    #[test]
    fn an_explicit_value_is_never_overridden() {
        let mut policy = all(ArmPolicy::Draw);
        policy.remove(&Arm::RefineLeaves);
        policy.remove(&Arm::HiddenFeatures);
        let cfg = ForestsConfig {
            arm_policy: policy,
            refine_leaves: 0,
            hidden_features: 3,
            ..Default::default()
        };
        for seed in 0..32 {
            let (r, rec) = resolve(&cfg, seed);
            assert_eq!(r.refine_leaves, 0, "explicit 0 stays 0");
            assert_eq!(r.hidden_features, 3, "explicit value stays");
            let refine = rec.0.iter().find(|c| c.arm == Arm::RefineLeaves).unwrap();
            assert_eq!(refine.source, ArmSource::Set);
        }
        // With no policy at all (a programmatic config), nothing is drawn.
        let (r, rec) = resolve(&ForestsConfig::default(), 7);
        assert_eq!(r, ForestsConfig::default());
        assert!(
            rec.0
                .iter()
                .all(|c| c.source == ArmSource::Set && c.value == 0)
        );
    }

    #[test]
    fn enhancements_force_every_unset_arm_off() {
        let cfg = ForestsConfig {
            arm_policy: all(ArmPolicy::Draw),
            enhancements: true,
            ..Default::default()
        };
        for seed in 0..32 {
            let (r, rec) = resolve(&cfg, seed);
            assert!(
                rec.0
                    .iter()
                    .all(|c| c.source == ArmSource::Forced && c.value == 0)
            );
            r.validate().unwrap();
        }
        assert!(resolve(&cfg, 1).1.tag().contains("refine-leaves=0:forced"));
    }

    #[test]
    fn no_random_arms_draws_nothing() {
        let cfg = ForestsConfig {
            arm_policy: all(ArmPolicy::Default),
            ..Default::default()
        };
        for seed in 0..32 {
            let (r, rec) = resolve(&cfg, seed);
            assert!(
                rec.0
                    .iter()
                    .all(|c| c.source == ArmSource::Default && c.value == 0)
            );
            assert_eq!(r.refit_every, 0);
        }
    }

    #[test]
    fn one_arms_policy_does_not_shift_anothers_draw() {
        let draw_all = ForestsConfig {
            arm_policy: all(ArmPolicy::Draw),
            ..Default::default()
        };
        let mut policy = all(ArmPolicy::Draw);
        policy.remove(&Arm::HiddenFeatures);
        let one_set = ForestsConfig {
            arm_policy: policy,
            ..Default::default()
        };
        for seed in 0..32 {
            let a = resolve(&draw_all, seed).0;
            let b = resolve(&one_set, seed).0;
            assert_eq!(a.refine_leaves, b.refine_leaves);
            assert_eq!(a.refit_every, b.refit_every);
            assert_eq!(a.linear_leaf_conditions, b.linear_leaf_conditions);
        }
    }

    #[test]
    fn the_tag_is_sorted_and_machine_readable() {
        let rec = ArmsRecord(vec![
            ArmChoice {
                arm: Arm::HiddenFeatures,
                value: 256,
                source: ArmSource::Drawn,
            },
            ArmChoice {
                arm: Arm::LinearLeafConditions,
                value: 0,
                source: ArmSource::Set,
            },
            ArmChoice {
                arm: Arm::RefineLeaves,
                value: 8,
                source: ArmSource::Drawn,
            },
            ArmChoice {
                arm: Arm::RefitEvery,
                value: 0,
                source: ArmSource::Forced,
            },
        ]);
        assert_eq!(
            rec.tag(),
            "hidden-features=256:drawn,linear-leaf-conditions=0:set,refine-leaves=8:drawn,refit-every=0:forced"
        );
        let mut names: Vec<&str> = Arm::ALL.iter().map(|a| a.flag()).collect();
        let sorted = {
            let mut s = names.clone();
            s.sort_unstable();
            s
        };
        assert_eq!(names, sorted, "Arm::ALL is in tag (flag-name) order");
        names.dedup();
        assert_eq!(names.len(), 4);
    }
}

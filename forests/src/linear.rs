//! Linear leaves: a split whose two sides each carry an intercept **and** a
//! slope on a second input (Issue #138).
//!
//! Every leaf Forests otherwise builds is a constant: "in this region, add
//! `c`". It cannot say "in this region the slope on signal `k` is `b`", nor the
//! thing that matters most on a low-signal target — the same signal helps in
//! one regime and hurts in the other:
//!
//! ```text
//! IF  condition:  x_j − t
//!     positive:   a_R · 1  +  b_R · x_k
//!     negative:   a_L · 1  +  b_L · x_k
//! ```
//!
//! Averaged over both regimes a signal whose slope flips sign has almost no
//! correlation with the residual, and a split on the regime alone with
//! constant leaves shows almost no gain, so neither the greedy constant-leaf
//! search nor mutation finds it. The `IF` kernel already sums any sources per
//! role, so this needs no change to NEAT-AI-core or the scorer — only a search
//! and a leaf that carries slopes ([`Node::linear_leaf`]).
//!
//! ## Search
//!
//! For each condition in a pool of `R`, one masked histogram pass gives, per
//! side and per bin of every feature `k`, `Σw` and `Σw·r`. With `v_b` the bin's
//! representative value, the side's `Σw·x`, `Σw·x²` and `Σw·r·x` follow, and so
//! the closed-form weighted least-squares intercept, slope and SSE gain of
//! every `(condition, signal)` pair: `R × F` pairs for `R` passes.
//!
//! The pool is **not** ranked by constant-leaf gain alone, because that is the
//! very signal a sign flip hides: it mixes the iteration's best ordinary
//! splits, conditions already on the creature, and seeded median and zero
//! splits on a random feature subset.

use std::collections::HashSet;

use rand::Rng;
use rand::rngs::StdRng;

use crate::bins::BinCache;
use crate::candidates::Discovery;
use crate::histogram::{HistogramSet, SearchControls, StumpCandidate};
use crate::patch::{Condition, Node};
use crate::strategies::SearchSet;

/// One condition of the pool, in set-feature space: `bin(row[feature]) > bin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolCondition {
    /// Set-feature index.
    pub feature: usize,
    /// Split after this bin.
    pub bin: usize,
    /// Where it came from.
    pub origin: PoolOrigin,
}

/// Where a pool condition came from (journalled in the notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolOrigin {
    /// One of this iteration's best ordinary stumps.
    Stump,
    /// A condition already on the creature.
    Creature,
    /// A seeded median split.
    Median,
    /// A seeded split at zero.
    Zero,
}

impl PoolOrigin {
    fn label(self) -> &'static str {
        match self {
            Self::Stump => "stump",
            Self::Creature => "creature",
            Self::Median => "median",
            Self::Zero => "zero",
        }
    }
}

/// Build a pool of up to `size` conditions: a third from `stumps`, a third
/// from `existing` (axis conditions already on the creature, in original
/// feature space), the rest seeded median and zero splits on random features.
pub fn condition_pool(
    stumps: &[StumpCandidate],
    existing: &[Condition],
    set: &SearchSet,
    bins: &BinCache,
    size: usize,
    rng: &mut StdRng,
) -> Vec<PoolCondition> {
    let mut out: Vec<PoolCondition> = Vec::new();
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut push = |c: PoolCondition, out: &mut Vec<PoolCondition>| {
        if out.len() < size && seen.insert((c.feature, c.bin)) {
            out.push(c);
        }
    };
    let third = size.div_ceil(3);
    for s in stumps.iter().take(third) {
        push(
            PoolCondition {
                feature: s.feature,
                bin: s.bin,
                origin: PoolOrigin::Stump,
            },
            &mut out,
        );
    }
    let set_index: std::collections::HashMap<usize, usize> = set
        .feature_map
        .iter()
        .enumerate()
        .map(|(i, &f)| (f, i))
        .collect();
    let limit = out.len() + third;
    for c in existing {
        if out.len() >= limit {
            break;
        }
        if !c.is_axis_aligned() {
            continue;
        }
        let Some(&f) = set_index.get(&c.terms[0].feature) else {
            continue;
        };
        let edges = &bins.edges[set.feature_map[f]];
        if edges.is_empty() {
            continue;
        }
        let b = edges
            .partition_point(|&e| e < c.threshold)
            .min(edges.len() - 1);
        push(
            PoolCondition {
                feature: f,
                bin: b,
                origin: PoolOrigin::Creature,
            },
            &mut out,
        );
    }
    let features = set.features();
    let mut attempts = 0;
    while out.len() < size && features > 0 && attempts < 8 * size {
        attempts += 1;
        let f = rng.random_range(0..features);
        let edges = &bins.edges[set.feature_map[f]];
        if edges.len() < 2 {
            continue;
        }
        push(
            PoolCondition {
                feature: f,
                bin: edges.len() / 2,
                origin: PoolOrigin::Median,
            },
            &mut out,
        );
        let zero = edges.partition_point(|&e| e < 0.0);
        if zero > 0 && zero < edges.len() {
            push(
                PoolCondition {
                    feature: f,
                    bin: zero,
                    origin: PoolOrigin::Zero,
                },
                &mut out,
            );
        }
    }
    out
}

/// Representative value of every bin of a feature with `edges`: the midpoint
/// of its two edges, and half a neighbouring width beyond the outer ones.
pub fn bin_values(edges: &[f32]) -> Vec<f64> {
    let m = edges.len();
    let e = |i: usize| f64::from(edges[i]);
    (0..=m)
        .map(|b| match (b, m) {
            (_, 0) => 0.0,
            (0, 1) | (1, 1) => e(0),
            (0, _) => e(0) - (e(1) - e(0)) / 2.0,
            (b, m) if b == m => e(m - 1) + (e(m - 1) - e(m - 2)) / 2.0,
            (b, _) => (e(b - 1) + e(b)) / 2.0,
        })
        .collect()
}

/// One side's sufficient statistics for a slope on one feature.
#[derive(Debug, Clone, Copy, Default)]
struct Side {
    n: f64,
    sx: f64,
    sxx: f64,
    sr: f64,
    srx: f64,
}

impl Side {
    fn var(&self) -> f64 {
        self.sxx - self.sx * self.sx / self.n
    }
    fn cov(&self) -> f64 {
        self.srx - self.sx * self.sr / self.n
    }
    /// Intercept, slope and SSE reduction of `r ≈ a + b·x` against `0`.
    fn fit(&self) -> (f64, f64, f64) {
        if self.n <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let var = self.var();
        let mean_gain = self.sr * self.sr / self.n;
        if var <= 1e-12 * self.n.max(1.0) {
            return (self.sr / self.n, 0.0, mean_gain);
        }
        let b = self.cov() / var;
        let a = (self.sr - b * self.sx) / self.n;
        (a, b, mean_gain + self.cov() * self.cov() / var)
    }
}

/// A fitted linear stump.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearStump {
    /// The condition (set-feature space).
    pub condition: PoolCondition,
    /// Set-feature index of the slope signal.
    pub slope_feature: usize,
    /// `(intercept, slope)` left and right.
    pub left: (f64, f64),
    /// `(intercept, slope)` right.
    pub right: (f64, f64),
    /// Predicted SSE reduction.
    pub gain: f64,
    /// What constant leaves on the same condition would gain.
    pub constant_gain: f64,
    /// Weighted records left and right.
    pub records: (f64, f64),
    /// Shared-intercept fit `(a, b_L, b_R)` and its gain, when solvable.
    pub shared: Option<((f64, f64, f64), f64)>,
}

fn side(hist_feature: &crate::histogram::FeatureHistogram, values: &[f64]) -> Side {
    let mut s = Side::default();
    for (b, (&c, &r)) in hist_feature.count.iter().zip(&hist_feature.sum).enumerate() {
        let v = values.get(b).copied().unwrap_or(0.0);
        s.n += c;
        s.sx += c * v;
        s.sxx += c * v * v;
        s.sr += r;
        s.srx += r * v;
    }
    s
}

/// `(a, b_L, b_R)` minimising both sides' SSE with one shared intercept, and
/// its gain.
fn shared_fit(l: &Side, r: &Side) -> Option<((f64, f64, f64), f64)> {
    // Normal equations, with Σ over the left (ₗ) and right (ᵣ) sides:
    // [ n      Σxₗ    Σxᵣ  ] [a ]   [ Σr      ]
    // [ Σxₗ    Σx²ₗ   0    ] [bL] = [ Σr·xₗ   ]
    // [ Σxᵣ    0      Σx²ᵣ ] [bR]   [ Σr·xᵣ   ]
    let (n, x_left, x_right) = (l.n + r.n, l.sx, r.sx);
    if l.sxx <= 0.0 || r.sxx <= 0.0 {
        return None;
    }
    let det = n - x_left * x_left / l.sxx - x_right * x_right / r.sxx;
    if det.abs() <= 1e-12 * n.max(1.0) {
        return None;
    }
    let a = (l.sr + r.sr - x_left * l.srx / l.sxx - x_right * r.srx / r.sxx) / det;
    let bl = (l.srx - x_left * a) / l.sxx;
    let br = (r.srx - x_right * a) / r.sxx;
    // SSE reduction = 2·θᵀc − θᵀMθ with θ = (a, bL, bR).
    let c = 2.0 * (a * (l.sr + r.sr) + bl * l.srx + br * r.srx);
    let m = a * a * n + bl * bl * l.sxx + br * br * r.sxx + 2.0 * a * (bl * x_left + br * x_right);
    let gain = c - m;
    (a.is_finite() && bl.is_finite() && br.is_finite() && gain.is_finite())
        .then_some(((a, bl, br), gain))
}

/// Fit every `(condition, slope feature)` pair of `pool` on `set` and return
/// the best `keep`, by gain.
///
/// # Errors
///
/// A histogram accumulation failure.
pub fn search_linear_stumps(
    set: &SearchSet,
    bins: &BinCache,
    pool: &[PoolCondition],
    controls: &SearchControls,
    input: usize,
    threads: usize,
    keep: usize,
) -> Result<Vec<LinearStump>, String> {
    let bins_per_feature = set.bins_per_feature(bins);
    let total = HistogramSet::from_source_threads(&set.source, &bins_per_feature, threads)?;
    let values: Vec<Vec<f64>> = set
        .feature_map
        .iter()
        .map(|&f| bin_values(&bins.edges[f]))
        .collect();
    let nf = set.features();
    let mut found = Vec::new();
    for &c in pool {
        let mut right = HistogramSet::new(&bins_per_feature);
        for chunk in &set.source.chunks {
            let mask: Vec<bool> = (0..chunk.records)
                .map(|r| usize::from(chunk.bins[r * nf + c.feature]) > c.bin)
                .collect();
            right.accumulate(chunk, Some(&mask));
        }
        let mut left = total.clone();
        left.subtract(&right);
        if left.total_count < controls.min_leaf_records
            || right.total_count < controls.min_leaf_records
        {
            continue;
        }
        let constant_gain = left.total_sum * left.total_sum / left.total_count
            + right.total_sum * right.total_sum / right.total_count;
        let mut best: Option<LinearStump> = None;
        for (k, value) in values.iter().enumerate() {
            // A slope must read an input: hidden columns (Issue #139) have no
            // raw value the graft could wire in.
            if set.feature_map[k] >= input {
                continue;
            }
            let l = side(&left.features[k], value);
            let r = side(&right.features[k], value);
            let (al, bl, gl) = l.fit();
            let (ar, br, gr) = r.fit();
            let gain = gl + gr;
            if !gain.is_finite()
                || al.abs() > controls.max_correction
                || ar.abs() > controls.max_correction
                || (bl == 0.0 && br == 0.0)
            {
                continue;
            }
            if best.as_ref().is_none_or(|b| gain > b.gain) {
                best = Some(LinearStump {
                    condition: c,
                    slope_feature: k,
                    left: (al, bl),
                    right: (ar, br),
                    gain,
                    constant_gain,
                    records: (l.n, r.n),
                    shared: shared_fit(&l, &r),
                });
            }
        }
        if let Some(b) = best
            && b.gain > constant_gain
            && b.gain > controls.min_gain
        {
            found.push(b);
        }
    }
    found.sort_by(|a, b| {
        b.gain
            .total_cmp(&a.gain)
            .then(a.condition.feature.cmp(&b.condition.feature))
            .then(a.condition.bin.cmp(&b.condition.bin))
            .then(a.slope_feature.cmp(&b.slope_feature))
    });
    found.truncate(keep);
    Ok(found)
}

impl LinearStump {
    fn node(
        &self,
        set: &SearchSet,
        bins: &BinCache,
        left: (f64, f64),
        right: (f64, f64),
    ) -> Option<Node> {
        let f = set.feature_map[self.condition.feature];
        let k = set.feature_map[self.slope_feature];
        let threshold = bins.threshold(f, self.condition.bin)?;
        let leaf = |(a, b): (f64, f64)| {
            if b == 0.0 {
                Node::leaf(a as f32)
            } else {
                Node::linear_leaf(a as f32, vec![(k, b as f32)])
            }
        };
        Some(Node::Split {
            condition: Condition::axis(f, threshold),
            left: Box::new(leaf(left)),
            right: Box::new(leaf(right)),
        })
    }

    /// The stump and, when `variants`, its shrunk-slope, one-sided-slope and
    /// shared-intercept variants, as discoveries.
    pub fn discoveries(&self, set: &SearchSet, bins: &BinCache, variants: bool) -> Vec<Discovery> {
        let mut out = Vec::new();
        let mut push = |strategy: &str, left: (f64, f64), right: (f64, f64), gain: f64| {
            if let Some(root) = self.node(set, bins, left, right) {
                out.push(Discovery {
                    root,
                    strategy: strategy.to_string(),
                    backend: "cpu".into(),
                    gain,
                    affected: self.records.0 + self.records.1,
                    side_records: Some(self.records),
                    notes: vec![
                        format!(
                            "condition-origin={} slope-input={}",
                            self.condition.origin.label(),
                            set.feature_map[self.slope_feature]
                        ),
                        format!("constant-gain={:.6e}", self.constant_gain),
                    ],
                    parent: None,
                });
            }
        };
        push("linear-leaf", self.left, self.right, self.gain);
        if variants {
            let half = |(a, b): (f64, f64)| (a, 0.5 * b);
            push(
                "linear-leaf/half-slope",
                half(self.left),
                half(self.right),
                0.0,
            );
            push(
                "linear-leaf/right-slope",
                (self.left.0, 0.0),
                self.right,
                0.0,
            );
            push(
                "linear-leaf/left-slope",
                self.left,
                (self.right.0, 0.0),
                0.0,
            );
            if let Some(((a, bl, br), gain)) = self.shared {
                push("linear-leaf/shared-intercept", (a, bl), (a, br), gain);
            }
        }
        out
    }
}

/// Test and parity fixtures (Issue #138).
pub mod fixtures {
    use neat_core::CreatureExport;

    use crate::graft::graft_patch;
    use crate::patch::{Condition, Node, Patch, Provenance};

    /// The sign-flip shape: `x_j > 0 ? a_R + b·x_k : a_L − b·x_k`, the same
    /// input in both branch roles with opposite signs.
    pub fn sign_flip(j: usize, k: usize) -> Node {
        Node::Split {
            condition: Condition::axis(j, 0.0),
            left: Box::new(Node::linear_leaf(0.05, vec![(k, -0.4)])),
            right: Box::new(Node::linear_leaf(-0.02, vec![(k, 0.4)])),
        }
    }

    /// `base` with [`sign_flip`]`(0, 1)` and, stacked on it, a linear stump
    /// whose slope reads the condition's own input — three roles from one
    /// source into one `IF`.
    ///
    /// # Panics
    ///
    /// When `base` cannot take the graft; this is a fixture.
    pub fn linear_stumps(base: &CreatureExport) -> CreatureExport {
        let first = graft_patch(base, &Patch::new(0, sign_flip(0, 1), Provenance::default()))
            .expect("graft")
            .creature;
        let same_input = Node::Split {
            condition: Condition::axis(2, 0.1),
            left: Box::new(Node::leaf(0.03)),
            right: Box::new(Node::linear_leaf(0.0, vec![(2, -0.3)])),
        };
        graft_patch(&first, &Patch::new(0, same_input, Provenance::default()))
            .expect("graft")
            .creature
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bins::{BinBuildOptions, build_bin_cache};
    use crate::config::ForestsConfig;
    use crate::corpus::{corpus_info, write_bin_file};
    use crate::graft::fixtures::{identity_creature, if_output_creature, min_clamped_if_creature};
    use crate::graft::graft_patch;
    use crate::histogram::search_stumps;
    use crate::incumbent::Incumbent;
    use crate::patch::{Patch, Provenance};
    use crate::residuals::compute_residuals;
    use crate::strategies::build_search_set;
    use neat_core::compile_creature;
    use neat_core::training_data::TrainingDataConfig;
    use rand::SeedableRng;

    fn uniform(n: usize, width: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut seed = seed;
        (0..n)
            .map(|_| {
                (0..width)
                    .map(|_| {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((seed >> 40) as f32) / (1u64 << 23) as f32 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn bin_values_sit_inside_their_bins() {
        let edges = [-1.0f32, 0.0, 0.5, 2.0];
        let v = bin_values(&edges);
        assert_eq!(v, vec![-1.5, -0.5, 0.25, 1.25, 2.75]);
        assert_eq!(bin_values(&[0.3]), vec![f64::from(0.3f32); 2]);
        assert_eq!(bin_values(&[]), vec![0.0]);
    }

    /// Issue #138 — linear stumps graft to a creature that agrees with the
    /// abstract evaluator record by record, including one source in both
    /// branch roles with opposite signs and one source in all three roles.
    #[test]
    fn linear_leaves_match_the_abstract_evaluator_record_by_record() {
        for base in [identity_creature(4, 1), if_output_creature(4)] {
            let flip = Patch::new(0, fixtures::sign_flip(0, 1), Provenance::default());
            let grafted = fixtures::linear_stumps(&base);
            let same_input = Node::Split {
                condition: Condition::axis(2, 0.1),
                left: Box::new(Node::leaf(0.03)),
                right: Box::new(Node::linear_leaf(0.0, vec![(2, -0.3)])),
            };
            let mut b = compile_creature(&base).unwrap();
            let mut g = compile_creature(&grafted).unwrap();
            for x in uniform(600, 4, 11) {
                let expected = flip.evaluate(&x) + same_input.evaluate(&x);
                let got = g.activate(&x, 1)[0] - b.activate(&x, 1)[0];
                assert!((got - expected).abs() < 1e-6, "{got} vs {expected}");
            }
        }
        // Behind a clamp the graft attaches at the body, gain 0.5, and the
        // slopes stay in output space wherever the clamp does not bind.
        let clamped = graft_patch(
            &min_clamped_if_creature(4),
            &Patch::new(0, fixtures::sign_flip(0, 1), Provenance::default()),
        )
        .unwrap();
        assert!(
            clamped
                .creature
                .synapses
                .iter()
                .any(|s| s.from_uuid == "input-1"
                    && s.synapse_type.as_deref() == Some("positive")
                    && s.to_uuid != "body-if")
        );
    }

    /// Issue #138 — a target of `sign(x_1) · x_2` plus noise. The greedy
    /// constant-leaf search sees nothing at the root for either feature and
    /// ranks them with the noise features; the linear-leaf search finds the
    /// planted pair, and grafting it removes most of the residual.
    #[test]
    fn a_sign_flipped_slope_is_found_only_by_the_linear_leaf_search() {
        let tmp = tempfile::tempdir().unwrap();
        let noise = uniform(8000, 1, 99);
        let recs: Vec<(Vec<f32>, Vec<f32>)> = uniform(8000, 5, 5)
            .into_iter()
            .zip(noise)
            .map(|(x, e)| {
                let flip = if x[1] > 0.0 { x[2] } else { -x[2] };
                let y = x[0] + 0.5 * flip + 0.05 * e[0];
                (x, vec![y])
            })
            .collect();
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let tcfg = TrainingDataConfig::new(5, 1);
        let corpus = corpus_info(tmp.path(), &tcfg).unwrap();
        let bins = build_bin_cache(
            tmp.path(),
            &tcfg,
            &corpus,
            &BinBuildOptions {
                bins: 32,
                ..Default::default()
            },
        )
        .unwrap();
        let cfg = ForestsConfig {
            search_records: 0,
            min_leaf_records: 50.0,
            ..Default::default()
        };
        let base = identity_creature(5, 1);
        let inc = Incumbent::from_creature(base.clone(), "t").unwrap();
        let res = compute_residuals(&inc, tmp.path(), &corpus, 512, 1).unwrap();
        let before = res.meta.local_mse;
        let set = build_search_set(&cfg, &bins, &res, tmp.path(), 0, 1).unwrap();
        let controls = cfg.search_controls();
        let hist = HistogramSet::from_source(&set.source, &set.bins_per_feature(&bins)).unwrap();
        // Best constant stump per feature: the planted pair is no better than noise.
        let per_feature = |f: usize| {
            let c = SearchControls {
                top_k: usize::MAX,
                max_per_feature: 0,
                ..controls.clone()
            };
            search_stumps(&hist, &set.threshold(&bins), &c, "cpu")
                .into_iter()
                .filter(|s| set.feature_map[s.feature] == f)
                .map(|s| s.gain)
                .fold(0.0, f64::max)
        };
        let noise_best = per_feature(3).max(per_feature(4));
        for f in [1, 2] {
            assert!(
                per_feature(f) < 3.0 * noise_best.max(1e-9),
                "feature {f} gains {} against noise {noise_best}",
                per_feature(f)
            );
        }
        let mut rng = StdRng::seed_from_u64(3);
        let stumps = search_stumps(&hist, &set.threshold(&bins), &controls, "cpu");
        let pool = condition_pool(&stumps, &[], &set, &bins, 24, &mut rng);
        assert!(pool.iter().any(|c| c.origin == PoolOrigin::Median));
        let found = search_linear_stumps(&set, &bins, &pool, &controls, 5, 1, 4).unwrap();
        let best = &found[0];
        assert_eq!(set.feature_map[best.condition.feature], 1, "{best:?}");
        assert_eq!(set.feature_map[best.slope_feature], 2, "{best:?}");
        assert!(
            best.left.1 < 0.0 && best.right.1 > 0.0,
            "the slope flips: {best:?}"
        );
        assert!(best.gain > 20.0 * best.constant_gain.max(1e-9));
        let d = &best.discoveries(&set, &bins, true)[0];
        let patch = Patch::new(0, d.root.clone(), Provenance::default());
        let grafted = graft_patch(&base, &patch).unwrap().creature;
        let inc = Incumbent::from_creature(grafted, "g").unwrap();
        let after = compute_residuals(&inc, tmp.path(), &corpus, 512, 1)
            .unwrap()
            .meta
            .local_mse;
        assert!(
            after < 0.25 * before,
            "linear leaf left {after} from {before}"
        );
        // Every variant grafts.
        for v in best.discoveries(&set, &bins, true) {
            graft_patch(&base, &Patch::new(0, v.root, Provenance::default()))
                .unwrap_or_else(|e| panic!("{}: {e}", v.strategy));
        }
    }

    #[test]
    fn the_shared_intercept_fit_solves_its_normal_equations() {
        let l = Side {
            n: 10.0,
            sx: 2.0,
            sxx: 5.0,
            sr: 1.0,
            srx: 3.0,
        };
        let r = Side {
            n: 8.0,
            sx: -1.0,
            sxx: 4.0,
            sr: -0.5,
            srx: 2.0,
        };
        let ((a, bl, br), gain) = shared_fit(&l, &r).unwrap();
        let eq1 = 18.0 * a + 2.0 * bl - 1.0 * br - 0.5;
        let eq2 = 2.0 * a + 5.0 * bl - 3.0;
        let eq3 = -a + 4.0 * br - 2.0;
        for e in [eq1, eq2, eq3] {
            assert!(e.abs() < 1e-12, "{e}");
        }
        assert!(gain > 0.0);
    }
}

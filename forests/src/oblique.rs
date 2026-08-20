//! Oblique (multi-feature linear) `IF` splits (Issue #14).
//!
//! `w1·x1 + w2·x2 (+ w3·x3) > threshold`. Candidates are generated from a
//! promising axis-aligned split plus a second feature, or as random sparse
//! combinations, with coefficients normalised by each feature's robust scale.
//! Evaluation is brute force on the raw sample (project, sort, prefix scan) and
//! a few rounds of coordinate jitter refine the hyperplane. All of it is a
//! heuristic proposal; the scorer judges.

use rand::Rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

use crate::histogram::{SearchControls, StumpKind};
use crate::patch::{Condition, Node, Term};
use crate::strategies::RawSample;

/// An oblique split proposal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObliqueCandidate {
    /// Root split.
    pub root: Node,
    /// Predicted SSE reduction.
    pub gain: f64,
    /// Records corrected.
    pub affected_records: f64,
    /// Records in the sample.
    pub total_records: f64,
    /// How it was generated (`axis+1`, `random-sparse`, `jitter`).
    pub origin: String,
}

/// Oblique search controls.
#[derive(Debug, Clone, PartialEq)]
pub struct ObliqueControls {
    /// Stump controls (min leaf, clamp).
    pub stump: SearchControls,
    /// Candidates to return.
    pub count: usize,
    /// Random sparse combinations to try.
    pub random_combos: usize,
    /// Coordinate-jitter rounds per hyperplane.
    pub jitter_rounds: usize,
    /// Maximum terms (2 or 3).
    pub max_terms: usize,
}

/// Best threshold for the projection `p = Σ w·x` (one-sided and two-leaf kinds).
fn best_threshold(
    p: &[f32],
    residual: &[f32],
    weight: &[f32],
    controls: &SearchControls,
) -> Option<(f32, f32, f32, f64, f64)> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let total_n: f64 = weight.iter().map(|&w| f64::from(w)).sum();
    let total_s: f64 = residual
        .iter()
        .zip(weight)
        .map(|(&r, &w)| f64::from(r) * f64::from(w))
        .sum();
    let (mut nl, mut sl) = (0.0, 0.0);
    let mut best: Option<(f32, f32, f32, f64, f64)> = None;
    for (k, &i) in idx.iter().enumerate() {
        nl += f64::from(weight[i]);
        sl += f64::from(residual[i]) * f64::from(weight[i]);
        let t = p[i];
        if !t.is_finite() || idx.get(k + 1).is_some_and(|&j| p[j] == t) {
            continue;
        }
        let (nr, sr) = (total_n - nl, total_s - sl);
        if nl < controls.min_leaf_records || nr < controls.min_leaf_records {
            continue;
        }
        let clamp =
            |s: f64, n: f64| (s / n).clamp(-controls.max_correction, controls.max_correction);
        let (cl, cr) = (clamp(sl, nl), clamp(sr, nr));
        let (gl, gr) = (2.0 * cl * sl - nl * cl * cl, 2.0 * cr * sr - nr * cr * cr);
        for kind in &controls.kinds {
            let (l, r, g, a) = match kind {
                StumpKind::LeftOnly => (cl, 0.0, gl, nl),
                StumpKind::RightOnly => (0.0, cr, gr, nr),
                StumpKind::TwoLeaf => (cl, cr, gl + gr, nl + nr),
            };
            if g > 0.0 && best.as_ref().is_none_or(|b| g > b.3) {
                best = Some((t, l as f32, r as f32, g, a));
            }
        }
    }
    best
}

fn project(sample: &RawSample, cols: &[usize], weights: &[f32], out: &mut Vec<f32>) {
    let k = sample.features.len();
    out.clear();
    out.extend(sample.values.chunks_exact(k).map(|row| {
        let mut s = 0.0f32;
        for (&c, &w) in cols.iter().zip(weights) {
            s += row[c] * w;
        }
        s
    }));
}

fn evaluate(
    sample: &RawSample,
    cols: &[usize],
    weights: &[f32],
    controls: &SearchControls,
    origin: &str,
    buf: &mut Vec<f32>,
) -> Option<ObliqueCandidate> {
    project(sample, cols, weights, buf);
    let (t, l, r, gain, affected) =
        best_threshold(buf, &sample.residual, &sample.weight, controls)?;
    let root = Node::Split {
        condition: Condition {
            terms: cols
                .iter()
                .zip(weights)
                .map(|(&c, &w)| Term {
                    feature: sample.features[c],
                    weight: w,
                })
                .collect(),
            threshold: t,
        },
        left: Box::new(Node::leaf(l)),
        right: Box::new(Node::leaf(r)),
    };
    Some(ObliqueCandidate {
        root,
        gain,
        affected_records: affected,
        total_records: sample.residual.len() as f64,
        origin: origin.into(),
    })
}

/// Search oblique splits over `sample`. `scales[c]` is the robust scale of
/// column `c`; `seed_axis` optionally names a column whose axis split is the
/// starting point.
pub fn search_oblique(
    sample: &RawSample,
    scales: &[f32],
    controls: &ObliqueControls,
    seed_axis: Option<usize>,
    rng: &mut StdRng,
) -> Vec<ObliqueCandidate> {
    let k = sample.features.len();
    if k < 2 || sample.records() == 0 {
        return Vec::new();
    }
    let mut buf = Vec::with_capacity(sample.records());
    let mut found: Vec<ObliqueCandidate> = Vec::new();
    let norm = |c: usize, w: f32| w / scales[c].max(1e-12);
    // 1. axis split + one correlated/random second feature.
    if let Some(a) = seed_axis {
        for b in 0..k {
            if b == a {
                continue;
            }
            for sign in [1.0f32, -1.0] {
                let cols = [a, b];
                let w = [norm(a, 1.0), norm(b, 0.5 * sign)];
                if let Some(c) = evaluate(sample, &cols, &w, &controls.stump, "axis+1", &mut buf) {
                    found.push(c);
                }
            }
        }
    }
    // 2. random sparse combinations.
    let max_terms = controls.max_terms.clamp(2, 3).min(k);
    for _ in 0..controls.random_combos {
        let terms = rng.random_range(2..=max_terms);
        let mut cols: Vec<usize> = Vec::new();
        while cols.len() < terms {
            let c = rng.random_range(0..k);
            if !cols.contains(&c) {
                cols.push(c);
            }
        }
        let w: Vec<f32> = cols
            .iter()
            .map(|&c| norm(c, rng.random_range(-1.0f32..1.0)))
            .collect();
        if let Some(c) = evaluate(
            sample,
            &cols,
            &w,
            &controls.stump,
            "random-sparse",
            &mut buf,
        ) {
            found.push(c);
        }
    }
    // 3. coordinate jitter around the best so far.
    found.sort_by(|a, b| b.gain.total_cmp(&a.gain));
    found.truncate(controls.count.max(1));
    let mut refined = Vec::new();
    for cand in &found {
        let Node::Split { condition, .. } = &cand.root else {
            continue;
        };
        let cols: Vec<usize> = condition
            .terms
            .iter()
            .map(|t| {
                sample
                    .features
                    .iter()
                    .position(|&f| f == t.feature)
                    .unwrap()
            })
            .collect();
        let mut weights: Vec<f32> = condition.terms.iter().map(|t| t.weight).collect();
        let mut best = cand.clone();
        for _ in 0..controls.jitter_rounds {
            for i in 0..weights.len() {
                for delta in [0.8f32, 1.25] {
                    let mut w = weights.clone();
                    w[i] *= delta;
                    if let Some(c) =
                        evaluate(sample, &cols, &w, &controls.stump, "jitter", &mut buf)
                        && c.gain > best.gain
                    {
                        best = c;
                        weights = w;
                    }
                }
            }
        }
        refined.push(best);
    }
    refined.sort_by(|a, b| b.gain.total_cmp(&a.gain));
    refined.truncate(controls.count);
    refined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::histogram::brute_force_best_stump;
    use rand::SeedableRng;

    #[test]
    fn oblique_boundary_beats_every_single_feature_stump() {
        // residual = 0.5 if x0 + x1 > 0 else -0.5
        let mut seed = 3u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        let mut values = Vec::new();
        let mut residual = Vec::new();
        for _ in 0..1500 {
            let (a, b, c) = (next(), next(), next());
            values.push(vec![a, b, c]);
            residual.push(if a + b > 0.0 { 0.5 } else { -0.5 });
        }
        let sample = RawSample {
            features: vec![0, 1, 2],
            values: values.iter().flatten().copied().collect(),
            residual: residual.clone(),
            weight: vec![1.0; 1500],
        };
        let controls = SearchControls {
            min_leaf_records: 20.0,
            ..Default::default()
        };
        let sse: f64 = residual.iter().map(|r| f64::from(*r) * f64::from(*r)).sum();
        let best_stump = (0..3)
            .filter_map(|f| brute_force_best_stump(&values, &residual, f, &controls))
            .map(|c| c.gain)
            .fold(0.0, f64::max);
        let ob = ObliqueControls {
            stump: controls,
            count: 3,
            random_combos: 20,
            jitter_rounds: 3,
            max_terms: 2,
        };
        let mut rng = StdRng::seed_from_u64(1);
        let found = search_oblique(&sample, &[1.0, 1.0, 1.0], &ob, Some(0), &mut rng);
        let top = &found[0];
        assert!(
            top.gain > 1.5 * best_stump,
            "oblique {} vs stump {best_stump}",
            top.gain
        );
        assert!(top.gain > 0.7 * sse);
        let hits = values
            .iter()
            .zip(&residual)
            .filter(|(v, r)| (top.root.evaluate(v) - **r).abs() < 0.25)
            .count();
        assert!(hits > 1300, "{hits}");
        // Reproducible with the seed.
        let mut rng2 = StdRng::seed_from_u64(1);
        assert_eq!(
            found,
            search_oblique(&sample, &[1.0, 1.0, 1.0], &ob, Some(0), &mut rng2)
        );
    }
}

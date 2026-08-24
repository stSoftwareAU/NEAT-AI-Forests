//! Depth-2/3 residual trees (Issue #11).
//!
//! Trees are grown on the same quantised search set and histogram machinery
//! as stumps. Child-region statistics are recomputed only for records that
//! reach the branch (a per-record membership walk over the path), either
//! level-wise (one pass per depth) or best-first (one pass per split).
//! Leaf corrections are the clamped region mean; the reported gain is the
//! total SSE reduction `Σ s²/n` over leaves — a proxy, never an acceptance
//! metric.

use serde::{Deserialize, Serialize};

use crate::config::GrowthPolicy;
use crate::histogram::{
    ChunkSource, HistogramSet, SearchControls, StumpCandidate, StumpKind, rank_order, search_stumps,
};
use crate::patch::{Condition, Node};

/// Tree-growth controls.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeSearchControls {
    /// Stump controls (min leaf, clamp, kinds…).
    pub stump: SearchControls,
    /// Maximum depth (≤ 3).
    pub max_depth: usize,
    /// Growth policy.
    pub growth: GrowthPolicy,
}

/// A grown tree with diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TreeCandidate {
    /// Root.
    pub root: Node,
    /// Predicted SSE reduction.
    pub gain: f64,
    /// Records with non-zero correction.
    pub affected_records: f64,
    /// Records in the search set.
    pub total_records: f64,
    /// Depth.
    pub depth: usize,
    /// Growth policy label.
    pub growth: String,
}

/// One path condition in set-feature space.
#[derive(Debug, Clone, PartialEq)]
struct PathStep {
    feature: usize,
    bin: usize,
    right: bool,
}

#[derive(Debug, Clone)]
struct Leaf {
    path: Vec<PathStep>,
    count: f64,
    sum: f64,
    /// Best split found for this leaf in the last pass.
    best: Option<StumpCandidate>,
    frozen: bool,
}

fn reaches(path: &[PathStep], bins: &[u8]) -> bool {
    path.iter()
        .all(|s| (usize::from(bins[s.feature]) > s.bin) == s.right)
}

fn leaf_correction(sum: f64, count: f64, max: f64) -> f32 {
    if count <= 0.0 {
        0.0
    } else {
        (sum / count).clamp(-max, max) as f32
    }
}

/// Accumulate one histogram set per open leaf in a single pass.
fn leaf_histograms(
    source: &dyn ChunkSource,
    bins_per_feature: &[usize],
    leaves: &[Leaf],
) -> Result<Vec<HistogramSet>, String> {
    let mut sets: Vec<HistogramSet> = leaves
        .iter()
        .map(|_| HistogramSet::new(bins_per_feature))
        .collect();
    let mut mask = Vec::new();
    source.for_each_chunk(&mut |chunk| {
        let nf = chunk.features;
        for (leaf, set) in leaves.iter().zip(sets.iter_mut()) {
            if leaf.frozen {
                continue;
            }
            mask.clear();
            mask.extend(
                (0..chunk.records).map(|r| reaches(&leaf.path, &chunk.bins[r * nf..(r + 1) * nf])),
            );
            set.accumulate(chunk, Some(&mask));
        }
        Ok(())
    })?;
    Ok(sets)
}

fn build_node(
    leaf_paths: &[(Vec<PathStep>, f32)],
    prefix: &[PathStep],
    thresholds: &dyn Fn(usize, usize) -> Option<f32>,
    feature_map: &dyn Fn(usize) -> usize,
) -> Node {
    // Leaves whose path extends `prefix`.
    let mine: Vec<&(Vec<PathStep>, f32)> = leaf_paths
        .iter()
        .filter(|(p, _)| p.starts_with(prefix))
        .collect();
    if mine.len() == 1 && mine[0].0.len() == prefix.len() {
        return Node::leaf(mine[0].1);
    }
    let step = &mine[0].0[prefix.len()];
    let mut left_prefix = prefix.to_vec();
    left_prefix.push(PathStep {
        feature: step.feature,
        bin: step.bin,
        right: false,
    });
    let mut right_prefix = prefix.to_vec();
    right_prefix.push(PathStep {
        feature: step.feature,
        bin: step.bin,
        right: true,
    });
    Node::Split {
        condition: Condition::axis(
            feature_map(step.feature),
            thresholds(step.feature, step.bin).unwrap_or(0.0),
        ),
        left: Box::new(build_node(
            leaf_paths,
            &left_prefix,
            thresholds,
            feature_map,
        )),
        right: Box::new(build_node(
            leaf_paths,
            &right_prefix,
            thresholds,
            feature_map,
        )),
    }
}

/// The `(feature, bin)` roots to grow trees from: the best-ranked stumps, one
/// per distinct feature, at most `limit` of them.
///
/// Growing two trees from the same feature would explore the same region twice
/// — a deeper tree on that feature already covers what the second stump found —
/// so distinct features are what the limit counts.
///
/// The limit is worth tuning rather than fixing. Measured across 23 production
/// runs and roughly 3,600 full-corpus scorer calls, a depth-3 tree returned
/// `3.5e-5` of score per call against `2.1e-6` for a stump: trees are the most
/// valuable thing a scorer call can be spent on, and how many of them exist to
/// spend it on is decided here.
pub fn root_features(stumps: &[StumpCandidate], limit: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for s in stumps {
        if out.len() >= limit {
            break;
        }
        if !out.iter().any(|(f, _)| *f == s.feature) {
            out.push((s.feature, s.bin));
        }
    }
    out
}

/// Grow a tree. `root` optionally fixes the first split (feature, bin) in
/// set-feature space; `feature_map` converts set features to creature inputs.
///
/// Returns one candidate per depth reached (depth 1 … `max_depth`) so the
/// caller can compare shallow and deep versions under the scorer.
pub fn grow_tree(
    source: &dyn ChunkSource,
    bins_per_feature: &[usize],
    thresholds: &dyn Fn(usize, usize) -> Option<f32>,
    feature_map: &dyn Fn(usize) -> usize,
    controls: &TreeSearchControls,
    root: Option<(usize, usize)>,
) -> Result<Vec<TreeCandidate>, String> {
    let max_depth = controls.max_depth.clamp(1, 3);
    let two_leaf = SearchControls {
        kinds: vec![StumpKind::TwoLeaf],
        top_k: usize::MAX,
        max_per_feature: 0,
        ..controls.stump.clone()
    };
    let mut leaves = vec![Leaf {
        path: Vec::new(),
        count: 0.0,
        sum: 0.0,
        best: None,
        frozen: false,
    }];
    let mut out = Vec::new();
    let max_splits = (1usize << max_depth) - 1;
    let mut splits = 0;
    let mut current_depth = 0;
    loop {
        // Evaluate every open leaf.
        let sets = leaf_histograms(source, bins_per_feature, &leaves)?;
        let total_records = sets
            .iter()
            .map(|s| s.total_count)
            .sum::<f64>()
            .max(leaves.iter().map(|l| l.count).sum());
        for (leaf, set) in leaves.iter_mut().zip(&sets) {
            if leaf.frozen {
                continue;
            }
            leaf.count = set.total_count;
            leaf.sum = set.total_sum;
            let mut found = if let (true, Some((f, b))) = (leaf.path.is_empty(), root) {
                search_stumps(set, thresholds, &two_leaf, "tree")
                    .into_iter()
                    .find(|c| c.feature == f && c.bin == b)
            } else {
                search_stumps(set, thresholds, &two_leaf, "tree")
                    .into_iter()
                    .next()
            };
            if let Some(c) = &found
                && c.left_records < controls.stump.min_leaf_records
            {
                found = None;
            }
            leaf.best = found;
        }
        // Choose which leaves to expand.
        let expand: Vec<usize> = match controls.growth {
            GrowthPolicy::LevelWise => leaves
                .iter()
                .enumerate()
                .filter(|(_, l)| !l.frozen && l.best.is_some())
                .map(|(i, _)| i)
                .collect(),
            GrowthPolicy::BestFirst => {
                let mut best: Option<(usize, &StumpCandidate)> = None;
                for (i, l) in leaves.iter().enumerate() {
                    if let Some(c) = &l.best
                        && best.is_none_or(|(_, b)| rank_order(c, b) == std::cmp::Ordering::Less)
                    {
                        best = Some((i, c));
                    }
                }
                best.map(|(i, _)| vec![i]).unwrap_or_default()
            }
        };
        if expand.is_empty() {
            break;
        }
        let mut next = Vec::new();
        for (i, leaf) in leaves.into_iter().enumerate() {
            if expand.contains(&i) && splits < max_splits && leaf.path.len() < max_depth {
                let c = leaf.best.clone().unwrap();
                splits += 1;
                for right in [false, true] {
                    let mut path = leaf.path.clone();
                    path.push(PathStep {
                        feature: c.feature,
                        bin: c.bin,
                        right,
                    });
                    let (count, sum) = if right {
                        (c.right_records, c.right_correction as f64 * c.right_records)
                    } else {
                        (c.left_records, c.left_correction as f64 * c.left_records)
                    };
                    next.push(Leaf {
                        path,
                        count,
                        sum,
                        best: None,
                        frozen: false,
                    });
                }
            } else {
                next.push(Leaf {
                    frozen: true,
                    ..leaf
                });
            }
        }
        leaves = next;
        let depth = leaves.iter().map(|l| l.path.len()).max().unwrap_or(0);
        if depth > current_depth || matches!(controls.growth, GrowthPolicy::BestFirst) {
            current_depth = depth;
            out.push(snapshot(
                &leaves,
                thresholds,
                feature_map,
                controls,
                total_records,
            ));
        }
        if depth >= max_depth || splits >= max_splits {
            break;
        }
    }
    // Best-first may snapshot the same depth repeatedly; keep the last per depth.
    let mut by_depth: Vec<TreeCandidate> = Vec::new();
    for c in out {
        if let Some(existing) = by_depth.iter_mut().find(|e| e.depth == c.depth) {
            *existing = c;
        } else {
            by_depth.push(c);
        }
    }
    Ok(by_depth)
}

fn snapshot(
    leaves: &[Leaf],
    thresholds: &dyn Fn(usize, usize) -> Option<f32>,
    feature_map: &dyn Fn(usize) -> usize,
    controls: &TreeSearchControls,
    total: f64,
) -> TreeCandidate {
    let max = controls.stump.max_correction;
    let leaf_paths: Vec<(Vec<PathStep>, f32)> = leaves
        .iter()
        .map(|l| (l.path.clone(), leaf_correction(l.sum, l.count, max)))
        .collect();
    let root = build_node(&leaf_paths, &[], thresholds, feature_map);
    let gain: f64 = leaves
        .iter()
        .map(|l| {
            let c = f64::from(leaf_correction(l.sum, l.count, max));
            2.0 * c * l.sum - l.count * c * c
        })
        .sum();
    let affected: f64 = leaves
        .iter()
        .filter(|l| leaf_correction(l.sum, l.count, max) != 0.0)
        .map(|l| l.count)
        .sum();
    TreeCandidate {
        depth: root.depth(),
        root,
        gain,
        affected_records: affected,
        total_records: total,
        growth: match controls.growth {
            GrowthPolicy::LevelWise => "level-wise".into(),
            GrowthPolicy::BestFirst => "best-first".into(),
        },
    }
}

#[cfg(test)]
mod tests {
    /// Issue #63 — how many distinct features get grown into trees. Depth-3
    /// trees returned 3.5e-5 of score per full-corpus scorer call over 23
    /// production runs against 2.1e-6 for a stump, so the supply of tree roots
    /// is worth controlling rather than hard-coding.
    #[test]
    fn tree_roots_takes_the_best_distinct_features_up_to_the_limit() {
        let stump = |feature: usize, gain: f64| StumpCandidate {
            feature,
            bin: 3,
            threshold: 0.0,
            kind: StumpKind::TwoLeaf,
            left_correction: -0.1,
            right_correction: 0.1,
            gain,
            left_records: 5.0,
            right_records: 5.0,
            affected_records: 10.0,
            affected_fraction: 1.0,
            backend: "cpu".into(),
        };
        // Ranked best first, with feature 2 repeated: a second stump on a
        // feature already grown adds nothing a deeper tree on it would not.
        let stumps = vec![
            stump(2, 9.0),
            stump(2, 8.0),
            stump(5, 7.0),
            stump(1, 6.0),
            stump(7, 5.0),
        ];
        assert_eq!(root_features(&stumps, 3), vec![(2, 3), (5, 3), (1, 3)]);
        assert_eq!(root_features(&stumps, 1), vec![(2, 3)]);
        assert_eq!(
            root_features(&stumps, 99),
            vec![(2, 3), (5, 3), (1, 3), (7, 3)],
            "never more roots than there are distinct features"
        );
        assert!(root_features(&stumps, 0).is_empty());
    }

    use super::*;
    use crate::bins::{BinCache, BinMeta, quantile_edges};
    use crate::histogram::{BinnedChunk, MemorySource};

    fn and_fixture() -> (BinCache, MemorySource, Vec<Vec<f32>>, Vec<f32>) {
        // residual = 0.6 when x0 > 0 AND x1 > 0, else 0. The best stump can
        // capture at most half of the SSE; a depth-2 tree captures it all.
        // (Pure XOR is deliberately not used: greedy growth cannot find it —
        // a known limitation of CART-style search, not a Forests bug.)
        let mut values = Vec::new();
        let mut residual = Vec::new();
        let mut seed = 11u64;
        for _ in 0..2000 {
            let mut next = || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
            };
            let (a, b, c) = (next(), next(), next());
            values.push(vec![a, b, c]);
            residual.push(if a > 0.0 && b > 0.0 { 0.6 } else { 0.0 });
        }
        let edges: Vec<Vec<f32>> = (0..3)
            .map(|f| quantile_edges(&mut values.iter().map(|v| v[f]).collect(), 64))
            .collect();
        let cache = BinCache {
            meta: BinMeta {
                format_version: 1,
                algorithm_version: 1,
                input_count: 3,
                output_count: 1,
                record_count: 2000,
                corpus_identity: "t".into(),
                requested_bins: 64,
                effective_bins: edges.iter().map(|e| (e.len() + 1) as u16).collect(),
                non_finite_counts: vec![0; 3],
                non_finite_policy: String::new(),
                sample_records: 2000,
                sample_stride: 1,
                created_at_unix: 0,
            },
            edges,
        };
        let flat: Vec<f32> = values.iter().flatten().copied().collect();
        let mut bins = Vec::new();
        cache.bin_rows(&flat, 3, &mut bins);
        let src = MemorySource {
            chunks: vec![BinnedChunk {
                records: 2000,
                features: 3,
                bins,
                residual: residual.clone(),
                weight: None,
                first_index: 0,
            }],
            label: "t".into(),
        };
        (cache, src, values, residual)
    }

    #[test]
    fn depth2_finds_interaction_a_stump_cannot() {
        let (cache, src, values, residual) = and_fixture();
        let bins: Vec<usize> = (0..3).map(|f| cache.bins(f)).collect();
        let stump_controls = SearchControls {
            min_leaf_records: 20.0,
            top_k: 5,
            ..Default::default()
        };
        let set = HistogramSet::from_source(&src, &bins).unwrap();
        let stumps = search_stumps(&set, &|f, b| cache.threshold(f, b), &stump_controls, "cpu");
        let sse: f64 = residual.iter().map(|r| f64::from(*r) * f64::from(*r)).sum();
        let stump_gain = stumps.first().map_or(0.0, |s| s.gain);
        assert!(stump_gain < 0.6 * sse, "stump gain {stump_gain} of {sse}");
        for growth in [GrowthPolicy::LevelWise, GrowthPolicy::BestFirst] {
            let controls = TreeSearchControls {
                stump: stump_controls.clone(),
                max_depth: 2,
                growth,
            };
            let trees = grow_tree(
                &src,
                &bins,
                &|f, b| cache.threshold(f, b),
                &|f| f,
                &controls,
                None,
            )
            .unwrap();
            let deep = trees.iter().find(|t| t.depth == 2).expect("depth-2 tree");
            assert!(
                deep.gain > 0.85 * sse,
                "{growth:?} gain {} of {sse}",
                deep.gain
            );
            assert!(deep.gain > 1.5 * stump_gain);
            // Abstract evaluator reproduces most records.
            let hits = values
                .iter()
                .zip(&residual)
                .filter(|(v, r)| (deep.root.evaluate(v) - **r).abs() < 0.15)
                .count();
            assert!(hits > 1850, "{growth:?} hits {hits}");
            assert_eq!(deep.root.features().len(), 2);
        }
    }

    #[test]
    fn depth3_obeys_leaf_and_depth_limits() {
        let (cache, src, _, _) = and_fixture();
        let bins: Vec<usize> = (0..3).map(|f| cache.bins(f)).collect();
        let controls = TreeSearchControls {
            stump: SearchControls {
                min_leaf_records: 150.0,
                ..Default::default()
            },
            max_depth: 3,
            growth: GrowthPolicy::LevelWise,
        };
        let trees = grow_tree(
            &src,
            &bins,
            &|f, b| cache.threshold(f, b),
            &|f| f,
            &controls,
            None,
        )
        .unwrap();
        for t in &trees {
            assert!(t.depth <= 3);
            assert!(t.root.split_count() <= 7);
        }
        // With a huge leaf minimum nothing can split.
        let strict = TreeSearchControls {
            stump: SearchControls {
                min_leaf_records: 5000.0,
                ..Default::default()
            },
            ..controls
        };
        assert!(
            grow_tree(
                &src,
                &bins,
                &|f, b| cache.threshold(f, b),
                &|f| f,
                &strict,
                None
            )
            .unwrap()
            .is_empty()
        );
    }
}

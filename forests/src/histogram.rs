//! CPU reference histogram search for depth-1 residual stumps (Issue #5).
//!
//! This is the correctness oracle for every other search backend. It is
//! written for clarity: per feature and bin it accumulates `count`, `Σr` and
//! `Σr²` of the correction-space residual `r`, then evaluates every threshold
//! with prefix/suffix scans.
//!
//! ## Gain model (squared error)
//!
//! Adding a constant `c` to `n` records with residual sum `s` and sum of
//! squares `q` changes the residual SSE from `q` to `q - 2cs + nc²`; the
//! reduction is `2cs - nc²`, maximised at `c* = s/n` giving `s²/n`. With a
//! clamp `|c| ≤ max_correction` the reduction is evaluated at the clamped `c`.
//! The gain is **only a ranking signal**; the scorer decides what is better.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

/// One chunk of quantised records plus their residual for the output being searched.
#[derive(Debug, Clone, Default)]
pub struct BinnedChunk {
    /// Records in the chunk.
    pub records: usize,
    /// Features per record.
    pub features: usize,
    /// Row-major bin indices (`records × features`).
    pub bins: Vec<u8>,
    /// Correction-space residual per record.
    pub residual: Vec<f32>,
    /// Optional per-record weight (1.0 when absent).
    pub weight: Option<Vec<f32>>,
    /// Global index of the first record (for diagnostics).
    pub first_index: u64,
}

/// A re-iterable source of [`BinnedChunk`]s (in-memory sample or streaming corpus).
pub trait ChunkSource {
    /// Visit every chunk in order.
    fn for_each_chunk(
        &self,
        f: &mut dyn FnMut(&BinnedChunk) -> Result<(), String>,
    ) -> Result<(), String>;
    /// Number of features per record.
    fn features(&self) -> usize;
    /// Total records.
    fn records(&self) -> u64;
    /// Human label for provenance (`memory-sample`, `streaming-full`, …).
    fn label(&self) -> String;
    /// In-memory chunks, when the source holds them (enables parallel accumulation).
    fn chunk_slice(&self) -> Option<&[BinnedChunk]> {
        None
    }
}

/// In-memory chunk source.
#[derive(Debug, Clone, Default)]
pub struct MemorySource {
    /// Chunks in order.
    pub chunks: Vec<BinnedChunk>,
    /// Label for provenance.
    pub label: String,
}

impl ChunkSource for MemorySource {
    fn for_each_chunk(
        &self,
        f: &mut dyn FnMut(&BinnedChunk) -> Result<(), String>,
    ) -> Result<(), String> {
        for c in &self.chunks {
            f(c)?;
        }
        Ok(())
    }
    fn features(&self) -> usize {
        self.chunks.first().map_or(0, |c| c.features)
    }
    fn records(&self) -> u64 {
        self.chunks.iter().map(|c| c.records as u64).sum()
    }
    fn label(&self) -> String {
        self.label.clone()
    }
    fn chunk_slice(&self) -> Option<&[BinnedChunk]> {
        Some(&self.chunks)
    }
}

/// Per-feature sufficient statistics, indexed `[bin]`.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureHistogram {
    /// Weighted record count per bin.
    pub count: Vec<f64>,
    /// Σ w·r per bin.
    pub sum: Vec<f64>,
    /// Σ w·r² per bin.
    pub sum_sq: Vec<f64>,
}

impl FeatureHistogram {
    /// Empty histogram with `bins` bins.
    pub fn new(bins: usize) -> Self {
        Self {
            count: vec![0.0; bins],
            sum: vec![0.0; bins],
            sum_sq: vec![0.0; bins],
        }
    }

    /// Add another histogram (same shape).
    pub fn merge(&mut self, other: &Self) {
        for b in 0..self.count.len() {
            self.count[b] += other.count[b];
            self.sum[b] += other.sum[b];
            self.sum_sq[b] += other.sum_sq[b];
        }
    }
}

/// Histograms for every feature.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramSet {
    /// One histogram per feature.
    pub features: Vec<FeatureHistogram>,
    /// Records accumulated.
    pub records: u64,
    /// Total Σ w·r² (SSE before any correction).
    pub total_sum_sq: f64,
    /// Total Σ w·r.
    pub total_sum: f64,
    /// Total Σ w.
    pub total_count: f64,
}

impl HistogramSet {
    /// Empty set; `bins_per_feature[f]` is the bin count of feature `f`.
    pub fn new(bins_per_feature: &[usize]) -> Self {
        Self {
            features: bins_per_feature
                .iter()
                .map(|&b| FeatureHistogram::new(b))
                .collect(),
            records: 0,
            total_sum_sq: 0.0,
            total_sum: 0.0,
            total_count: 0.0,
        }
    }

    /// Accumulate one chunk. `mask` (if given) selects which records count.
    pub fn accumulate(&mut self, chunk: &BinnedChunk, mask: Option<&[bool]>) {
        let nf = chunk.features;
        for r in 0..chunk.records {
            if mask.is_some_and(|m| !m[r]) {
                continue;
            }
            let w = chunk.weight.as_ref().map_or(1.0, |w| f64::from(w[r]));
            let res = f64::from(chunk.residual[r]);
            let row = &chunk.bins[r * nf..(r + 1) * nf];
            for (f, &b) in row.iter().enumerate() {
                let h = &mut self.features[f];
                let b = usize::from(b).min(h.count.len() - 1);
                h.count[b] += w;
                h.sum[b] += w * res;
                h.sum_sq[b] += w * res * res;
            }
            self.records += 1;
            self.total_count += w;
            self.total_sum += w * res;
            self.total_sum_sq += w * res * res;
        }
    }

    /// Merge another set (same shape).
    pub fn merge(&mut self, other: &Self) {
        for (a, b) in self.features.iter_mut().zip(&other.features) {
            a.merge(b);
        }
        self.records += other.records;
        self.total_sum_sq += other.total_sum_sq;
        self.total_sum += other.total_sum;
        self.total_count += other.total_count;
    }

    /// Accumulate every chunk of a source (single thread).
    pub fn from_source(
        source: &dyn ChunkSource,
        bins_per_feature: &[usize],
    ) -> Result<Self, String> {
        let mut set = Self::new(bins_per_feature);
        source.for_each_chunk(&mut |c| {
            set.accumulate(c, None);
            Ok(())
        })?;
        Ok(set)
    }

    /// Accumulate with up to `threads` workers, each owning a contiguous
    /// **feature range** and scanning every record. Per-feature accumulation
    /// order is identical to [`Self::from_source`], so the result is
    /// bit-identical regardless of thread count; splitting by feature (not by
    /// chunk) keeps each worker's histograms small enough to stay cache-resident,
    /// which is what makes it faster (see `docs/benchmarks.md`). Falls back to
    /// the single-thread path when the source is not in memory.
    pub fn from_source_threads(
        source: &dyn ChunkSource,
        bins_per_feature: &[usize],
        threads: usize,
    ) -> Result<Self, String> {
        let Some(chunks) = source.chunk_slice() else {
            return Self::from_source(source, bins_per_feature);
        };
        let features = bins_per_feature.len();
        let threads = threads.max(1).min(features.max(1));
        if threads <= 1 || chunks.is_empty() {
            return Self::from_source(source, bins_per_feature);
        }
        let per = features.div_ceil(threads);
        let partials: Vec<Vec<FeatureHistogram>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..features)
                .step_by(per)
                .map(|lo| {
                    let hi = (lo + per).min(features);
                    scope.spawn(move || {
                        let mut hists: Vec<FeatureHistogram> = bins_per_feature[lo..hi]
                            .iter()
                            .map(|&b| FeatureHistogram::new(b))
                            .collect();
                        for chunk in chunks {
                            let nf = chunk.features;
                            for r in 0..chunk.records {
                                let w = chunk.weight.as_ref().map_or(1.0, |w| f64::from(w[r]));
                                let res = f64::from(chunk.residual[r]);
                                let row = &chunk.bins[r * nf + lo..r * nf + hi];
                                for (h, &b) in hists.iter_mut().zip(row) {
                                    let b = usize::from(b).min(h.count.len() - 1);
                                    h.count[b] += w;
                                    h.sum[b] += w * res;
                                    h.sum_sq[b] += w * res * res;
                                }
                            }
                        }
                        hists
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .map_err(|_| "histogram worker panicked".to_string())
                })
                .collect::<Result<Vec<_>, String>>()
        })?;
        let mut set = Self::new(&[]);
        set.features = partials.into_iter().flatten().collect();
        for chunk in chunks {
            for r in 0..chunk.records {
                let w = chunk.weight.as_ref().map_or(1.0, |w| f64::from(w[r]));
                let res = f64::from(chunk.residual[r]);
                set.records += 1;
                set.total_count += w;
                set.total_sum += w * res;
                set.total_sum_sq += w * res * res;
            }
        }
        Ok(set)
    }
}

/// Which side(s) receive a correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash, PartialOrd, Ord)]
#[serde(rename_all = "kebab-case")]
pub enum StumpKind {
    /// Correct records with `x <= threshold`; right leaf is exactly zero.
    LeftOnly,
    /// Correct records with `x > threshold`; left leaf is exactly zero.
    RightOnly,
    /// Correct both sides.
    TwoLeaf,
}

impl StumpKind {
    /// All kinds in deterministic order.
    pub const ALL: [StumpKind; 3] = [
        StumpKind::LeftOnly,
        StumpKind::RightOnly,
        StumpKind::TwoLeaf,
    ];

    /// Kebab-case label (`left-only`, `right-only`, `two-leaf`).
    pub fn label(self) -> &'static str {
        match self {
            Self::LeftOnly => "left-only",
            Self::RightOnly => "right-only",
            Self::TwoLeaf => "two-leaf",
        }
    }
}

impl std::fmt::Display for StumpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Search controls.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchControls {
    /// Minimum (weighted) records in any corrected leaf.
    pub min_leaf_records: f64,
    /// Clamp on |leaf correction|.
    pub max_correction: f64,
    /// Minimum predicted gain to report.
    pub min_gain: f64,
    /// Kinds to evaluate.
    pub kinds: Vec<StumpKind>,
    /// Stumps to return.
    pub top_k: usize,
    /// Diversity cap: at most this many stumps per feature (0 = unlimited).
    pub max_per_feature: usize,
}

impl Default for SearchControls {
    fn default() -> Self {
        Self {
            min_leaf_records: 50.0,
            max_correction: 1.0,
            min_gain: 0.0,
            kinds: StumpKind::ALL.to_vec(),
            top_k: 32,
            max_per_feature: 0,
        }
    }
}

/// A ranked stump discovery.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StumpCandidate {
    /// Feature index.
    pub feature: usize,
    /// Split is "after bin `bin`" (`x > edges[bin]`).
    pub bin: usize,
    /// Threshold value carried into the creature.
    pub threshold: f32,
    /// Which side(s) are corrected.
    pub kind: StumpKind,
    /// Correction for `x <= threshold`.
    pub left_correction: f32,
    /// Correction for `x > threshold`.
    pub right_correction: f32,
    /// Predicted SSE reduction.
    pub gain: f64,
    /// Records on the left.
    pub left_records: f64,
    /// Records on the right.
    pub right_records: f64,
    /// Records that receive a non-zero correction.
    pub affected_records: f64,
    /// `affected / total` — how concentrated the patch is.
    pub affected_fraction: f64,
    /// Backend label.
    pub backend: String,
}

fn clamp_correction(s: f64, n: f64, max_abs: f64) -> (f64, f64) {
    if n <= 0.0 {
        return (0.0, 0.0);
    }
    let c = (s / n).clamp(-max_abs, max_abs);
    (c, 2.0 * c * s - n * c * c)
}

/// Evaluate one threshold of one feature for a given kind.
fn evaluate_split(
    nl: f64,
    sl: f64,
    nr: f64,
    sr: f64,
    kind: StumpKind,
    controls: &SearchControls,
) -> Option<(f64, f64, f64, f64)> {
    if nl <= 0.0 || nr <= 0.0 {
        return None;
    }
    let (cl, gl) = clamp_correction(sl, nl, controls.max_correction);
    let (cr, gr) = clamp_correction(sr, nr, controls.max_correction);
    let (left, right, gain, affected) = match kind {
        StumpKind::LeftOnly => (cl, 0.0, gl, nl),
        StumpKind::RightOnly => (0.0, cr, gr, nr),
        StumpKind::TwoLeaf => (cl, cr, gl + gr, nl + nr),
    };
    let min = controls.min_leaf_records;
    let ok = match kind {
        StumpKind::LeftOnly => nl >= min,
        StumpKind::RightOnly => nr >= min,
        StumpKind::TwoLeaf => nl >= min && nr >= min,
    };
    if !ok || !gain.is_finite() || gain < controls.min_gain || gain <= 0.0 {
        return None;
    }
    Some((left, right, gain, affected))
}

/// Deterministic ordering: gain desc, then feature, bin, kind.
pub fn rank_order(a: &StumpCandidate, b: &StumpCandidate) -> Ordering {
    b.gain
        .total_cmp(&a.gain)
        .then(a.feature.cmp(&b.feature))
        .then(a.bin.cmp(&b.bin))
        .then(a.kind.cmp(&b.kind))
}

/// Evaluate every threshold of every feature and return the top-K stumps.
///
/// `thresholds(f, bin)` maps a bin boundary to its threshold value; features
/// with fewer than two bins are skipped.
pub fn search_stumps(
    set: &HistogramSet,
    thresholds: &dyn Fn(usize, usize) -> Option<f32>,
    controls: &SearchControls,
    backend: &str,
) -> Vec<StumpCandidate> {
    let mut out: Vec<StumpCandidate> = Vec::new();
    let total = set.total_count;
    for (f, h) in set.features.iter().enumerate() {
        let bins = h.count.len();
        if bins < 2 {
            continue;
        }
        let mut per_feature: Vec<StumpCandidate> = Vec::new();
        let (mut nl, mut sl) = (0.0, 0.0);
        let n_total: f64 = h.count.iter().sum();
        let s_total: f64 = h.sum.iter().sum();
        for b in 0..bins - 1 {
            nl += h.count[b];
            sl += h.sum[b];
            let nr = n_total - nl;
            let sr = s_total - sl;
            let Some(threshold) = thresholds(f, b) else {
                continue;
            };
            for &kind in &controls.kinds {
                if let Some((left, right, gain, affected)) =
                    evaluate_split(nl, sl, nr, sr, kind, controls)
                {
                    per_feature.push(StumpCandidate {
                        feature: f,
                        bin: b,
                        threshold,
                        kind,
                        left_correction: left as f32,
                        right_correction: right as f32,
                        gain,
                        left_records: nl,
                        right_records: nr,
                        affected_records: affected,
                        affected_fraction: if total > 0.0 { affected / total } else { 0.0 },
                        backend: backend.to_string(),
                    });
                }
            }
        }
        per_feature.sort_by(rank_order);
        if controls.max_per_feature > 0 {
            per_feature.truncate(controls.max_per_feature);
        }
        out.extend(per_feature);
    }
    out.sort_by(rank_order);
    out.truncate(controls.top_k);
    out
}

/// Brute-force oracle: evaluates every distinct per-record threshold directly
/// (no histogram). Intended for tests and backend parity checks on small data.
pub fn brute_force_best_stump(
    values: &[Vec<f32>],
    residual: &[f32],
    feature: usize,
    controls: &SearchControls,
) -> Option<StumpCandidate> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[a][feature].total_cmp(&values[b][feature]));
    let total_n = values.len() as f64;
    let total_s: f64 = residual.iter().map(|&r| f64::from(r)).sum();
    let mut best: Option<StumpCandidate> = None;
    let (mut nl, mut sl) = (0.0, 0.0);
    for (k, &i) in idx.iter().enumerate() {
        nl += 1.0;
        sl += f64::from(residual[i]);
        let t = values[i][feature];
        if t.is_nan() || idx.get(k + 1).is_some_and(|&j| values[j][feature] == t) {
            continue;
        }
        for &kind in &controls.kinds {
            if let Some((left, right, gain, affected)) =
                evaluate_split(nl, sl, total_n - nl, total_s - sl, kind, controls)
            {
                let cand = StumpCandidate {
                    feature,
                    bin: k,
                    threshold: t,
                    kind,
                    left_correction: left as f32,
                    right_correction: right as f32,
                    gain,
                    left_records: nl,
                    right_records: total_n - nl,
                    affected_records: affected,
                    affected_fraction: affected / total_n,
                    backend: "brute-force".into(),
                };
                if best
                    .as_ref()
                    .is_none_or(|b| rank_order(&cand, b) == Ordering::Less)
                {
                    best = Some(cand);
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bins::{BinCache, BinMeta, quantile_edges};

    fn cache_for(values: &[Vec<f32>], bins: usize) -> BinCache {
        let features = values[0].len();
        let edges: Vec<Vec<f32>> = (0..features)
            .map(|f| quantile_edges(&mut values.iter().map(|v| v[f]).collect(), bins))
            .collect();
        BinCache {
            meta: BinMeta {
                format_version: 1,
                algorithm_version: 1,
                input_count: features,
                output_count: 1,
                record_count: values.len() as u64,
                corpus_identity: "test".into(),
                requested_bins: bins,
                effective_bins: edges.iter().map(|e| (e.len() + 1) as u16).collect(),
                non_finite_counts: vec![0; features],
                non_finite_policy: String::new(),
                sample_records: values.len() as u64,
                sample_stride: 1,
                created_at_unix: 0,
            },
            edges,
        }
    }

    fn source(cache: &BinCache, values: &[Vec<f32>], residual: &[f32]) -> MemorySource {
        let features = values[0].len();
        let flat: Vec<f32> = values.iter().flatten().copied().collect();
        let mut bins = Vec::new();
        cache.bin_rows(&flat, features, &mut bins);
        MemorySource {
            chunks: vec![BinnedChunk {
                records: values.len(),
                features,
                bins,
                residual: residual.to_vec(),
                weight: None,
                first_index: 0,
            }],
            label: "test".into(),
        }
    }

    #[test]
    fn finds_known_optimal_split_and_leaf_value() {
        // Feature 1 carries the signal: residual = +0.5 when x1 > 0.3 else 0.
        let values: Vec<Vec<f32>> = (0..200)
            .map(|i| vec![(i % 7) as f32, i as f32 / 200.0])
            .collect();
        let residual: Vec<f32> = values
            .iter()
            .map(|v| if v[1] > 0.3 { 0.5 } else { 0.0 })
            .collect();
        let cache = cache_for(&values, 20);
        let src = source(&cache, &values, &residual);
        let bins: Vec<usize> = (0..2).map(|f| cache.bins(f)).collect();
        let set = HistogramSet::from_source(&src, &bins).unwrap();
        let controls = SearchControls {
            min_leaf_records: 5.0,
            top_k: 5,
            ..Default::default()
        };
        let top = search_stumps(&set, &|f, b| cache.threshold(f, b), &controls, "cpu");
        let best = &top[0];
        assert_eq!(best.feature, 1);
        assert!(
            best.threshold >= 0.295 && best.threshold < 0.305,
            "threshold {}",
            best.threshold
        );
        assert_eq!(best.kind, StumpKind::RightOnly);
        assert!((best.right_correction - 0.5).abs() < 1e-6);
        assert_eq!(best.left_correction, 0.0);
        let expected_gain = 0.25 * residual.iter().filter(|&&r| r > 0.0).count() as f64;
        assert!((best.gain - expected_gain).abs() < 1e-9);
        assert_eq!(
            best.affected_records,
            residual.iter().filter(|&&r| r > 0.0).count() as f64
        );
    }

    #[test]
    fn histogram_matches_brute_force_when_bins_resolve_every_value() {
        let values: Vec<Vec<f32>> = (0..60)
            .map(|i| {
                vec![
                    ((i * 37) % 61) as f32 / 10.0,
                    ((i * 11) % 13) as f32,
                    ((i * 5) % 17) as f32 - 8.0,
                ]
            })
            .collect();
        let residual: Vec<f32> = values
            .iter()
            .map(|v| (v[0] * 0.3 - 1.0).sin() + if v[2] > 0.0 { 0.4 } else { -0.1 })
            .collect();
        let cache = cache_for(&values, 256);
        let src = source(&cache, &values, &residual);
        let bins: Vec<usize> = (0..3).map(|f| cache.bins(f)).collect();
        let set = HistogramSet::from_source(&src, &bins).unwrap();
        let controls = SearchControls {
            min_leaf_records: 3.0,
            top_k: 1000,
            ..Default::default()
        };
        let hist = search_stumps(&set, &|f, b| cache.threshold(f, b), &controls, "cpu");
        for f in 0..3 {
            let brute = brute_force_best_stump(&values, &residual, f, &controls).unwrap();
            let h = hist.iter().find(|c| c.feature == f).unwrap();
            assert!(
                (h.gain - brute.gain).abs() < 1e-6,
                "feature {f}: {} vs {}",
                h.gain,
                brute.gain
            );
            assert_eq!(h.kind, brute.kind);
            assert_eq!(h.threshold, brute.threshold);
        }
    }

    #[test]
    fn threaded_accumulation_matches_sequential() {
        let values: Vec<Vec<f32>> = (0..500)
            .map(|i| vec![((i * 37) % 61) as f32, ((i * 11) % 13) as f32])
            .collect();
        let residual: Vec<f32> = values
            .iter()
            .map(|v| (v[0] * 0.1).sin() + v[1] * 0.01)
            .collect();
        let cache = cache_for(&values, 32);
        let mut src = source(&cache, &values, &residual);
        // Split into 7 chunks.
        let one = src.chunks.remove(0);
        let nf = one.features;
        for (k, rows) in (0..one.records).collect::<Vec<_>>().chunks(73).enumerate() {
            let mut c = BinnedChunk {
                records: rows.len(),
                features: nf,
                first_index: k as u64,
                ..Default::default()
            };
            for &r in rows {
                c.bins.extend_from_slice(&one.bins[r * nf..(r + 1) * nf]);
                c.residual.push(one.residual[r]);
            }
            src.chunks.push(c);
        }
        let bins: Vec<usize> = (0..2).map(|f| cache.bins(f)).collect();
        let seq = HistogramSet::from_source(&src, &bins).unwrap();
        let par = HistogramSet::from_source_threads(&src, &bins, 4).unwrap();
        assert_eq!(seq.records, par.records);
        for (a, b) in seq.features.iter().zip(&par.features) {
            for k in 0..a.count.len() {
                assert_eq!(a.count[k], b.count[k]);
                assert!((a.sum[k] - b.sum[k]).abs() < 1e-9);
                assert!((a.sum_sq[k] - b.sum_sq[k]).abs() < 1e-9);
            }
        }
        assert_eq!(
            par,
            HistogramSet::from_source_threads(&src, &bins, 4).unwrap()
        );
    }

    #[test]
    fn controls_are_enforced_and_ranking_is_deterministic() {
        let values: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32]).collect();
        let residual: Vec<f32> = values
            .iter()
            .map(|v| if v[0] > 94.0 { 10.0 } else { 0.0 })
            .collect();
        let cache = cache_for(&values, 100);
        let src = source(&cache, &values, &residual);
        let set = HistogramSet::from_source(&src, &[cache.bins(0)]).unwrap();
        let strict = SearchControls {
            min_leaf_records: 10.0,
            max_correction: 1.0,
            top_k: 3,
            ..Default::default()
        };
        let top = search_stumps(&set, &|f, b| cache.threshold(f, b), &strict, "cpu");
        for c in &top {
            assert!(c.left_correction.abs() <= 1.0 && c.right_correction.abs() <= 1.0);
            assert!(c.affected_records >= 10.0);
        }
        let again = search_stumps(&set, &|f, b| cache.threshold(f, b), &strict, "cpu");
        assert_eq!(top, again);
        let none = SearchControls {
            min_gain: 1e12,
            ..strict.clone()
        };
        assert!(search_stumps(&set, &|f, b| cache.threshold(f, b), &none, "cpu").is_empty());
        let one_per = SearchControls {
            max_per_feature: 1,
            top_k: 10,
            ..strict
        };
        assert_eq!(
            search_stumps(&set, &|f, b| cache.threshold(f, b), &one_per, "cpu").len(),
            1
        );
    }
}

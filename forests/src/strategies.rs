//! Search-set construction and "dirty trick" sampling strategies (Issue #12).
//!
//! Everything here only changes **what is searched**; nothing here can accept a
//! candidate. Every strategy names itself in provenance: random search is
//! called random search, sampled search is called sampled search.
//!
//! A [`SearchSet`] is the quantised view of (a sample of) the corpus for one
//! output: per-record bins for the selected features, the correction-space
//! residual, an optional importance weight, and the record indices so raw
//! values can be re-read for oblique search.

use std::path::Path;

use neat_core::training_data::TrainingDataConfig;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bins::BinCache;
use crate::config::{FeatureSelection, ForestsConfig, RowSampling};
use crate::corpus::for_each_chunk;
use crate::histogram::{BinnedChunk, ChunkSource, HistogramSet, MemorySource};
use crate::residuals::ResidualCache;

/// Quantised search set.
#[derive(Debug, Clone)]
pub struct SearchSet {
    /// Chunks (in memory).
    pub source: MemorySource,
    /// Selected features (index into the set → original feature index).
    pub feature_map: Vec<usize>,
    /// Global record indices included, ascending.
    pub record_indices: Vec<u64>,
    /// Row sampling used.
    pub row_sampling: RowSampling,
    /// Feature selection used.
    pub feature_selection: FeatureSelection,
    /// Output searched.
    pub output: usize,
    /// Human description for provenance notes.
    pub notes: Vec<String>,
}

impl SearchSet {
    /// Bins per selected feature.
    pub fn bins_per_feature(&self, cache: &BinCache) -> Vec<usize> {
        self.feature_map.iter().map(|&f| cache.bins(f)).collect()
    }

    /// Records in the set.
    pub fn records(&self) -> u64 {
        self.source.records()
    }

    /// Features in the set.
    pub fn features(&self) -> usize {
        self.feature_map.len()
    }

    /// Threshold lookup in set-feature space.
    pub fn threshold<'a>(
        &'a self,
        cache: &'a BinCache,
    ) -> impl Fn(usize, usize) -> Option<f32> + 'a {
        move |f, b| cache.threshold(self.feature_map[f], b)
    }

    /// Label for the journal.
    pub fn label(&self) -> String {
        self.source.label.clone()
    }
}

/// Decide which records enter the search set and with what weight.
struct RowPlan {
    /// `keep[i]` for record `i`, with importance weight.
    keep: Vec<Option<f32>>,
}

fn row_plan(
    cfg: &ForestsConfig,
    residual: &[f32],
    rng: &mut StdRng,
    notes: &mut Vec<String>,
) -> RowPlan {
    let n = residual.len();
    let target = if cfg.search_records == 0 || cfg.search_records as usize >= n {
        n
    } else {
        cfg.search_records as usize
    };
    let mut keep = vec![None; n];
    match cfg.row_sampling {
        RowSampling::Stride => {
            let stride = n.div_ceil(target.max(1)).max(1);
            for (i, k) in keep.iter_mut().enumerate() {
                if i % stride == 0 {
                    *k = Some(1.0);
                }
            }
            notes.push(format!("row-sampling=stride/{stride}"));
        }
        RowSampling::Uniform => {
            let p = target as f64 / n as f64;
            for k in keep.iter_mut() {
                if rng.random::<f64>() < p {
                    *k = Some(1.0);
                }
            }
            notes.push(format!("row-sampling=uniform p={p:.4}"));
        }
        RowSampling::Stratified => {
            // Four strata by |residual| quartile; equal sample per stratum,
            // weight = stratum population / stratum sample.
            let mut abs: Vec<(f32, usize)> = residual.iter().map(|r| r.abs()).zip(0..).collect();
            abs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let strata = 4usize;
            let per = target.div_ceil(strata).max(1);
            for s in 0..strata {
                let lo = s * n / strata;
                let hi = ((s + 1) * n / strata).min(n);
                let pop = hi - lo;
                if pop == 0 {
                    continue;
                }
                let p = (per as f64 / pop as f64).min(1.0);
                let w = (1.0 / p) as f32;
                for &(_, i) in &abs[lo..hi] {
                    if rng.random::<f64>() < p {
                        keep[i] = Some(w);
                    }
                }
            }
            notes.push(format!(
                "row-sampling=stratified strata={strata} per-stratum≈{per}"
            ));
        }
        RowSampling::ResidualWeighted => {
            let eps = 1e-6f64;
            let total: f64 = residual.iter().map(|r| f64::from(r.abs()) + eps).sum();
            for (i, k) in keep.iter_mut().enumerate() {
                let p = ((f64::from(residual[i].abs()) + eps) / total * target as f64).min(1.0);
                if rng.random::<f64>() < p {
                    *k = Some((1.0 / p) as f32);
                }
            }
            notes.push(format!("row-sampling=residual-weighted target={target}"));
        }
    }
    RowPlan { keep }
}

/// Build the search set for `output` by streaming the corpus once.
pub fn build_search_set(
    cfg: &ForestsConfig,
    cache: &BinCache,
    residuals: &ResidualCache,
    training_dir: &Path,
    output: usize,
    seed: u64,
) -> Result<SearchSet, String> {
    let features = cache.features();
    let config = TrainingDataConfig::new(features, residuals.meta.output_count);
    let column = residuals.correction_column(output);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut notes = Vec::new();
    let plan = row_plan(cfg, &column, &mut rng, &mut notes);
    let mut chunks = Vec::new();
    let mut record_indices = Vec::new();
    let mut bins_buf = Vec::new();
    for_each_chunk(training_dir, &config, cfg.chunk_records, |chunk| {
        let mut out = BinnedChunk {
            features,
            first_index: chunk.first_index,
            ..Default::default()
        };
        let mut weights = Vec::new();
        for r in 0..chunk.records {
            let idx = chunk.first_index as usize + r;
            let Some(w) = plan.keep[idx] else { continue };
            cache.bin_rows(
                &chunk.inputs[r * features..(r + 1) * features],
                features,
                &mut bins_buf,
            );
            out.bins.extend_from_slice(&bins_buf);
            out.residual.push(column[idx]);
            weights.push(w);
            record_indices.push(idx as u64);
            out.records += 1;
        }
        if out.records > 0 {
            if weights.iter().any(|&w| w != 1.0) {
                out.weight = Some(weights);
            }
            chunks.push(out);
        }
        Ok(())
    })?;
    let label =
        if cfg.search_records == 0 || record_indices.len() as u64 >= residuals.meta.record_count {
            "memory-full".to_string()
        } else {
            format!("memory-sample/{}", record_indices.len())
        };
    let mut set = SearchSet {
        source: MemorySource { chunks, label },
        feature_map: (0..features).collect(),
        record_indices,
        row_sampling: cfg.row_sampling,
        feature_selection: cfg.feature_selection,
        output,
        notes,
    };
    select_features(&mut set, cfg, cache, &mut rng)?;
    Ok(set)
}

/// Apply the configured feature selection, projecting the set onto a subset.
fn select_features(
    set: &mut SearchSet,
    cfg: &ForestsConfig,
    cache: &BinCache,
    rng: &mut StdRng,
) -> Result<(), String> {
    let all = set.feature_map.len();
    let keep = ((all as f64 * cfg.feature_fraction).ceil() as usize).clamp(1, all);
    let selected: Vec<usize> = match cfg.feature_selection {
        FeatureSelection::All => return Ok(()),
        FeatureSelection::Random => {
            let mut idx: Vec<usize> = (0..all).collect();
            // Fisher–Yates prefix.
            for i in 0..keep {
                let j = i + rng.random_range(0..all - i);
                idx.swap(i, j);
            }
            let mut s = idx[..keep].to_vec();
            s.sort_unstable();
            set.notes
                .push(format!("feature-selection=random {keep}/{all}"));
            s
        }
        FeatureSelection::ErrorRanked => {
            let bins = set.bins_per_feature(cache);
            let hist = HistogramSet::from_source(&set.source, &bins)?;
            let mut scored: Vec<(f64, usize)> = hist
                .features
                .iter()
                .enumerate()
                .map(|(f, h)| {
                    // Between-bin variance of the mean residual = how much the
                    // binned feature explains (a correlation-ratio proxy).
                    let n: f64 = h.count.iter().sum();
                    let mean = if n > 0.0 {
                        h.sum.iter().sum::<f64>() / n
                    } else {
                        0.0
                    };
                    let explained: f64 = h
                        .count
                        .iter()
                        .zip(&h.sum)
                        .filter(|(c, _)| **c > 0.0)
                        .map(|(c, s)| c * (s / c - mean).powi(2))
                        .sum();
                    (explained, f)
                })
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let mut s: Vec<usize> = scored[..keep].iter().map(|x| x.1).collect();
            s.sort_unstable();
            set.notes
                .push(format!("feature-selection=error-ranked {keep}/{all}"));
            s
        }
    };
    project_features(set, &selected);
    Ok(())
}

/// Keep only `selected` (indices into the current set-feature space).
pub fn project_features(set: &mut SearchSet, selected: &[usize]) {
    let old = set.feature_map.len();
    for chunk in &mut set.source.chunks {
        let mut bins = Vec::with_capacity(chunk.records * selected.len());
        for r in 0..chunk.records {
            let row = &chunk.bins[r * old..(r + 1) * old];
            bins.extend(selected.iter().map(|&f| row[f]));
        }
        chunk.bins = bins;
        chunk.features = selected.len();
    }
    set.feature_map = selected.iter().map(|&f| set.feature_map[f]).collect();
}

/// Raw feature values for a subset of features over the search-set records
/// (used by oblique search, which cannot work on bins).
#[derive(Debug, Clone)]
pub struct RawSample {
    /// Original feature indices (columns).
    pub features: Vec<usize>,
    /// Row-major values, `records × features.len()`.
    pub values: Vec<f32>,
    /// Correction-space residual per record.
    pub residual: Vec<f32>,
    /// Importance weight per record.
    pub weight: Vec<f32>,
}

impl RawSample {
    /// Records.
    pub fn records(&self) -> usize {
        self.residual.len()
    }
}

/// Re-read raw values of `features` for the set's records.
pub fn raw_sample(
    set: &SearchSet,
    features: &[usize],
    training_dir: &Path,
    cache: &BinCache,
    outputs: usize,
    chunk_records: usize,
) -> Result<RawSample, String> {
    let width = cache.features();
    let config = TrainingDataConfig::new(width, outputs);
    let mut values = Vec::with_capacity(set.record_indices.len() * features.len());
    let mut residual = Vec::with_capacity(set.record_indices.len());
    let mut weight = Vec::with_capacity(set.record_indices.len());
    let mut next = 0usize;
    let mut chunk_iter = set.source.chunks.iter();
    let mut cur = chunk_iter.next();
    let mut cur_pos = 0usize;
    for_each_chunk(training_dir, &config, chunk_records, |chunk| {
        let end = chunk.first_index + chunk.records as u64;
        while next < set.record_indices.len() && set.record_indices[next] < end {
            let idx = set.record_indices[next];
            let r = (idx - chunk.first_index) as usize;
            let row = &chunk.inputs[r * width..(r + 1) * width];
            values.extend(features.iter().map(|&f| row[f]));
            // residual/weight come from the set chunks in the same order.
            while let Some(c) = cur {
                if cur_pos < c.records {
                    break;
                }
                cur = chunk_iter.next();
                cur_pos = 0;
            }
            let c = cur.ok_or("search set / corpus record mismatch")?;
            residual.push(c.residual[cur_pos]);
            weight.push(c.weight.as_ref().map_or(1.0, |w| w[cur_pos]));
            cur_pos += 1;
            next += 1;
        }
        Ok(())
    })?;
    Ok(RawSample {
        features: features.to_vec(),
        values,
        residual,
        weight,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bins::{BinBuildOptions, build_bin_cache};
    use crate::corpus::{corpus_info, write_bin_file};
    use crate::graft::fixtures::identity_creature;
    use crate::incumbent::Incumbent;
    use crate::residuals::compute_residuals;

    fn fixture(n: usize) -> (tempfile::TempDir, BinCache, ResidualCache) {
        let tmp = tempfile::tempdir().unwrap();
        let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..n)
            .map(|i| {
                let x0 = (i % 17) as f32 / 17.0;
                let x1 = ((i * 7) % 23) as f32 / 23.0;
                let x2 = (i % 3) as f32;
                (
                    vec![x0, x1, x2],
                    vec![x0 + if x1 > 0.6 { 0.5 } else { 0.0 }],
                )
            })
            .collect();
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let cfg = TrainingDataConfig::new(3, 1);
        let corpus = corpus_info(tmp.path(), &cfg).unwrap();
        let cache = build_bin_cache(
            tmp.path(),
            &cfg,
            &corpus,
            &BinBuildOptions {
                bins: 32,
                ..Default::default()
            },
        )
        .unwrap();
        let inc = Incumbent::from_creature(identity_creature(3, 1), "t").unwrap();
        let res = compute_residuals(&inc, tmp.path(), &corpus, 64, 1).unwrap();
        (tmp, cache, res)
    }

    #[test]
    fn stride_set_is_deterministic_and_complete_when_unsampled() {
        let (tmp, cache, res) = fixture(300);
        let cfg = ForestsConfig {
            search_records: 0,
            ..Default::default()
        };
        let a = build_search_set(&cfg, &cache, &res, tmp.path(), 0, 1).unwrap();
        let b = build_search_set(&cfg, &cache, &res, tmp.path(), 0, 2).unwrap();
        assert_eq!(a.records(), 300);
        assert_eq!(a.record_indices, b.record_indices);
        assert_eq!(a.label(), "memory-full");
        let cfg = ForestsConfig {
            search_records: 100,
            ..Default::default()
        };
        let s = build_search_set(&cfg, &cache, &res, tmp.path(), 0, 1).unwrap();
        assert_eq!(s.records(), 100);
        assert!(s.label().starts_with("memory-sample/100"));
    }

    #[test]
    fn random_strategies_reproduce_with_seed_and_weight_correctly() {
        let (tmp, cache, res) = fixture(400);
        for sampling in [
            RowSampling::Uniform,
            RowSampling::Stratified,
            RowSampling::ResidualWeighted,
        ] {
            let cfg = ForestsConfig {
                search_records: 120,
                row_sampling: sampling,
                ..Default::default()
            };
            let a = build_search_set(&cfg, &cache, &res, tmp.path(), 0, 42).unwrap();
            let b = build_search_set(&cfg, &cache, &res, tmp.path(), 0, 42).unwrap();
            let c = build_search_set(&cfg, &cache, &res, tmp.path(), 0, 43).unwrap();
            assert_eq!(a.record_indices, b.record_indices, "{sampling:?}");
            assert_ne!(a.record_indices, c.record_indices, "{sampling:?}");
            assert!(
                a.records() > 30 && a.records() < 300,
                "{sampling:?} {}",
                a.records()
            );
            if sampling != RowSampling::Uniform {
                assert!(a.source.chunks.iter().any(|c| c.weight.is_some()));
            }
            assert!(a.notes.iter().any(|n| n.contains("row-sampling")));
        }
    }

    #[test]
    fn feature_selection_projects_and_error_ranked_keeps_the_signal() {
        let (tmp, cache, res) = fixture(400);
        let cfg = ForestsConfig {
            feature_selection: FeatureSelection::ErrorRanked,
            feature_fraction: 0.3,
            ..Default::default()
        };
        let s = build_search_set(&cfg, &cache, &res, tmp.path(), 0, 1).unwrap();
        assert_eq!(s.feature_map, vec![1]);
        assert_eq!(s.source.chunks[0].features, 1);
        let cfg = ForestsConfig {
            feature_selection: FeatureSelection::Random,
            feature_fraction: 0.67,
            ..Default::default()
        };
        let r = build_search_set(&cfg, &cache, &res, tmp.path(), 0, 9).unwrap();
        assert_eq!(
            r.feature_map.len(),
            3usize.min(((3.0f64 * 0.67).ceil()) as usize)
        );
        let raw = raw_sample(&r, &[1], tmp.path(), &cache, 1, 64).unwrap();
        assert_eq!(raw.records(), 400);
        // raw feature 1 of record i equals ((i*7)%23)/23
        assert!((raw.values[5] - ((5 * 7 % 23) as f32 / 23.0)).abs() < 1e-6);
        assert_eq!(raw.residual[5], res.correction_at(5, 0));
    }
}

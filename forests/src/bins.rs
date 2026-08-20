//! Versioned quantile-bin cache (Issue #3).
//!
//! Histogram split search needs every continuous observation quantised into a
//! small number of approximately equal-population bins. The cache is built in
//! bounded memory (features are processed in blocks, each block one streaming
//! pass over the corpus, using deterministic stride sub-sampling) and persisted
//! beside the corpus in a compact binary file with a JSON metadata header.
//!
//! ## Binary format (`forests-bins.cache`)
//!
//! ```text
//! magic       4 bytes   b"NFBN"
//! format      u32 LE    FORMAT_VERSION
//! json_len    u32 LE
//! json        json_len bytes  (BinMeta, camelCase)
//! per feature:
//!   edge_count u32 LE
//!   edges      edge_count × f32 LE, strictly ascending
//! ```
//!
//! ## Mapping
//!
//! `bin(x) = |{ e : x > e }|` (a `partition_point`). Thus `x <= e_0` is bin 0,
//! `x > e_last` is bin `edge_count`, and a split "after bin b" is exactly the
//! `IF` condition `x > e_b`. **`NaN` maps to bin 0** — the `IF` kernel sends a
//! `NaN` condition sum to the negative (left) branch, so this keeps the
//! histogram consistent with the creature. `±∞` fall out naturally.
//!
//! ## Ties
//!
//! Duplicate values collapse into one edge, so heavily repeated values (e.g.
//! zeros) yield fewer effective bins than requested; `effectiveBins` records
//! it per feature. Equal-width binning is deliberately not used — skewed
//! observations would waste most bins.

use std::fmt;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use neat_core::training_data::TrainingDataConfig;
use serde::{Deserialize, Serialize};

use crate::corpus::{CorpusInfo, for_each_chunk};

/// Binary format version.
pub const FORMAT_VERSION: u32 = 1;
/// Quantisation algorithm version (stride sample + sorted quantile edges).
pub const ALGORITHM_VERSION: u32 = 1;
/// Default requested bins per observation.
pub const DEFAULT_BINS: usize = 256;
/// Hard cap so bin indices fit in a `u8` (255 edges → bins 0..=255).
pub const MAX_BINS: usize = 256;
/// Default per-pass memory budget for bin building.
pub const DEFAULT_BIN_MEMORY_BUDGET_BYTES: usize = 256 * 1024 * 1024;
/// Default maximum sampled records per feature for quantile estimation.
pub const DEFAULT_BIN_SAMPLE_RECORDS: u64 = 65_536;
/// File name beside the corpus.
pub const CACHE_FILE_NAME: &str = "forests-bins.cache";
const MAGIC: &[u8; 4] = b"NFBN";

/// Metadata header.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BinMeta {
    /// Binary format version.
    pub format_version: u32,
    /// Quantisation algorithm version.
    pub algorithm_version: u32,
    /// Input width.
    pub input_count: usize,
    /// Output width.
    pub output_count: usize,
    /// Corpus record count.
    pub record_count: u64,
    /// Corpus identity (see [`crate::corpus::corpus_info`]).
    pub corpus_identity: String,
    /// Bins requested.
    pub requested_bins: usize,
    /// Effective bins per feature (`edges + 1`).
    pub effective_bins: Vec<u16>,
    /// Non-finite values seen per feature in the sample.
    pub non_finite_counts: Vec<u64>,
    /// Policy string for non-finite handling.
    pub non_finite_policy: String,
    /// Records sampled per feature for quantile estimation.
    pub sample_records: u64,
    /// Stride used for deterministic sampling (1 = every record).
    pub sample_stride: u64,
    /// Unix seconds.
    pub created_at_unix: u64,
}

/// Loaded bin cache.
#[derive(Debug, Clone, PartialEq)]
pub struct BinCache {
    /// Header.
    pub meta: BinMeta,
    /// Ascending edges per feature.
    pub edges: Vec<Vec<f32>>,
}

/// Cache errors.
#[derive(Debug)]
pub enum BinCacheError {
    /// I/O failure.
    Io(PathBuf, std::io::Error),
    /// File exists but is not a valid cache (bad magic, truncated, bad JSON …).
    Corrupt(String),
    /// Valid cache for a different corpus/version/bin count.
    Stale(String),
    /// Corpus streaming failure.
    Corpus(String),
}

impl fmt::Display for BinCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(p, e) => write!(f, "{}: {e}", p.display()),
            Self::Corrupt(m) => write!(f, "bin cache corrupt: {m}"),
            Self::Stale(m) => write!(f, "bin cache stale: {m}"),
            Self::Corpus(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for BinCacheError {}

impl BinCache {
    /// Bin index of `x` for `feature` (see module docs for semantics).
    #[inline]
    pub fn bin(&self, feature: usize, x: f32) -> u8 {
        bin_of(&self.edges[feature], x)
    }

    /// Threshold for a split "after bin `b`" (the `IF` condition is `x > threshold`).
    pub fn threshold(&self, feature: usize, bin: usize) -> Option<f32> {
        self.edges[feature].get(bin).copied()
    }

    /// Bins for `feature` (`edges + 1`).
    pub fn bins(&self, feature: usize) -> usize {
        self.edges[feature].len() + 1
    }

    /// Largest bin count over all features.
    pub fn max_bins(&self) -> usize {
        self.edges.iter().map(|e| e.len() + 1).max().unwrap_or(1)
    }

    /// Number of features.
    pub fn features(&self) -> usize {
        self.edges.len()
    }

    /// Quantise a row-major block of records into `out` (`records × features` u8).
    pub fn bin_rows(&self, inputs: &[f32], features: usize, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(inputs.len());
        for row in inputs.chunks_exact(features) {
            for (f, &x) in row.iter().enumerate() {
                out.push(bin_of(&self.edges[f], x));
            }
        }
    }

    /// Robust scale of a feature (inter-quartile range of its edges, or 1.0).
    pub fn scale(&self, feature: usize) -> f32 {
        let e = &self.edges[feature];
        if e.len() < 4 {
            return 1.0;
        }
        let q1 = e[e.len() / 4];
        let q3 = e[e.len() * 3 / 4];
        let iqr = q3 - q1;
        if iqr.is_finite() && iqr > 0.0 {
            iqr
        } else {
            1.0
        }
    }

    /// Serialise to `path` atomically (write temp + rename).
    pub fn write(&self, path: &Path) -> Result<(), BinCacheError> {
        let tmp = path.with_extension("cache.tmp");
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        let json =
            serde_json::to_vec(&self.meta).map_err(|e| BinCacheError::Corrupt(e.to_string()))?;
        buf.extend_from_slice(&(json.len() as u32).to_le_bytes());
        buf.extend_from_slice(&json);
        for edges in &self.edges {
            buf.extend_from_slice(&(edges.len() as u32).to_le_bytes());
            for e in edges {
                buf.extend_from_slice(&e.to_le_bytes());
            }
        }
        std::fs::File::create(&tmp)
            .and_then(|mut f| f.write_all(&buf))
            .map_err(|e| BinCacheError::Io(tmp.clone(), e))?;
        std::fs::rename(&tmp, path).map_err(|e| BinCacheError::Io(path.to_path_buf(), e))
    }

    /// Read and structurally validate a cache file (no staleness check).
    pub fn read(path: &Path) -> Result<Self, BinCacheError> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|e| BinCacheError::Io(path.to_path_buf(), e))?;
        Self::from_bytes(&bytes)
    }

    /// Parse the documented binary layout.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BinCacheError> {
        let corrupt = |m: &str| BinCacheError::Corrupt(m.to_string());
        if bytes.len() < 12 || &bytes[..4] != MAGIC {
            return Err(corrupt("bad magic"));
        }
        let format = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if format != FORMAT_VERSION {
            return Err(BinCacheError::Stale(format!(
                "format {format} != {FORMAT_VERSION}"
            )));
        }
        let json_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut pos = 12;
        let json = bytes
            .get(pos..pos + json_len)
            .ok_or_else(|| corrupt("truncated header"))?;
        let meta: BinMeta =
            serde_json::from_slice(json).map_err(|e| corrupt(&format!("header json: {e}")))?;
        pos += json_len;
        let mut edges = Vec::with_capacity(meta.input_count);
        for f in 0..meta.input_count {
            let n = bytes
                .get(pos..pos + 4)
                .ok_or_else(|| corrupt(&format!("truncated at feature {f}")))?;
            let n = u32::from_le_bytes(n.try_into().unwrap()) as usize;
            pos += 4;
            let raw = bytes
                .get(pos..pos + n * 4)
                .ok_or_else(|| corrupt(&format!("truncated edges for feature {f}")))?;
            let mut e = Vec::with_capacity(n);
            for c in raw.chunks_exact(4) {
                e.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
            if e.windows(2)
                .any(|w| w[0] >= w[1] || w[0].is_nan() || w[1].is_nan())
            {
                return Err(corrupt(&format!(
                    "edges not strictly ascending for feature {f}"
                )));
            }
            if meta.effective_bins.get(f).copied() != Some((n + 1) as u16) {
                return Err(corrupt(&format!("effectiveBins mismatch for feature {f}")));
            }
            pos += n * 4;
            edges.push(e);
        }
        if pos != bytes.len() {
            return Err(corrupt("trailing bytes"));
        }
        Ok(Self { meta, edges })
    }

    /// Check the cache belongs to `corpus` at the requested bin count.
    pub fn check_compatible(
        &self,
        corpus: &CorpusInfo,
        requested_bins: usize,
    ) -> Result<(), BinCacheError> {
        let m = &self.meta;
        let stale = |m: String| Err(BinCacheError::Stale(m));
        if m.algorithm_version != ALGORITHM_VERSION {
            return stale(format!(
                "algorithm {} != {ALGORITHM_VERSION}",
                m.algorithm_version
            ));
        }
        if m.corpus_identity != corpus.identity {
            return stale(format!(
                "corpus identity {} != {}",
                m.corpus_identity, corpus.identity
            ));
        }
        if m.record_count != corpus.record_count
            || m.input_count != corpus.input_count
            || m.output_count != corpus.output_count
        {
            return stale("corpus shape differs".into());
        }
        if m.requested_bins != requested_bins {
            return stale(format!(
                "requested bins {} != {requested_bins}",
                m.requested_bins
            ));
        }
        Ok(())
    }
}

/// Bin index of `x` given ascending `edges`.
#[inline]
pub fn bin_of(edges: &[f32], x: f32) -> u8 {
    if x.is_nan() {
        return 0;
    }
    edges.partition_point(|&e| x > e) as u8
}

/// Options for building the cache.
#[derive(Debug, Clone)]
pub struct BinBuildOptions {
    /// Requested bins (≤ [`MAX_BINS`]).
    pub bins: usize,
    /// Maximum sampled records per feature.
    pub sample_records: u64,
    /// Per-pass memory budget in bytes.
    pub memory_budget_bytes: usize,
    /// Records per streaming chunk.
    pub chunk_records: usize,
}

impl Default for BinBuildOptions {
    fn default() -> Self {
        Self {
            bins: DEFAULT_BINS,
            sample_records: DEFAULT_BIN_SAMPLE_RECORDS,
            memory_budget_bytes: DEFAULT_BIN_MEMORY_BUDGET_BYTES,
            chunk_records: 4096,
        }
    }
}

/// Build quantile edges from one feature's sampled values.
///
/// Returns strictly ascending edges; duplicates collapse (tie policy).
pub fn quantile_edges(values: &mut Vec<f32>, bins: usize) -> Vec<f32> {
    values.retain(|v| v.is_finite());
    values.sort_by(|a, b| a.total_cmp(b));
    let n = values.len();
    let mut edges = Vec::with_capacity(bins);
    if n == 0 || bins < 2 {
        return edges;
    }
    for b in 1..bins {
        let idx = (b * n / bins).min(n - 1);
        let e = values[idx];
        if e == -0.0 && edges.last().copied() == Some(0.0) {
            continue;
        }
        let e = if e == -0.0 { 0.0 } else { e };
        if edges.last().is_none_or(|&last| e > last) {
            edges.push(e);
        }
    }
    // The top value must remain reachable as "x > e_last": drop an edge equal
    // to the maximum so the last bin is never empty by construction.
    if let (Some(&last), Some(&max)) = (edges.last(), values.last())
        && last >= max
    {
        edges.pop();
    }
    edges
}

/// Build the cache with bounded memory: features are processed in blocks,
/// each block one streaming pass with deterministic stride sampling.
pub fn build_bin_cache(
    training_dir: &Path,
    config: &TrainingDataConfig,
    corpus: &CorpusInfo,
    opts: &BinBuildOptions,
) -> Result<BinCache, BinCacheError> {
    let bins = opts.bins.clamp(2, MAX_BINS);
    let features = config.num_inputs;
    let stride = corpus
        .record_count
        .div_ceil(opts.sample_records.max(1))
        .max(1);
    let sample_records = corpus.record_count.div_ceil(stride);
    let per_feature_bytes = (sample_records as usize).saturating_mul(4).max(4);
    let block = (opts.memory_budget_bytes / per_feature_bytes).clamp(1, features.max(1));
    let mut edges: Vec<Vec<f32>> = Vec::with_capacity(features);
    let mut non_finite = vec![0u64; features];
    let mut start = 0;
    while start < features {
        let end = (start + block).min(features);
        let mut cols: Vec<Vec<f32>> = (start..end)
            .map(|_| Vec::with_capacity(sample_records as usize))
            .collect();
        for_each_chunk(training_dir, config, opts.chunk_records, |chunk| {
            for r in 0..chunk.records {
                if !(chunk.first_index + r as u64).is_multiple_of(stride) {
                    continue;
                }
                let row = &chunk.inputs[r * features..(r + 1) * features];
                for (k, col) in cols.iter_mut().enumerate() {
                    let x = row[start + k];
                    if x.is_finite() {
                        col.push(x);
                    } else {
                        non_finite[start + k] += 1;
                    }
                }
            }
            Ok(())
        })
        .map_err(BinCacheError::Corpus)?;
        for mut col in cols {
            edges.push(quantile_edges(&mut col, bins));
        }
        start = end;
    }
    let meta = BinMeta {
        format_version: FORMAT_VERSION,
        algorithm_version: ALGORITHM_VERSION,
        input_count: features,
        output_count: config.num_outputs,
        record_count: corpus.record_count,
        corpus_identity: corpus.identity.clone(),
        requested_bins: bins,
        effective_bins: edges.iter().map(|e| (e.len() + 1) as u16).collect(),
        non_finite_counts: non_finite,
        non_finite_policy:
            "NaN→bin 0 (left of every threshold, matching IF NaN semantics); ±inf ordered naturally"
                .into(),
        sample_records,
        sample_stride: stride,
        created_at_unix: crate::incumbent::now_unix(),
    };
    Ok(BinCache { meta, edges })
}

/// Default cache path for a corpus directory.
pub fn cache_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join(CACHE_FILE_NAME)
}

/// Load a compatible cache from `cache_dir`, or (re)build and persist one.
///
/// A stale or corrupt cache is reported with a warning and rebuilt; it is
/// never silently reused.
pub fn ensure_bin_cache(
    training_dir: &Path,
    cache_dir: &Path,
    config: &TrainingDataConfig,
    corpus: &CorpusInfo,
    opts: &BinBuildOptions,
) -> Result<BinCache, BinCacheError> {
    let path = cache_path(cache_dir);
    if path.exists() {
        match BinCache::read(&path).and_then(|c| {
            c.check_compatible(corpus, opts.bins.clamp(2, MAX_BINS))
                .map(|()| c)
        }) {
            Ok(cache) => {
                crate::log::detail(&format!(
                    "reusing bin cache {} ({} features)",
                    path.display(),
                    cache.features()
                ));
                return Ok(cache);
            }
            Err(e @ (BinCacheError::Stale(_) | BinCacheError::Corrupt(_))) => {
                crate::log::warn(&format!("{e}; rebuilding {}", path.display()));
            }
            Err(e) => return Err(e),
        }
    }
    let cache = build_bin_cache(training_dir, config, corpus, opts)?;
    std::fs::create_dir_all(cache_dir)
        .map_err(|e| BinCacheError::Io(cache_dir.to_path_buf(), e))?;
    cache.write(&path)?;
    crate::log::detail(&format!(
        "built bin cache {} ({} features, {} sampled records, stride {})",
        path.display(),
        cache.features(),
        cache.meta.sample_records,
        cache.meta.sample_stride
    ));
    Ok(cache)
}

/// Occupancy diagnostics for a binning (used by tests/benchmarks).
pub fn occupancy(edges: &[f32], values: &[f32]) -> Vec<u64> {
    let mut counts = vec![0u64; edges.len() + 1];
    for &v in values {
        counts[usize::from(bin_of(edges, v))] += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{corpus_info, write_bin_file};

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / (u32::MAX as f32 / 2.0)
    }

    fn skewed_corpus(dir: &Path, n: usize) {
        let mut seed = 7u64;
        let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..n)
            .map(|i| {
                let u = lcg(&mut seed).max(1e-6);
                let skew = (-u.ln()).powi(3); // heavy right tail
                let nan = if i % 50 == 0 { f32::NAN } else { i as f32 };
                (vec![skew, nan], vec![0.0])
            })
            .collect();
        write_bin_file(&dir.join("0.bin"), &recs).unwrap();
    }

    #[test]
    fn edges_are_deterministic_and_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        skewed_corpus(tmp.path(), 2000);
        let cfg = TrainingDataConfig::new(2, 1);
        let corpus = corpus_info(tmp.path(), &cfg).unwrap();
        let opts = BinBuildOptions {
            bins: 16,
            ..Default::default()
        };
        let a = build_bin_cache(tmp.path(), &cfg, &corpus, &opts).unwrap();
        let b = build_bin_cache(tmp.path(), &cfg, &corpus, &opts).unwrap();
        assert_eq!(a.edges, b.edges);
        assert_eq!(a.meta.non_finite_counts[1], 40);
        let bytes_path = tmp.path().join(CACHE_FILE_NAME);
        a.write(&bytes_path).unwrap();
        let back = BinCache::read(&bytes_path).unwrap();
        assert_eq!(back.edges, a.edges);
        assert_eq!(back.meta.corpus_identity, corpus.identity);
        back.check_compatible(&corpus, 16).unwrap();
    }

    #[test]
    fn skewed_feature_gets_far_better_occupancy_than_equal_width() {
        let tmp = tempfile::tempdir().unwrap();
        skewed_corpus(tmp.path(), 5000);
        let cfg = TrainingDataConfig::new(2, 1);
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
        let mut values = Vec::new();
        for_each_chunk(tmp.path(), &cfg, 1000, |c| {
            values.extend(c.inputs.chunks(2).map(|r| r[0]));
            Ok(())
        })
        .unwrap();
        let q = occupancy(&cache.edges[0], &values);
        let (min, max) = values
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        let width: Vec<f32> = (1..32)
            .map(|i| min + (max - min) * i as f32 / 32.0)
            .collect();
        let w = occupancy(&width, &values);
        let share = |c: &[u64]| *c.iter().max().unwrap() as f64 / values.len() as f64;
        assert!(share(&q) < 0.08, "quantile max share {}", share(&q));
        assert!(share(&w) > 0.5, "equal-width max share {}", share(&w));
        assert!(q.iter().filter(|&&c| c == 0).count() <= 1);
    }

    #[test]
    fn mapping_handles_boundaries_duplicates_and_non_finite() {
        let edges = vec![0.0f32, 1.0, 2.0];
        assert_eq!(bin_of(&edges, -5.0), 0);
        assert_eq!(bin_of(&edges, 0.0), 0);
        assert_eq!(bin_of(&edges, 0.0000001), 1);
        assert_eq!(bin_of(&edges, 1.0), 1);
        assert_eq!(bin_of(&edges, 2.0), 2);
        assert_eq!(bin_of(&edges, 2.5), 3);
        assert_eq!(bin_of(&edges, f32::NAN), 0);
        assert_eq!(bin_of(&edges, f32::INFINITY), 3);
        assert_eq!(bin_of(&edges, f32::NEG_INFINITY), 0);
        let mut dup = vec![1.0f32; 100];
        dup.extend([2.0f32; 100]);
        let e = quantile_edges(&mut dup, 8);
        assert_eq!(e, vec![1.0]);
        let mut constant = vec![3.0f32; 50];
        assert!(quantile_edges(&mut constant, 8).is_empty());
    }

    #[test]
    fn stale_and_corrupt_caches_fail_clearly() {
        let tmp = tempfile::tempdir().unwrap();
        skewed_corpus(tmp.path(), 300);
        let cfg = TrainingDataConfig::new(2, 1);
        let corpus = corpus_info(tmp.path(), &cfg).unwrap();
        let opts = BinBuildOptions {
            bins: 8,
            ..Default::default()
        };
        let cache = ensure_bin_cache(tmp.path(), tmp.path(), &cfg, &corpus, &opts).unwrap();
        assert!(matches!(
            cache.check_compatible(&corpus, 9),
            Err(BinCacheError::Stale(_))
        ));
        let mut other = corpus.clone();
        other.identity = "ffff".into();
        assert!(matches!(
            cache.check_compatible(&other, 8),
            Err(BinCacheError::Stale(_))
        ));
        let path = cache_path(tmp.path());
        let bytes = std::fs::read(&path).unwrap();
        assert!(matches!(
            BinCache::from_bytes(&bytes[..bytes.len() - 3]),
            Err(BinCacheError::Corrupt(_))
        ));
        assert!(matches!(
            BinCache::from_bytes(b"junk"),
            Err(BinCacheError::Corrupt(_))
        ));
        std::fs::write(&path, &bytes[..20]).unwrap();
        // ensure() rebuilds a corrupt file rather than reusing it.
        let rebuilt = ensure_bin_cache(tmp.path(), tmp.path(), &cfg, &corpus, &opts).unwrap();
        assert_eq!(rebuilt.edges, cache.edges);
        // A different corpus invalidates reuse.
        write_bin_file(&tmp.path().join("1.bin"), &[(vec![1.0, 2.0], vec![0.0])]).unwrap();
        let corpus2 = corpus_info(tmp.path(), &cfg).unwrap();
        assert!(matches!(
            cache.check_compatible(&corpus2, 8),
            Err(BinCacheError::Stale(_))
        ));
    }
}

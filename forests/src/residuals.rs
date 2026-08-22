//! Incumbent residual extraction and regional diagnostics (Issue #4).
//!
//! ## Sign convention
//!
//! `residual = target − prediction` in **output space**. A positive residual
//! means the incumbent predicts too low, so a positive correction helps.
//!
//! ## Correction space
//!
//! A graft adds to the output neuron's *pre-squash* sum, so the search needs
//! the residual expressed there. For each output with squash `s` and pre-squash
//! value `h` (the "hint" NEAT-AI-core traces), the correction-space residual is
//! `unsquash_s(target, h) − h`, using NEAT-AI-core's own `apply_unsquash`. For
//! an `IDENTITY` output — and for aggregate outputs such as an `IF` output
//! neuron, whose activation is linear in the winning branch sum — the two
//! residuals coincide. Both are stored; the
//! correction-space one drives histogram search, the output-space one drives
//! the MSE parity check against the scorer.
//!
//! Residual statistics are **search signals**, never acceptance metrics.
//!
//! ## Sidecar format (`forests-residuals-<checksum12>.cache`)
//!
//! ```text
//! magic "NFRS" | u32 format | u32 json_len | json ResidualMeta
//! residual    records×outputs f32 LE
//! correction  records×outputs f32 LE
//! ```

use std::fmt;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use neat_core::training_data::TrainingDataConfig;
use neat_core::unsquash::apply_unsquash;
use neat_core::{CompiledNetwork, CreatureExport, SquashType, compile_creature};
use serde::{Deserialize, Serialize};

use crate::corpus::{CorpusInfo, for_each_chunk};
use crate::incumbent::Incumbent;

/// Sidecar format version.
pub const FORMAT_VERSION: u32 = 1;
/// Residual algorithm version.
pub const ALGORITHM_VERSION: u32 = 1;
const MAGIC: &[u8; 4] = b"NFRS";
/// Largest-|residual| records retained for concentration diagnostics.
pub const TOP_TAIL_RECORDS: usize = 32;

/// Per-output residual statistics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ResidualStats {
    /// Records.
    pub count: u64,
    /// Mean of `target − prediction`.
    pub mean: f64,
    /// Population variance of the residual.
    pub variance: f64,
    /// Mean absolute residual.
    pub mae: f64,
    /// Sum of squared residuals (output space).
    pub sse: f64,
    /// `sse / count`.
    pub mse: f64,
    /// Mean of squared correction-space residual.
    pub correction_mse: f64,
    /// Smallest residual.
    pub min: f64,
    /// Largest residual.
    pub max: f64,
    /// |residual| quantiles p50/p90/p99/p999.
    pub abs_quantiles: [f64; 4],
    /// Records with |residual| > 2σ and > 3σ.
    pub tail_counts: [u64; 2],
    /// Share of total SSE contributed by the worst 1 % of records.
    pub top_percent_sse_share: f64,
    /// Records with the largest |residual| (`record index`, `residual`).
    pub top_records: Vec<(u64, f32)>,
    /// Records where the incumbent output or target was non-finite.
    pub non_finite: u64,
}

/// Sidecar header.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResidualMeta {
    /// Format version.
    pub format_version: u32,
    /// Algorithm version.
    pub algorithm_version: u32,
    /// Incumbent checksum.
    pub incumbent_checksum: String,
    /// Corpus identity.
    pub corpus_identity: String,
    /// Records.
    pub record_count: u64,
    /// Outputs per record.
    pub output_count: usize,
    /// Output squash names (correction space depends on them).
    pub output_squashes: Vec<String>,
    /// Whole-corpus statistics per output.
    pub stats: Vec<ResidualStats>,
    /// Mean per-record MSE exactly as NEAT-AI-core defines it (parity proxy).
    pub local_mse: f64,
    /// `true` when the training format supplied sample weights (it never does today).
    pub has_sample_weights: bool,
    /// Unix seconds.
    pub created_at_unix: u64,
}

/// Residuals for every record of the corpus under one incumbent.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualCache {
    /// Header.
    pub meta: ResidualMeta,
    /// Output-space residuals, `records × outputs`.
    pub residual: Vec<f32>,
    /// Correction-space residuals, `records × outputs`.
    pub correction: Vec<f32>,
}

/// Errors.
#[derive(Debug)]
pub enum ResidualError {
    /// I/O.
    Io(PathBuf, std::io::Error),
    /// Corrupt sidecar.
    Corrupt(String),
    /// Sidecar for a different incumbent/corpus.
    Stale(String),
    /// Creature/corpus failure.
    Other(String),
}

impl fmt::Display for ResidualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(p, e) => write!(f, "{}: {e}", p.display()),
            Self::Corrupt(m) => write!(f, "residual cache corrupt: {m}"),
            Self::Stale(m) => write!(f, "residual cache stale: {m}"),
            Self::Other(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ResidualError {}

/// Output squash types of a compiled creature (last `outputs` neurons).
pub fn output_squashes(network: &CompiledNetwork, outputs: usize) -> Vec<SquashType> {
    let n = network.neurons.len();
    network.neurons[n - outputs..]
        .iter()
        .map(|x| SquashType::from(x.squash_type))
        .collect()
}

/// Per-record residuals in output and correction space.
fn record_residuals(
    network: &mut CompiledNetwork,
    squashes: &[SquashType],
    inputs: &[f32],
    targets: &[f32],
    residual: &mut [f32],
    correction: &mut [f32],
) {
    let outputs = targets.len();
    let traced = network.activate_and_trace(inputs, outputs);
    let non_input = network.num_neurons() - network.num_inputs();
    let hints_start = outputs + non_input + (non_input - outputs);
    for j in 0..outputs {
        let prediction = traced[j];
        let hint = traced[hints_start + j];
        let target = targets[j];
        residual[j] = target - prediction;
        // Aggregate outputs (e.g. an `IF` output neuron) apply no squash:
        // the activation is linear in the winning branch sum, so the
        // correction space is the output space. `unsquash` has no inverse for
        // them and must not be used.
        correction[j] = if squashes[j].is_aggregate() {
            residual[j]
        } else {
            let pre = apply_unsquash(squashes[j], target, hint);
            if pre.is_finite() { pre - hint } else { 0.0 }
        };
    }
}

/// Compute residuals for the whole corpus (streaming, `threads` workers per chunk).
pub fn compute_residuals(
    incumbent: &Incumbent,
    training_dir: &Path,
    corpus: &CorpusInfo,
    chunk_records: usize,
    threads: usize,
) -> Result<ResidualCache, ResidualError> {
    let creature: &CreatureExport = &incumbent.creature;
    let config = TrainingDataConfig::new(creature.input, creature.output);
    let template = compile_creature(creature).map_err(|e| ResidualError::Other(e.to_string()))?;
    let squashes = output_squashes(&template, creature.output);
    let outputs = creature.output;
    let inputs = creature.input;
    let total = corpus.record_count as usize;
    let mut residual = vec![0f32; total * outputs];
    let mut correction = vec![0f32; total * outputs];
    let threads = threads.max(1);
    for_each_chunk(training_dir, &config, chunk_records, |chunk| {
        let base = chunk.first_index as usize * outputs;
        let res_out = &mut residual[base..base + chunk.records * outputs];
        let cor_out = &mut correction[base..base + chunk.records * outputs];
        let per = chunk.records.div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for ((r_part, c_part), t) in res_out
                .chunks_mut(per * outputs)
                .zip(cor_out.chunks_mut(per * outputs))
                .zip(0..)
            {
                let start = t * per;
                let rows = r_part.len() / outputs;
                let in_slice = &chunk.inputs[start * inputs..(start + rows) * inputs];
                let tg_slice = &chunk.targets[start * outputs..(start + rows) * outputs];
                let mut net = template.clone();
                let squashes = &squashes;
                handles.push(scope.spawn(move || {
                    for r in 0..rows {
                        record_residuals(
                            &mut net,
                            squashes,
                            &in_slice[r * inputs..(r + 1) * inputs],
                            &tg_slice[r * outputs..(r + 1) * outputs],
                            &mut r_part[r * outputs..(r + 1) * outputs],
                            &mut c_part[r * outputs..(r + 1) * outputs],
                        );
                    }
                }));
            }
            for h in handles {
                h.join()
                    .map_err(|_| "residual worker panicked".to_string())?;
            }
            Ok(())
        })
    })
    .map_err(ResidualError::Other)?;
    let stats: Vec<ResidualStats> = (0..outputs)
        .map(|j| stats_for(&residual, &correction, outputs, j))
        .collect();
    let local_mse = if total == 0 {
        0.0
    } else {
        residual
            .chunks_exact(outputs)
            .map(|row| {
                row.iter()
                    .map(|&d| f64::from(d) * f64::from(d))
                    .sum::<f64>()
                    / outputs as f64
            })
            .sum::<f64>()
            / total as f64
    };
    let meta = ResidualMeta {
        format_version: FORMAT_VERSION,
        algorithm_version: ALGORITHM_VERSION,
        incumbent_checksum: incumbent.checksum.clone(),
        corpus_identity: corpus.identity.clone(),
        record_count: corpus.record_count,
        output_count: outputs,
        output_squashes: squashes
            .iter()
            .map(|s| neat_core::squash_name_from(*s).to_string())
            .collect(),
        stats,
        local_mse,
        has_sample_weights: false,
        created_at_unix: crate::incumbent::now_unix(),
    };
    Ok(ResidualCache {
        meta,
        residual,
        correction,
    })
}

fn stats_for(residual: &[f32], correction: &[f32], outputs: usize, j: usize) -> ResidualStats {
    let vals: Vec<f32> = residual.iter().skip(j).step_by(outputs).copied().collect();
    let cors: Vec<f32> = correction
        .iter()
        .skip(j)
        .step_by(outputs)
        .copied()
        .collect();
    let mut st = ResidualStats {
        count: vals.len() as u64,
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
        ..Default::default()
    };
    if vals.is_empty() {
        st.min = 0.0;
        st.max = 0.0;
        return st;
    }
    let mut sum = 0.0;
    let mut sse = 0.0;
    let mut mae = 0.0;
    let mut csse = 0.0;
    for (&v, &c) in vals.iter().zip(&cors) {
        if !v.is_finite() {
            st.non_finite += 1;
            continue;
        }
        let v = f64::from(v);
        sum += v;
        sse += v * v;
        mae += v.abs();
        csse += f64::from(c) * f64::from(c);
        st.min = st.min.min(v);
        st.max = st.max.max(v);
    }
    let n = (vals.len() as u64 - st.non_finite).max(1) as f64;
    st.mean = sum / n;
    st.variance = (sse / n - st.mean * st.mean).max(0.0);
    st.mae = mae / n;
    st.sse = sse;
    st.mse = sse / n;
    st.correction_mse = csse / n;
    let sigma = st.variance.sqrt();
    let mut abs: Vec<(f32, u64)> = vals
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, v)| (v.abs(), i as u64))
        .collect();
    abs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let q = |p: f64| -> f64 {
        if abs.is_empty() {
            return 0.0;
        }
        let k = ((1.0 - p) * (abs.len() - 1) as f64).round() as usize;
        f64::from(abs[k.min(abs.len() - 1)].0)
    };
    st.abs_quantiles = [q(0.5), q(0.9), q(0.99), q(0.999)];
    st.tail_counts = [
        abs.iter()
            .filter(|(a, _)| f64::from(*a) > 2.0 * sigma)
            .count() as u64,
        abs.iter()
            .filter(|(a, _)| f64::from(*a) > 3.0 * sigma)
            .count() as u64,
    ];
    let top1 = (abs.len() / 100).max(1);
    let top_sse: f64 = abs
        .iter()
        .take(top1)
        .map(|(a, _)| f64::from(*a) * f64::from(*a))
        .sum();
    st.top_percent_sse_share = if sse > 0.0 { top_sse / sse } else { 0.0 };
    st.top_records = abs
        .iter()
        .take(TOP_TAIL_RECORDS)
        .map(|&(_, i)| (i, vals[i as usize]))
        .collect();
    st
}

impl ResidualCache {
    /// Residual (output space) of `record` for `output`.
    pub fn residual_at(&self, record: usize, output: usize) -> f32 {
        self.residual[record * self.meta.output_count + output]
    }

    /// Correction-space residual of `record` for `output`.
    pub fn correction_at(&self, record: usize, output: usize) -> f32 {
        self.correction[record * self.meta.output_count + output]
    }

    /// Correction-space residuals for one output as a contiguous vector.
    pub fn correction_column(&self, output: usize) -> Vec<f32> {
        self.correction
            .iter()
            .skip(output)
            .step_by(self.meta.output_count)
            .copied()
            .collect()
    }

    /// Serialise to `path` atomically.
    pub fn write(&self, path: &Path) -> Result<(), ResidualError> {
        let tmp = path.with_extension("cache.tmp");
        let json =
            serde_json::to_vec(&self.meta).map_err(|e| ResidualError::Corrupt(e.to_string()))?;
        let mut out = std::io::BufWriter::new(
            std::fs::File::create(&tmp).map_err(|e| ResidualError::Io(tmp.clone(), e))?,
        );
        let io = |e: std::io::Error| ResidualError::Io(tmp.clone(), e);
        out.write_all(MAGIC).map_err(io)?;
        out.write_all(&FORMAT_VERSION.to_le_bytes()).map_err(io)?;
        out.write_all(&(json.len() as u32).to_le_bytes())
            .map_err(io)?;
        out.write_all(&json).map_err(io)?;
        for v in self.residual.iter().chain(&self.correction) {
            out.write_all(&v.to_le_bytes()).map_err(io)?;
        }
        out.flush().map_err(io)?;
        drop(out);
        std::fs::rename(&tmp, path).map_err(|e| ResidualError::Io(path.to_path_buf(), e))
    }

    /// Read from `path`.
    pub fn read(path: &Path) -> Result<Self, ResidualError> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|e| ResidualError::Io(path.to_path_buf(), e))?;
        let corrupt = |m: &str| ResidualError::Corrupt(m.to_string());
        if bytes.len() < 12 || &bytes[..4] != MAGIC {
            return Err(corrupt("bad magic"));
        }
        let format = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if format != FORMAT_VERSION {
            return Err(ResidualError::Stale(format!("format {format}")));
        }
        let json_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let json = bytes
            .get(12..12 + json_len)
            .ok_or_else(|| corrupt("truncated header"))?;
        let meta: ResidualMeta =
            serde_json::from_slice(json).map_err(|e| corrupt(&e.to_string()))?;
        let n = meta.record_count as usize * meta.output_count;
        let body = bytes
            .get(12 + json_len..)
            .ok_or_else(|| corrupt("truncated"))?;
        if body.len() != n * 8 {
            return Err(corrupt(&format!(
                "expected {} payload bytes, found {}",
                n * 8,
                body.len()
            )));
        }
        // Rust 1.98's clippy::chunks_exact_to_as_chunks — see bins.rs: the
        // fixed-size chunks drop the fallible conversion entirely.
        let read = |s: &[u8]| {
            s.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect::<Vec<f32>>()
        };
        Ok(Self {
            meta,
            residual: read(&body[..n * 4]),
            correction: read(&body[n * 4..]),
        })
    }

    /// Check the sidecar belongs to `incumbent` × `corpus`.
    pub fn check_compatible(
        &self,
        incumbent: &Incumbent,
        corpus: &CorpusInfo,
    ) -> Result<(), ResidualError> {
        if self.meta.algorithm_version != ALGORITHM_VERSION {
            return Err(ResidualError::Stale("algorithm version".into()));
        }
        if self.meta.incumbent_checksum != incumbent.checksum {
            return Err(ResidualError::Stale("incumbent checksum differs".into()));
        }
        if self.meta.corpus_identity != corpus.identity
            || self.meta.record_count != corpus.record_count
        {
            return Err(ResidualError::Stale("corpus differs".into()));
        }
        Ok(())
    }
}

/// Sidecar path for an incumbent.
pub fn cache_path(cache_dir: &Path, incumbent: &Incumbent) -> PathBuf {
    cache_dir.join(format!(
        "forests-residuals-{}.cache",
        incumbent.short_checksum()
    ))
}

/// Load a compatible residual sidecar or compute and persist one.
pub fn ensure_residual_cache(
    incumbent: &Incumbent,
    training_dir: &Path,
    cache_dir: &Path,
    corpus: &CorpusInfo,
    chunk_records: usize,
    threads: usize,
) -> Result<ResidualCache, ResidualError> {
    let path = cache_path(cache_dir, incumbent);
    if path.exists() {
        match ResidualCache::read(&path)
            .and_then(|c| c.check_compatible(incumbent, corpus).map(|()| c))
        {
            Ok(c) => {
                crate::log::detail(&format!("reusing residual cache {}", path.display()));
                return Ok(c);
            }
            Err(e @ (ResidualError::Stale(_) | ResidualError::Corrupt(_))) => {
                crate::log::warn(&format!("{e}; recomputing residuals"));
            }
            Err(e) => return Err(e),
        }
    }
    let cache = compute_residuals(incumbent, training_dir, corpus, chunk_records, threads)?;
    std::fs::create_dir_all(cache_dir)
        .map_err(|e| ResidualError::Io(cache_dir.to_path_buf(), e))?;
    cache.write(&path)?;
    Ok(cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{corpus_info, write_bin_file};
    use crate::graft::fixtures::{identity_creature, small_mlp};

    #[test]
    fn identity_residuals_match_hand_calculation() {
        let tmp = tempfile::tempdir().unwrap();
        let recs = vec![
            (vec![1.0f32, 0.0], vec![1.5f32]),
            (vec![2.0, 0.0], vec![1.0]),
            (vec![-1.0, 0.0], vec![-1.0]),
            (vec![0.5, 0.0], vec![0.0]),
        ];
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let inc = Incumbent::from_creature(identity_creature(2, 1), "t").unwrap();
        let cfg = TrainingDataConfig::new(2, 1);
        let corpus = corpus_info(tmp.path(), &cfg).unwrap();
        let cache = compute_residuals(&inc, tmp.path(), &corpus, 3, 2).unwrap();
        assert_eq!(cache.residual, vec![0.5, -1.0, 0.0, -0.5]);
        assert_eq!(cache.correction, cache.residual);
        let s = &cache.meta.stats[0];
        assert_eq!(s.count, 4);
        assert!((s.mean + 0.25).abs() < 1e-9);
        assert!((s.sse - 1.5).abs() < 1e-9);
        assert!((cache.meta.local_mse - 0.375).abs() < 1e-9);
        assert_eq!(s.top_records[0], (1, -1.0));
        assert_eq!(s.max, 0.5);
        assert_eq!(s.min, -1.0);
    }

    #[test]
    fn logistic_output_yields_pre_squash_correction() {
        let tmp = tempfile::tempdir().unwrap();
        let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..20)
            .map(|i| {
                (
                    vec![i as f32 / 10.0 - 1.0, 0.3],
                    vec![if i % 2 == 0 { 0.8 } else { 0.2 }],
                )
            })
            .collect();
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let inc = Incumbent::from_creature(small_mlp(2), "t").unwrap();
        let cfg = TrainingDataConfig::new(2, 1);
        let corpus = corpus_info(tmp.path(), &cfg).unwrap();
        let cache = compute_residuals(&inc, tmp.path(), &corpus, 7, 1).unwrap();
        assert_eq!(cache.meta.output_squashes, ["LOGISTIC"]);
        let mut net = compile_creature(&inc.creature).unwrap();
        for (i, (x, t)) in recs.iter().enumerate() {
            let traced = net.activate_and_trace(x, 1);
            let non_input = net.num_neurons() - net.num_inputs();
            let hint = traced[1 + non_input + non_input - 1];
            let pred = traced[0];
            assert!((cache.residual_at(i, 0) - (t[0] - pred)).abs() < 1e-6);
            // Applying the correction pre-squash reproduces the target.
            let fixed = neat_core::squash::apply_squash(
                SquashType::Logistic,
                hint + cache.correction_at(i, 0),
            );
            assert!((fixed - t[0]).abs() < 1e-4, "{fixed} vs {}", t[0]);
            assert_ne!(cache.residual_at(i, 0), cache.correction_at(i, 0));
        }
    }

    #[test]
    fn if_output_uses_output_space_as_correction_space() {
        let tmp = tempfile::tempdir().unwrap();
        let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..12)
            .map(|i| (vec![i as f32 - 6.0, 0.5, 0.25], vec![i as f32 * 0.1]))
            .collect();
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let inc =
            Incumbent::from_creature(crate::graft::fixtures::if_output_creature(3), "t").unwrap();
        let cfg = TrainingDataConfig::new(3, 1);
        let corpus = corpus_info(tmp.path(), &cfg).unwrap();
        let cache = compute_residuals(&inc, tmp.path(), &corpus, 5, 2).unwrap();
        assert_eq!(cache.meta.output_squashes, ["IF"]);
        assert_eq!(cache.residual, cache.correction);
        assert!(cache.correction.iter().any(|&c| c != 0.0));
        assert!(cache.meta.stats[0].correction_mse > 0.0);
    }

    #[test]
    fn sidecar_round_trips_and_invalidates() {
        let tmp = tempfile::tempdir().unwrap();
        let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..9)
            .map(|i| (vec![i as f32], vec![i as f32 * 0.5]))
            .collect();
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let inc = Incumbent::from_creature(identity_creature(1, 1), "t").unwrap();
        let cfg = TrainingDataConfig::new(1, 1);
        let corpus = corpus_info(tmp.path(), &cfg).unwrap();
        let cache_dir = tmp.path().join("cache");
        let a = ensure_residual_cache(&inc, tmp.path(), &cache_dir, &corpus, 4, 1).unwrap();
        let b = ensure_residual_cache(&inc, tmp.path(), &cache_dir, &corpus, 4, 1).unwrap();
        assert_eq!(a, b);
        let other = Incumbent::from_creature(identity_creature(1, 1), "other").unwrap();
        let mut c2 = other.creature.clone();
        c2.neurons[0].bias = 0.1;
        let other = Incumbent::from_creature(c2, "other").unwrap();
        assert!(matches!(
            a.check_compatible(&other, &corpus),
            Err(ResidualError::Stale(_))
        ));
        write_bin_file(&tmp.path().join("1.bin"), &[(vec![1.0], vec![1.0])]).unwrap();
        let corpus2 = corpus_info(tmp.path(), &cfg).unwrap();
        assert!(matches!(
            a.check_compatible(&inc, &corpus2),
            Err(ResidualError::Stale(_))
        ));
        let path = cache_path(&cache_dir, &inc);
        let bytes = std::fs::read(&path).unwrap();
        assert!(matches!(
            ResidualCache::read(&{
                std::fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
                path.clone()
            }),
            Err(ResidualError::Corrupt(_))
        ));
    }
}

//! Training-corpus identity and bounded-memory streaming.
//!
//! The NEAT-AI corpus is a directory of headerless little-endian `f32` `.bin`
//! files; each record is `input + output` values. Nothing here loads the whole
//! corpus into RAM: [`for_each_chunk`] streams fixed-size record chunks.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use neat_core::training_data::{TrainingDataConfig, find_bin_files};
use serde::{Deserialize, Serialize};

/// Deterministic identity of a corpus directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusInfo {
    /// 64-bit FNV-style mix over widths, file names, sizes and head/tail bytes.
    pub identity: String,
    /// Total records across all `.bin` files.
    pub record_count: u64,
    /// Number of `.bin` files.
    pub file_count: usize,
    /// Input width used to interpret records.
    pub input_count: usize,
    /// Output width used to interpret records.
    pub output_count: usize,
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

fn mix(state: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *state ^= u64::from(*b);
        *state = state.wrapping_mul(FNV_PRIME);
    }
}

/// Compute the corpus identity and record count without reading every byte.
///
/// Identity covers the widths, each file name, length, and its first/last 64
/// bytes, matching the NEAT-AI-Lamarck convention so caches built by either
/// tool agree on "same corpus".
pub fn corpus_info(dir: &Path, config: &TrainingDataConfig) -> Result<CorpusInfo, String> {
    if !dir.is_dir() {
        return Err(format!(
            "training data path '{}' is not a directory",
            dir.display()
        ));
    }
    let files = find_bin_files(dir).map_err(|e| format!("cannot list '{}': {e}", dir.display()))?;
    if files.is_empty() {
        return Err(format!(
            "no .bin files found in training data directory '{}'",
            dir.display()
        ));
    }
    let record_bytes = config.bytes_per_record() as u64;
    let mut state = FNV_OFFSET;
    mix(&mut state, &(config.num_inputs as u64).to_le_bytes());
    mix(&mut state, &(config.num_outputs as u64).to_le_bytes());
    let mut records = 0u64;
    for path in &files {
        let mut file =
            File::open(path).map_err(|e| format!("cannot open '{}': {e}", path.display()))?;
        let len = file.metadata().map_err(|e| e.to_string())?.len();
        if len % record_bytes != 0 {
            return Err(format!(
                "'{}' is {len} bytes, not a multiple of the {record_bytes}-byte record size",
                path.display()
            ));
        }
        records += len / record_bytes;
        mix(&mut state, &len.to_le_bytes());
        if let Some(name) = path.file_name() {
            mix(&mut state, name.to_string_lossy().as_bytes());
        }
        let mut head = [0u8; 64];
        let n = read_up_to(&mut file, &mut head)?;
        mix(&mut state, &head[..n]);
        if len > 64 {
            file.seek(SeekFrom::End(-64)).map_err(|e| e.to_string())?;
            let mut tail = [0u8; 64];
            let n = read_up_to(&mut file, &mut tail)?;
            mix(&mut state, &tail[..n]);
        }
    }
    Ok(CorpusInfo {
        identity: format!("{state:016x}"),
        record_count: records,
        file_count: files.len(),
        input_count: config.num_inputs,
        output_count: config.num_outputs,
    })
}

fn read_up_to(file: &mut File, buf: &mut [u8]) -> Result<usize, String> {
    let mut filled = 0;
    while filled < buf.len() {
        let n = file.read(&mut buf[filled..]).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    Ok(filled)
}

/// A chunk of consecutive records handed to [`for_each_chunk`] callbacks.
#[derive(Debug)]
pub struct RecordChunk<'a> {
    /// Global index of the first record in this chunk.
    pub first_index: u64,
    /// Number of records in the chunk.
    pub records: usize,
    /// Row-major inputs, `records * input_count` values.
    pub inputs: &'a [f32],
    /// Row-major expected outputs, `records * output_count` values.
    pub targets: &'a [f32],
}

/// Stream the corpus in chunks of at most `chunk_records` records.
///
/// Memory is bounded by one chunk. The callback may return `Err` to abort.
pub fn for_each_chunk<F>(
    dir: &Path,
    config: &TrainingDataConfig,
    chunk_records: usize,
    mut on_chunk: F,
) -> Result<u64, String>
where
    F: FnMut(RecordChunk<'_>) -> Result<(), String>,
{
    let files = find_bin_files(dir).map_err(|e| format!("cannot list '{}': {e}", dir.display()))?;
    let values = config.values_per_record();
    let record_bytes = config.bytes_per_record();
    let chunk_records = chunk_records.max(1);
    let mut raw = vec![0u8; chunk_records * record_bytes];
    let mut inputs = vec![0f32; chunk_records * config.num_inputs];
    let mut targets = vec![0f32; chunk_records * config.num_outputs];
    let mut next_index = 0u64;
    for path in &files {
        let mut file =
            File::open(path).map_err(|e| format!("cannot open '{}': {e}", path.display()))?;
        let len = file.metadata().map_err(|e| e.to_string())?.len();
        if len % record_bytes as u64 != 0 {
            return Err(format!(
                "'{}' is {len} bytes, not a multiple of the {record_bytes}-byte record size",
                path.display()
            ));
        }
        let mut remaining = (len / record_bytes as u64) as usize;
        while remaining > 0 {
            let take = remaining.min(chunk_records);
            let bytes = &mut raw[..take * record_bytes];
            file.read_exact(bytes)
                .map_err(|e| format!("read '{}': {e}", path.display()))?;
            for r in 0..take {
                let base = r * values;
                for i in 0..config.num_inputs {
                    let o = (base + i) * 4;
                    inputs[r * config.num_inputs + i] =
                        f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
                }
                for j in 0..config.num_outputs {
                    let o = (base + config.num_inputs + j) * 4;
                    targets[r * config.num_outputs + j] =
                        f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
                }
            }
            on_chunk(RecordChunk {
                first_index: next_index,
                records: take,
                inputs: &inputs[..take * config.num_inputs],
                targets: &targets[..take * config.num_outputs],
            })?;
            next_index += take as u64;
            remaining -= take;
        }
    }
    Ok(next_index)
}

/// Write records as a `.bin` file (test/fixture helper; also used by benches).
pub fn write_bin_file(path: &Path, records: &[(Vec<f32>, Vec<f32>)]) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = std::io::BufWriter::new(File::create(path)?);
    for (inputs, outputs) in records {
        for v in inputs.iter().chain(outputs.iter()) {
            out.write_all(&v.to_le_bytes())?;
        }
    }
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus(dir: &Path, n: usize) {
        let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..n)
            .map(|i| (vec![i as f32, -(i as f32)], vec![i as f32 * 2.0]))
            .collect();
        write_bin_file(&dir.join("0.bin"), &recs[..n / 2]).unwrap();
        write_bin_file(&dir.join("1.bin"), &recs[n / 2..]).unwrap();
    }

    #[test]
    fn identity_is_stable_and_changes_with_content() {
        let tmp = tempfile::tempdir().unwrap();
        corpus(tmp.path(), 10);
        let cfg = TrainingDataConfig::new(2, 1);
        let a = corpus_info(tmp.path(), &cfg).unwrap();
        let b = corpus_info(tmp.path(), &cfg).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.record_count, 10);
        assert_eq!(a.file_count, 2);
        write_bin_file(&tmp.path().join("1.bin"), &[(vec![9.0, 9.0], vec![9.0])]).unwrap();
        let c = corpus_info(tmp.path(), &cfg).unwrap();
        assert_ne!(a.identity, c.identity);
    }

    #[test]
    fn chunks_cover_every_record_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        corpus(tmp.path(), 11);
        let cfg = TrainingDataConfig::new(2, 1);
        let mut seen = Vec::new();
        let total = for_each_chunk(tmp.path(), &cfg, 3, |c| {
            assert_eq!(c.first_index as usize, seen.len());
            for r in 0..c.records {
                assert_eq!(c.inputs[r * 2], -c.inputs[r * 2 + 1]);
                assert_eq!(c.targets[r], c.inputs[r * 2] * 2.0);
                seen.push(c.inputs[r * 2]);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(total, 11);
        assert_eq!(seen, (0..11).map(|i| i as f32).collect::<Vec<_>>());
    }

    #[test]
    fn misaligned_file_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("0.bin"), [0u8; 7]).unwrap();
        let cfg = TrainingDataConfig::new(1, 1);
        assert!(
            corpus_info(tmp.path(), &cfg)
                .unwrap_err()
                .contains("multiple")
        );
    }
}

//! Production-shaped histogram search benchmark (Issues #6, #12, #15).
//!
//! Builds a synthetic search set with the Enceladus width (2461 features by
//! default) and times: CPU exhaustive accumulation, GPU accumulation (when the
//! `gpu` feature and an adapter are available), stump ranking, and sampled
//! variants (row / feature fractions). Prints one JSON object so the numbers
//! can be pasted into `docs/benchmarks.md`.
//!
//! ```text
//! cargo run --release --example stump_search_bench [--features gpu] -- [records] [features] [threads]
//! ```

use std::time::Instant;

use neat_ai_forests::gpu;
use neat_ai_forests::histogram::{
    BinnedChunk, HistogramSet, MemorySource, SearchControls, search_stumps,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let records: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let features: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2461);
    let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let bins = 256usize;
    let chunk = 4096usize;
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let started = Instant::now();
    let mut chunks = Vec::new();
    let mut done = 0;
    while done < records {
        let n = chunk.min(records - done);
        let mut b = Vec::with_capacity(n * features);
        let mut r = Vec::with_capacity(n);
        for _ in 0..n {
            for _ in 0..features {
                b.push((next() % bins as u32) as u8);
            }
            let signal = if b[b.len() - features + 7] > 200 {
                0.05
            } else {
                0.0
            };
            r.push((next() as f32 / u32::MAX as f32 - 0.5) * 0.2 + signal);
        }
        chunks.push(BinnedChunk {
            records: n,
            features,
            bins: b,
            residual: r,
            weight: None,
            first_index: done as u64,
        });
        done += n;
    }
    let build_ms = started.elapsed().as_millis();
    let src = MemorySource {
        chunks,
        label: "bench".into(),
    };
    let bins_per_feature = vec![bins; features];
    let controls = SearchControls {
        min_leaf_records: 50.0,
        top_k: 16,
        max_per_feature: 2,
        ..Default::default()
    };
    let thresholds = |_: usize, b: usize| Some(b as f32);

    let t = Instant::now();
    let cpu = HistogramSet::from_source(&src, &bins_per_feature).unwrap();
    let cpu_ms = t.elapsed().as_millis();
    let t = Instant::now();
    let top = search_stumps(&cpu, &thresholds, &controls, "cpu");
    let rank_ms = t.elapsed().as_millis();
    let t = Instant::now();
    let cpu_par = HistogramSet::from_source_threads(&src, &bins_per_feature, threads).unwrap();
    let cpu_par_ms = t.elapsed().as_millis();
    let par_agrees = search_stumps(&cpu_par, &thresholds, &controls, "cpu")
        .iter()
        .zip(&top)
        .all(|(a, b)| a.feature == b.feature && a.bin == b.bin);

    let (gpu_ms, gpu_backend, gpu_agrees) = match gpu::GpuAccumulator::new() {
        Ok(g) => {
            let _warm = g.accumulate(&src, &bins_per_feature);
            let t = Instant::now();
            let set = g.accumulate(&src, &bins_per_feature).unwrap();
            let ms = t.elapsed().as_millis();
            let top_gpu = search_stumps(&set, &thresholds, &controls, "gpu");
            let agrees = top.iter().zip(&top_gpu).all(|(a, b)| {
                a.feature == b.feature
                    && a.bin == b.bin
                    && (a.gain - b.gain).abs() <= 1e-4 * a.gain.abs().max(1.0)
            });
            (Some(ms), g.adapter_name().to_string(), Some(agrees))
        }
        Err(e) => (None, format!("unavailable: {e}"), None),
    };

    // Sampled variants: row fraction 0.25, feature fraction 0.25.
    let quarter = MemorySource {
        chunks: src
            .chunks
            .iter()
            .take(src.chunks.len().div_ceil(4))
            .cloned()
            .collect(),
        label: "rows/4".into(),
    };
    let t = Instant::now();
    let _ = HistogramSet::from_source(&quarter, &bins_per_feature).unwrap();
    let rows_quarter_ms = t.elapsed().as_millis();
    let fq = features / 4;
    let feat_quarter = MemorySource {
        chunks: src
            .chunks
            .iter()
            .map(|c| {
                let mut b = Vec::with_capacity(c.records * fq);
                for r in 0..c.records {
                    b.extend_from_slice(&c.bins[r * features..r * features + fq]);
                }
                BinnedChunk {
                    records: c.records,
                    features: fq,
                    bins: b,
                    residual: c.residual.clone(),
                    weight: None,
                    first_index: c.first_index,
                }
            })
            .collect(),
        label: "features/4".into(),
    };
    let t = Instant::now();
    let _ = HistogramSet::from_source(&feat_quarter, &bins_per_feature[..fq]).unwrap();
    let feat_quarter_ms = t.elapsed().as_millis();

    let cells = records as f64 * features as f64;
    let report = serde_json::json!({
        "records": records,
        "features": features,
        "bins": bins,
        "searchSetBytes": records * features,
        "buildMs": build_ms,
        "cpu": {"accumulateMs": cpu_ms, "rankMs": rank_ms, "cellsPerSecond": cells / (cpu_ms.max(1) as f64 / 1000.0), "recordsPerSecond": records as f64 / (cpu_ms.max(1) as f64 / 1000.0)},
        "cpuThreads": {"threads": threads, "accumulateMs": cpu_par_ms, "speedupVsSingle": cpu_ms as f64 / cpu_par_ms.max(1) as f64, "agreesWithSingleTop16": par_agrees},
        "gpu": {"backend": gpu_backend, "accumulateMs": gpu_ms, "agreesWithCpuTop16": gpu_agrees, "speedupVsSingleCpu": gpu_ms.map(|g| cpu_ms as f64 / g.max(1) as f64), "speedupVsThreadedCpu": gpu_ms.map(|g| cpu_par_ms as f64 / g.max(1) as f64)},
        "sampled": {"rowsQuarterMs": rows_quarter_ms, "featuresQuarterMs": feat_quarter_ms},
        "topStump": top.first().map(|s| serde_json::json!({"feature": s.feature, "bin": s.bin, "gain": s.gain, "kind": s.kind})),
        "plantedFeature": 7,
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

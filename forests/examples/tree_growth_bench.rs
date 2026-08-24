//! Time tree growth on a production-shaped search set (Issue #69).
//!
//! cargo run --release --example tree_growth_bench -- [records] [features] [roots]
use neat_ai_forests::bins::quantile_edges;
use neat_ai_forests::config::GrowthPolicy;
use neat_ai_forests::histogram::{BinnedChunk, MemorySource, SearchControls};
use neat_ai_forests::tree::{TreeSearchControls, grow_tree};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let records: usize = a.get(1).map_or(200_000, |s| s.parse().unwrap());
    let features: usize = a.get(2).map_or(300, |s| s.parse().unwrap());
    let roots: usize = a.get(3).map_or(8, |s| s.parse().unwrap());
    let mut seed = 42u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut values = vec![0f32; records * features];
    for v in values.iter_mut() {
        *v = next() * 2.0 - 1.0;
    }
    // Every root the bench uses must be a feature the residual actually
    // depends on, or the tree stops at depth 1 and the run measures nothing.
    let signal = (roots + 2).min(features);
    let mut residual = vec![0f32; records];
    for (r, res) in residual.iter_mut().enumerate() {
        let row = &values[r * features..(r + 1) * features];
        let mut v = next() * 0.05;
        for (k, x) in row.iter().take(signal).enumerate() {
            if *x > 0.0 {
                v += 0.6 / (k + 1) as f32;
            }
        }
        *res = v;
    }
    let mut bins = vec![0u8; records * features];
    let mut edges = Vec::with_capacity(features);
    for f in 0..features {
        let mut col: Vec<f32> = (0..records).map(|r| values[r * features + f]).collect();
        edges.push(quantile_edges(&mut col, 64));
    }
    for r in 0..records {
        for f in 0..features {
            let v = values[r * features + f];
            bins[r * features + f] = edges[f].partition_point(|&e| e <= v) as u8;
        }
    }
    let per_feature: Vec<usize> = edges.iter().map(|e| e.len() + 1).collect();
    let src = MemorySource {
        chunks: bins
            .chunks(8192 * features)
            .enumerate()
            .map(|(i, b)| BinnedChunk {
                records: b.len() / features,
                features,
                bins: b.to_vec(),
                residual: residual[i * 8192..(i * 8192 + b.len() / features)].to_vec(),
                weight: None,
                first_index: (i * 8192) as u64,
            })
            .collect(),
        label: "bench".into(),
    };
    let controls = TreeSearchControls {
        stump: SearchControls {
            min_leaf_records: 50.0,
            top_k: usize::MAX,
            ..Default::default()
        },
        max_depth: 3,
        growth: GrowthPolicy::BestFirst,
    };
    let started = Instant::now();
    let mut grown = 0;
    for root in std::iter::once(None).chain((0..roots).map(|f| Some((f, 31)))) {
        grown += grow_tree(
            &src,
            &per_feature,
            &|f, b| edges[f].get(b).copied(),
            &|f| f,
            &controls,
            root,
        )
        .unwrap()
        .len();
    }
    println!(
        "{{\"records\":{records},\"features\":{features},\"roots\":{roots},\"trees\":{grown},\"ms\":{}}}",
        started.elapsed().as_millis()
    );
}

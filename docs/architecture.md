# Architecture

Forests is one Rust crate (`forests/`, package `neat_ai_forests`) organised as
a pipeline. The search side may be approximate; the acceptance side is a single
function that only trusts the full-corpus NEAT-AI-scorer.

```mermaid
flowchart LR
    subgraph immutable["immutable inputs"]
        C[creature.json]
        T[training/*.bin]
    end
    C --> I[incumbent<br/>checksum + workspace copy]
    T --> K[bins<br/>quantile-bin cache]
    I --> R[residuals<br/>output + correction space]
    T --> R
    I --> B[baseline<br/>scorer full + parity gate]
    R --> S[strategies<br/>search set: rows / features / weights]
    K --> S
    S --> H[histogram / gpu<br/>per-feature sufficient statistics]
    H --> ST[stumps]
    H --> TR[tree depth 2-3]
    S --> OB[oblique raw-sample search]
    ST --> CA[candidates<br/>variants + random controls]
    TR --> CA
    OB --> CA
    CA --> G[graft<br/>patch → IF on a clone]
    G --> P[promote<br/>screen → full scorer]
    B --> P
    P -->|accepted| W[best.json + winners/]
    P --> J[journal → report]
    W -->|new incumbent| R

    classDef immutable fill:#dbeafe,stroke:#1d4ed8,color:#0b2545
    classDef search fill:#fef3c7,stroke:#b45309,color:#451a03
    classDef judge fill:#dcfce7,stroke:#15803d,color:#052e16
    class C,T,I immutable
    class K,R,S,H,ST,TR,OB,CA,G search
    class B,P,W judge
```

## Modules

| Module | Responsibility | Trusts |
|---|---|---|
| `incumbent` | load, validate (compile + round-trip + trailing outputs), SHA-256, byte-exact workspace copy | NEAT-AI-core |
| `corpus` | corpus identity (FNV mix over widths/file names/sizes/head+tail bytes, Lamarck-compatible) and bounded-memory chunk streaming | — |
| `bins` | quantile-bin cache: feature-block streaming passes, stride sampling, tie collapse, binary file with JSON header | — |
| `residuals` | per-record residual in output space and in pre-squash **correction space** via `neat_core::unsquash`; stats, tails, largest-residual records; sidecar keyed by checksum × corpus | NEAT-AI-core kernels |
| `baseline` | full-corpus scorer call on the incumbent alone; local-MSE parity gate (abort / skip) | **scorer** |
| `strategies` | search-set construction: stride / uniform / stratified / residual-weighted rows, all / random / error-ranked features, raw-value re-read for oblique | — |
| `histogram` | CPU reference: `count / Σr / Σr²` per bin, prefix scans, left-only / right-only / two-leaf gains, deterministic top-K, brute-force oracle | — |
| `gpu` | wgpu/WGSL accumulation with per-invocation private partials (no atomics), CPU fold in slice order | — |
| `tree` | level-wise / best-first growth on path-masked histograms, depth ≤ 3 | — |
| `oblique` | 2–3 feature linear conditions on a raw sample; projection sort + coordinate jitter | — |
| `patch` | portable `Leaf` / `Split(Condition{terms, threshold})` tree, `f32` evaluator mirroring the `IF` kernel, provenance, content id | — |
| `graft` | patch → `constant` + `IF` neurons + typed synapses appended before the outputs on a clone; NEAT-AI-core compile + structural validation | NEAT-AI-core |
| `candidates` | analytical optimum, one-sided variants, magnitude scales, threshold jitter, random stumps; dedup + cap + graft | — |
| `promote` | screen (scorer sample mode) → full-corpus cohort with baseline; strict threshold; same-call baseline drift veto; FP/FN bookkeeping | **scorer** |
| `run` | the loop, outputs, journal, promotion of winners, residual recomputation | — |
| `journal`, `report` | append-only JSONL; economics aggregation | — |
| `xgboost`, `tools` | matrix export, dump → patch conversion with exact `>=` mapping, control run through `promote` | **scorer** |

## Determinism

- A run's seed is drawn (or supplied) and written to the `runHeader`; per-iteration seeds derive from it. Sampling, random stumps, bypass selection and oblique combos all use the seeded RNG.
- Histogram accumulation is order-independent in `f64` on the CPU; on the GPU every partial is a sequential loop and the CPU fold is in slice order, so results are reproducible run to run.
- Ranking is always on the CPU with the tie-break `gain desc, feature, bin, kind`.
- Candidate ids are content hashes of `(output, root)`; the same incumbent, caches and seed yield the same ids (tested).

## Memory

- The corpus is never loaded whole: streaming chunks of `--chunk-records` records.
- The bin cache is built in feature blocks sized by `--bin-memory-budget-mib`.
- The residual sidecar is `records × outputs × 2 × f32` (≈18 MB for 2.26 M records × 1 output).
- The in-memory search set is `--search-records × features` bytes of `u8` bins (≈490 MB at 200 k × 2461); `--search-records 0` uses every record (≈5.5 GB at production scale — prefer sampling, or reduce `--feature-fraction`).

# Benchmarks and economics (Issues #6, #12, #15)

The metric that matters is **scorer-verified improvement per wall-clock hour**.
Everything below is either that, or an input to it. Negative results are
results.

## Representative hardware baseline

Host: Apple M4 (10 cores), 24 GiB unified memory, macOS, `rustc` per
`rust-toolchain.toml`, release profile (`opt-level=3`, `lto=fat`).
Command: `scripts/run-benchmark.sh 200000 2461 8` → `stump_search_bench`
(synthetic search set at the Enceladus width: 200 000 records × 2461 features
× 256 bins, planted residual region on feature 7).

| Stage | Time | Throughput | Notes |
|---|---|---|---|
| build synthetic `u8` search set (490 MB) | 0.50 s | — | not part of a run; the real set is built by one corpus stream |
| CPU histogram accumulation, 1 thread | 1.27–1.42 s | 360–390 M cells/s, ≈150 k records/s | `HistogramSet::from_source` |
| CPU histogram accumulation, 4 threads (feature split) | 0.44 s | 1.1 G cells/s | bit-identical to 1 thread |
| CPU histogram accumulation, 8 threads (feature split) | **0.35 s** | **1.4 G cells/s**, 4.1× | bit-identical to 1 thread |
| CPU accumulation, 10 threads | 0.37 s | — | no further gain past the performance cores |
| stump ranking (2461 × 255 thresholds × 3 kinds, top-16) | 0.07 s | — | CPU, deterministic tie-break |
| GPU accumulation, Metal (Apple M4), incl. upload | 2.09–2.16 s | — | top-16 agrees with CPU; **0.65× single-thread CPU, 0.17× 8-thread CPU** |
| rows/4 sample (50 k records) | 0.34 s (1 thread) | — | linear in rows |
| features/4 sample (615 features) | 0.32 s (1 thread) | — | linear in features |

### Verdicts

- **GPU histogram accumulation is not worthwhile on this host.** The one-shot
  upload of the `u8` matrix (490 MB) plus partial read-back dominates, and the
  kernel itself cannot beat 8 CPU threads writing into cache-resident
  histograms. `--gpu off` is therefore the default; the kernel remains behind
  the `gpu` feature for discrete-GPU hosts where the upload can be amortised
  across many accumulations (trees re-accumulate per level; that reuse is the
  first optimisation to try if the GPU is ever revisited).
- **Chunk-parallel CPU accumulation was 2× slower than single-threaded**
  (8 private 15 MB histogram sets thrash the cache). Splitting by feature range
  instead keeps each worker's histograms in L2 and is bit-identical to the
  sequential result. Kept.
- **Search is not the bottleneck.** At production width a full histogram pass
  over a 200 k-record search set costs ≈ 0.35 s + 0.07 s ranking. A single
  full-corpus `rust_scorer` call over 2.26 M records × a 13 k-synapse creature
  is measured in tens of seconds per cohort. Expect `searchTimeFraction` in
  the report to be well under 10 %; the economics are set by how many
  candidates each authoritative call can carry and how well the screen picks
  them.
- **Sampling strategies scale linearly** with rows and features as expected;
  at these speeds their value is diversity (more distinct cohorts per hour),
  not raw search time.

## Production evidence (2026-08-21)

Run on the production champion (2511 inputs, 1 `IF` output, 1761 neurons,
22 928 synapses, authoritative score 0.353158958) against the current 100 %
corpus (2 266 178 records, 21 GB) on the M4, CPU only, default flags
(`--combo-candidates 4`), `--analysis-threads 8`, 45-minute budget. Paths and
data are private and are not referenced from this repository.

**A first run with the pre-fix graft layout claimed +1.875e-3 under
`rust_scorer` but re-scored at −1.2e-5 under NEAT-AI's TypeScript
`Creature.scoreDir`**: the layout repeated `(from, to)` synapse pairs, which
`Creature.fromJSON` silently drops (159 of 23 246 synapses) while
`rust_scorer` sums them. The numbers below are from the fixed layout and were
verified by **both** engines.

| Metric | Value |
|---|---|
| iterations / acceptances | 24 / 23 |
| opening → final authoritative score | 0.353158958 → 0.355655238 (**Δ +2.50e-3**) |
| TypeScript `ConfirmScore.ts` re-score | ✅ verified 0.3556552382939058; `validate()` passes; 23 229 synapses before and after `fromJSON` |
| improvement per wall-clock hour | ≈3.2e-3 |
| winners by strategy | 10 single stumps (3 magnitude-scale variants), 13 combinations (`combo/2` 6, `combo/3` 2, `combo/4` 5) |
| per-iteration gain | 1–3e-4 while combinations win, tapering to ≈2e-5 by iteration 16 |
| wall split | search ≈15 %, screen ≈20 %, full scorer ≈65 % |
| concentration | 21 of 23 winners touch every record; two touch 0.8 % |
| structure added | 23 patches → +105 neurons, +301 synapses (4 neurons / 5 synapses per stump, +1 relay neuron / +1 synapse into the `IF` output) |

Context: recent Lamarck sampler check-ins on the same creature family are
≈1e-5 per 45-minute run. The creature was checked into the GRQ sampler
population after the TypeScript verification.

Follow-ups: the screen (#17) — on the first-generation run the 5 % sample
screen rejected half of the exploratory bypasses that later cleared the full
threshold; a 15-minute sweep showed neither a 20 % sample nor a 16-wide
promotion improved Δ per hour (both shift time into the scorer). Depth-2
trees and oblique splits remain unmeasured on the champion.

## Production run protocol

No production corpus is checked in or available on the development machine,
so the scorer-verified numbers must be produced on the training host:

```bash
SCORER=../NEAT-AI-scorer/target/release/rust_scorer
C=path/to/champion.json
D=path/to/training-bin-dir

# A. exhaustive stumps, CPU, 45 minutes (the first milestone question)
neat_ai_forests $C $D --scorer $SCORER --scorer-arg=--gpu=off --output-dir runs/stumps --seed 1

# B. sampled dirty tricks, same seed
neat_ai_forests $C $D --scorer $SCORER --scorer-arg=--gpu=off --output-dir runs/sampled --seed 1 \
  --row-sampling residual-weighted --feature-selection error-ranked --feature-fraction 0.25 \
  --threshold-jitter 2 --random-candidates 16

# C. depth-2/3 trees
neat_ai_forests $C $D --scorer $SCORER --scorer-arg=--gpu=off --output-dir runs/trees --seed 1 --max-depth 3

# D. oblique splits
neat_ai_forests $C $D --scorer $SCORER --scorer-arg=--gpu=off --output-dir runs/oblique --seed 1 --oblique-candidates 8

# E. random-only control (no histogram guidance)
neat_ai_forests $C $D --scorer $SCORER --scorer-arg=--gpu=off --output-dir runs/random --seed 1 --top-k 0 --random-candidates 48

# F. XGBoost control — see docs/xgboost-control.md

scripts/report-experiments.sh runs/*/experiments.jsonl
```

Record for each run: opening/final authoritative score, acceptances,
`improvementPerWallHour`, `timeToFirstAcceptanceMs`, `searchTimeFraction`,
screen FP/FN rates, and the `affectedFraction` of every accepted gain (a
winner that touches 0.1 % of records is suspicious and should be inspected
with the journal's patch before anyone trusts it). Compare with a Lamarck run
of the same budget on the same champion rather than assuming either dominates.

## What to do with the answer

- **Stumps win repeatedly:** keep going; measure saturation via
  `scoreTrajectory`; try depth 2.
- **Stumps never win but XGBoost's converted trees do:** the native search is
  missing something — compare thresholds/features in the two journals.
- **Nothing wins economically:** that is the documented negative result. Do not
  add complexity; record it here and stop.

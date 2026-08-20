# Search strategies ("dirty tricks", Issue #12)

Every strategy changes only **what is searched** and identifies itself in the
candidate's provenance (`strategy`, `backend`, `notes`). None can accept a
candidate; the full-corpus scorer gate in `promote` is unconditional.

| Flag | Strategy | Provenance |
|---|---|---|
| `--search-records N` | in-memory search sample (0 = whole corpus) | `searchRecords`, search set label `memory-sample/N` or `memory-full` |
| `--row-sampling stride` | deterministic every-k-th record | `row-sampling=stride/k` |
| `--row-sampling uniform` | seeded Bernoulli subset | `row-sampling=uniform p=…` |
| `--row-sampling stratified` | four \|residual\| quartile strata, equal sample each, importance weight = population / sample | `row-sampling=stratified …` |
| `--row-sampling residual-weighted` | inclusion ∝ \|residual\|, importance weight 1/p | `row-sampling=residual-weighted …` |
| `--feature-selection random --feature-fraction f` | seeded feature subset | `feature-selection=random k/n` |
| `--feature-selection error-ranked` | top features by between-bin variance of mean residual (a correlation-ratio proxy) | `feature-selection=error-ranked k/n` |
| `--magnitude-scales 1,0.5,1.5,-1` | leaf scales around the analytical optimum (negative = opposite) | strategy suffix `/scale`, note `scale=…` |
| one-sided variants (always) | zero one leaf of a two-leaf stump so most records stay untouched | strategy suffix `/one-sided` |
| `--threshold-jitter j` | neighbouring bin edges ±1..j around each top stump | strategy `threshold-jitter` |
| `--random-candidates n` | uniformly random feature/edge/kind, magnitude ~ residual σ | strategy `random-stump`, backend `random` |
| `--max-per-feature m` | diversity cap in the top-K | — |
| `--explore-quota q` | screen rejects fully scored anyway to measure false negatives | `bypass: true` in the journal |
| `--max-depth 2 or 3`, `--growth …` | trees (Issue #11) | `histogram-tree-depthD`, note `growth=…` |
| `--oblique-candidates n` | 2–3 feature linear conditions (Issue #14) | `oblique-split`, note `origin=axis+1`, `random-sparse` or `jitter` |

Weights from stratified / residual-weighted sampling flow into the histogram
statistics (`Σw`, `Σw·r`, `Σw·r²`), so gains estimate the full-corpus SSE
reduction rather than the sample's.

## Measuring them

`neat_ai_forests report` aggregates per strategy: generated, promoted, fully
scored, winners, accepted gain and mean full Δ; plus screen false-positive and
false-negative rates, time to first winner and improvement per wall-clock hour.
Run the same creature/corpus with different flags (same `--seed`), then
`scripts/report-experiments.sh run-a/experiments.jsonl run-b/experiments.jsonl`.

`examples/stump_search_bench.rs` compares exhaustive accumulation against
row-quarter and feature-quarter samples at production width (see
[benchmarks.md](benchmarks.md)).

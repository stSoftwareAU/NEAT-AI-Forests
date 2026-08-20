# Caches

Both caches live in `--cache-dir` (default: the training directory) and are
validated before reuse. A stale or corrupt cache is reported and rebuilt; it is
never silently reused. Any other error (permissions, I/O) aborts.

## `forests-bins.cache` — quantile bins (Issue #3)

```text
magic       4 bytes   b"NFBN"
format      u32 LE    1
json_len    u32 LE
json        json_len bytes  — BinMeta (camelCase)
per feature (inputCount times):
  edge_count u32 LE
  edges      edge_count × f32 LE, strictly ascending
```

`BinMeta` fields: `formatVersion`, `algorithmVersion`, `inputCount`,
`outputCount`, `recordCount`, `corpusIdentity`, `requestedBins`,
`effectiveBins[]` (edges + 1 per feature), `nonFiniteCounts[]`,
`nonFinitePolicy`, `sampleRecords`, `sampleStride`, `createdAtUnix`.

**Algorithm (v1).** Records are sampled with a deterministic stride
`ceil(records / --bin-sample-records)`; features are processed in blocks that
fit `--bin-memory-budget-mib`, one streaming pass per block. Per feature the
finite sample is sorted and edges are taken at `floor(b·n/bins)` for
`b = 1..bins`; duplicate edge values collapse (tie policy), `-0.0` folds into
`0.0`, and an edge equal to the maximum is dropped so the top bin is reachable.

**Mapping.** `bin(x) = |{e : x > e}|` (a `partition_point`), so a split
"after bin *b*" is exactly the `IF` condition `x > e_b`. `NaN → bin 0`
(left of every threshold) because the `IF` kernel sends a `NaN` condition sum
to the negative branch; `±∞` order naturally.

**Compatibility** requires identical `algorithmVersion`, `corpusIdentity`,
record count, widths and `requestedBins`.

**Reader.** `neat_ai_forests::bins::BinCache::from_bytes` parses and validates
(magic, trailing bytes, ascending edges, `effectiveBins` agreement). The
`u8` bin index and contiguous `f32` edge arrays are what the GPU kernel uploads.

## `forests-residuals-<checksum12>.cache` — residual sidecar (Issue #4)

```text
magic       4 bytes   b"NFRS"
format      u32 LE    1
json_len    u32 LE
json        ResidualMeta (camelCase)
residual    records × outputs × f32 LE      target − prediction
correction  records × outputs × f32 LE      unsquash(target, hint) − hint
```

`ResidualMeta`: `incumbentChecksum`, `corpusIdentity`, `recordCount`,
`outputCount`, `outputSquashes[]`, `stats[]` (count, mean, variance, MAE, SSE,
MSE, correction MSE, min/max, |r| quantiles p50/p90/p99/p999, 2σ/3σ tail
counts, top-1 % SSE share, 32 largest-|r| record indices, non-finite count),
`localMse` (exactly NEAT-AI-core's per-record MSE, the parity proxy),
`hasSampleWeights` (always `false`: the `.bin` format carries none),
`createdAtUnix`.

**Sign convention.** `residual = target − prediction`; a positive residual
means the incumbent is too low and a positive correction helps.

**Correction space.** A graft adds to the output neuron's pre-squash sum, so
the search target is `unsquash(target, hint) − hint` using NEAT-AI-core's own
`apply_unsquash`. For `IDENTITY` outputs it equals the output-space residual.

**Invalidation.** Keyed by incumbent checksum × corpus identity; an accepted
winner has a new checksum, so its residuals are recomputed by construction.

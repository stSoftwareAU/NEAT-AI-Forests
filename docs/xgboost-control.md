# XGBoost external control (Issue #13)

XGBoost is a **scientific control**, not a dependency. The question is whether
conventional gradient-boosted trees find residual structure that the native
histogram search misses, and whether such trees survive conversion into
ordinary `IF` grafts and the authoritative scorer.

## Protocol

```bash
# 1. export the incumbent's correction-space residuals (deterministic stride sample)
neat_ai_forests creature.json training/ export-matrix --out matrix.csv --max-records 200000
#    → matrix.csv (f0..fN,residual,correction) + matrix.meta.json (checksums, stride)

# 2. train shallow trees with base_score=0 (pure additive correction)
pip install xgboost pandas
scripts/xgboost-control.py matrix.csv --depth 1 --rounds 8 --out dump.json
#    → dump.json (booster.get_dump(dump_format="json")) + dump.json.meta.json (params, train RMSE, seed)

# 3. convert every tree to a patch, graft, screen, full-score — same path as native candidates
neat_ai_forests creature.json training/ import-xgboost --dump dump.json \
  --scorer ../NEAT-AI-scorer/target/release/rust_scorer --output-dir runs/xgb
#    → runs/xgb/experiments.jsonl (strategy "xgboost-import"), best.json only if the scorer agrees
```

## Conversion rules

- `x < split_condition → yes` becomes `x > prev_f32(split_condition) → right`, which is exactly `x >= split_condition`; `yes → left`, `no → right` with no boundary divergence.
- Leaf values are used as emitted (already scaled by `eta`).
- A node whose `missing` child is not its `yes` child is rejected with the reason recorded (the `IF` kernel routes `NaN` left); `--allow-missing-divergence` accepts it and notes the divergence in provenance.
- Features must be named `fN` (the export header), and `N` must be below the creature's input width.
- Single-leaf trees are rejected (a split-free graft is a no-op).

## Reading the result

Compare `runs/xgb/experiments.jsonl` with a native run on the same incumbent
using `scripts/report-experiments.sh`: search time (the `.meta.json` files
carry training seconds), proxy gain, and — the only number that matters —
authoritative scorer Δ. Record in [benchmarks.md](benchmarks.md) which XGBoost
ideas were adopted natively (histograms, min child weight, column/row
subsampling, shallow depth are already in) and which were rejected.

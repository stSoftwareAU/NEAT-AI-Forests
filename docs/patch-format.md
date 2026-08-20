# Patch format and `IF` graft

## Patch JSON (version 1)

```json
{
  "version": 1,
  "output": 0,
  "root": {
    "kind": "split",
    "condition": { "terms": [{ "feature": 317, "weight": 1.0 }], "threshold": 0.283 },
    "left":  { "kind": "leaf", "correction": 0.0 },
    "right": { "kind": "leaf", "correction": 0.013 }
  },
  "provenance": {
    "strategy": "histogram-stump",
    "backend": "gpu:Apple M4 (Metal)",
    "predictedGain": 12.7,
    "affectedRecords": 4021,
    "searchRecords": 200000,
    "incumbentChecksum": "…sha256…",
    "seed": 42,
    "notes": ["kind=RightOnly bin=201", "row-sampling=stride/12"]
  }
}
```

- A `Condition` with one unit-weight term is axis-aligned; 2–3 weighted terms
  form an oblique split (Issue #14).
- The patch **id** is the first 16 hex chars of SHA-256 over `(output, root)`;
  provenance does not affect identity, so the same discovery found by two
  strategies de-duplicates.
- Every candidate's patch is stored in the journal, so a creature that is
  ordinary NEAT-AI JSON can still be traced back to what produced it.

## Evaluation semantics

The abstract evaluator mirrors the NEAT-AI-core `IF` kernel exactly:

```text
sum = 0 (f32)
for term in terms:      sum += x[term.feature] * term.weight
sum += 1.0 * (-threshold)
right branch  ⇔  sum > 0.0
```

`NaN` never satisfies `> 0`, so it always takes the left branch — in the
evaluator and in the creature. Thresholds are bin edges (values that occur in
the data) carried as `f32`, so `x − t` is computed exactly enough that the
strict comparison agrees between the evaluator and the compiled network.

## Graft layout

For a patch with id `P` the following is appended to a **clone** of the
incumbent, inserted immediately before the first output neuron so listed order
stays topological:

```text
forest-P-one            type "constant", bias 1.0, no squash, no inward synapses
forest-P-ifN            type "hidden", squash "IF", bias 0   (children emitted before parents)
    condition:  input-f  weight w_f   (one per term)
    condition:  forest-P-one  weight -threshold
    positive:   right child   (child IF, weight 1.0  |  forest-P-one, weight right_leaf)
    negative:   left  child   (child IF, weight 1.0  |  forest-P-one, weight left_leaf)
root IF ──(weight 1.0, untyped)──▶ output-j
```

When the target output neuron is itself an **`IF` aggregate** (as the
production champion's is), an untyped synapse would feed only its positive
branch, so the root is wired in twice — once `positive`, once `negative` —
and the correction reaches every record. Other aggregate outputs
(`MINIMUM`/`MAXIMUM`/`MEAN`/`HYPOT`) are not additive in a new synapse and the
graft is refused.

Because `IF` applies no squash and `condition_sum > 0 ? positive_sum + bias :
negative_sum + bias`, the root's activation is exactly the patch's correction,
and it enters the output neuron's pre-squash sum with weight 1. A leaf of
`0.0` therefore leaves that region's behaviour unchanged (up to one ulp of
SIMD summation order inside NEAT-AI-core, which the scorer sees identically).

Every graft must (a) compile through `neat_core::compile_creature` and
(b) pass `neat_core::topology_ops::validate_structural_integrity` (≥ 3 inward
synapses per `IF`, one each of condition / positive / negative, constants with
no inward links). Anything else fails closed before the scorer is involved.

Pre-existing neurons and synapses are never edited, reordered or removed.

## Cost of a graft under the scorer

NEAT-AI-scorer charges `growthCost × (hidden + synapses/10 + …)` and an extra
`3 × growthCost / 100` per `IF` neuron (≈ `3e-9`). A stump adds 2 neurons and
5 synapses, so its complexity penalty is ≈ `2.5e-7`; an accepted patch must
beat that *and* `--min-improvement` on the authoritative score.

## XGBoost mapping

XGBoost routes `x < split_condition` to `yes`. With
`threshold = prev_f32(split_condition)` the creature routes
`x > threshold ⇔ x >= split_condition` to the right, so `yes → left` and
`no → right` exactly. Nodes whose `missing` child is not the `yes` child are
rejected unless `--allow-missing-divergence` is given.

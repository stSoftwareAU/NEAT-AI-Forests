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

NEAT-AI's TypeScript loader keys synapses by `(from, to)` and silently
collapses duplicates; `rust_scorer` does not. A graft must therefore never
repeat a pair, or the two judges disagree (this bit the first production run:
three synapses from one constant into each `IF` made `rust_scorer` report
gains the fleet's TypeScript re-score could not see). Layout per patch `P`;
new hidden neurons are inserted before the first output neuron so listed order
stays topological, and new constants ahead of the first hidden neuron:

```text
shared:     one_c / one_p / one_n    bias-1 constants, one per synapse role, reused from the
                             creature where it has them, else created once (`forest-one-a/b/c`);
                             a threshold or leaf is the synapse WEIGHT from one of them (#43)
per split:  forest-P-ifN     hidden IF, bias 0
                condition:   input-f (weight w_f, one per term)  and  one_c (weight = −threshold)
                             → Σ w·x − threshold > 0 ⇔ right
                positive:    right child ifN (weight 1)  |  one_p (weight = right leaf)
                negative:    left  child ifN (weight 1)  |  one_n (weight = left leaf)
root ifN ──(weight 1, untyped)──▶ output-j                       point-wise output squash
root ifN ──(positive)──▶ output-j,  root ifN → relayN (IDENTITY) ──(negative)──▶ output-j    IF output
```

That is NEAT-AI-core's own canonical shape (`neat_core::decision_tree`,
NEAT-AI-core #555): every node is described as a `neat_core::IfNodeSpec` and
every shape is built by NEAT-AI-core (#42, #48). The post-order batch goes to
`neat_core::graft_if_nodes` in one all-or-nothing graft — a child carries no
outward edge of its own, the parent that reads it supplies one — and where the
target output is itself an `IF` aggregate the root's outward edge carries the
`positive` role and `neat_core::graft_relay_node` adds the IDENTITY relay that
carries the same correction into the `negative` branch.

The duplicate-pair rule — NEAT-AI-core's `validate_no_duplicate_synapses`
(#556), wrapped as `check_no_duplicate_synapses` — is part of the validation
gate every grafted creature clears (Issue #50), and a condition naming the same
feature twice is refused. The three shared bias-1 constants are what keep the
three synapse roles of one `IF` node reading three different neurons; a
creature that already carries the names `forest-one-a/b/c` on neurons the graft
must not repurpose gets fresh names (`forest-one-a2`, …) rather than a refused
graft. Other aggregate outputs
(`MINIMUM`/`MAXIMUM`/`MEAN`/`HYPOT`) are not additive in a new synapse and the
graft is refused.

Because `IF` applies no squash and `condition_sum > 0 ? positive_sum + bias :
negative_sum + bias`, the root's activation is exactly the patch's correction,
and it enters the output neuron's pre-squash sum with weight 1. A leaf of
`0.0` therefore leaves that region's behaviour unchanged (up to one ulp of
SIMD summation order inside NEAT-AI-core, which the scorer sees identically).

Every graft must (a) compile through `neat_core::compile_creature`,
(b) pass `neat_core::topology_ops::validate_structural_integrity` (≥ 3 inward
synapses per `IF`, one each of condition / positive / negative, constants with
no inward links), (c) contain no repeated `(from, to)` pair, and (d) satisfy
`neat_core::creature_validate` — the shared definition of a valid creature
(#39, see [architecture.md](architecture.md#creature-validation)). Anything
else fails closed before the scorer is involved. Parity between `rust_scorer`
and the TypeScript `Creature.scoreDir` was checked by hand on stump / depth-2 /
oblique fixtures (agreement to 1e-7); see Forests issue on automating it.

Pre-existing neurons and synapses are never edited or removed, and their
relative order is preserved. Their *position* in the list can move: the result
is emitted in NEAT-AI's canonical order (#39), so new constants are listed
ahead of the first hidden neuron and the whole synapse list is sorted ascending
by `(from index, to index)` — the two ordering rules `creature_validate`
enforces.

## Cost of a graft under the scorer

NEAT-AI-scorer charges `growthCost × (hidden + synapses/10 + …)` and an extra
`3 × growthCost / 100` per `IF` neuron (≈ `3e-9`). A stump adds 1 neuron and
5 synapses (2 neurons / 7 synapses into an `IF` output; plus the three shared constants once per creature), so its complexity penalty is ≈ `5e-7`; an accepted patch must
beat that *and* `--min-improvement` on the authoritative score.

## XGBoost mapping

XGBoost routes `x < split_condition` to `yes`. With
`threshold = prev_f32(split_condition)` the creature routes
`x > threshold ⇔ x >= split_condition` to the right, so `yes → left` and
`no → right` exactly. Nodes whose `missing` child is not the `yes` child are
rejected unless `--allow-missing-divergence` is given.

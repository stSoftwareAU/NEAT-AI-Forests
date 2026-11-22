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
| `--refine-leaves n`, `--refine-max-depth d` | split `n` leaves already on the creature, read back from its structure, on the rows that reach each one (Issue #140) | `refine-stump` / `refine-tree-depthD`, notes `refine=<parent uuid>:<role> leaf-depth=…` and `leaf-records=…`; the patch carries `parent` |
| `--refit-every n`, `--refit-records`, `--refit-steps`, `--refit-ridges` | refit every leaf on the creature jointly (Issue #137) | strategy `refit`, backend `cpu-cg`, notes `ridge=… step=… leaves=… moved=… clamped-rows=… corpus=…`; the journalled patch is a placeholder bare leaf |

### Refitting every leaf at once (Issue #137)

With every split fixed, the output's correction-space sum is linear in the
leaf values and each record selects one leaf per tree, so the least-squares
step for **all** leaves is a sparse problem with a closed form. `refit` reads
the trees back from structure (the same `refine::ForestView`), so leaves other
tools changed and patches from other hosts are included at their current
values, and solves

```text
(AᵀWA + ρ·diag(AᵀWA)) δ = AᵀW r,   A_il = m_i · g_l · [record i reaches leaf l]
```

by Jacobi-preconditioned conjugate gradient on a stride sample of the search
set (`--refit-records`, default 20,000 rows held as a compressed row list).
`g_l` is the leaf's gain to the output, so the solve works in correction space
with the anchor's gain (#58), and `m_i` is zero where an output clamp binds —
found by probing the anchor with a small bias step and checking the output
follows at the walk's gain — because no leaf can move those records. The
ridge is relative to each leaf's own curvature, so `δ = 0` (the incumbent) is
the limit of a large ridge and the knobs are scale-free. Every `(ridge, step)`
pair is one candidate; a leaf moves by changing the weight of one of its
constant edges, so a refit changes weights and never structure.

A refit runs on each output's first iteration whenever the incumbent's
`forests-refit-corpus` tag names a different corpus (or is missing), and then
every `--refit-every` iterations. An accepted refit writes the tag. Candidates
are judged by the full-corpus scorer like everything else; they are never
carried forward, filed in the learnings cache or replayed, and
`--refit-every` is refused with `--enhancements`: a refit is not a patch
NEAT-AI-Rebase could carry, and it is cheap enough to re-run on the moved
champion instead.

## Refining existing leaves (Issue #140)

Forests otherwise only plants new trees at the anchor, at most three
conditions deep, so the creature grows wider and never deeper. With
`--refine-leaves n` each iteration reads every tree hanging off the output's
anchor back from the creature's **structure** (`refine::ForestView`) — so
patches from every host and every earlier run count, at the weights they
carry now, whether they reach an `IF` anchor as a typed pair or through the
`--if-correction relay` IDENTITY neuron — and:

1. routes every search-set record through every readable tree, keeping
   per-leaf `Σw`, `Σw·r`, `Σw·r²`;
2. picks the `n` leaves with enough records for two new leaves, room under
   `--refine-max-depth`, and the most residual spread a constant cannot fit
   (`Σw·r² − (Σw·r)²/Σw`);
3. runs the ordinary stump search, and the tree search where depth allows, on
   the rows that reach each chosen leaf.

The refinement is additive: the leaf keeps its constant edge and a child `IF`
is added to the same branch sum of the parent, its outward edge carrying
`1 / gain` so its leaves stay in output space. Zero corrections leave every
record bit-identical. Both passes stream the corpus for the search set's raw
rows, so the cost is about two oblique-search reads per iteration.

The minimum record count per new leaf is `--min-leaf-records`, counted on the
sample; the full-corpus scorer then judges the candidate as it judges every
other. A parent that is gone, rewired or unreadable is refused at graft time
(`GraftError::NoSuchParent`).

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

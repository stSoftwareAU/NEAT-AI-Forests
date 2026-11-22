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
| `--linear-leaf-conditions R` | split + intercept and slope on a second input per side (Issue #138) | `linear-leaf`, `linear-leaf/half-slope`, `/right-slope`, `/left-slope`, `/shared-intercept`; notes `condition-origin=stump\|creature\|median\|zero slope-input=k`, `constant-gain=…`; leaves carry `slopes` |
| `--hidden-features n` | the creature's own hidden neurons as extra split columns (Issue #139) | strategy suffix `+hidden`, note `hidden-sources=<uuid,…>`; search-set note `hidden-features=k/eligible`; the term carries `neuron` |
| `--refit-every n`, `--refit-records`, `--refit-steps`, `--refit-ridges` | refit every leaf on the creature jointly (Issue #137) | strategy `refit`, backend `cpu-cg`, notes `ridge=… step=… leaves=… moved=… clamped-rows=… corpus=…`; the journalled patch is a placeholder bare leaf |

## Linear leaves (Issue #138)

Every other leaf is a constant. A linear leaf also carries a slope on a second
input, so one split can say "the same signal helps in one regime and hurts in
the other":

```text
IF  condition:  x_j − t
    positive:   a_R · 1  +  b_R · x_k
    negative:   a_L · 1  +  b_L · x_k
```

Averaged over both regimes such a signal has almost no correlation with the
residual, and a split on the regime alone with constant leaves has almost no
gain, so the constant-leaf search never finds it. With
`--linear-leaf-conditions R` each iteration builds a pool of `R` conditions —
a third from the iteration's best ordinary stumps, a third from axis
conditions already on the creature, the rest seeded median and zero splits on
random features, so the pool is never ranked by constant-leaf gain alone. One
masked histogram pass per condition gives, per side and per bin of every
feature `k`, `Σw` and `Σw·r`; with the bin's midpoint `v_b` that is `Σw·x`,
`Σw·x²` and `Σw·r·x`, and so the closed-form weighted least-squares intercept,
slope and SSE gain of every `(condition, k)` pair. Each condition keeps its best
`k` if it beats constant leaves on the same condition; the best `R` pairs become
discoveries, and the top two also get half-slope, one-sided-slope and
shared-intercept variants. Magnitude variants scale slopes and intercepts
together.

A slope is one extra synapse from `input-k` into the branch, a tenth of a
neuron in the scorer's growth cost. A sign flip puts one input in both branch
roles, which NEAT-AI-core keys by `(from, to, type)` and allows for an `IF`
target (#68). Slopes read inputs only, never hidden columns. A refinement
(#140) and a refit (#137) treat a branch with a slope as not a constant leaf
and leave it alone. A linear leaf never seeds a boosting bundle, since the
residual update works in bin space and cannot apply a slope. Slopes are not
capped by `--max-correction` (intercepts are); a steep slope is judged by the
full-corpus scorer like any other candidate. NEAT-AI-Rebase
cannot replay a slope yet, so the flag is refused with `--enhancements`.

## Splitting on hidden neurons (Issue #139)

With `--hidden-features n` each iteration traces the incumbent on the search
set and appends up to `n` of its hidden neurons as extra binned columns, after
the inputs. Every search — stumps, trees, refinements — then sees them like
any input, so a depth-3 tree can split on a neuron that already combines
dozens of inputs.

Eligible neurons are hidden neurons listed before the output's anchor, so the
graft stays forward-only; the anchor and everything it feeds, the output
included, would make a graft that feeds itself and are refused at graft time
(`GraftError::DownstreamSource`). Forests' own `forest-…` neurons are skipped,
and so is any neuron that is constant on the sample (no quantile edge); a
two-valued neuron such as a STEP has one edge and is offered. The rest are
ranked by the `error-ranked` proxy — between-bin variance of the mean residual
— on a 4,096-row stride sample, which also places their quantile edges; then
every row is traced again and the top `n` are binned. Memory is one `u8` per
row per column: 200,000 rows × 1,500 columns is about 300 MB.

A discovery that splits on one is written into the patch by UUID
(`{"neuron": "<uuid>", "feature": 0, "weight": 1}`), as format v2, with
`+hidden` appended to its strategy so `report` keeps hidden-source economics
apart. The graft emits the condition edge from that neuron. A patch that names
a neuron which no longer exists is refused at graft time
(`GraftError::NoSuchSource`), so a replay onto another creature fails cleanly.
Oblique search re-reads raw inputs and skips hidden columns. NEAT-AI-Rebase
cannot replay a hidden-source term yet, so `--hidden-features` is refused with
`--enhancements`.

## Refitting every leaf at once (Issue #137)

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
every `--refit-every` iterations on that output — the cadence is counted per
output, so on a multi-output creature each output's anchor gets its turn. An
accepted refit writes the tag. Candidates
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

## Random arms (Issue #147)

The four newest families (`--refine-leaves`, `--refit-every`,
`--hidden-features`, `--linear-leaf-conditions`) are **arms**: a flag left
unset is drawn on or off with probability ½ per run, from the run seed with a
fixed salt, so `--seed` reproduces the draw. Drawn on, each takes its
production-protocol value (`docs/benchmarks.md` lines G–J): refine 8, refit
100000 (the opening refit only), hidden 256, linear 16.

- An explicit value, `0` included, is always used as given.
- Under `--enhancements` every unset arm is forced off: none of them is
  something NEAT-AI-Rebase can replay yet.
- `--no-random-arms` leaves every unset arm off (today's defaults).

Every creature the run publishes carries the creature-level tag
`forests-arms`, beside `score`, `forests` and `forests-detail`:

```text
hidden-features=256:drawn,linear-leaf-conditions=0:set,refine-leaves=8:drawn,refit-every=0:forced
```

Each arm is `flag=value:source`, sorted by flag name; the source is `drawn`,
`set` (explicit), `forced` (off under `--enhancements`) or `default` (unset
under `--no-random-arms`). The tag is written only on a creature the run
changed — on acceptance — so a run that accepts nothing leaves an inherited
tag untouched and a tag always describes the run that produced the creature.
The journal's run header carries the same string (`arms`), and its `config` is
the resolved one.

`scripts/report-experiments.sh` adds a per-arm table over every journal that
has arms: for each family, runs drawn on against runs drawn off, with
acceptances, Δ per wall hour, Δ per full scorer call and how many accepted
patches the family itself produced. Only `drawn` arms count. One run proves
nothing: the champion and corpus move between runs, so expect a few dozen per
arm before reading a verdict.

## Measuring them

`neat_ai_forests report` aggregates per strategy: generated, promoted, fully
scored, winners, accepted gain and mean full Δ; plus screen false-positive and
false-negative rates, time to first winner and improvement per wall-clock hour.
Run the same creature/corpus with different flags (same `--seed`), then
`scripts/report-experiments.sh run-a/experiments.jsonl run-b/experiments.jsonl`.

`examples/stump_search_bench.rs` compares exhaustive accumulation against
row-quarter and feature-quarter samples at production width (see
[benchmarks.md](benchmarks.md)).

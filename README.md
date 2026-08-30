# NEAT-AI-Forests

![NEAT-AI-Forests](https://raw.githubusercontent.com/stSoftwareAU/NEAT-AI/Develop/docs/brand/social-previews/neat-ai-forests.png)

An experimental Rust optimiser for already-fit
[NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) creatures. It reads a mature
creature and its training corpus, measures the errors the creature still makes,
searches for tree-shaped corrections to them, grafts the promising ones onto
**copies** of the creature as ordinary NEAT-AI `IF` structure, and lets the
[NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) decide which — if
any — is genuinely fitter.

> **Find all the dirty tricks that uncover real improvements — but trust only the scorer.**

Candidate generation may be adventurous. **Acceptance is deliberately boring.**

## What it does

```text
fittest creature ─▶ residuals ─▶ split search ─▶ patches ─▶ grafted clones ─▶ scorer ─▶ best.json
```

- **Reads, never writes, the creature you give it.** Every candidate is a clone.
- **Only adds structure.** A patch becomes an `IF` subtree feeding the output it
  corrects; nothing evolved is deleted, simplified or rewired.
- **Never accepts its own opinion.** Histogram gain, sampled scores and every
  other proxy rank candidates; only a full-corpus NEAT-AI-scorer result accepts
  one.
- **Writes down what it tried.** Every candidate, proxy, score and verdict lands
  in `experiments.jsonl`, and `neat_ai_forests report` turns that into the
  economics of each strategy.
- **Shares what it learns.** With `--learnings-dir`, a fleet replays each
  other's verified wins and stops paying twice for the same failure.

It is a genuine experiment, and the question it asks is narrow:

> **Can a fundamentally different search process discover small improvements that
> normal evolution is now finding only slowly?**

So far, yes — see [docs/benchmarks.md](docs/benchmarks.md) for what has been
measured, including the strategies that turned out not to be worth their
runtime.

NEAT-AI-Forests is **not** an attempt to replace that creature, retrain it from
scratch, or redesign the working system. The motivating creature has been
evolved for years to predict **90-day stock movement** and is useful in
production.

---

## Status

Every phase in the [issue list](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues)
has landed as code: immutable incumbent and scorer-parity gate, quantile-bin and
residual caches, CPU stump search,
portable patches grafted as ordinary `IF` structure, two-phase screening with
authoritative promotion, the 45-minute evolution loop and journal, depth-2/3
trees, sampling and random "dirty tricks", oblique splits, the XGBoost external
control, and the shared learnings cache. The phase table and the outstanding
work are at the bottom of this file.

One shared-family prerequisite is still open upstream:

- [NEAT-AI-scorer #574](https://github.com/stSoftwareAU/NEAT-AI-scorer/issues/574) — CPU/GPU parity for `IF`-heavy creatures. Until it lands, production runs should pass `--scorer-arg=--gpu=off` or watch the `baselineDrift` field in the journal.

---

## Quick start

Forests is a Rust workspace that depends on the sibling
[NEAT-AI-core](https://github.com/stSoftwareAU/NEAT-AI-core) checkout and
invokes the [NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer)
binary (`rust_scorer`) as the judge:

```text
parent/
├── NEAT-AI-core/      # path dependency: ../../NEAT-AI-core/neat-core
├── NEAT-AI-scorer/    # build it: cargo build --release  →  target/release/rust_scorer
└── NEAT-AI-Forests/
```

```bash
cargo build --release                       # CPU build

./target/release/neat_ai_forests creature.json training/ \
  --scorer ../NEAT-AI-scorer/target/release/rust_scorer \
  --output-dir runs/first --timeout-seconds 2700

./target/release/neat_ai_forests report runs/first/experiments.jsonl
```

The source `creature.json` is never written to. `best.json` starts as a
byte-for-byte copy and is only replaced by a creature the scorer verified on
the full corpus in the same call as its parent.

### Command line

```text
neat_ai_forests <creature.json> <training-data-dir> [OPTIONS]
neat_ai_forests report <experiments.jsonl>
neat_ai_forests <creature.json> <training-data-dir> export-matrix  [--out CSV] [--max-records N] [--output J]
neat_ai_forests <creature.json> <training-data-dir> import-xgboost --dump dump.json [--output J] [--allow-missing-divergence]
neat_ai_forests prune-learnings --dir <learnings-dir> [--corpus ID] [--host H] [--dry-run]
```

`prune-learnings` keeps the shared cache (#61) from growing without bound, and
is safe to run from cron on an idle host. It only ever touches the file this
host writes, so it needs no coordination with the rest of the fleet, and
dropping an old rejection is the point rather than a side effect: it puts that
experiment back on the table.

| Flag | Default | Meaning |
|---|---|---|
| `--dir` | — | the shared learnings cache root, as passed to `--learnings-dir` |
| `--corpus` | every corpus found | which corpus directory to prune |
| `--host` | hostname | whose file to prune; only ever this one |
| `--rejected-after-hours` | `720` | drop rejections older than this — must exceed `--learnings-retry-after-hours`, or a rejection is dropped before it is ever retried |
| `--accepted-after-hours` | `4320` | drop acceptances older than this; far longer, since wins are what the cache is for |
| `--max-records` | `0` | cap on the records a host keeps, newest first (0 = uncapped) |
| `--dry-run` | off | report what would go without writing anything |

| Flag | Default | Meaning |
|---|---|---|
| `--output-dir` | `.` | `best.json`, `experiments.jsonl`, `winners/`, `workspace/` |
| `--cache-dir` | training dir | where `forests-bins.cache` / `forests-residuals-*.cache` live |
| `--scorer` | `rust_scorer` | NEAT-AI-scorer binary |
| `--scorer-arg` | — | extra scorer argument, repeatable (e.g. `--scorer-arg=--gpu=off`) |
| `--timeout-seconds` | `2700` | 45-minute wall-clock budget |
| `--max-iterations` | — | stop after N iterations |
| `--seed` | drawn | RNG seed; printed for replay |
| `--min-improvement` | `1e-6` | strict authoritative Δscore required to accept |
| `--bins` | `256` | quantile bins per observation |
| `--bin-sample-records` | `65536` | records sampled per feature for quantiles |
| `--bin-memory-budget-mib` | `256` | memory per bin-cache pass |
| `--chunk-records` | `4096` | records per streaming chunk |
| `--analysis-threads` | `4` | threads for residual extraction and CPU histogram accumulation |
| `--search-records` | `200000` | in-memory search sample (0 = whole corpus) |
| `--row-sampling` | `stride` | `stride`, `uniform`, `stratified`, `residual-weighted` |
| `--feature-selection` | `all` | `all`, `random`, `error-ranked` |
| `--feature-fraction` | `0.25` | fraction kept under random / error-ranked |
| `--min-leaf-records` | `50` | minimum records in a corrected leaf |
| `--max-correction` | `1` | clamp on leaf corrections (pre-squash units) |
| `--min-gain` | `0` | minimum proxy gain reported |
| `--stump-kinds` | all three | `left-only`, `right-only`, `two-leaf` |
| `--top-k` | `16` | stumps kept per search |
| `--max-per-feature` | `2` | diversity cap per feature (0 = off) |
| `--max-depth` | `3` | tree depth 1–3 |
| `--growth` | `best-first` | `level-wise` or `best-first` |
| `--tree-roots` | `8` | distinct stump features grown into trees each iteration, on top of the unconstrained best-first tree; trees are the most valuable candidates per scorer call |
| `--magnitude-scales` | `1.0,0.5,0.25` | leaf scales around the analytical optimum |
| `--threshold-jitter` | `0` | neighbouring bins tried per top stump |
| `--random-candidates` | `4` | deliberately random stumps per iteration |
| `--oblique-candidates` | `0` | oblique 2–3 feature splits per iteration |
| `--boost-rounds` | `1` | boosting rounds on the sample: subtract the best patch, search again; bundle prefixes verified in one scorer call |
| `--combo-candidates` | `4` | combinations: top-2…top-N distinct discoveries stacked on one clone, plus last iteration's near-winners carried forward |
| `--candidates` | `64` | maximum grafted candidates per iteration |
| `--graft-constants` | `shared` | who owns a graft's three bias-1 constants: `shared` (one set per creature) or `per-patch` (`forest-<patch id>-one-c/p/n`, three extra constant neurons per patch, blast radius of one patch) |
| `--if-correction` | `typed-pair` | how a correction reaches both branches of an `IF` anchor: `typed-pair` (one source feeding both roles — a neuron cheaper, and what every engine agrees on from @stsoftware/neat-ai 6.6.40 and a `rust_scorer` built against neat-core 0.10.6) or `relay` (an IDENTITY neuron per graft, for creatures that must load under an older engine) |
| `--screen-sample-rate` | `0.05` | scorer sample rate for the screen; 0 disables; skipped automatically when the cohort already fits `--promote-count` |
| `--screen-threshold` | `0` | sampled Δ required to promote |
| `--promote-count` | `8` | candidates promoted to full scoring |
| `--explore-quota` | `1` | screen rejects fully scored to measure false negatives |
| `--baseline-drift-epsilon` | `1e-6` | tolerated same-call vs stored baseline disagreement |
| `--skip-parity` | off | skip the local-MSE vs scorer parity gate (non-MSE costs) |
| `--parity-abs` / `--parity-rel` | `1e-7` / `1e-4` | parity tolerances |
| `--preserve-candidates` | off | keep per-iteration cohort directories |
| `--max-consecutive-scorer-failures` | `3` | abort after this many failures in a row |
| `--enhancements` | off | file every accepted patch as `enhancements.json` beside `best.json`, for [population re-entry through Rebase](#population-re-entry) |
| `--learnings-dir` | off | shared cache of full-corpus verdicts (#60); point every host at one git checkout and the fleet replays each other's wins |
| `--learnings-host` | hostname | the file this machine appends to, so no two hosts ever conflict |
| `--learnings-replay` | `8` | cached candidates replayed per iteration (0 = write only) |
| `--learnings-retry-after-hours` | `168` | how long a candidate that only ever failed is left alone before it is offered again |

`--help` and `--version` are the usual clap extras. The scorer's own
`--sample-rate`, `--sample-phase`, `--gpu` and `--cost` flags are passed by
Forests, not by you.

---

## How a run works

1. **Incumbent** — load through NEAT-AI-core, refuse anything that does not compile or round-trip, checksum it (SHA-256), copy it byte-for-byte into `workspace/incumbent.json` with `incumbent.meta.json`.
2. **Bin cache** — stream the corpus once per feature block and persist ~256 equal-population quantile edges per observation (`forests-bins.cache`, reused only for the identical corpus/version/bin count).
3. **Residuals** — run the incumbent over every record; store `target − prediction` and the pre-squash *correction-space* residual in a sidecar keyed by incumbent checksum × corpus identity.
4. **Baseline** — score the incumbent alone with the full-corpus scorer; compare its `error` with the local MSE (parity gate, fail closed); journal the result.
5. **Search** — build the quantised search set (sampled rows/features as configured), accumulate per-feature histograms, rank stumps; optionally grow depth-2/3 trees and oblique splits.
6. **Candidates** — expand discoveries into patches (analytical optimum, one-sided variants, magnitude scales, threshold jitter, random controls), graft each onto a clone, discard anything NEAT-AI-core rejects — including anything `neat_core::creature_validate` rules invalid, with the reason journalled (see [docs/architecture.md](docs/architecture.md#creature-validation)).
7. **Screen** — the scorer's record-sampling mode ranks the cohort; the top `--promote-count` plus an exploratory bypass quota go on.
8. **Promote** — full-corpus scorer call with the baseline in the same cohort; accept only `Δscore > --min-improvement` and only if the same-call baseline matches the stored one.
9. **Repeat** — a winner becomes the experimental incumbent, `best.json` and `winners/winner-NNNN.json` are written, residuals are recomputed, and the loop continues until the budget ends.

---

## Outputs

| Path | Content |
|---|---|
| `best.json` | best scorer-verified creature (pretty JSON — see [what a published creature keeps](#what-a-published-creature-keeps)) |
| `experiments.jsonl` | append-only journal, one JSON object per line with a `record` discriminator |
| `winners/winner-NNNN.json` | every accepted intermediate |
| `enhancements.json` | `--enhancements` only, and only when the run accepted something: the accepted patches as a Rebase v1 bundle (see [population re-entry](#population-re-entry)) |
| `workspace/incumbent.json`, `incumbent.meta.json`, `baseline.json` | immutable copy, checksum metadata, authoritative baseline record |
| `<cache-dir>/forests-bins.cache` | quantile-bin cache (see [docs/caches.md](docs/caches.md)) |
| `<cache-dir>/forests-residuals-<checksum>.cache` | residual sidecar per incumbent |

### Population re-entry

A run opens on the fleet's champion `A` and finishes up to 45 minutes later.
By then the fleet has usually moved `A` on to `B`, and publishing this run's
own descendant — `A` plus its patches — quietly deletes whatever `B` gained.
`--enhancements` is the way out: the run files the **patches** it accepted
rather than the creature it reached, and
[NEAT-AI-Rebase](https://github.com/stSoftwareAU/NEAT-AI-Rebase) grafts them
onto a freshly fetched champion, where the scorer decides again.

```mermaid
sequenceDiagram
    autonumber
    participant F as Forests run
    participant L as PatchLog
    participant P as Population
    participant R as Rebase
    F->>P: fetch champion → A
    F->>L: opening(A, baseline score, corpus identity)
    Note over P: the fleet evolves A → B independently
    F->>L: accept(winner patches, authoritative score) — at each acceptance
    F->>L: write_bundle(enhancements.json) — beside best.json
    F->>P: fetch champion again → B
    F->>R: rebase enhancements.json onto champion B
    R-->>F: a candidate, only when B + Δ beats B
```

The switch changes how a run *publishes*, never what it searches for or
accepts: `run::tests::with_enhancements_off_nothing_is_written_and_the_run_is_unchanged`
runs the loop both ways and asserts the same candidates, the same acceptances
and the same final creature. The bundle is written once, at the end, and only
when the run accepted something — no file means there is nothing to rebase.

Everything filed is the patch **as accepted**: the bytes are not rebuilt or
rounded, so the patch id stays the id the graft named its `forest-<id>-…`
structure with, and a champion that already carries the patch is recognised as
carrying it rather than grafted twice. Combination winners are filed as their
members, in the order the winner applies them, appending only what the run has
not already filed.

### What a published creature keeps

A creature is only worth checking in if it still carries the provenance it
arrived with — a better score does not buy the right to delete the discovery
and intelligent-design history of neurons Forests never touched. Every creature
Forests writes (`best.json`, `winners/winner-NNNN.json`) therefore holds to
this, which is also what the fleet's check-in guard verifies before committing:

| Metadata | What happens to it |
|---|---|
| creature-level `tags` | preserved in their original order; `score`, `error`, `forests` and `forests-detail` upserted |
| per-neuron `tags` | preserved by neuron uuid, for every neuron the source carried |
| neurons the graft appended | tagged `forests` / `forests-patch` with the run, iteration, patch and verified Δscore that created them |
| creature-level `uuid` | dropped — NEAT-AI derives it from content, and the content changed |
| `memetic` | dropped — it describes fine-tuning of the structure the graft has just altered, and its keys resolve by runtime neuron id, which the constants a graft inserts shift |

`run::tests::published_creature_keeps_source_provenance_and_drops_stale_identity`
asserts all five on a creature a whole run published, not on the serialiser
alone.

### The check-in message

The publishing tool reads two tags off the creature: `forests` is the commit
**subject** and `forests-detail` is the commit **body** (#98). The subject is
deliberately short — one score and one signed delta in Rust's scientific
rendering, the format the rest of the fleet is standardising on — so a wall of
check-ins stays readable:

```text
🌳 Forests · score: 0.407038 (+1.27e-4)

7 accepts / 9 iters · last: histogram-tree-depth3/scale · 🎯 output-0
```

The delta is measured against the score the run opened on, so `+0.00e0` is a
run that accepted nothing and a negative delta is reported rather than hidden.

### Journal records

- `runHeader` — timestamp, seed + source, version, incumbent checksum, corpus identity, bin-cache identity, the complete effective configuration.
- `baseline` — authoritative score/error, scorer identity, cost name, parity verdict (written at start and after every acceptance).
- `experiment` — per iteration: search backend/set/records/features/strategies, timings, every candidate's patch + provenance, its **screen** score and its **full** score recorded separately, promotion/bypass flags, winner, improvement, acceptance, new incumbent checksum, screen false-positive/false-negative counts, scorer error or baseline veto, and `discarded` — every rejected candidate with the verbatim graft/validation reason.
- `summary` — stop reason, iterations, acceptances, opening/final score, wall time.

`neat_ai_forests report` folds the journal into the economics metrics: cumulative improvement, **improvement per wall-clock hour**, time to first acceptance, candidates per minute, search vs scorer time, screen false-positive/negative rates, and per-strategy / per-backend / per-depth winner counts and accepted gain. `scripts/report-experiments.sh` compares several journals side by side.

---

## Safety invariants

This experiment exists to continue evolution without risking a creature that already works.

1. **The supplied incumbent is immutable.** Forests never modifies the source creature in place.
2. **Every candidate starts from a clone of a known incumbent.**
3. **Version 1 only adds small corrective structure.** It does not delete, simplify or rewire mature evolved structure.
4. **Cheap search is allowed to be wrong.** Histogram gain, residual reduction, sampled scoring and other proxies are ranking signals only.
5. **Full-corpus NEAT-AI-scorer is the final authority.** A candidate is not an improvement until the scorer says it is.
6. **Scorer failure means no winner.** Missing, malformed or inconsistent results fail closed.
7. **`best.json` may never be worse than the opening scorer-verified baseline.**
8. **After an accepted change, residuals are recomputed.** Forests never assumes predicted gains remain valid after the creature changes.
9. **Random accidents are legitimate discoveries.** We care about measurable improvement, not whether the winning idea looked clever beforehand.
10. **Negative results are results.** A dirty trick that consumes time but produces no verified improvements should be measured and discarded.

### What the invariants do not cover: adaptive data analysis

Every invariant above makes a **single** decision honest. None of them makes
**thousands** of decisions against the same corpus honest.

Acceptance is measured on the full production corpus every time — the same
records, for every iteration of every run, and for the sibling optimisers
(Discovery, Lamarck and ordinary NEAT evolution) that select against it too.
That is **adaptive data analysis**: once a corpus has answered thousands of
queries, the accepted set is fitted to that corpus in a way no individual
scorer call can reveal. With a target as low-SNR as 90-day price movement it is
the dominant risk in this experiment — larger than any choice of split search.
The names for it are
[Dwork et al. 2015, *The reusable holdout*](https://doi.org/10.1126/science.aaa9375)
and [Blum & Hardt 2015, *The Ladder*](https://arxiv.org/abs/1502.04585).

**Is any corpus slice held back from every optimiser? No — there is no holdout
today.** `--search-records` samples rows for the *search* and
`--screen-sample-rate` samples records for the *screen*, but the authoritative
call scores the whole corpus, and no slice is withheld from Forests or from any
sibling. So "the scorer decides" is an **in-corpus** guarantee: `best.json` is
the creature that scored best on the corpus everything else was also selected
on. Until a shared, never-optimised-against slice exists, read a reported Δ as
an upper bound on out-of-sample gain, and treat the smallest accepted gains —
those near `--min-improvement` — as the least trustworthy.

---

## Non-goals

Forests is **not**:

- a replacement for NEAT-AI;
- a replacement for the production creature;
- a new stock-prediction model trained independently from scratch;
- permission to rewrite or simplify years of evolved structure because another algorithm thinks it looks cleaner;
- an optimiser allowed to accept its own estimated gain;
- an online/live-trading decision engine;
- proof that decision trees are inherently better than neural/evolutionary methods;
- committed to keeping any strategy that fails to produce real improvements economically.

---

## Why residual trees

```mermaid
flowchart TD
    FIT(["current fittest creature\nimmutable source"]) --> RES["measure remaining residual/error structure"]

    RES --> HIST["histogram / quantile split search"]
    RES --> SAMPLE["sampling / random exploration"]
    RES --> OTHER["other dirty tricks"]

    HIST --> PATCH[["small residual tree patches"]]
    SAMPLE --> PATCH
    OTHER --> PATCH

    PATCH --> GRAFT["graft each patch onto a clone"]
    GRAFT --> SCREEN{"cheap screen\noptional / non-authoritative"}

    SCREEN -- "unlikely" --> DROP["discard"]
    SCREEN -- "interesting" --> FULL{"NEAT-AI-scorer\nfull canonical corpus"}

    FULL -- "not better" --> DROP2["discard"]
    FULL -- "score really improves" --> WIN(["new experimental incumbent"])

    WIN --> REPEAT((("repeat from new residuals")))
    REPEAT --> RES

    classDef creature fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0b2545
    classDef search fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03
    classDef candidate fill:#ede9fe,stroke:#6d28d9,stroke-width:2px,color:#2e1065
    classDef win fill:#dcfce7,stroke:#15803d,stroke-width:2px,color:#052e16
    classDef reject fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#450a0a

    class FIT,RES creature
    class HIST,SAMPLE,OTHER,SCREEN,FULL search
    class PATCH,GRAFT,REPEAT candidate
    class WIN win
    class DROP,DROP2 reject
```

The search mechanism and the acceptance mechanism are intentionally independent.

Forests may use approximate arithmetic, samples, heuristics, statistics, random guesses or approximations to decide **what is worth trying**. None of those may decide **what is better**.

Only the authoritative scorer gets that vote.

### Residual evolution

The useful mental model is not "replace the neural network with a forest".

The mature creature already represents a valuable function:

```text
f(x)
```

Forests searches for a small conditional correction:

```text
g(x)
```

and tests the complete candidate:

```text
f'(x) = f(x) + g(x)
```

For example, a first-generation patch might effectively say:

```text
if observation_317 > 0.283:
    correction = +0.013
else:
    correction = 0
```

Most of observation-space can therefore retain the incumbent's existing behaviour exactly, while Forests asks whether a particular region contains a systematic residual error worth correcting.

This is much closer to **boosting a mature evolved model** than training a conventional random forest from scratch.

### Why decision trees?

Decision trees offer a type of search that ordinary gradient methods find awkward: **hard, discontinuous partitions**.

NEAT-AI already has the building block required to represent them: the `IF` aggregate and typed condition/positive/negative synapses.

A conventional split:

```text
RSI_14 > 63
```

can therefore become ordinary NEAT-AI creature structure rather than requiring a second model runtime.

Nested `IF` nodes can represent deeper trees, and NEAT-AI has an additional capability that conventional axis-aligned trees usually do not: an `IF` condition can eventually be an **oblique split** such as:

```text
0.8 * RSI_14 - 1.3 * PE + 0.4 * momentum > threshold
```

That is deliberately later research. The experiment starts with simple one-feature stumps because they are easy to verify and cheap to search.

### Why "Forests"?

The name is playful rather than a commitment to a conventional Random Forest algorithm.

Forests may explore many competing tree-shaped corrections at once, but the likely evolutionary pattern is closer to sequential boosting:

```text
mature creature
    ↓
find a residual pattern
    ↓
try many small tree patches
    ↓
scorer accepts one (or none)
    ↓
recompute residuals
    ↓
search again
```

The final creature remains a normal NEAT-AI creature.

### What we want to steal from XGBoost

[XGBoost](https://github.com/dmlc/xgboost) is an important source of ideas, not a planned runtime dependency.

The most interesting concepts for this experiment include:

- **quantile/binning of continuous observations** so every raw value is not tested as a threshold;
- **histogram split search** using compact sufficient statistics;
- **row subsampling**;
- **feature/column subsampling**;
- **error/gradient-biased sampling** so search effort concentrates on informative records;
- **minimum leaf support and complexity controls** to avoid chasing tiny pathological subsets;
- **sequential residual correction / boosting**;
- using a real XGBoost model as an **external scientific control** to see whether conventional boosted trees can find residual structure our native search misses.

We want to understand and adapt useful techniques to the NEAT-AI problem, not copy XGBoost's architecture wholesale.

Forests is Rust-first. A GPU histogram kernel was built and measured at 0.17x the CPU path on unified memory, and deleted (#67): search is under a fifth of wall clock, so the lever is reusing accumulations across tree levels (#69), not another backend.

### Search philosophy: dirty is fine, dishonest is not

Examples of fair experimental tactics include:

- exhaustive quantile stump search;
- random feature subsets;
- random/stratified record subsets;
- residual-magnitude-weighted sampling;
- threshold jitter around promising boundaries;
- several correction magnitudes around a statistical optimum;
- deliberately random stumps;
- diversity selection so one promising feature does not monopolise a batch;
- shallow depth-2/3 trees;
- eventually sparse multi-feature/oblique splits;
- importing a small XGBoost tree as a candidate control.

Every candidate must record what produced it. Random search must be called random search. Approximate search must be called approximate search.

The metric that matters is not elegance or proxy gain. It is:

> **scorer-verified improvement per wall-clock hour.**

---

## Where this sits in the literature

Everything above is in house terms — 🌳 Forests, "dirty tricks", "trust only the
scorer" — and those stay. But every mechanism in the pipeline already has a name
in the research literature, and naming them makes the experiment legible to
anyone who has not read the issue history. It also bounds the novelty claim
honestly: most of this is established method, and exactly one part is unusual.

```mermaid
flowchart LR
    RES["residuals → shallow tree → add"] --> L1["gradient boosting<br/>Friedman 2001 · GrowNet 2020"]
    SPLIT["split search"] --> L2["histogram / oblique splits<br/>XGBoost 2016 · OC1 1994 · Breiman 2001"]
    GRAFT["patch → native IF structure"] --> L3["automated software transplantation<br/>Barr 2015 · genetic improvement 2018"]
    SCREEN["sampled screen → full score"] --> L4["racing<br/>Hoeffding 1994 · F-Race 2002 · Hyperband 2017"]
    CACHE["shared learnings cache"] --> L5["memory-based search<br/>tabu 1986 · adaptive operator selection 2010"]
    ACCEPT["accept on the same corpus, every time"] --> L6["adaptive data analysis<br/>Dwork 2015 · Blum &amp; Hardt 2015"]

    classDef ours fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px,color:#0b2545
    classDef lit fill:#fef3c7,stroke:#b45309,stroke-width:2px,color:#451a03
    classDef risk fill:#fee2e2,stroke:#b91c1c,stroke-width:2px,color:#450a0a
    class RES,SPLIT,GRAFT,SCREEN,CACHE ours
    class L1,L2,L3,L4,L5 lit
    class ACCEPT,L6 risk
```

### The core loop is gradient boosting

Measure the residuals of a fitted model, fit a shallow tree to them, add it to
the model, recompute the residuals — that is **gradient boosting**
([Friedman 2001, *Greedy function approximation: a gradient boosting machine*](https://doi.org/10.1214/aos/1013203451)).
The `--boost-rounds` flag, the shrinkage result in
[docs/benchmarks.md](docs/benchmarks.md#shrinkage), and the whole "recompute
residuals after every acceptance" invariant are that method, not a house
invention.

Doing it with a **neural network as the base model** is **GrowNet**
([Badirli et al. 2020, *Gradient Boosting Neural Networks*](https://arxiv.org/abs/2002.07971)),
which boosts shallow networks as weak learners; Forests inverts the pairing and
boosts tree-shaped corrections onto a mature network. Putting trees and nets in
one differentiable model is a line of its own:
[Kontschieder et al. 2015, *Deep Neural Decision Forests*](https://doi.org/10.1109/ICCV.2015.172)
and [Popov et al. 2019, *NODE*](https://arxiv.org/abs/1909.06312). Forests
differs from all of these in what it optimises — nothing here is trained by
gradient descent end to end, and the base model is frozen — but the loop is the
same loop.

**The XGBoost control is the incumbent method, and should be read that way.**
Gradient-boosted trees ([Chen & Guestrin 2016, *XGBoost*](https://arxiv.org/abs/1603.02754))
are what a practitioner would reach for on exactly this problem shape:
tabular features, a scalar target, residual structure left in a fitted model.
The control in [docs/xgboost-control.md](docs/xgboost-control.md) is therefore
not a curiosity — it is the comparison that decides whether native search is
worth having.

### The graft is software transplantation

Turning a patch into **native `IF` structure inside the creature**, rather than
carrying a second model at runtime, is the unusual part, and its closest
precedent is not from machine learning:
[Barr et al. 2015, *Automated Software Transplantation* (ISSTA)](https://doi.org/10.1145/2771783.2771796)
moves a working feature from one program into another and validates the host by
running it. The wider field is genetic improvement
([Petke et al. 2018, *Genetic Improvement of Software: A Comprehensive Survey*](https://doi.org/10.1109/TEVC.2017.2693219)),
which searches for edits to an existing program under a test-based acceptance
gate — structurally the same contract as "graft onto a clone, let the scorer
decide". Citing it bounds the claim: the novelty is real, and it is narrow.

### Two-phase screening is racing

Running every candidate on a cheap sample, dropping what the sample can resolve
as a loser, and spending the expensive evaluation only on survivors is
**racing**:
[Maron & Moore 1994, *Hoeffding races*](https://proceedings.neurips.cc/paper/1993/hash/02a32ad2669e6fe298e607fe7cc0e1a0-Abstract.html),
[Birattari et al. 2002, *F-Race*](https://dl.acm.org/doi/10.5555/2955491.2955494),
[Jamieson & Talwalkar 2016, successive halving](https://arxiv.org/abs/1502.07943)
and [Li et al. 2017, *Hyperband*](https://arxiv.org/abs/1603.06560).

That literature also raises the question we have to answer here: **a sampled
screen only earns the right to drop an arm once it has the power to resolve the
effect size in play.** Ours are around 1e-4 per accepted iteration and smaller,
against a 5 % sample — see
[docs/benchmarks.md](docs/benchmarks.md#where-the-screen-sits) for what the
screen is and is not powered for, and why `--explore-quota` exists. An
under-powered screen does not merely waste calls; it silently vetoes true wins.

### The shared learnings cache is memory-based search

Replaying verified wins and refusing to pay twice for a known failure is
**memory-based search**: a tabu list of recently rejected moves
([Glover 1986, *Future paths for integer programming and links to artificial intelligence*](https://doi.org/10.1016/0305-0548%2886%2990048-1)),
with `--learnings-retry-after-hours` as its tenure. Choosing what to try next
from the measured payoff of past choices is **adaptive operator selection**
([Fialho et al. 2010, *Analyzing bandit-based adaptive operator selection mechanisms*](https://doi.org/10.1007/s10472-010-9213-y))
— which is what `neat_ai_forests report`'s per-strategy economics are for.

### Splits

Axis-aligned quantile stumps are the ordinary decision-tree/random-forest
baseline ([Breiman 2001, *Random Forests*](https://doi.org/10.1023/A:1010933404324)).
The `--oblique-candidates` path — a split on a weighted sum of two or three
features — is oblique induction
([Murthy et al. 1994, *A System for Induction of Oblique Decision Trees* (OC1)](https://doi.org/10.1613/jair.63)),
and NEAT-AI's `IF` aggregate can express one natively. Their measured economics
are in [docs/benchmarks.md](docs/benchmarks.md), and so far the axis-aligned
baseline wins.

### The exposure

The one thing in the pipeline that is **not** covered by any of the methods
above is that acceptance is measured against the same corpus every time.
That is adaptive data analysis, and it is written up with the
[safety invariants](#what-the-invariants-do-not-cover-adaptive-data-analysis).

---

## What success looks like

The experiment succeeds if it increases the **rate at which verified improvements are found** for a creature whose ordinary evolutionary progress has become difficult.

Useful measurements include:

- time to first accepted improvement;
- accepted improvements per 45-minute run;
- cumulative scorer improvement;
- candidates searched/scored per minute;
- improvement rate by strategy;
- depth-1 vs deeper-tree economics;
- how concentrated each gain is across the corpus;
- whether repeated residual grafting continues producing improvements or quickly saturates;
- comparison with Lamarck, Discovery, random search and the XGBoost control.

If the evidence says a clever technique is useless, remove it.

If dumb luck repeatedly wins, generate more dumb luck.

**The scorer does not care about our theory, and neither should the experiment.** 🌳🧬

---

## Repository layout

```text
NEAT-AI-Forests/
├── Cargo.toml                 # workspace
├── forests/
│   ├── Cargo.toml             # neat_ai_forests (lib + bin)
│   ├── src/
│   │   ├── main.rs            # CLI
│   │   ├── lib.rs
│   │   ├── baseline.rs        # authoritative baseline + parity gate (#2)
│   │   ├── bins.rs            # quantile-bin cache (#3)
│   │   ├── cancel.rs          # SIGINT/SIGTERM cooperative cancellation
│   │   ├── candidates.rs      # candidate population (#8)
│   │   ├── config.rs          # ForestsConfig + validation
│   │   ├── corpus.rs          # corpus identity + bounded-memory streaming
│   │   ├── enhancements.rs    # accepted patches filed for Rebase (Rebase#65)
│   │   ├── graft.rs           # patch → IF structure on a clone (#7), validated (#39)
│   │   ├── histogram.rs       # CPU reference stump search (#5)
│   │   ├── incumbent.rs       # immutable incumbent + checksum (#2)
│   │   ├── journal.rs         # experiments.jsonl records (#10)
│   │   ├── learnings.rs       # fleet-shared cache of what worked / failed (#60)
│   │   ├── log.rs             # stderr logging
│   │   ├── meta.rs            # creature tag preservation
│   │   ├── oblique.rs         # multi-feature linear splits (#14)
│   │   ├── patch.rs           # portable patch format + evaluator (#7)
│   │   ├── promote.rs         # screen + authoritative promotion (#9)
│   │   ├── report.rs          # journal economics report (#15)
│   │   ├── residuals.rs       # residual extraction + sidecar (#4)
│   │   ├── run.rs             # the evolution loop (#10)
│   │   ├── scorer.rs          # rust_scorer integration
│   │   ├── strategies.rs      # sampling / feature selection (#12)
│   │   ├── tools.rs           # export-matrix / import-xgboost (#13)
│   │   ├── tree.rs            # depth-2/3 growth (#11)
│   │   └── xgboost.rs         # XGBoost dump conversion (#13)
│   ├── examples/stump_search_bench.rs
│   └── tests/                 # README contract, real-scorer integration
├── docs/                      # architecture, caches, patch format, strategies, xgboost control, benchmarks, learnings cache
│   └── archive/pr-summaries/  # one summary per merged PR
├── scripts/                   # quality helpers, auto-version.sh, report-experiments.sh, run-benchmark.sh, xgboost-control.py
├── quality.sh                 # local gate mirroring CI
└── .github/workflows/         # CI, security, gitleaks, markdown lint, actionlint, dependency review, SBOM, semgrep
```

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — module map and data flow.
- [docs/caches.md](docs/caches.md) — bin-cache and residual-sidecar binary formats and invalidation rules.
- [docs/patch-format.md](docs/patch-format.md) — patch JSON, `IF` graft layout, exact routing semantics.
- [docs/learnings.md](docs/learnings.md) — the shared cache of what worked and what failed, and how a fleet uses it.
- [docs/strategies.md](docs/strategies.md) — sampling, jitter, diversity, random controls and how they report themselves.
- [docs/xgboost-control.md](docs/xgboost-control.md) — the external control experiment.
- [docs/benchmarks.md](docs/benchmarks.md) — measured economics and the production-run protocol.
- [docs/archive/pr-summaries/](docs/archive/pr-summaries/) — the PR summary for each merged change.

---

## Development

The toolchain is pinned in `rust-toolchain.toml` (the same channel as the
NEAT-AI Rust family). `./quality.sh` mirrors CI: shellcheck, codespell,
markdownlint, actionlint, cargo-deny, `cargo fmt --check`, clippy with
`-D warnings`, tests (`--all-features`) and rustdoc. See
[CONTRIBUTING.md](CONTRIBUTING.md).

### Versioning

The unattended machines rebuild the binary only when the crate version changes,
so every PR must leave `forests/Cargo.toml` ahead of the base branch. CI's
`version-increment` job does that for you: `scripts/auto-version.sh` bumps the
patch — in the manifest and in `Cargo.lock` — and pushes the bump back onto the
PR branch. Bump the version yourself (a minor or major, say) and the job leaves
it alone; let it slip behind the base branch and the job fails rather than
shipping a version the machines have already built.

```mermaid
flowchart LR
    PR[PR commit] --> CMP{"forests/Cargo.toml vs base branch"}
    CMP -- behind --> FAIL["fail loud:<br/>machines would skip the rebuild"]
    CMP -- ahead --> KEEP["no-op: the author bumped it"]
    CMP -- level --> BUMP["bump patch in<br/>Cargo.toml + Cargo.lock"]
    BUMP --> PUSH["commit and push<br/>onto the PR branch"]
    PUSH --> BUILD["merged: unattended machines<br/>see a new version and rebuild"]
    KEEP --> BUILD
```

---

## Planned phases

The phases the experiment was founded on, all of them delivered. Kept here
because they are the shape of the argument as much as the plan: each one had to
work before the next was worth trying.

| Phase | Purpose | Issues |
|---|---|---|
| 0 | Bootstrap, immutable baseline and scorer contracts | [#1](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/1), [#2](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/2) |
| 1 | Quantile cache and incumbent residuals | [#3](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/3), [#4](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/4) |
| 2 | Correct CPU stump search | [#5](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/5) |
| 3 | GPU histogram acceleration (built, measured 0.17x the CPU path, deleted in [#67](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/67)) | [#6](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/6) |
| 4 | Portable tree patches and conservative IF grafts | [#7](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/7), [#8](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/8) |
| 5 | Cheap screening + authoritative promotion | [#9](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/9) |
| 6 | First 45-minute iterative Forest evolution loop | [#10](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/10) |
| 7 | Depth-2/3 trees and sequential boosting | [#11](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/11) |
| 8 | Aggressive sampling/random dirty tricks | [#12](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/12) |
| 9 | XGBoost control experiment | [#13](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/13) |
| 10 | Oblique multi-feature IF splits | [#14](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/14) |
| 11 | Measure what actually improves evolution | [#15](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/15) |

Two shared-family prerequisites deliberately live outside this repository:

- [NEAT-AI-core #555](https://github.com/stSoftwareAU/NEAT-AI-core/issues/555) — canonical decision-tree/`IF` fixture and safe construction helpers.
- [NEAT-AI-scorer #574](https://github.com/stSoftwareAU/NEAT-AI-scorer/issues/574) — lock CPU/GPU parity for `IF`-heavy candidate creatures.

---

## Outstanding work

- Prune the shared learnings cache, so old failures become worth retrying and the directory stops growing ([#61](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/61)).
- Drop the `--scorer-arg=--gpu=off` advice once [NEAT-AI-scorer #574](https://github.com/stSoftwareAU/NEAT-AI-scorer/issues/574) lands.
- Make CODEOWNERS review actually required rather than advisory ([#38](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/38)).

---

## Related NEAT-AI experiments

- [NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) — TypeScript evolutionary trainer and the long-running evolutionary process.
- [NEAT-AI-core](https://github.com/stSoftwareAU/NEAT-AI-core) — shared Rust creature/network representation and inference primitives.
- [NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) — authoritative Rust scorer; **the final judge**.
- [NEAT-AI-Discovery](https://github.com/stSoftwareAU/NEAT-AI-Discovery) — statistical/structural discovery of promising mutations.
- [NEAT-AI-Lamarck](https://github.com/stSoftwareAU/NEAT-AI-Lamarck) — acquired-learning/backprop/statistics-guided optimisation of mature creatures.

These experiments are complementary rather than mutually exclusive:

```text
normal NEAT evolution
        │
        ├── Discovery  → structural/statistical opportunities
        ├── Lamarck    → continuous learning / parameter opportunities
        └── Forests    → discontinuous conditional/residual opportunities
                              │
                              ▼
                       NEAT-AI-scorer
                       decides what survives
```

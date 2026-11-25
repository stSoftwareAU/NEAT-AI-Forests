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

- **GPU histogram accumulation was not worthwhile, and the path is gone (#67).** The one-shot
  upload of the `u8` matrix (490 MB) plus partial read-back dominates, and the
  kernel itself cannot beat 8 CPU threads writing into cache-resident
  histograms. A flag whose only correct setting was `off`, behind a cargo
  feature nothing compiled, was complexity for nothing — deleted along with the
  `wgpu` / `pollster` / `bytemuck` dependencies. Search is under a fifth of wall
  clock, so the lever is reusing accumulations across tree levels (#69), not
  another backend. The kernel is in the git history if a host ever makes the
  upload worth amortising.
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

## What a scorer call is worth, by candidate family (Issue #63)

Full scoring is roughly half of wall-clock, so the resource to allocate is the
**full-corpus scorer call**, and the question is what each family of candidate
returns per call. Measured across 23 production runs and about 3,600 calls:

| strategy | full scores | winners | win rate | score gained per call |
|---|---|---|---|---|
| `histogram-tree-depth3` | 94 | 22 | 23.4 % | **3.50e-5** |
| `histogram-tree-depth3/one-sided` | 81 | 15 | 18.5 % | 2.71e-5 |
| `histogram-tree-depth2/scale` | 30 | 13 | 43.3 % | 1.74e-5 |
| `histogram-tree-depth2` | 111 | 17 | 15.3 % | 1.28e-5 |
| `histogram-stump` | 1008 | 66 | 6.5 % | 2.15e-6 |
| `histogram-stump/scale` | 1057 | 149 | 14.1 % | 2.11e-6 |
| `oblique-split/one-sided` | 61 | 1 | 1.6 % | 3.56e-7 |
| `histogram-stump/one-sided` | 858 | 11 | 1.3 % | 2.48e-7 |

A depth-3 tree is worth 141× a one-sided stump per call and 16× a plain stump.
Two things follow, and both were accidents of implementation order rather than
decisions: the tree-root count was hard-coded at 3, and the cohort cap cut
exactly where the magnitude variants began. `--tree-roots` and the reordered
expansion in `candidates::expand_discoveries` are the response.

### Shrinkage

Split the same runs by leaf magnitude:

| scale | full scores | winners | win rate |
|---|---|---|---|
| 1.0 (the analytical optimum) | 2539 | 157 | 6.18 % |
| **0.5** | **1057** | **168** | **15.89 %** |
| 1.5 | 56 | 0 | 0 % |

Shrunk leaves win 2.6× as often, and beat the unscaled leaf in 15 of 17 runs
where both were tried at least 20 times (two-proportion z = 9.3). This is the
familiar boosting result — the analytically optimal step overshoots, and a
shrunk step generalises better
([Friedman 2001](https://doi.org/10.1214/aos/1013203451), §5: the shrinkage
parameter `ν`) — and it is why magnitude variants are now emitted before
one-sided ones. `--magnitude-scales` is that shrinkage parameter, searched
rather than fixed.

### Where the screen sits

Two-phase screening with authoritative promotion is **racing**: evaluate every
arm cheaply, drop the arms the cheap evaluation can resolve as losers, and spend
the expensive evaluation only on the survivors —
[Maron & Moore 1994 (Hoeffding races)](https://proceedings.neurips.cc/paper/1993/hash/02a32ad2669e6fe298e607fe7cc0e1a0-Abstract.html),
[Birattari et al. 2002 (F-Race)](https://dl.acm.org/doi/10.5555/2955491.2955494),
[Jamieson & Talwalkar 2016 (successive halving)](https://arxiv.org/abs/1502.07943),
[Li et al. 2017 (Hyperband)](https://arxiv.org/abs/1603.06560). Every one of
those methods drops an arm only when the cheap evaluation has the **power** to
resolve the difference, which is the precondition to check here.

**What the 5 % screen is powered for.** The screen scores a 5 % stride sample
(≈113 k of 2.27 M records), so on the same Δ it carries roughly
√20 ≈ 4.5× the standard error of the authoritative call. The effects it is
asked to resolve are far smaller than that margin: accepted gains run at
1e-4 per iteration early and taper to ≈2e-5 (production evidence below),
per-call family gains are 3.5e-5 down to 2.5e-7 (above), and
`--min-improvement` accepts at 1e-6. So the screen is powered to reject
candidates that are **clearly worse**, and is not powered to order candidates at
the 1e-4-and-below scale where real acceptances live. The measurements agree:
winners are spread almost evenly across screen ranks 1–8 (rank 1 holds only
17 % of them; all eight are needed for 94.5 %), and on the first-generation
production run the screen rejected half of the exploratory bypass candidates
that later cleared the full threshold. Read a screen rejection as a veto on the
obviously bad, never as a verdict — an under-powered screen silently vetoes true
wins, which is exactly what `--explore-quota` exists to keep measuring.

Modelling the cost — 27 s search, 29 s
screen, 85 s full scoring per iteration — `--promote-count` 4 would cost more
winners than the extra iterations return, and 12 would waste calls. The default
of 8 is where it should be; the lever is the cohort, not the cut.

### Variant economics (22 runs)

| variant | runs | Δscore per wall hour |
|---|---|---|
| depth-3 best-first | 5 | 1.94e-3 |
| depth-3 + sampling tricks | 2 | 1.69e-3 |
| sampling tricks | 5 | 7.37e-4 |
| depth-2 | 3 | 6.52e-4 |
| depth-2 + boosting | 1 | 6.09e-4 |
| oblique | 1 | 3.26e-4 |
| boosting only | 1 | 1.42e-4 |
| stumps only (`--max-depth 1`) | 1 | **0** |

Depth is what pays. A rotation that spends cycles on stumps alone is spending
them on the one arm measured at zero.

### The A/B (2026-08-24)

Same source creature, same seed, same 25-minute budget, one arm per binary:

| arm | iterations | acceptances | Δscore | wall | Δ per hour | full scores |
|---|---|---|---|---|---|---|
| control (roots 3, one-sided first) | 8 | 7 | 2.4483e-4 | 27.1 min | 5.42e-4 | 72 |
| **`--tree-roots 8`, magnitude first** | 7 | 7 | **4.9213e-4** | 28.4 min | **1.0396e-3** | 63 |

1.9× the improvement per wall hour, on *fewer* scorer calls and fewer
iterations — the extra tree roots cost about 20 s of search per iteration and
paid for it several times over. The cohort composition tells the story:

| | trees | shrunk | one-sided |
|---|---|---|---|
| control | 18.8 % | 15.8 % | 43.6 % |
| experiment | 45.3 % | 45.3 % | 0 % |

The experiment's first acceptance was a `histogram-tree-depth3/scale` — a
family that never appeared in the control's cohort at all, because the cap cut
before the magnitude variants of trees were reached.

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

# G. refine existing leaves (Issue #140) — compare against A at the same seed and budget
neat_ai_forests $C $D --scorer $SCORER --scorer-arg=--gpu=off --output-dir runs/refine --seed 1 --refine-leaves 8

# H. one refit at the start of the run (Issue #137) — compare against A at the same seed and budget;
#    --refit-every larger than the run's iteration count means "only the opening refit"
neat_ai_forests $C $D --scorer $SCORER --scorer-arg=--gpu=off --output-dir runs/refit --seed 1 --refit-every 100000

# I. hidden neurons as split sources (Issue #139) — compare against A; report the share of
#    accepted patches whose strategy ends in `+hidden` and their score gained per scorer call
neat_ai_forests $C $D --scorer $SCORER --scorer-arg=--gpu=off --output-dir runs/hidden --seed 1 --hidden-features 256

# J. linear leaves (Issue #138) — compare against A at equal budget; add a `linear-leaf` row to
#    "What a scorer call is worth" from this run's report
neat_ai_forests $C $D --scorer $SCORER --scorer-arg=--gpu=off --output-dir runs/linear --seed 1 --linear-leaf-conditions 16

# E. random-only control (no histogram guidance)
neat_ai_forests $C $D --scorer $SCORER --scorer-arg=--gpu=off --output-dir runs/random --seed 1 --top-k 0 --random-candidates 48

# F. XGBoost control — see docs/xgboost-control.md

scripts/report-experiments.sh runs/*/experiments.jsonl
```

Lines G–J are optional controlled checks: ordinary runs already draw those
four families at random and stamp the draw in `forests-arms` (Issue #147), so
`scripts/report-experiments.sh` over the fleet's journals gives the on/off
comparison without hand-run A/Bs.

Record for each run: opening/final authoritative score, acceptances,
`improvementPerWallHour`, `timeToFirstAcceptanceMs`, `searchTimeFraction`,
screen FP/FN rates, and the `affectedFraction` of every accepted gain (a
winner that touches 0.1 % of records is suspicious and should be inspected
with the journal's patch before anyone trusts it). Compare with a Lamarck run
of the same budget on the same champion rather than assuming either dominates.

## Exact cohort ranking experiment (Issue #136)

The scorer prerequisite is delivered: [scorer PR #684](https://github.com/stSoftwareAU/NEAT-AI-scorer/pull/684)
merged on 22 November 2026 with cohort-versus-standalone parity tests. Forests
probes the binary's help once for an exact run. `--ranking-mode exact` uses
full-corpus cohort scores to rank the whole valid generated population, with
candidate-id tie breaking, then gives the highest-ranked eligible candidate
an ordinary full-corpus call with its parent. The scorer's documented directory
fallback is recorded rather than silently treated as hosted execution.

`--explore-quota 1` retains a diagnostic 5 % screen and seeded bypass selection.
Its hypothetical decisions do not remove candidates from exact ranking; the
extra call counts against the same wall budget. Use quota 0 to turn diagnostics
off. `fullScore` and `fullDelta` still mean ordinary verification; exact scores
alone neither file learnings nor qualify a near-winner for combination carry-forward.

### Production A/B status and decision rule

**Not run on production.** No production creature/corpus paths were supplied,
and the available checkout and data directories contain no production corpus.
The generated fixtures below cannot measure production improvement per wall
hour. There is no production ratio or production negative result to report.
The default remains `--ranking-mode legacy`, with 64 candidates, 8 tree roots,
scales `1.0,0.5,0.25` and 4 combinations.

The predeclared switch condition remains **at least 1.5× control improvement
per wall hour**. A ratio cannot be measured when control improvement is zero;
report that explicitly rather than dividing by zero or treating it as a pass.
A measured miss is recorded here with its production numbers and retains the
legacy default. Missing production data also retains that default.

Use fresh output directories; do not append another run to an earlier journal.
Run both arms from the identical source creature and corpus, at seed 1 and the
same 2700-second budget; pin the candidate families and CPU backend identically.
The treatment widens only the search/cohort knobs being evaluated:

```bash
COMMON=("$C" "$D" --scorer "$SCORER" --scorer-arg=--gpu=off
  --seed 1 --timeout-seconds 2700 --no-random-arms --explore-quota 1)

# Control: current legacy defaults, made explicit.
neat_ai_forests "${COMMON[@]}" --ranking-mode legacy --output-dir runs/cohort-control \
  --candidates 64 --tree-roots 8 --top-k 16 --magnitude-scales 1.0,0.5,0.25 --combo-candidates 4

# Treatment: wider exact cohort; preserve all other control settings.
neat_ai_forests "${COMMON[@]}" --ranking-mode exact --output-dir runs/cohort-exact \
  --candidates 256 --tree-roots 32 --top-k 64 \
  --magnitude-scales 1.0,0.875,0.75,0.5,0.25,0.125 --combo-candidates 16

scripts/report-experiments.sh runs/cohort-control/experiments.jsonl runs/cohort-exact/experiments.jsonl
```

Keep the source/corpus identities and resolved family settings from the journals.
Report `improvementPerWallHour`, acceptances and `exactCandidatesPerIteration`
for both arms. `rankComparisons` lists each exact candidate's exact rank beside
its hypothetical screen rank and promotion/bypass decisions. Diagnostic FP/FN
counts are separate from ordinary verified screen counts.

The complete wall split is opening baseline + search + graft + screen scorer +
ranking scorer + ordinary full scorer + promotion overhead + setup/other.
`verificationScorerMs` is the exact winner/parent subset of `fullScorerMs`, not
an extra term. `promotionOverheadMs` includes cohort file I/O and failed
screen/ranking calls without separately recorded results. `setupAndOtherMs`
covers bin/residual work, identity hashing, accepted-creature output and the
remaining wall time. Failures and baseline vetoes invalidate a clean performance
comparison; record them separately. Runs can finish their in-flight iteration
after the budget expires, so compare actual `wallMs`, not the requested timeout.

### Generated-fixture width sweep

`forests/examples/cohort_ranking_bench.rs` exercises the real evolution loop
with a generated 32-input MLP, deterministic residual regions, seed 1, the same
starting creature and fixed family settings. It alternates profile order across
repeats and gives each run its own cold cache. It measures legacy-64, exact-64,
exact-128 and exact-256, widening roots, top-K, scales and combinations together.
These small synthetic inputs test the protocol and candidate reach, not the
production speed or default-selection criterion.

```bash
cargo run --release --example cohort_ranking_bench -- \
  --scorer "$SCORER" --records 20000 --repeats 3
```

Measured on 25 November 2026 on shared aarch64 Linux host `vibe-coder-55874`,
release builds of Forests and scorer commit `a164969ac4d96ff39eace7086ad22c0538d95705`.
Three alternating-order repeats, 20 000 records, one iteration per profile,
cold caches, diagnostics enabled. Times are **indicative on a shared, parallel
host**, not test thresholds. Each profile found the same planted winner; the
ordinary multi-candidate control differed from pair verification by less than
6e-14 in score. This tiny fixture supplies no production-economics evidence.

| Profile | Exact candidates / iteration | Acceptances across 3 repeats | Total wall (ms) | Total verified gain | Gain / wall hour |
|---|---:|---:|---:|---:|---:|
| legacy-64 | 9 | 3 | 1094 | 0.002257920549 | 7.4301 |
| exact-64 | 64 | 3 | 1155 | 0.002257920549 | 7.0377 |
| exact-128 | 128 | 3 | 1646 | 0.002257920549 | 4.9383 |
| exact-256 | 256 | 3 | 2546 | 0.002257920549 | 3.1927 |

Complete wall split, totals over the same three repeats (ms); every row sums
to its wall total. Ordinary full scoring includes the separate verification
in exact mode:

| Profile | Opening baseline | Search | Graft | Diagnostic/legacy screen | Ranking | Ordinary full | Promotion overhead | Setup/other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy-64 | 32 | 473 | 42 | 76 | 0 | 118 | 14 | 339 |
| exact-64 | 33 | 421 | 42 | 75 | 116 | 53 | 21 | 394 |
| exact-128 | 23 | 731 | 93 | 114 | 221 | 50 | 43 | 371 |
| exact-256 | 57 | 1325 | 207 | 149 | 308 | 48 | 75 | 377 |

The 128-wide ranking call averaged 74 ms versus 69 ms for the control's whole
promotion stage (screen + full + overhead). Verification, diagnostic screening
and wider search still add cost, and widening found no additional gain on this
fixture. This supports keeping widths experimental; it does not establish a
production negative result or change the legacy defaults. Zero scorer failures,
zero baseline vetoes and zero cohort fallbacks were observed in this sweep.

## What to do with the answer

- **Stumps win repeatedly:** keep going; measure saturation via
  `scoreTrajectory`; try depth 2.
- **Stumps never win but XGBoost's converted trees do:** the native search is
  missing something — compare thresholds/features in the two journals.
- **Nothing wins economically:** that is the documented negative result. Do not
  add complexity; record it here and stop.

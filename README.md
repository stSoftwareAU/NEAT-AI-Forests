# NEAT-AI-Forests

> **Experimental:** can decision trees, boosting-style residual search, GPU histogram tricks, random exploration and other techniques increase the rate of evolution of an already highly evolved NEAT-AI creature?

## 🌳 Motto

> **Find all the dirty tricks that uncover real improvements — but trust only the scorer.**

NEAT-AI-Forests is an experimental Rust optimiser for already-fit
[NEAT-AI](https://github.com/stSoftwareAU/NEAT-AI) creatures.

The motivating creature has already been evolved for years to predict **90-day stock movement** and is useful in production. Forests is **not** an attempt to replace that creature, retrain it from scratch, or redesign the working system.

The experiment asks a narrower question:

> **Can a fundamentally different search process discover small improvements that normal evolution is now finding only slowly?**

Forests starts with the current fittest creature, studies the errors it still makes, searches aggressively for conditional/tree-shaped residual corrections, grafts promising candidates onto **copies** of the incumbent, and asks the existing
[NEAT-AI-scorer](https://github.com/stSoftwareAU/NEAT-AI-scorer) to decide whether any complete candidate is genuinely fitter.

Candidate generation may be adventurous. **Acceptance is deliberately boring.**

---

## Status

🚧 **Bootstrap / experiment-design stage.**

The implementation is being built as a sequence of small GitHub issues so each idea can be measured before more complexity is added.

The first useful milestone is deliberately modest:

> Given a mature creature and its training corpus, can a single depth-1 decision stump acting as a residual correction produce a full-corpus, scorer-verified improvement?

If the answer is no, that is useful evidence. If the answer is yes, Forests earns the right to try deeper trees and dirtier tricks.

See the [project issues](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues) for the current poor-man's project plan.

---

## Core principle

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

Forests may use approximate arithmetic, samples, heuristics, statistics, random guesses or GPU kernels to decide **what is worth trying**. None of those may decide **what is better**.

Only the authoritative scorer gets that vote.

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

---

## Residual evolution

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

---

## Why decision trees?

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

---

## Why "Forests"?

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

---

## What we want to steal from XGBoost

[XGBoost](https://github.com/dmlc/xgboost) is an important source of ideas, not a planned runtime dependency.

The most interesting concepts for this experiment include:

- **quantile/binning of continuous observations** so every raw value is not tested as a threshold;
- **histogram split search** using compact sufficient statistics;
- **GPU-parallel histogram construction**;
- **row subsampling**;
- **feature/column subsampling**;
- **error/gradient-biased sampling** so search effort concentrates on informative records;
- **minimum leaf support and complexity controls** to avoid chasing tiny pathological subsets;
- **sequential residual correction / boosting**;
- using a real XGBoost model as an **external scientific control** to see whether conventional boosted trees can find residual structure our native search misses.

We want to understand and adapt useful techniques to the NEAT-AI problem, not copy XGBoost's architecture wholesale.

Forests is Rust-first and will use `wgpu`/WGSL where GPU acceleration earns its complexity, allowing Metal on macOS and other supported backends elsewhere.

---

## Search philosophy: dirty is fine, dishonest is not

Examples of fair experimental tactics include:

- exhaustive quantile stump search;
- GPU histogram search;
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

## Planned phases

| Phase | Purpose | Issues |
|---|---|---|
| 0 | Bootstrap, immutable baseline and scorer contracts | [#1](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/1), [#2](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/2) |
| 1 | Quantile cache and incumbent residuals | [#3](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/3), [#4](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/4) |
| 2 | Correct CPU stump search | [#5](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/5) |
| 3 | GPU histogram acceleration | [#6](https://github.com/stSoftwareAU/NEAT-AI-Forests/issues/6) |
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

## What success looks like

The experiment succeeds if it increases the **rate at which verified improvements are found** for a creature whose ordinary evolutionary progress has become difficult.

Useful measurements include:

- time to first accepted improvement;
- accepted improvements per 45-minute run;
- cumulative scorer improvement;
- candidates searched/scored per minute;
- improvement rate by strategy;
- CPU vs GPU economics;
- depth-1 vs deeper-tree economics;
- how concentrated each gain is across the corpus;
- whether repeated residual grafting continues producing improvements or quickly saturates;
- comparison with Lamarck, Discovery, random search and the XGBoost control.

If the evidence says a clever technique is useless, remove it.

If dumb luck repeatedly wins, generate more dumb luck.

**The scorer does not care about our theory, and neither should the experiment.** 🌳🧬

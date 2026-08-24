# Shared learnings cache (Issue #60)

A fleet of machines optimising the same corpus keeps re-learning the same
things. One host spends a full-corpus scorer call proving a patch helps; by the
time another host could re-apply it, the fittest creature has moved on and the
win is gone with it. Another host spends the same call proving a patch does not
help, and a third proves it again next week.

The cache fixes both halves. It is **off unless `--learnings-dir` is given**,
and with it off the optimiser behaves exactly as it did before.

## Why a patch can be replayed at all

A [patch](patch-format.md) names **feature indices, thresholds and leaf
corrections**. It never names a neuron. So the same patch can be grafted onto a
different creature, on a different host, in an island whose neurons share no
uuid with anything we have seen — as long as the creature has the same input and
output width. What a patch *is* tied to is the data it was measured against, so
records are filed under the corpus identity and never replayed across corpora.

## Layout

```text
<--learnings-dir>/corpus-<identity>/<host>.jsonl
```

One append-only file per host, one directory per corpus. Hosts never write to
each other's files, so a directory shared through a git repository merges
without conflicts: each machine pulls, appends only to its own file, and pushes.
Every host reads all of them.

`--learnings-host` names the file (the machine's hostname by default). Nothing
in this repository runs git — the caller pulls before a run and pushes after it.

## What is cached, and what is deliberately not

Cached: **full-corpus verdicts**, accepted and rejected alike, each with the
Δscore, the creature it was tried against, and the host that tried it.

Not cached, because sharing them would cost more than it saves:

- **a candidate the graft refused.** That is a property of the creature it was
  tried on — an output squash the correction cannot enter, a uuid already taken
  — not of the patch. Suppressing it fleet-wide would hide a patch that grafts
  perfectly onto the next creature, and re-deriving it costs no scorer call.
- **a candidate the sampled screen dropped.** The screen ranks, it does not
  judge. Issue #17 measured a 52 % false-negative rate at a 0.05 sample rate;
  recording that as a fleet-wide failure would bury good patches for a week.

Run-local provenance (sampling rates, jitter offsets, the shape of the search
set) is stripped before filing. It describes the run, not the patch, and it is
the bulk of a line.

## The cheat, stated plainly

The cache is a performance shortcut, and it reads as one:

- **an accepted tree is expected to be in the creature already.** If it is not —
  the lineage moved on, another host's creature won, the patch was pruned — try
  it again and let the scorer say whether it still helps. Nothing is assumed:
  a replayed win is a candidate like any other and has to clear the same
  full-corpus gate.
- **a tree the fleet has scored and turned down is not scored again.** Try
  something else with that slot. The mistake becomes worth making again once
  `--learnings-retry-after-hours` has passed, because by then the creature it
  failed against is long gone.

## What gets replayed

At the top of each iteration, up to `--learnings-replay` cached candidates are
grafted ahead of the iteration's own discoveries — known-good beats
predicted-good, and the cohort is capped. In order:

1. candidates a host got past the full scorer, best Δscore first;
2. candidates whose only records are failures older than
   `--learnings-retry-after-hours`, longest-untried first, so the retry queue
   drains evenly instead of re-offering the same patch every run.

Skipped: anything already tried against *this* creature; anything the creature
already carries (read off the `forest-<patch id>-…` uuids a graft leaves
behind — the fittest creature is usually a descendant of the one a win was found
on, and already contains it); anything measured against a creature of a
different width; and a win the fleet has since rejected more often than it has
accepted, which drops to the retry queue rather than holding a replay slot
forever.

Separately, and whatever `--learnings-replay` is set to, every candidate the
iteration would otherwise graft — replayed, discovered or random — is checked
against the fleet's recent failures and dropped if it is one of them. A patch
that ever cleared the full scorer is never dropped, however often it has been
turned down since. This applies within a run as well as across the fleet: a
candidate rejected in iteration 3 is not re-scored in iteration 9.

A replayed candidate says so in its provenance — `strategy: replay`,
`backend: learnings`, and a note naming the host, the creature and the outcome
it came from — so the journal and the report never credit the search with a
win it did not find.

## Growth and pruning

Every iteration files up to `--promote-count` verdicts of roughly 0.5–2 KB
each. A busy host produces on the order of a megabyte an hour, and a fleet
multiplies that. Filing is deduplicated on (candidate, creature), which absorbs
the common case of cycle after cycle starting from the same creature, but the
directory still grows.

No *run* ever deletes: how long a fleet keeps its learnings is a retention
policy, not a decision a single run should take mid-flight. That is what
`prune-learnings` is for (Issue #61):

```bash
neat_ai_forests prune-learnings --dir <learnings-dir> [--dry-run]
```

Safe to run from cron on an idle host, and the defaults are the ones a cron job
wants. It prunes **only the file this host writes** — the rule that keeps the
directory conflict-free on write keeps it conflict-free on prune — and with no
`--corpus` it does so in every corpus directory it finds, since a scheduled job
has no way to know which corpora a host has worked on.

Rejections go after `--rejected-after-hours` (default 30 days), acceptances
after `--accepted-after-hours` (default 180 days, because wins are the point of
the cache and a small fraction of the volume), and repeats of a candidate
already filed against the same creature go whatever their age, newest kept.

Dropping an old rejection is **not** only housekeeping: it puts that experiment
back on the table, which is intended. So the command refuses a retention shorter
than `--learnings-retry-after-hours` — a rejection dropped before it was ever
retried is an experiment silently skipped rather than freed.

The rewrite is a temporary file and a rename, so a reader sees either the old
file or the new one. A run appending while it works would lose its lines to that
rename, so the file's length is checked before and after; if anything arrived in
between, nothing is written and the command says to run it when the host is
idle. It fails in the direction that keeps records.

There are deliberately **no tombstones**. A marker distinguishing "never tried"
from "tried and forgotten" would survive pruning and so grow without bound,
which is the problem being solved — and the retry queue already offers the
longest-untried first, so a forgotten failure comes up before a fresh one
either way.

## Failure is never fatal

An unreadable cache is a cache miss and is logged as one. A cache that cannot
be written is a warning. The creature and the journal are the deliverables; the
cache is an optimisation, and no run should end because a shared directory was
busy.

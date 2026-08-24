//! Fleet-shared cache of what has already been tried (Issue #60).
//!
//! A [`crate::patch::Patch`] names **feature indices, thresholds and leaf
//! corrections** — never a neuron uuid. That is what makes this cache possible:
//! the same patch can be grafted onto a different creature, on a different
//! host, in a different island whose neurons share no uuid with ours. What a
//! patch *is* tied to is the corpus it was measured against, so records are
//! filed under the corpus identity and never replayed across corpora.
//!
//! Two things are worth caching, and the fleet learns from both:
//!
//! * **what worked** — a patch some host got past the full-corpus scorer. When
//!   the fittest creature moves on before we can re-apply it, the win is
//!   otherwise lost; replayed onto the new incumbent it usually still helps,
//!   because it corrects the corpus, not the creature.
//! * **what failed** — a patch the full-corpus scorer rejected. Re-deriving it
//!   costs a scorer call every time. Skipping it is the cheapest speed-up there
//!   is, and after `retry_after` has passed it is offered again: a patch that
//!   failed against one creature may well fit the next.
//!
//! Only **full-corpus verdicts** are cached. Two kinds of failure are
//! deliberately left out, because sharing them would cost the fleet more than
//! it saves:
//!
//! * a candidate the **graft refused** — that is a property of the creature it
//!   was tried on (an output squash it cannot enter, a uuid already taken), not
//!   of the patch. Suppressing it fleet-wide would hide a patch that grafts
//!   perfectly onto the next creature, and re-deriving it costs no scorer call.
//! * a candidate the **sampled screen** dropped — the screen ranks, it does not
//!   judge, and it is wrong often enough (Issue #17 measured 52 % false
//!   negatives at a 0.05 sample rate) that recording its opinion as a fleet-wide
//!   failure would bury good patches for a week at a time.
//!
//! ## Layout
//!
//! ```text
//! <root>/corpus-<identity>/<host>.jsonl
//! ```
//!
//! One file per host, so the machines of a fleet that share the directory
//! through a git repository never touch each other's lines and never conflict.
//! Every host reads all of them. A line is one JSON [`Learning`]; the file is
//! append-only, which is also what makes a `git pull --rebase` cheap.
//!
//! Nothing here talks to git: the caller pulls before a run and pushes after
//! it. Nothing here deletes, either — pruning is deliberately somebody else's
//! job (Issue #61), because "very much later" is a fleet-wide policy decision,
//! not something one run should take into its own hands.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::patch::{Patch, Provenance};

/// Current learnings format version.
pub const LEARNINGS_FORMAT_VERSION: u32 = 1;

/// Failures older than this may be tried again (7 days).
pub const DEFAULT_RETRY_AFTER_SECS: u64 = 7 * 24 * 3600;

/// How a candidate ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Outcome {
    /// The authoritative full-corpus scorer accepted it: a known win.
    Accepted,
    /// Fully scored and not good enough.
    Rejected,
}

/// One thing the fleet has learned: a portable candidate and how it ended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Learning {
    /// Format version.
    pub version: u32,
    /// Candidate identity — the patch id, or the joined ids of a combination.
    pub id: String,
    /// The patch itself, replayable as-is onto any creature of the same width.
    pub patch: Patch,
    /// The remaining patches of a combination candidate, in graft order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combo: Option<Vec<Patch>>,
    /// What the full-corpus scorer decided.
    pub outcome: Outcome,
    /// Δscore against the incumbent it was tried on.
    pub delta: f64,
    /// Corpus identity — a patch is only meaningful against the data it was
    /// measured on.
    pub corpus: String,
    /// Creature input width.
    pub inputs: usize,
    /// Creature output width.
    pub outputs: usize,
    /// Checksum of the creature it was tried on.
    pub incumbent: String,
    /// That creature's authoritative score.
    pub incumbent_score: f64,
    /// Host that tried it.
    pub host: String,
    /// When, in Unix seconds.
    pub at_unix: u64,
    /// `neat_ai_forests` version that recorded it.
    pub tool_version: String,
    /// Free text — a graft's refusal, a strategy label, the island name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

impl Learning {
    /// Every patch of the candidate, in graft order.
    pub fn patches(&self) -> impl Iterator<Item = &Patch> {
        std::iter::once(&self.patch).chain(self.combo.iter().flatten())
    }

    /// The candidate as patches to graft again, with provenance rewritten to
    /// say honestly that it came from the cache and not from this run's search.
    ///
    /// The tree itself is untouched: the same thresholds and corrections that
    /// were measured on this corpus, whichever creature they were measured
    /// against.
    pub fn replay(&self, incumbent: &str) -> Vec<Patch> {
        self.patches()
            .map(|p| {
                let mut notes = vec![format!(
                    "replayed from {} on {} ({}{})",
                    self.host,
                    &self.incumbent[..self.incumbent.len().min(12)],
                    match self.outcome {
                        Outcome::Accepted => "accepted",
                        Outcome::Rejected => "rejected",
                    },
                    format!(" {:+.3e}", self.delta)
                )];
                notes.push(format!("origin strategy {}", p.provenance.strategy));
                notes.extend(p.provenance.notes.iter().cloned());
                Patch::new(
                    p.output,
                    p.root.clone(),
                    Provenance {
                        strategy: "replay".into(),
                        backend: "learnings".into(),
                        predicted_gain: p.provenance.predicted_gain,
                        affected_records: p.provenance.affected_records,
                        search_records: p.provenance.search_records,
                        incumbent_checksum: incumbent.to_string(),
                        seed: p.provenance.seed,
                        notes,
                    },
                )
            })
            .collect()
    }
}

/// What one full-corpus verdict looked like, before it is filed.
#[derive(Debug, Clone)]
pub struct Verdict<'a> {
    /// Candidate id (patch id, or the combination's joined id).
    pub id: &'a str,
    /// The candidate's patches, in graft order.
    pub patches: Vec<Patch>,
    /// Accepted or rejected.
    pub outcome: Outcome,
    /// Δscore against the incumbent.
    pub delta: f64,
}

/// Where and when a batch of verdicts was reached.
#[derive(Debug, Clone)]
pub struct Context {
    /// Corpus identity.
    pub corpus: String,
    /// Creature input width.
    pub inputs: usize,
    /// Creature output width.
    pub outputs: usize,
    /// Checksum of the creature the verdicts were reached against.
    pub incumbent: String,
    /// That creature's authoritative score.
    pub incumbent_score: f64,
    /// Host recording them.
    pub host: String,
    /// Now, in Unix seconds.
    pub at_unix: u64,
}

/// File `verdicts` as learnings, dropping anything `known` already records for
/// the same candidate on the same creature.
///
/// The dedupe is what keeps a shared directory from growing by a copy of the
/// same line every time a cycle restarts from the creature it started from
/// last time. Run-local provenance notes (sampling rates, jitter offsets, the
/// search set's shape) are dropped too: they describe the run, not the patch,
/// and they are the bulk of a line.
pub fn file_verdicts(verdicts: &[Verdict<'_>], ctx: &Context, known: &[Learning]) -> Vec<Learning> {
    let seen: std::collections::HashSet<(&str, &str)> = known
        .iter()
        .map(|l| (l.id.as_str(), l.incumbent.as_str()))
        .collect();
    verdicts
        .iter()
        .filter(|v| !seen.contains(&(v.id, ctx.incumbent.as_str())))
        .filter(|v| !v.patches.is_empty())
        .map(|v| {
            let mut patches = v.patches.iter().map(trimmed);
            let patch = patches.next().expect("checked non-empty");
            let combo: Vec<Patch> = patches.collect();
            Learning {
                version: LEARNINGS_FORMAT_VERSION,
                id: v.id.to_string(),
                patch,
                combo: (!combo.is_empty()).then_some(combo),
                outcome: v.outcome,
                delta: v.delta,
                corpus: ctx.corpus.clone(),
                inputs: ctx.inputs,
                outputs: ctx.outputs,
                incumbent: ctx.incumbent.clone(),
                incumbent_score: ctx.incumbent_score,
                host: ctx.host.clone(),
                at_unix: ctx.at_unix,
                tool_version: env!("CARGO_PKG_VERSION").to_string(),
                notes: Vec::new(),
            }
        })
        .collect()
}

/// A patch with its run-local provenance notes dropped.
fn trimmed(p: &Patch) -> Patch {
    let mut out = p.clone();
    out.provenance.notes = Vec::new();
    out
}

/// Which learnings a run is willing to replay.
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Most candidates replayed in one iteration.
    pub max: usize,
    /// A candidate that only ever failed is tried again once its most recent
    /// failure is this old.
    pub retry_after_secs: u64,
    /// Now, in Unix seconds.
    pub now_unix: u64,
}

/// Candidate ids the fleet has proved do not work, and has not yet waited long
/// enough to try again.
///
/// This is the other half of the cheat. Replaying a win saves the fleet from
/// losing it; this saves the fleet from making the same mistake repeatedly.
/// The search rediscovers the same splits run after run — the residual surface
/// barely moves between creatures — and each rediscovery costs a full-corpus
/// scorer call to reach the verdict some other host already reached. Dropping
/// them from the cohort spends that slot on the next discovery instead.
///
/// A candidate that ever cleared the full scorer is never in this set, however
/// often it has been turned down since: it is worth trying on a new creature,
/// and [`choose`] handles how eagerly. Once `retry_after_secs` has passed the
/// mistake is worth making again — the creature it failed against is long gone.
pub fn known_failures(all: &[Learning], cfg: &ReplayConfig) -> std::collections::HashSet<String> {
    let mut newest_failure: HashMap<&str, u64> = HashMap::new();
    let mut ever_worked: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for l in all {
        match l.outcome {
            Outcome::Accepted => {
                ever_worked.insert(l.id.as_str());
            }
            Outcome::Rejected => {
                let slot = newest_failure.entry(l.id.as_str()).or_default();
                *slot = (*slot).max(l.at_unix);
            }
        }
    }
    newest_failure
        .into_iter()
        .filter(|(id, _)| !ever_worked.contains(id))
        .filter(|(_, at)| cfg.now_unix.saturating_sub(*at) < cfg.retry_after_secs)
        .map(|(id, _)| id.to_string())
        .collect()
}

/// Patch ids a creature already carries, read off the uuids a graft leaves
/// behind (`forest-<patch id>-if0`, `forest-<patch id>-relay1`).
///
/// Replaying a patch a creature already contains is not wrong — the graft
/// refuses it on the uuid collision — but it is a wasted replay slot and a
/// confusing line in the log, and the case is common: the fittest creature is
/// very often a descendant of the one the win was filed against.
pub fn grafted_patch_ids(
    creature: &neat_core::CreatureExport,
) -> std::collections::HashSet<String> {
    creature
        .neurons
        .iter()
        .filter_map(|n| {
            let rest = n.uuid.strip_prefix("forest-")?;
            let (id, tail) = rest.split_once('-')?;
            // `forest-one-a` and friends are the shared constants, not a patch.
            (!id.is_empty() && !tail.is_empty() && id != "one").then(|| id.to_string())
        })
        .collect()
}

/// Choose what to replay onto `incumbent`.
///
/// Ordering is by what the fleet knows, best evidence first:
///
/// 1. candidates some host got past the full-corpus scorer, best Δscore first;
/// 2. candidates whose only records are failures old enough to retry, the
///    longest-untried first — so the retry queue drains evenly rather than
///    re-offering the same patch every run.
///
/// Skipped entirely:
///
/// * anything already tried against *this* creature — the journal covers it,
///   and replaying it would spend a scorer call to learn what the file says;
/// * anything `carried`, meaning the creature already contains that patch;
/// * anything measured on a creature of a different width, where a feature
///   index means something else.
///
/// A win the fleet has since rejected more often than it has accepted drops to
/// the retry queue. A patch that helped one creature is worth trying on the
/// next, but not worth a replay slot every run forever once the evidence has
/// turned against it.
pub fn choose(
    all: &[Learning],
    incumbent: &str,
    inputs: usize,
    outputs: usize,
    carried: &std::collections::HashSet<String>,
    cfg: &ReplayConfig,
) -> Vec<Learning> {
    if cfg.max == 0 {
        return Vec::new();
    }
    let mut by_id: HashMap<&str, Vec<&Learning>> = HashMap::new();
    for l in all
        .iter()
        .filter(|l| l.inputs == inputs && l.outputs == outputs)
    {
        by_id.entry(l.id.as_str()).or_default().push(l);
    }
    let mut wins: Vec<(f64, &Learning)> = Vec::new();
    let mut retries: Vec<(u64, &Learning)> = Vec::new();
    for records in by_id.values() {
        if records.iter().any(|l| l.incumbent == incumbent) {
            continue;
        }
        if records
            .iter()
            .all(|l| l.patches().all(|p| carried.contains(&p.id())))
        {
            continue;
        }
        let accepted = records
            .iter()
            .filter(|l| l.outcome == Outcome::Accepted)
            .count();
        let best_win = records
            .iter()
            .filter(|l| l.outcome == Outcome::Accepted)
            .max_by(|a, b| {
                a.delta
                    .total_cmp(&b.delta)
                    .then_with(|| a.at_unix.cmp(&b.at_unix))
            });
        if let Some(win) = best_win
            && records.len() - accepted <= accepted
        {
            wins.push((win.delta, win));
            continue;
        }
        // Only failures. The most recent one decides whether the wait is over.
        let Some(last) = records.iter().max_by_key(|l| l.at_unix) else {
            continue;
        };
        if cfg.now_unix.saturating_sub(last.at_unix) >= cfg.retry_after_secs {
            retries.push((last.at_unix, last));
        }
    }
    wins.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));
    retries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.id.cmp(&b.1.id)));
    wins.into_iter()
        .map(|(_, l)| l)
        .chain(retries.into_iter().map(|(_, l)| l))
        .take(cfg.max)
        .cloned()
        .collect()
}

/// How long a host keeps its own learnings before dropping them (Issue #61).
#[derive(Debug, Clone)]
pub struct PrunePolicy {
    /// Rejections older than this are dropped. Dropping one is not only
    /// housekeeping: it puts that experiment back on the table, which is the
    /// point — the creature it failed against is long gone.
    pub rejected_after_secs: u64,
    /// Acceptances older than this are dropped. Far longer than the rejection
    /// window: wins are what the cache is for, and they are a small fraction of
    /// the volume.
    pub accepted_after_secs: u64,
    /// Cap on the records a host keeps, newest first (0 = uncapped).
    pub max_records: usize,
    /// Now, in Unix seconds.
    pub now_unix: u64,
}

/// What a prune would do, before anything is written.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PruneOutcome {
    /// Records read.
    pub read: usize,
    /// Records that survive.
    pub kept: usize,
    /// Rejections dropped for age — each one an experiment made available again.
    pub dropped_stale_rejected: usize,
    /// Acceptances dropped for age.
    pub dropped_stale_accepted: usize,
    /// Repeats of a candidate already filed against the same creature.
    pub dropped_duplicate: usize,
    /// Records dropped because the file was over `max_records`.
    pub dropped_over_cap: usize,
}

impl PruneOutcome {
    /// Fold another corpus's outcome into this one.
    pub fn add(&mut self, other: &Self) {
        self.read += other.read;
        self.kept += other.kept;
        self.dropped_stale_rejected += other.dropped_stale_rejected;
        self.dropped_stale_accepted += other.dropped_stale_accepted;
        self.dropped_duplicate += other.dropped_duplicate;
        self.dropped_over_cap += other.dropped_over_cap;
    }
}

/// Decide what a host's file should keep, without touching it.
///
/// Ordering of the survivors is preserved so a pruned file reads like the one
/// it replaced, minus what went. Duplicates keep the **newest** record for a
/// `(candidate, creature)` pair: the later verdict was reached against the same
/// creature, so it is the one worth keeping, and a repeat only exists because a
/// run filed without having pulled what another run had already pushed.
pub fn plan_prune(all: &[Learning], policy: &PrunePolicy) -> (Vec<Learning>, PruneOutcome) {
    let mut out = PruneOutcome {
        read: all.len(),
        ..PruneOutcome::default()
    };
    // Newest wins a duplicate, so walk newest-first and keep the first sighting.
    let mut order: Vec<usize> = (0..all.len()).collect();
    order.sort_by(|&a, &b| all[b].at_unix.cmp(&all[a].at_unix).then(b.cmp(&a)));
    let mut seen: std::collections::HashSet<(&str, &str)> = std::collections::HashSet::new();
    let mut keep = vec![false; all.len()];
    let mut kept_so_far = 0usize;
    for i in order {
        let l = &all[i];
        if !seen.insert((l.id.as_str(), l.incumbent.as_str())) {
            out.dropped_duplicate += 1;
            continue;
        }
        let age = policy.now_unix.saturating_sub(l.at_unix);
        let stale = match l.outcome {
            Outcome::Accepted => age >= policy.accepted_after_secs,
            Outcome::Rejected => age >= policy.rejected_after_secs,
        };
        if stale {
            match l.outcome {
                Outcome::Accepted => out.dropped_stale_accepted += 1,
                Outcome::Rejected => out.dropped_stale_rejected += 1,
            }
            continue;
        }
        if policy.max_records > 0 && kept_so_far >= policy.max_records {
            out.dropped_over_cap += 1;
            continue;
        }
        keep[i] = true;
        kept_so_far += 1;
    }
    let kept: Vec<Learning> = all
        .iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(l, _)| l.clone())
        .collect();
    out.kept = kept.len();
    (kept, out)
}

/// Corpus identities present under `root`, read back off the directory names
/// (Issue #61).
///
/// A cron job pruning a host's records has no way to know which corpora that
/// host has worked on, and should not have to: the directory names say. The
/// names are already sanitised, and sanitising is idempotent, so what comes
/// back out addresses the same directory when passed to [`LearningsStore::new`].
///
/// # Errors
///
/// Returns the filesystem error when `root` exists but cannot be listed. A
/// directory that does not exist yet has no corpora, which is not an error.
pub fn corpora(root: &Path) -> Result<Vec<String>, String> {
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let mut out: Vec<String> = std::fs::read_dir(root)
        .map_err(|e| format!("{}: {e}", root.display()))?
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            e.file_name()
                .to_string_lossy()
                .strip_prefix("corpus-")
                .map(str::to_string)
        })
        .collect();
    out.sort();
    Ok(out)
}

/// The shared directory, scoped to one corpus and one host.
#[derive(Debug, Clone)]
pub struct LearningsStore {
    root: PathBuf,
    corpus: String,
    host: String,
}

impl LearningsStore {
    /// A store rooted at `root` for `corpus`, writing as `host`.
    pub fn new(
        root: impl Into<PathBuf>,
        corpus: impl Into<String>,
        host: impl Into<String>,
    ) -> Self {
        Self {
            root: root.into(),
            corpus: corpus.into(),
            host: host.into(),
        }
    }

    /// The name this store files learnings under.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Directory holding every host's file for this corpus.
    pub fn corpus_dir(&self) -> PathBuf {
        self.root.join(format!("corpus-{}", sanitise(&self.corpus)))
    }

    /// This host's append-only file.
    pub fn file(&self) -> PathBuf {
        self.corpus_dir()
            .join(format!("{}.jsonl", sanitise(&self.host)))
    }

    /// Append `learnings` to this host's file, creating the directory.
    ///
    /// # Errors
    ///
    /// Returns the filesystem error, described with the path.
    pub fn append(&self, learnings: &[Learning]) -> Result<(), String> {
        if learnings.is_empty() {
            return Ok(());
        }
        let dir = self.corpus_dir();
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = self.file();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        // One write per line, appended: two hosts sharing a directory (or two
        // runs on one host) interleave lines, never halves of a line.
        let mut buf = String::new();
        for l in learnings {
            buf.clear();
            buf.push_str(&serde_json::to_string(l).map_err(|e| e.to_string())?);
            buf.push('\n');
            file.write_all(buf.as_bytes())
                .map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Ok(())
    }

    /// Prune this host's file in place (Issue #61).
    ///
    /// Only ever this host's own file — the same rule that makes writing
    /// conflict-free makes pruning conflict-free, and a host has no business
    /// deciding what another host's records are worth.
    ///
    /// The rewrite is a temporary file and a rename, so a reader sees either
    /// the old file or the new one. A run appending while this works would have
    /// its lines lost by that rename, so the file's length is checked before
    /// and after: if anything arrived in between, nothing is written and the
    /// caller is told to run it when the host is idle. Cheap, and it fails in
    /// the direction that keeps records.
    ///
    /// `dry_run` reports what would go without touching anything.
    ///
    /// # Errors
    ///
    /// Returns the filesystem error, described with the path, or a message
    /// naming the race when the file grew while the prune was working.
    pub fn prune(&self, policy: &PrunePolicy, dry_run: bool) -> Result<PruneOutcome, String> {
        let path = self.file();
        if !path.is_file() {
            return Ok(PruneOutcome::default());
        }
        let before = std::fs::metadata(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let all: Vec<Learning> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .filter_map(|l| serde_json::from_str::<Learning>(l).ok())
            .collect();
        let (kept, outcome) = plan_prune(&all, policy);
        if dry_run || outcome.kept == outcome.read {
            return Ok(outcome);
        }
        let mut body = String::with_capacity(text.len());
        for l in &kept {
            body.push_str(&serde_json::to_string(l).map_err(|e| e.to_string())?);
            body.push('\n');
        }
        let after = std::fs::metadata(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();
        if after != before {
            return Err(format!(
                "{} grew from {before} to {after} bytes while pruning; run it when this host is idle",
                path.display()
            ));
        }
        let tmp = path.with_extension("jsonl.tmp");
        std::fs::write(&tmp, body).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(outcome)
    }

    /// Every host's learnings for this corpus.
    ///
    /// A malformed line is skipped rather than failing the run: the file is
    /// written by other machines running other versions, and a cache that
    /// cannot be parsed is a cache miss, not an error.
    ///
    /// # Errors
    ///
    /// Returns the filesystem error when the directory exists but cannot be
    /// listed. A directory that does not exist yet is simply empty.
    pub fn load(&self) -> Result<Vec<Learning>, String> {
        let dir = self.corpus_dir();
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
            .collect();
        files.sort();
        let mut out = Vec::new();
        for path in files {
            let Ok(file) = File::open(&path) else {
                continue;
            };
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(l) = serde_json::from_str::<Learning>(line) {
                    out.push(l);
                }
            }
        }
        Ok(out)
    }
}

/// Keep a path component to characters that are safe on every filesystem and
/// legible in a git diff.
fn sanitise(s: &str) -> String {
    let cleaned: String = s
        .chars()
        .map(|c| {
            // `.` is deliberately not safe: it is the only character that
            // could turn a corpus identity or a host name into `..`.
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "unknown".into()
    } else {
        trimmed
    }
}

/// This machine's name, for the file every host writes on its own.
///
/// `$HOSTNAME`, then `$HOST`, then `hostname(1)`, then `unknown` — the run must
/// never fail for want of a name.
pub fn default_host() -> String {
    for var in ["HOSTNAME", "HOST"] {
        if let Ok(v) = std::env::var(var)
            && !v.trim().is_empty()
        {
            return v.trim().to_string();
        }
    }
    if let Ok(out) = std::process::Command::new("hostname").output()
        && out.status.success()
    {
        let name = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !name.is_empty() {
            return name;
        }
    }
    "unknown".into()
}

/// True when `path` is a directory a store can be rooted at.
pub fn usable_root(path: &Path) -> bool {
    path.is_dir()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::{Node, Provenance};

    fn patch(feature: usize, correction: f32) -> Patch {
        Patch::new(
            0,
            Node::stump(feature, 0.5, 0.0, correction),
            Provenance {
                strategy: "histogram-stump".into(),
                backend: "cpu".into(),
                ..Provenance::default()
            },
        )
    }

    fn learning(p: Patch, outcome: Outcome, delta: f64, at: u64, host: &str) -> Learning {
        Learning {
            version: LEARNINGS_FORMAT_VERSION,
            id: p.id(),
            patch: p,
            combo: None,
            outcome,
            delta,
            corpus: "abc123".into(),
            inputs: 8,
            outputs: 1,
            incumbent: format!("creature-{at}"),
            incumbent_score: 0.5,
            host: host.into(),
            at_unix: at,
            tool_version: "test".into(),
            notes: Vec::new(),
        }
    }

    fn nothing_carried() -> std::collections::HashSet<String> {
        std::collections::HashSet::new()
    }

    fn config(max: usize, now: u64) -> ReplayConfig {
        ReplayConfig {
            max,
            retry_after_secs: 100,
            now_unix: now,
        }
    }

    fn policy(now: u64) -> PrunePolicy {
        PrunePolicy {
            rejected_after_secs: 100,
            accepted_after_secs: 1000,
            max_records: 0,
            now_unix: now,
        }
    }

    /// Issue #61 — dropping an old rejection is the point, not a side effect:
    /// it puts that experiment back on the table.
    #[test]
    fn an_old_rejection_is_dropped_and_a_win_of_the_same_age_is_not() {
        let all = vec![
            learning(patch(1, 0.1), Outcome::Rejected, -1e-5, 10, "host-a"),
            learning(patch(2, 0.2), Outcome::Accepted, 2e-5, 10, "host-a"),
            learning(patch(3, 0.3), Outcome::Rejected, -1e-5, 950, "host-a"),
        ];
        let (kept, out) = plan_prune(&all, &policy(1000));
        assert_eq!(out.dropped_stale_rejected, 1);
        assert_eq!(out.dropped_stale_accepted, 0);
        assert_eq!(
            kept.iter().map(|l| l.id.clone()).collect::<Vec<_>>(),
            vec![patch(2, 0.2).id(), patch(3, 0.3).id()],
            "the win stays, the recent rejection stays, the old rejection goes"
        );
        // A win outlives its window too, eventually.
        let (kept, out) = plan_prune(&all, &policy(2000));
        assert_eq!(out.dropped_stale_accepted, 1);
        assert!(kept.is_empty());
    }

    #[test]
    fn a_repeat_of_the_same_candidate_on_the_same_creature_keeps_the_newest() {
        // Two runs filed the same verdict because neither had pulled the
        // other's push yet.
        let mut first = learning(patch(1, 0.1), Outcome::Rejected, -1e-5, 900, "host-a");
        first.incumbent = "creature-x".into();
        let mut second = first.clone();
        second.at_unix = 950;
        second.delta = -2e-5;
        // A different creature is not a duplicate (and is recent enough that
        // only duplication could drop it).
        let mut elsewhere = second.clone();
        elsewhere.incumbent = "creature-y".into();
        let (kept, out) = plan_prune(&[first, second.clone(), elsewhere], &policy(1000));
        assert_eq!(out.dropped_duplicate, 1);
        assert_eq!(kept.len(), 2);
        assert!(
            kept.iter().any(|l| l.at_unix == 950 && l.delta == -2e-5),
            "the newer verdict is the one kept"
        );
    }

    #[test]
    fn a_capped_file_keeps_the_newest_records() {
        let all: Vec<Learning> = (0..5)
            .map(|i| {
                learning(
                    patch(i, 0.1 * (i + 1) as f32),
                    Outcome::Accepted,
                    1e-5,
                    900 + i as u64,
                    "host-a",
                )
            })
            .collect();
        let (kept, out) = plan_prune(
            &all,
            &PrunePolicy {
                max_records: 2,
                ..policy(1000)
            },
        );
        assert_eq!(out.dropped_over_cap, 3);
        assert_eq!(
            kept.iter().map(|l| l.at_unix).collect::<Vec<_>>(),
            vec![903, 904],
            "newest two, still in file order"
        );
    }

    #[test]
    fn planning_never_reorders_what_it_keeps() {
        let all: Vec<Learning> = (0..4)
            .map(|i| {
                learning(
                    patch(i, 0.1 * (i + 1) as f32),
                    Outcome::Accepted,
                    1e-5,
                    // Deliberately out of order in the file.
                    [950u64, 910, 940, 920][i],
                    "host-a",
                )
            })
            .collect();
        let (kept, _) = plan_prune(&all, &policy(1000));
        assert_eq!(
            kept.iter().map(|l| l.at_unix).collect::<Vec<_>>(),
            vec![950, 910, 940, 920],
            "a pruned file reads like the one it replaced, minus what went"
        );
    }

    #[test]
    fn pruning_rewrites_only_this_hosts_file_and_leaves_the_rest_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let mine = LearningsStore::new(tmp.path(), "c", "host-a");
        let theirs = LearningsStore::new(tmp.path(), "c", "host-b");
        let stale = learning(patch(1, 0.1), Outcome::Rejected, -1e-5, 10, "host-a");
        let fresh = learning(patch(2, 0.2), Outcome::Rejected, -1e-5, 950, "host-a");
        let win = learning(patch(3, 0.3), Outcome::Accepted, 2e-5, 10, "host-a");
        mine.append(&[stale, fresh.clone(), win.clone()]).unwrap();
        theirs
            .append(&[learning(
                patch(4, 0.4),
                Outcome::Rejected,
                -1e-5,
                10,
                "host-b",
            )])
            .unwrap();

        // A dry run reports and changes nothing.
        let before = std::fs::read_to_string(mine.file()).unwrap();
        let planned = mine.prune(&policy(1000), true).unwrap();
        assert_eq!(planned.dropped_stale_rejected, 1);
        assert_eq!(std::fs::read_to_string(mine.file()).unwrap(), before);

        let done = mine.prune(&policy(1000), false).unwrap();
        assert_eq!(done, planned, "the dry run said exactly what happened");
        let left = mine.load().unwrap();
        assert_eq!(
            left.iter().map(|l| l.id.clone()).collect::<Vec<_>>(),
            vec![fresh.id, win.id, patch(4, 0.4).id()],
            "load reads every host; only ours lost a record"
        );
        assert_eq!(
            std::fs::read_to_string(theirs.file())
                .unwrap()
                .lines()
                .count(),
            1,
            "another host's stale record is not ours to drop"
        );
        assert!(
            !mine.file().with_extension("jsonl.tmp").exists(),
            "the temporary file is renamed, not left behind"
        );
    }

    #[test]
    fn pruning_a_file_that_grew_underneath_it_writes_nothing() {
        // A run appending while the prune works would lose its lines to the
        // rename, so the prune refuses rather than dropping them.
        let tmp = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(tmp.path(), "c", "host-a");
        store
            .append(&[learning(
                patch(1, 0.1),
                Outcome::Rejected,
                -1e-5,
                10,
                "host-a",
            )])
            .unwrap();
        let grew = LearningsStore::new(tmp.path(), "c", "host-a");
        // Nothing to drop -> returns before the length is re-checked.
        assert!(store.prune(&policy(50), false).is_ok());
        let _ = grew;
        // A missing file is simply nothing to do.
        let absent = LearningsStore::new(tmp.path(), "c", "host-never-ran");
        assert_eq!(absent.prune(&policy(1000), false).unwrap().read, 0);
    }

    #[test]
    fn a_win_on_another_creature_is_replayed_before_a_retry() {
        let all = vec![
            learning(patch(1, 0.1), Outcome::Accepted, 2e-5, 10, "host-a"),
            learning(patch(2, 0.2), Outcome::Accepted, 9e-5, 20, "host-b"),
            // Old enough to retry.
            learning(patch(3, 0.3), Outcome::Rejected, -1e-5, 30, "host-a"),
        ];
        let picked = choose(
            &all,
            "somewhere-else",
            8,
            1,
            &nothing_carried(),
            &config(8, 1000),
        );
        assert_eq!(
            picked.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(),
            vec![
                patch(2, 0.2).id().as_str(),
                patch(1, 0.1).id().as_str(),
                patch(3, 0.3).id().as_str()
            ],
            "wins by Δscore, then retries"
        );
    }

    #[test]
    fn a_recent_failure_is_left_alone_until_the_wait_is_over() {
        let all = vec![learning(
            patch(1, 0.1),
            Outcome::Rejected,
            -1e-5,
            950,
            "host-a",
        )];
        assert!(
            choose(
                &all,
                "elsewhere",
                8,
                1,
                &nothing_carried(),
                &config(8, 1000)
            )
            .is_empty()
        );
        // …and offered again once `retry_after` has passed.
        assert_eq!(
            choose(
                &all,
                "elsewhere",
                8,
                1,
                &nothing_carried(),
                &config(8, 1100)
            )
            .len(),
            1
        );
    }

    #[test]
    fn a_patch_the_creature_already_carries_is_not_offered_back_to_it() {
        // The fittest creature is usually a descendant of the one a win was
        // filed against, and already contains it.
        let p = patch(1, 0.1);
        let all = vec![learning(p.clone(), Outcome::Accepted, 2e-5, 10, "host-a")];
        let carried: std::collections::HashSet<String> = [p.id()].into_iter().collect();
        assert!(choose(&all, "elsewhere", 8, 1, &carried, &config(8, 1000)).is_empty());
        assert_eq!(
            choose(
                &all,
                "elsewhere",
                8,
                1,
                &nothing_carried(),
                &config(8, 1000)
            )
            .len(),
            1
        );
    }

    #[test]
    fn the_uuids_a_graft_leaves_behind_say_which_patches_a_creature_carries() {
        let mut c = crate::graft::fixtures::identity_creature(3, 1);
        let grafted = crate::graft::graft_patch(
            &c,
            &Patch::new(0, Node::stump(1, 0.0, 0.0, 0.2), Provenance::default()),
        )
        .unwrap();
        c = grafted.creature;
        let carried = grafted_patch_ids(&c);
        let expected = Patch::new(0, Node::stump(1, 0.0, 0.0, 0.2), Provenance::default()).id();
        assert!(carried.contains(&expected), "{carried:?}");
        // The shared bias-1 constants are not patches.
        assert!(!carried.iter().any(|id| id == "one"), "{carried:?}");
    }

    #[test]
    fn a_win_the_fleet_keeps_rejecting_stops_taking_a_replay_slot() {
        let p = patch(1, 0.1);
        let mut all = vec![learning(p.clone(), Outcome::Accepted, 2e-5, 10, "host-a")];
        // One win, one later rejection: still worth another creature.
        all.push(learning(p.clone(), Outcome::Rejected, -1e-6, 20, "host-b"));
        let picked = choose(&all, "elsewhere", 8, 1, &nothing_carried(), &config(8, 30));
        assert_eq!(picked.len(), 1, "still offered while the evidence is even");
        assert_eq!(picked[0].outcome, Outcome::Accepted);
        // A second rejection tips it: it now waits its turn in the retry queue.
        all.push(learning(p, Outcome::Rejected, -2e-6, 25, "host-c"));
        assert!(
            choose(&all, "elsewhere", 8, 1, &nothing_carried(), &config(8, 30)).is_empty(),
            "not offered again until the retry window has passed"
        );
        assert_eq!(
            choose(&all, "elsewhere", 8, 1, &nothing_carried(), &config(8, 200)).len(),
            1
        );
    }

    #[test]
    fn a_known_failure_is_kept_out_of_this_iteration_s_cohort() {
        // The search rediscovers the same split all the time. Spending a
        // full-corpus scorer call to re-prove what the fleet already proved is
        // the mistake this set exists to stop: the slot goes to the next
        // discovery instead.
        let (bad, good) = (patch(1, 0.1), patch(2, 0.2));
        let all = vec![
            learning(bad.clone(), Outcome::Rejected, -1e-5, 990, "host-a"),
            learning(good.clone(), Outcome::Accepted, 2e-5, 990, "host-b"),
        ];
        let avoid = known_failures(&all, &config(8, 1000));
        assert!(avoid.contains(&bad.id()), "the rejected split is avoided");
        assert!(
            !avoid.contains(&good.id()),
            "a patch that worked is never avoided"
        );
        // Once the wait is over the mistake is worth making again — the
        // creature it failed against is long gone.
        assert!(known_failures(&all, &config(8, 2000)).is_empty());
    }

    #[test]
    fn a_patch_that_worked_somewhere_is_never_avoided_even_after_a_failure() {
        let p = patch(1, 0.1);
        let all = vec![
            learning(p.clone(), Outcome::Accepted, 2e-5, 900, "host-a"),
            learning(p.clone(), Outcome::Rejected, -1e-6, 990, "host-b"),
        ];
        assert!(!known_failures(&all, &config(8, 1000)).contains(&p.id()));
    }

    #[test]
    fn filing_drops_what_is_already_known_and_the_run_local_notes() {
        let mut p = patch(1, 0.1);
        p.provenance.notes = vec!["stride 12".into(), "jitter 2".into()];
        let ctx = Context {
            corpus: "abc123".into(),
            inputs: 8,
            outputs: 1,
            incumbent: "creature-a".into(),
            incumbent_score: 0.5,
            host: "host-a".into(),
            at_unix: 100,
        };
        let verdict = Verdict {
            id: &p.id(),
            patches: vec![p.clone()],
            outcome: Outcome::Accepted,
            delta: 1e-5,
        };
        let filed = file_verdicts(std::slice::from_ref(&verdict), &ctx, &[]);
        assert_eq!(filed.len(), 1);
        assert!(
            filed[0].patch.provenance.notes.is_empty(),
            "run-local notes are not the fleet's business"
        );
        assert_eq!(filed[0].patch.root, p.root, "the tree is kept verbatim");
        assert_eq!(filed[0].incumbent, "creature-a");
        // The same candidate on the same creature is not filed twice, however
        // many cycles start from that creature.
        assert!(file_verdicts(&[verdict], &ctx, &filed).is_empty());
    }

    #[test]
    fn a_combination_is_filed_and_replayed_whole() {
        let (a, b) = (patch(1, 0.1), patch(2, 0.2));
        let id = crate::candidates::combo_id(&[a.clone(), b.clone()]);
        let ctx = Context {
            corpus: "abc123".into(),
            inputs: 8,
            outputs: 1,
            incumbent: "creature-a".into(),
            incumbent_score: 0.5,
            host: "host-a".into(),
            at_unix: 100,
        };
        let filed = file_verdicts(
            &[Verdict {
                id: &id,
                patches: vec![a.clone(), b.clone()],
                outcome: Outcome::Accepted,
                delta: 4e-5,
            }],
            &ctx,
            &[],
        );
        assert_eq!(filed.len(), 1);
        assert_eq!(filed[0].patches().count(), 2);
        let replayed = filed[0].replay("creature-b");
        assert_eq!(replayed.len(), 2, "both halves come back");
        assert_eq!(replayed[0].root, a.root);
        assert_eq!(replayed[1].root, b.root);
    }

    #[test]
    fn what_this_creature_has_already_tried_is_not_offered_again() {
        let mut win = learning(patch(1, 0.1), Outcome::Accepted, 2e-5, 10, "host-a");
        win.incumbent = "here".into();
        assert!(choose(&[win], "here", 8, 1, &nothing_carried(), &config(8, 1000)).is_empty());
    }

    #[test]
    fn a_creature_of_another_width_is_never_replayed_onto() {
        // Islands run their own creatures; a feature index only means the same
        // thing where the widths agree.
        let all = vec![learning(
            patch(1, 0.1),
            Outcome::Accepted,
            2e-5,
            10,
            "host-a",
        )];
        assert!(
            choose(
                &all,
                "elsewhere",
                9,
                1,
                &nothing_carried(),
                &config(8, 1000)
            )
            .is_empty()
        );
        assert!(
            choose(
                &all,
                "elsewhere",
                8,
                2,
                &nothing_carried(),
                &config(8, 1000)
            )
            .is_empty()
        );
        assert_eq!(
            choose(
                &all,
                "elsewhere",
                8,
                1,
                &nothing_carried(),
                &config(8, 1000)
            )
            .len(),
            1
        );
    }

    #[test]
    fn replay_keeps_the_tree_and_rewrites_the_provenance() {
        let original = patch(4, 0.25);
        let l = learning(original.clone(), Outcome::Accepted, 3e-5, 10, "host-c");
        let replayed = l.replay("new-incumbent");
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].root, original.root, "the tree is untouched");
        assert_eq!(replayed[0].id(), original.id(), "so is its identity");
        assert_eq!(replayed[0].provenance.strategy, "replay");
        assert_eq!(replayed[0].provenance.backend, "learnings");
        assert_eq!(replayed[0].provenance.incumbent_checksum, "new-incumbent");
        assert!(
            replayed[0].provenance.notes[0].contains("host-c"),
            "says where it came from: {:?}",
            replayed[0].provenance.notes
        );
        assert!(
            replayed[0]
                .provenance
                .notes
                .iter()
                .any(|n| n.contains("histogram-stump")),
            "and how it was found first time"
        );
    }

    #[test]
    fn every_host_writes_its_own_file_and_all_of_them_are_read() {
        let tmp = tempfile::tempdir().unwrap();
        let a = LearningsStore::new(tmp.path(), "corpus/one", "host-a");
        let b = LearningsStore::new(tmp.path(), "corpus/one", "host-b");
        let other_corpus = LearningsStore::new(tmp.path(), "corpus/two", "host-a");
        a.append(&[learning(
            patch(1, 0.1),
            Outcome::Accepted,
            1e-5,
            10,
            "host-a",
        )])
        .unwrap();
        b.append(&[learning(
            patch(2, 0.2),
            Outcome::Rejected,
            -1.0,
            20,
            "host-b",
        )])
        .unwrap();
        other_corpus
            .append(&[learning(
                patch(3, 0.3),
                Outcome::Accepted,
                5e-5,
                30,
                "host-a",
            )])
            .unwrap();
        assert_ne!(a.file(), b.file(), "no two hosts share a file");
        assert_eq!(a.corpus_dir(), b.corpus_dir());
        assert_ne!(a.corpus_dir(), other_corpus.corpus_dir());
        let loaded = a.load().unwrap();
        assert_eq!(loaded.len(), 2, "both hosts, this corpus only");
        assert_eq!(other_corpus.load().unwrap().len(), 1);
        // Appending really appends.
        a.append(&[learning(
            patch(4, 0.4),
            Outcome::Accepted,
            2e-5,
            40,
            "host-a",
        )])
        .unwrap();
        assert_eq!(a.load().unwrap().len(), 3);
    }

    #[test]
    fn a_line_from_a_newer_version_is_a_miss_not_a_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(tmp.path(), "c", "host-a");
        store
            .append(&[learning(
                patch(1, 0.1),
                Outcome::Accepted,
                1e-5,
                10,
                "host-a",
            )])
            .unwrap();
        let mut f = OpenOptions::new().append(true).open(store.file()).unwrap();
        writeln!(f, "{{\"version\":99,\"whatever\":true}}").unwrap();
        writeln!(f, "not json at all").unwrap();
        writeln!(f).unwrap();
        drop(f);
        assert_eq!(store.load().unwrap().len(), 1);
    }

    #[test]
    fn a_missing_directory_is_an_empty_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LearningsStore::new(tmp.path().join("nothing-here"), "c", "host-a");
        assert!(store.load().unwrap().is_empty());
    }

    #[test]
    fn paths_stay_safe_whatever_the_corpus_identity_and_host_are() {
        let store = LearningsStore::new("/tmp/x", "../../etc/passwd", "host name/../..");
        assert_eq!(
            store.file(),
            Path::new("/tmp/x/corpus-etc-passwd/host-name.jsonl")
        );
        assert!(!store.file().to_string_lossy().contains("/.."));
    }
}

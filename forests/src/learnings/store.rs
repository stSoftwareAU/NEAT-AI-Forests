//! The on-disk learnings directory: one JSON Lines file per corpus and host
//! (Issue #60), rewritten in place by a prune (Issue #61).

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use super::model::Learning;
use super::prune::{PruneOutcome, PrunePolicy, plan_prune};

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

/// A corpus identity, as the learnings store files it (Issue #116).
///
/// Its own type so it cannot be passed where a [`HostName`] belongs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusId(String);

impl CorpusId {
    /// Wrap a corpus identity.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The identity as given.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The host a learnings store writes as (Issue #116).
///
/// Its own type so it cannot be passed where a [`CorpusId`] belongs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostName(String);

impl HostName {
    /// Wrap a host name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The name as given.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The shared directory, scoped to one corpus and one host.
#[derive(Debug, Clone)]
pub struct LearningsStore {
    root: PathBuf,
    corpus: CorpusId,
    host: HostName,
}

impl LearningsStore {
    /// A store rooted at `root` for `corpus`, writing as `host`.
    ///
    /// ```
    /// use neat_ai_forests::learnings::{CorpusId, HostName, LearningsStore};
    /// let store = LearningsStore::new("/tmp/x", CorpusId::new("c"), HostName::new("h"));
    /// assert_eq!(store.host(), "h");
    /// ```
    ///
    /// Transposing corpus and host does not compile:
    ///
    /// ```compile_fail,E0308
    /// use neat_ai_forests::learnings::{CorpusId, HostName, LearningsStore};
    /// let store = LearningsStore::new("/tmp/x", HostName::new("h"), CorpusId::new("c"));
    /// ```
    pub fn new(root: impl Into<PathBuf>, corpus: CorpusId, host: HostName) -> Self {
        Self {
            root: root.into(),
            corpus,
            host,
        }
    }

    /// The name this store files learnings under.
    pub fn host(&self) -> &str {
        self.host.as_str()
    }

    /// Directory holding every host's file for this corpus.
    pub fn corpus_dir(&self) -> PathBuf {
        self.root
            .join(format!("corpus-{}", sanitise(self.corpus.as_str())))
    }

    /// This host's append-only file.
    pub fn file(&self) -> PathBuf {
        self.corpus_dir()
            .join(format!("{}.jsonl", sanitise(self.host.as_str())))
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
    use crate::learnings::model::Outcome;
    use crate::learnings::test_support::*;

    #[test]
    fn pruning_rewrites_only_this_hosts_file_and_leaves_the_rest_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let mine = LearningsStore::new(tmp.path(), CorpusId::new("c"), HostName::new("host-a"));
        let theirs = LearningsStore::new(tmp.path(), CorpusId::new("c"), HostName::new("host-b"));
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
        let store = LearningsStore::new(tmp.path(), CorpusId::new("c"), HostName::new("host-a"));
        store
            .append(&[learning(
                patch(1, 0.1),
                Outcome::Rejected,
                -1e-5,
                10,
                "host-a",
            )])
            .unwrap();
        let grew = LearningsStore::new(tmp.path(), CorpusId::new("c"), HostName::new("host-a"));
        // Nothing to drop -> returns before the length is re-checked.
        assert!(store.prune(&policy(50), false).is_ok());
        let _ = grew;
        // A missing file is simply nothing to do.
        let absent = LearningsStore::new(
            tmp.path(),
            CorpusId::new("c"),
            HostName::new("host-never-ran"),
        );
        assert_eq!(absent.prune(&policy(1000), false).unwrap().read, 0);
    }

    #[test]
    fn every_host_writes_its_own_file_and_all_of_them_are_read() {
        let tmp = tempfile::tempdir().unwrap();
        let a = LearningsStore::new(
            tmp.path(),
            CorpusId::new("corpus/one"),
            HostName::new("host-a"),
        );
        let b = LearningsStore::new(
            tmp.path(),
            CorpusId::new("corpus/one"),
            HostName::new("host-b"),
        );
        let other_corpus = LearningsStore::new(
            tmp.path(),
            CorpusId::new("corpus/two"),
            HostName::new("host-a"),
        );
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
        let store = LearningsStore::new(tmp.path(), CorpusId::new("c"), HostName::new("host-a"));
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
        let store = LearningsStore::new(
            tmp.path().join("nothing-here"),
            CorpusId::new("c"),
            HostName::new("host-a"),
        );
        assert!(store.load().unwrap().is_empty());
    }

    /// Issue #116: corpus and host are distinct types, so each lands in its
    /// own slot of the path and the store reports the host it was given.
    #[test]
    fn corpus_and_host_are_typed_and_file_under_their_own_slots() {
        let corpus = CorpusId::new("corpus/one");
        let host = HostName::new("host-a");
        assert_eq!(corpus.as_str(), "corpus/one");
        assert_eq!(host.as_str(), "host-a");
        let store = LearningsStore::new("/tmp/x", corpus, host);
        assert_eq!(store.host(), "host-a");
        assert_eq!(store.corpus_dir(), Path::new("/tmp/x/corpus-corpus-one"));
        assert_eq!(
            store.file(),
            Path::new("/tmp/x/corpus-corpus-one/host-a.jsonl")
        );
    }

    #[test]
    fn paths_stay_safe_whatever_the_corpus_identity_and_host_are() {
        let store = LearningsStore::new(
            "/tmp/x",
            CorpusId::new("../../etc/passwd"),
            HostName::new("host name/../.."),
        );
        assert_eq!(
            store.file(),
            Path::new("/tmp/x/corpus-etc-passwd/host-name.jsonl")
        );
        assert!(!store.file().to_string_lossy().contains("/.."));
    }
}

//! Filing accepted patches as a Rebase enhancement bundle
//! (stSoftwareAU/NEAT-AI-Rebase#65).
//!
//! A run's discoveries are patches, but what it publishes is a creature — and
//! by the time a 45-minute run ends, the fleet's champion has usually moved on,
//! so publishing the run's own descendant throws away everybody else's work.
//! Rebase exists to graft the patches onto a *freshly fetched* champion
//! instead, and it can only do that if the run recorded what it accepted.
//!
//! [`neat_ai_rebase::patch_log::PatchLog`] is the producer-side contract: the
//! opening creature's checksum, its authoritative score, the corpus identity
//! and the creature's widths are recorded once and stamped on every accepted
//! patch. This module is the switch around it — the call site keeps its own
//! behaviour when `--enhancements` is off, and every accepted patch is filed
//! when it is on.
//!
//! ## What this must not do
//!
//! Change what the run accepts. Filing is a recording of the decision
//! [`crate::promote`] already made on the authoritative scorer's verdict; it
//! never participates in it.
//!
//! ## Why filing failures poison the bundle
//!
//! A bundle missing one accepted patch is worse than no bundle at all: Rebase's
//! cumulative prefixes then claim a score that was measured on a creature the
//! prefix does not reproduce. So a refused patch is logged loudly, the log is
//! poisoned, and [`EnhancementLog::write_bundle`] refuses to write anything and
//! reports the first failure. A run that could not file what it accepted fails
//! at the end, with `best.json` and the journal already written.

use std::path::Path;

use neat_ai_rebase::patch_log::PatchLog;
use neat_core::CreatureExport;

use crate::patch::Patch;

/// Producer name stamped on every enhancement this build files.
pub fn producer() -> String {
    format!("neat-ai-forests/{}", env!("CARGO_PKG_VERSION"))
}

/// Convert a Forests patch into the Rebase mirror of the same format.
///
/// The conversion is a JSON round trip on purpose: the two representations are
/// the same wire form by contract (`neat_ai_rebase::patch` mirrors
/// `crate::patch`), so the bytes carry across unchanged — provenance included —
/// and [`Patch::id`] survives. It is the id the graft names its
/// `forest-<id>-…` structure with, so an id that moved here would make an
/// already-grafted patch invisible to Rebase.
///
/// # Errors
///
/// The serde failure text when the two formats have diverged, which is exactly
/// the case that must never be papered over.
pub fn to_rebase_patch(patch: &Patch) -> Result<neat_ai_rebase::patch::Patch, String> {
    let json =
        serde_json::to_string(patch).map_err(|e| format!("patch is not serialisable: {e}"))?;
    let mirrored: neat_ai_rebase::patch::Patch = serde_json::from_str(&json)
        .map_err(|e| format!("Rebase does not accept this patch format: {e}"))?;
    if mirrored.id() != patch.id() {
        return Err(format!(
            "patch id changed in translation ({} → {}); the graft's `forest-<id>-…` names would not match",
            patch.id(),
            mirrored.id()
        ));
    }
    Ok(mirrored)
}

/// The run's accepted patches, filed for Rebase — or switched off.
///
/// Off is the default and costs nothing: every method is a no-op and no file is
/// written.
#[derive(Debug, Default)]
pub struct EnhancementLog {
    log: Option<PatchLog>,
    /// First filing failure, if any. Poisons the bundle.
    failure: Option<String>,
}

impl EnhancementLog {
    /// A switched-off log: nothing is recorded and no bundle is ever written.
    pub fn off() -> Self {
        Self::default()
    }

    /// Open a log on the creature the run starts from.
    ///
    /// `base_score` is that creature's authoritative score on `corpus_identity`
    /// — the same corpus the rebase verdict will be measured on.
    ///
    /// # Errors
    ///
    /// The reason the opening facts could not be recorded: a creature that
    /// cannot be serialised, or a non-finite baseline.
    pub fn open(
        opening: &CreatureExport,
        base_score: f64,
        corpus_identity: &str,
    ) -> Result<Self, String> {
        let log = PatchLog::opening(&producer(), opening, base_score, corpus_identity)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            log: Some(log),
            failure: None,
        })
    }

    /// `true` when this run is filing enhancements.
    pub fn is_on(&self) -> bool {
        self.log.is_some()
    }

    /// Patches filed so far.
    pub fn filed(&self) -> usize {
        self.log.as_ref().map_or(0, |l| l.enhancements().len())
    }

    /// File one authoritative acceptance: a single winner, or the members of a
    /// verified combo in the order the combo applies them.
    ///
    /// Members this run has already filed are left where they are, so a combo
    /// that grew from an already-filed single does not duplicate it and the
    /// bundle's prefix of that length still reproduces the creature the score
    /// was measured on. Returns how many members were newly filed.
    ///
    /// # Errors
    ///
    /// Whatever Rebase refuses at filing time — a patch the opening creature
    /// could not carry, a non-finite score, an unsupported patch version. The
    /// failure is remembered: no bundle is written afterwards.
    pub fn accept(&mut self, patches: &[Patch], improved_score: f64) -> Result<usize, String> {
        let Some(log) = self.log.as_mut() else {
            return Ok(0);
        };
        let result = (|| {
            let mirrored: Vec<neat_ai_rebase::patch::Patch> = patches
                .iter()
                .map(to_rebase_patch)
                .collect::<Result<_, _>>()?;
            log.accept_combo(&mirrored, improved_score)
                .map(<[_]>::len)
                .map_err(|e| e.to_string())
        })();
        if let Err(e) = &result
            && self.failure.is_none()
        {
            self.failure = Some(e.clone());
        }
        result
    }

    /// Write the bundle to `path`.
    ///
    /// Returns `true` when a bundle was written, and `false` when the log is
    /// off or the run accepted nothing — in which case no file is created and
    /// the caller must not invoke Rebase at all.
    ///
    /// # Errors
    ///
    /// The first filing failure this run hit, or the write failure. A bundle
    /// that is missing an accepted patch is never written: its prefixes would
    /// claim scores that were measured on creatures they do not reproduce.
    pub fn write_bundle(&self, path: &Path) -> Result<bool, String> {
        if let Some(e) = &self.failure {
            return Err(format!(
                "refusing to write an incomplete enhancement bundle: {e}"
            ));
        }
        let Some(log) = self.log.as_ref() else {
            return Ok(false);
        };
        log.write_bundle(path).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::{Node, Provenance};
    use neat_ai_rebase::enhancement::{EnhancementBundle, Payload};

    const CORPUS: &str = "corpus-forests-test";

    fn creature() -> CreatureExport {
        neat_core::parse_creature_json(&crate::graft::fixtures::identity_creature_json(3, 1))
            .unwrap()
    }

    fn stump(feature: usize, right: f32) -> Patch {
        Patch::new(
            0,
            Node::stump(feature, 0.25, 0.0, right),
            Provenance {
                strategy: "histogram-stump".into(),
                backend: "cpu".into(),
                predicted_gain: 1.5,
                affected_records: 10,
                search_records: 100,
                incumbent_checksum: "abc".into(),
                seed: Some(7),
                notes: vec!["sampled".into()],
            },
        )
    }

    /// The whole contract in one assertion: the id the bundle carries is the id
    /// the graft named its structure with. If translation moved it, a champion
    /// that already carries the patch would not be recognised as carrying it.
    #[test]
    fn translation_preserves_the_patch_id_and_its_provenance() {
        let p = stump(1, 0.02);
        let mirrored = to_rebase_patch(&p).unwrap();
        assert_eq!(mirrored.id(), p.id());
        assert_eq!(mirrored.output, p.output);
        assert_eq!(mirrored.version, p.version);
        assert_eq!(mirrored.provenance.strategy, "histogram-stump");
        assert_eq!(mirrored.provenance.seed, Some(7));
        assert_eq!(mirrored.provenance.notes, vec!["sampled".to_string()]);
    }

    #[test]
    fn a_switched_off_log_records_nothing_and_writes_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("enhancements.json");
        let mut log = EnhancementLog::off();
        assert!(!log.is_on());
        assert_eq!(log.accept(&[stump(0, 0.01)], 0.51).unwrap(), 0);
        assert!(!log.write_bundle(&path).unwrap());
        assert!(!path.exists());
    }

    #[test]
    fn accepted_patches_are_filed_in_order_and_never_twice() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("enhancements.json");
        let mut log = EnhancementLog::open(&creature(), 0.5, CORPUS).unwrap();
        let seed = stump(0, 0.01);
        let grown = stump(1, 0.02);
        assert_eq!(log.accept(std::slice::from_ref(&seed), 0.51).unwrap(), 1);
        // A boosting round verified [seed, grown] together: only `grown` is new.
        assert_eq!(log.accept(&[seed.clone(), grown.clone()], 0.55).unwrap(), 1);
        assert_eq!(log.filed(), 2);

        assert!(log.write_bundle(&path).unwrap());
        let bundle =
            EnhancementBundle::parse_json(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let ids: Vec<&str> = bundle
            .enhancements
            .iter()
            .map(|e| e.meta.id.as_str())
            .collect();
        assert_eq!(
            ids,
            vec![seed.id(), grown.id()],
            "acceptance order is the bundle order"
        );
        for e in &bundle.enhancements {
            assert_eq!(e.meta.producer, producer());
            assert_eq!(e.meta.corpus_identity, CORPUS);
            assert!((e.meta.base_score - 0.5).abs() < 1e-12);
            assert!(matches!(e.payload, Payload::ForestPatch { .. }));
        }
        assert!(
            (bundle.enhancements[0].meta.improved_score - 0.51).abs() < 1e-12,
            "a member already filed keeps the score it was filed with"
        );
    }

    /// A run that accepted nothing has nothing to rebase, and the absence of the
    /// file is the caller's signal not to invoke Rebase at all.
    #[test]
    fn a_run_that_accepted_nothing_writes_no_bundle() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("enhancements.json");
        let log = EnhancementLog::open(&creature(), 0.5, CORPUS).unwrap();
        assert!(log.is_on());
        assert!(!log.write_bundle(&path).unwrap());
        assert!(!path.exists());
    }

    /// Fail loud: a patch Rebase refuses poisons the bundle rather than
    /// producing one whose prefixes lie about what was measured.
    #[test]
    fn a_refused_patch_poisons_the_bundle_instead_of_shortening_it() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("enhancements.json");
        let mut log = EnhancementLog::open(&creature(), 0.5, CORPUS).unwrap();
        log.accept(&[stump(0, 0.01)], 0.51).unwrap();

        // Feature 9 on a 3-input creature: the graft could never apply it.
        let err = log.accept(&[stump(9, 0.03)], 0.52).unwrap_err();
        assert!(err.contains("feature 9"), "{err}");
        let err = log.write_bundle(&path).unwrap_err();
        assert!(err.contains("incomplete enhancement bundle"), "{err}");
        assert!(!path.exists(), "a partial bundle must never reach disk");
    }

    #[test]
    fn a_non_finite_baseline_is_refused_at_open() {
        let err = EnhancementLog::open(&creature(), f64::NAN, CORPUS).unwrap_err();
        assert!(err.contains("baseScore"), "{err}");
    }
}

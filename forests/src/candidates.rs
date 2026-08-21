//! Candidate population generation (Issue #8, extended by #11/#12/#14).
//!
//! Turns search discoveries (stumps, trees, oblique splits) into complete
//! candidate creatures: a clone of the immutable incumbent with one grafted
//! patch each. Several leaf magnitudes are tried around the analytical
//! optimum, one-sided variants are preferred where a side can be exactly zero,
//! and deliberately random stumps are included as first-class controls.
//! Nothing here accepts anything.

use std::collections::HashSet;

use neat_core::CreatureExport;
use rand::Rng;
use rand::rngs::StdRng;

use crate::bins::BinCache;
use crate::graft::graft_patch;
use crate::histogram::{StumpCandidate, StumpKind};
use crate::incumbent::Incumbent;
use crate::oblique::ObliqueCandidate;
use crate::patch::{Node, Patch, Provenance};
use crate::tree::TreeCandidate;

/// Generation controls.
#[derive(Debug, Clone)]
pub struct CandidateConfig {
    /// Leaf scales tried (1.0 is the analytical optimum).
    pub magnitude_scales: Vec<f32>,
    /// Random stumps to add.
    pub random_candidates: usize,
    /// Hard cap on candidates returned.
    pub max_candidates: usize,
    /// Search backend label for histogram discoveries.
    pub backend: String,
    /// Records searched (for provenance).
    pub search_records: u64,
    /// Clamp for random corrections.
    pub max_correction: f32,
    /// Typical |correction| (e.g. residual σ) used to scale random stumps.
    pub random_scale: f32,
    /// Neighbouring bins tried around each stump threshold (0 = off).
    pub threshold_jitter: usize,
    /// Extra provenance notes from the search set.
    pub notes: Vec<String>,
}

/// A grafted candidate.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Candidate id (the patch id, or a hash of the member ids for a combination).
    pub id: String,
    /// Primary patch (first member for a combination).
    pub patch: Patch,
    /// Further patches stacked on top of `patch` (empty for a single graft).
    pub combo: Vec<Patch>,
    /// Complete candidate creature.
    pub creature: CreatureExport,
    /// Neuron uuids appended, per member patch (`patch` first).
    pub added_uuids: Vec<Vec<String>>,
}

impl Candidate {
    /// Every member patch, primary first.
    pub fn patches(&self) -> impl Iterator<Item = &Patch> {
        std::iter::once(&self.patch).chain(self.combo.iter())
    }

    /// Strategy label (`combo/<k>` for combinations).
    pub fn strategy(&self) -> String {
        if self.combo.is_empty() {
            self.patch.provenance.strategy.clone()
        } else {
            format!(
                "combo/{}:{}",
                self.combo.len() + 1,
                self.patch.provenance.strategy
            )
        }
    }

    /// Sum of member proxy gains.
    pub fn predicted_gain(&self) -> f64 {
        self.patches().map(|p| p.provenance.predicted_gain).sum()
    }

    /// Largest member affected-record count.
    pub fn affected_records(&self) -> u64 {
        self.patches()
            .map(|p| p.provenance.affected_records)
            .max()
            .unwrap_or(0)
    }

    /// Deepest member.
    pub fn depth(&self) -> usize {
        self.patches().map(|p| p.root.depth()).max().unwrap_or(0)
    }

    /// Union of member features, ascending.
    pub fn features(&self) -> Vec<usize> {
        let mut f: Vec<usize> = self.patches().flat_map(|p| p.root.features()).collect();
        f.sort_unstable();
        f.dedup();
        f
    }
}

/// A discovery waiting to become a candidate.
#[derive(Debug, Clone)]
pub struct Discovery {
    /// Root node (creature feature space).
    pub root: Node,
    /// Strategy label.
    pub strategy: String,
    /// Backend label.
    pub backend: String,
    /// Proxy gain.
    pub gain: f64,
    /// Affected records.
    pub affected: f64,
    /// Records on the (left, right) side of the root split, when known.
    pub side_records: Option<(f64, f64)>,
    /// Notes.
    pub notes: Vec<String>,
}

impl Discovery {
    /// From a histogram stump (set-feature index already mapped).
    pub fn from_stump(s: &StumpCandidate, feature: usize, backend: &str) -> Self {
        Self {
            root: Node::stump(feature, s.threshold, s.left_correction, s.right_correction),
            strategy: "histogram-stump".into(),
            backend: backend.into(),
            gain: s.gain,
            affected: s.affected_records,
            side_records: Some((s.left_records, s.right_records)),
            notes: vec![format!("kind={:?} bin={}", s.kind, s.bin)],
        }
    }

    /// From a grown tree.
    pub fn from_tree(t: &TreeCandidate, backend: &str) -> Self {
        Self {
            root: t.root.clone(),
            strategy: format!("histogram-tree-depth{}", t.depth),
            backend: backend.into(),
            gain: t.gain,
            affected: t.affected_records,
            side_records: None,
            notes: vec![format!("growth={}", t.growth)],
        }
    }

    /// From an oblique split.
    pub fn from_oblique(o: &ObliqueCandidate) -> Self {
        Self {
            root: o.root.clone(),
            strategy: "oblique-split".into(),
            backend: "cpu-raw-sample".into(),
            gain: o.gain,
            affected: o.affected_records,
            side_records: None,
            notes: vec![format!("origin={}", o.origin)],
        }
    }
}

fn zero_side(node: &Node, right: bool) -> Option<Node> {
    if let Node::Split {
        condition,
        left,
        right: r,
    } = node
    {
        let (l2, r2) = if right {
            (left.clone(), Box::new(Node::leaf(0.0)))
        } else {
            (Box::new(Node::leaf(0.0)), r.clone())
        };
        // Only a genuine variant: something changed and a non-zero leaf remains.
        if (**left != *l2 || **r != *r2)
            && (l2.max_abs_correction() > 0.0 || r2.max_abs_correction() > 0.0)
        {
            return Some(Node::Split {
                condition: condition.clone(),
                left: l2,
                right: r2,
            });
        }
    }
    None
}

/// Expand discoveries into patches: base, magnitude variants, one-sided variants, jitter.
pub fn expand_discoveries(
    discoveries: &[Discovery],
    cache: &BinCache,
    output: usize,
    incumbent_checksum: &str,
    seed: u64,
    cfg: &CandidateConfig,
) -> Vec<Patch> {
    let mut out = Vec::new();
    let prov = |d: &Discovery, strategy: String, extra: Vec<String>| Provenance {
        strategy,
        backend: d.backend.clone(),
        predicted_gain: d.gain,
        affected_records: d.affected as u64,
        search_records: cfg.search_records,
        incumbent_checksum: incumbent_checksum.to_string(),
        seed: Some(seed),
        notes: d
            .notes
            .iter()
            .chain(&cfg.notes)
            .cloned()
            .chain(extra)
            .collect(),
    };
    // Pass 1: the analytical optimum of every discovery, in rank order.
    for d in discoveries {
        out.push(Patch::new(
            output,
            d.root.clone(),
            prov(d, d.strategy.clone(), vec![]),
        ));
    }
    // Pass 2: one-sided variants of two-leaf stumps (most records untouched).
    for d in discoveries {
        for right in [false, true] {
            if let Some(root) = zero_side(&d.root, right) {
                let mut p = Patch::new(
                    output,
                    root,
                    prov(
                        d,
                        format!("{}/one-sided", d.strategy),
                        vec![format!("zeroed={}", if right { "right" } else { "left" })],
                    ),
                );
                // Only the kept side is affected now.
                if let Some((l, r)) = d.side_records {
                    p.provenance.affected_records = if right { l } else { r } as u64;
                }
                out.push(p);
            }
        }
    }
    // Pass 3: magnitude variants.
    for &scale in &cfg.magnitude_scales {
        if scale == 1.0 {
            continue;
        }
        for d in discoveries {
            let gain = if scale < 0.0 {
                -d.gain.abs()
            } else {
                d.gain * (2.0 * f64::from(scale) - f64::from(scale) * f64::from(scale))
            };
            let mut p = Patch::new(
                output,
                d.root.scaled(scale),
                prov(
                    d,
                    format!("{}/scale", d.strategy),
                    vec![format!("scale={scale}")],
                ),
            );
            p.provenance.predicted_gain = gain;
            out.push(p);
        }
    }
    // Pass 4: threshold jitter to neighbouring bins (axis stumps only).
    if cfg.threshold_jitter > 0 {
        for d in discoveries {
            if let Node::Split {
                condition,
                left,
                right,
            } = &d.root
                && condition.is_axis_aligned()
            {
                let f = condition.terms[0].feature;
                let edges = &cache.edges[f];
                if let Some(b) = edges.iter().position(|&e| e == condition.threshold) {
                    for j in 1..=cfg.threshold_jitter {
                        for nb in [b.checked_sub(j), Some(b + j)].into_iter().flatten() {
                            if let Some(&t) = edges.get(nb) {
                                let root = Node::Split {
                                    condition: crate::patch::Condition::axis(f, t),
                                    left: left.clone(),
                                    right: right.clone(),
                                };
                                let mut p = Patch::new(
                                    output,
                                    root,
                                    prov(
                                        d,
                                        "threshold-jitter".into(),
                                        vec![format!("jitter-bin={nb}")],
                                    ),
                                );
                                p.provenance.predicted_gain = 0.0;
                                out.push(p);
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Deliberately random stumps — honest controls, and legitimate winners.
pub fn random_stumps(
    cache: &BinCache,
    output: usize,
    incumbent_checksum: &str,
    seed: u64,
    rng: &mut StdRng,
    cfg: &CandidateConfig,
) -> Vec<Patch> {
    let features = cache.features();
    let mut out = Vec::new();
    let mut tries = 0;
    while out.len() < cfg.random_candidates && tries < cfg.random_candidates * 20 {
        tries += 1;
        let f = rng.random_range(0..features);
        if cache.edges[f].is_empty() {
            continue;
        }
        let b = rng.random_range(0..cache.edges[f].len());
        let t = cache.edges[f][b];
        let mag = (rng.random_range(-1.0f32..1.0) * cfg.random_scale)
            .clamp(-cfg.max_correction, cfg.max_correction);
        if mag == 0.0 {
            continue;
        }
        let kind = match rng.random_range(0..3) {
            0 => StumpKind::LeftOnly,
            1 => StumpKind::RightOnly,
            _ => StumpKind::TwoLeaf,
        };
        let (l, r) = match kind {
            StumpKind::LeftOnly => (mag, 0.0),
            StumpKind::RightOnly => (0.0, mag),
            StumpKind::TwoLeaf => (mag, -mag),
        };
        out.push(Patch::new(
            output,
            Node::stump(f, t, l, r),
            Provenance {
                strategy: "random-stump".into(),
                backend: "random".into(),
                predicted_gain: 0.0,
                affected_records: 0,
                search_records: cfg.search_records,
                incumbent_checksum: incumbent_checksum.to_string(),
                seed: Some(seed),
                notes: vec![format!("kind={kind:?} bin={b}")],
            },
        ));
    }
    out
}

/// Graft patches onto incumbent clones, de-duplicating by patch id and
/// capping at `max_candidates`. Returns the candidates and the discard reasons.
pub fn generate_candidates(
    incumbent: &Incumbent,
    patches: Vec<Patch>,
    cfg: &CandidateConfig,
) -> (Vec<Candidate>, Vec<(String, String)>) {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut discarded = Vec::new();
    for patch in patches {
        if out.len() >= cfg.max_candidates {
            break;
        }
        let id = patch.id();
        if !seen.insert(id.clone()) {
            continue;
        }
        if patch.root.max_abs_correction() == 0.0 {
            discarded.push((id, "all-zero correction".into()));
            continue;
        }
        match graft_patch(&incumbent.creature, &patch) {
            Ok(g) => out.push(Candidate {
                id,
                patch,
                combo: Vec::new(),
                creature: g.creature,
                added_uuids: vec![g.added_uuids],
            }),
            Err(e) => discarded.push((id, e.to_string())),
        }
    }
    (out, discarded)
}

/// Id of a combination: hash of the member ids in order.
pub fn combo_id(patches: &[Patch]) -> String {
    let ids: Vec<String> = patches.iter().map(Patch::id).collect();
    crate::incumbent::sha256_hex(ids.join("+").as_bytes())[..16].to_string()
}

/// Build combination candidates: each `groups[i]` is stacked onto one clone.
/// Groups with fewer than two distinct members, or that fail to graft, are
/// discarded with a reason. Strategy labels become `combo/<k>:<primary>`.
pub fn generate_combos(
    incumbent: &Incumbent,
    groups: Vec<Vec<Patch>>,
    strategy_note: &str,
) -> (Vec<Candidate>, Vec<(String, String)>) {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut discarded = Vec::new();
    for mut group in groups {
        let mut ids = HashSet::new();
        group.retain(|p| ids.insert(p.id()));
        if group.len() < 2 {
            continue;
        }
        let id = combo_id(&group);
        if !seen.insert(id.clone()) {
            continue;
        }
        match crate::graft::graft_patches(&incumbent.creature, &group) {
            Ok((creature, added_uuids)) => {
                let mut it = group.into_iter();
                let mut patch = it.next().unwrap();
                patch.provenance.notes.push(strategy_note.to_string());
                out.push(Candidate {
                    id,
                    patch,
                    combo: it.collect(),
                    creature,
                    added_uuids,
                });
            }
            Err(e) => discarded.push((id, e.to_string())),
        }
    }
    (out, discarded)
}

/// Combination groups from ranked discoveries: the top-2, top-3, … top-`max`
/// patches on **distinct features**, stacked cumulatively.
pub fn top_k_groups(ranked: &[Patch], max: usize) -> Vec<Vec<Patch>> {
    let mut picked: Vec<Patch> = Vec::new();
    let mut used: HashSet<Vec<usize>> = HashSet::new();
    for p in ranked {
        if picked.len() >= max {
            break;
        }
        if used.insert(p.root.features()) {
            picked.push(p.clone());
        }
    }
    (2..=picked.len()).map(|k| picked[..k].to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bins::{BinCache, BinMeta};
    use crate::graft::fixtures::identity_creature;
    use rand::SeedableRng;

    fn cache() -> BinCache {
        let edges = vec![vec![0.0f32, 0.5, 1.0], vec![-1.0, 0.0, 1.0, 2.0]];
        BinCache {
            meta: BinMeta {
                format_version: 1,
                algorithm_version: 1,
                input_count: 2,
                output_count: 1,
                record_count: 10,
                corpus_identity: "t".into(),
                requested_bins: 4,
                effective_bins: vec![4, 5],
                non_finite_counts: vec![0, 0],
                non_finite_policy: String::new(),
                sample_records: 10,
                sample_stride: 1,
                created_at_unix: 0,
            },
            edges,
        }
    }

    fn cfg() -> CandidateConfig {
        CandidateConfig {
            magnitude_scales: vec![1.0, 0.5, -1.0],
            random_candidates: 3,
            max_candidates: 100,
            backend: "cpu".into(),
            search_records: 10,
            max_correction: 1.0,
            random_scale: 0.2,
            threshold_jitter: 1,
            notes: vec![],
        }
    }

    #[test]
    fn population_is_deterministic_and_grafts_load() {
        let inc = Incumbent::from_creature(identity_creature(2, 1), "t").unwrap();
        let stump = StumpCandidate {
            feature: 1,
            bin: 1,
            threshold: 0.0,
            kind: StumpKind::TwoLeaf,
            left_correction: -0.1,
            right_correction: 0.2,
            gain: 1.0,
            left_records: 5.0,
            right_records: 5.0,
            affected_records: 10.0,
            affected_fraction: 1.0,
            backend: "cpu".into(),
        };
        let d = vec![Discovery::from_stump(&stump, 1, "cpu")];
        let c = cfg();
        let mut patches = expand_discoveries(&d, &cache(), 0, &inc.checksum, 7, &c);
        let mut rng = StdRng::seed_from_u64(7);
        patches.extend(random_stumps(&cache(), 0, &inc.checksum, 7, &mut rng, &c));
        let (cands, discarded) = generate_candidates(&inc, patches.clone(), &c);
        assert!(discarded.is_empty(), "{discarded:?}");
        // base + 2 one-sided + 2 scales + 2 jitter + 3 random
        assert_eq!(cands.len(), 10);
        let strategies: HashSet<&str> = cands
            .iter()
            .map(|c| c.patch.provenance.strategy.as_str())
            .collect();
        assert!(strategies.contains("random-stump"));
        assert!(strategies.contains("histogram-stump/one-sided"));
        assert!(strategies.contains("threshold-jitter"));
        for cand in &cands {
            neat_core::compile_creature(&cand.creature).unwrap();
            assert_eq!(cand.patch.provenance.incumbent_checksum, inc.checksum);
            assert_eq!(cand.patch.provenance.seed, Some(7));
        }
        let mut rng2 = StdRng::seed_from_u64(7);
        let mut again = expand_discoveries(&d, &cache(), 0, &inc.checksum, 7, &c);
        again.extend(random_stumps(&cache(), 0, &inc.checksum, 7, &mut rng2, &c));
        assert_eq!(again, patches);
        // Cap and dedup.
        let capped = CandidateConfig {
            max_candidates: 3,
            ..c
        };
        let doubled: Vec<Patch> = patches.iter().chain(&patches).cloned().collect();
        assert_eq!(generate_candidates(&inc, doubled, &capped).0.len(), 3);
    }

    #[test]
    fn invalid_patches_are_discarded_before_scoring() {
        let inc = Incumbent::from_creature(identity_creature(2, 1), "t").unwrap();
        let bad = Patch::new(0, Node::stump(9, 0.0, 0.0, 1.0), Provenance::default());
        let zero = Patch::new(0, Node::stump(0, 0.0, 0.0, 0.0), Provenance::default());
        let (c, d) = generate_candidates(&inc, vec![bad, zero], &cfg());
        assert!(c.is_empty());
        assert_eq!(d.len(), 2);
        assert!(d[0].1.contains("feature 9"));
    }

    /// Issue #39 — a candidate the shared validator rejects never reaches
    /// scoring, and its rejection is recorded with the `ValidationFailure`
    /// detail rather than dropped.
    #[test]
    fn validation_failures_are_recorded_against_the_candidate() {
        let inc = Incumbent::from_creature(
            crate::graft::fixtures::constant_after_hidden_creature(),
            "t",
        )
        .expect("an invalid-order incumbent still loads — Forests does not validate on ingest");
        let patch = Patch::new(0, Node::stump(0, 0.0, -0.2, 0.4), Provenance::default());
        let id = patch.id();
        let (cands, discarded) = generate_candidates(&inc, vec![patch], &cfg());
        assert!(cands.is_empty(), "an invalid creature must not be scored");
        assert_eq!(discarded.len(), 1);
        assert_eq!(discarded[0].0, id, "the rejection names the candidate");
        assert!(discarded[0].1.contains("NEURON_ORDER"), "{discarded:?}");
        assert!(discarded[0].1.contains("neuron index"), "{discarded:?}");
    }
}

//! Portable Forest patch format (Issue #7).
//!
//! A patch is a tiny tree of residual corrections for one output. It is
//! stored as JSON in the journal and converted to ordinary NEAT-AI `IF`
//! structure by [`crate::graft`]. The abstract evaluator here mirrors the
//! NEAT-AI-core `IF` kernel exactly: the condition is accumulated in `f32`, in
//! synapse order (feature terms first, then the constant `-threshold`), and the
//! right branch is taken when the sum is **strictly greater than zero**. `NaN`
//! therefore always falls to the left branch, exactly as a `NaN` condition sum
//! does inside the creature.

use serde::{Deserialize, Serialize};

/// Current patch format version.
pub const PATCH_FORMAT_VERSION: u32 = 1;

/// Format version of a patch that uses a v2 field: [`Patch::parent`] for a
/// refinement of a leaf already on the creature (Issue #140), or
/// [`Term::neuron`] for a condition on a hidden neuron (Issue #139). The
/// version moves so a reader that does not know the field refuses the patch
/// instead of misreading it.
pub const REFINE_PATCH_FORMAT_VERSION: u32 = 2;

/// Format version of a patch with a linear leaf (Issue #138) — the same v2.
pub const LINEAR_LEAF_PATCH_FORMAT_VERSION: u32 = REFINE_PATCH_FORMAT_VERSION;

/// Format version of a patch whose condition reads a hidden neuron (Issue
/// #139) — the same v2 as a refinement.
pub const HIDDEN_SOURCE_PATCH_FORMAT_VERSION: u32 = REFINE_PATCH_FORMAT_VERSION;

/// Which branch sum of an existing `IF` neuron a refinement feeds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BranchRole {
    /// Summed when the parent's condition is strictly positive (the right branch).
    Positive,
    /// Summed otherwise (the left branch).
    Negative,
}

impl BranchRole {
    /// Lower-case label, as the creature's synapse `type` spells it.
    pub fn label(self) -> &'static str {
        match self {
            Self::Positive => "positive",
            Self::Negative => "negative",
        }
    }
}

/// Where a refinement attaches: one branch of an `IF` neuron already on the
/// creature, addressed by UUID so a replay on a moved champion can find it or
/// refuse cleanly when it is gone (Issue #140).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Attach {
    /// UUID of the existing `IF` neuron.
    pub uuid: String,
    /// The branch sum the patch's root feeds.
    pub role: BranchRole,
}

/// One weighted source in a (possibly oblique) condition.
///
/// A source is an input (`feature`) or, since Issue #139, a neuron already on
/// the creature (`neuron`, by UUID) — one of its learned features. When
/// `neuron` is set, `feature` is unused and kept at `0`.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Term {
    /// Hidden neuron read by this term (Issue #139); `None` for an input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neuron: Option<String>,
    /// Input observation index (unused when `neuron` is set).
    pub feature: usize,
    /// Synapse weight (exactly `1.0` for axis-aligned splits).
    pub weight: f32,
}

/// An input term prints exactly as it did before `neuron` existed, so debug
/// fingerprints of input-only trees are unchanged.
impl std::fmt::Debug for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("Term");
        if let Some(neuron) = &self.neuron {
            d.field("neuron", neuron);
        }
        d.field("feature", &self.feature)
            .field("weight", &self.weight)
            .finish()
    }
}

impl Term {
    /// A unit-or-weighted term on input `feature`.
    pub fn input(feature: usize, weight: f32) -> Self {
        Self {
            neuron: None,
            feature,
            weight,
        }
    }

    /// A term on the hidden neuron `uuid` (Issue #139).
    pub fn hidden(uuid: impl Into<String>, weight: f32) -> Self {
        Self {
            neuron: Some(uuid.into()),
            feature: 0,
            weight,
        }
    }
}

/// `Σ weight·x > threshold` → right branch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Condition {
    /// Feature terms (one for axis-aligned splits, 2–3 for oblique).
    pub terms: Vec<Term>,
    /// Threshold (stored as the exact `f32` the creature will carry).
    pub threshold: f32,
}

impl Condition {
    /// Axis-aligned `x[feature] > threshold`.
    pub fn axis(feature: usize, threshold: f32) -> Self {
        Self {
            terms: vec![Term::input(feature, 1.0)],
            threshold,
        }
    }

    /// `hidden(uuid) > threshold` on a neuron already on the creature
    /// (Issue #139).
    pub fn hidden(uuid: impl Into<String>, threshold: f32) -> Self {
        Self {
            terms: vec![Term::hidden(uuid, 1.0)],
            threshold,
        }
    }

    /// `true` when the record takes the right (positive) branch.
    ///
    /// Accumulates exactly like the `IF` kernel: `f32`, in term order, with the
    /// constant term `1.0 * -threshold` added last. A hidden-source term reads
    /// `NaN` here — so the record goes left — because inputs alone cannot say
    /// what a neuron computes; use [`Condition::goes_right_with`].
    pub fn goes_right(&self, inputs: &[f32]) -> bool {
        self.goes_right_with(inputs, &|_| f32::NAN)
    }

    /// [`Condition::goes_right`], with `hidden(uuid)` giving the activation of
    /// each hidden source on this record (Issue #139).
    pub fn goes_right_with(&self, inputs: &[f32], hidden: &dyn Fn(&str) -> f32) -> bool {
        let mut sum = 0.0f32;
        for t in &self.terms {
            let x = match &t.neuron {
                Some(uuid) => hidden(uuid),
                None => inputs.get(t.feature).copied().unwrap_or(0.0),
            };
            sum += x * t.weight;
        }
        sum += 1.0 * -self.threshold;
        sum > 0.0
    }

    /// `true` when the condition is a single unit-weight **input** term.
    pub fn is_axis_aligned(&self) -> bool {
        self.terms.len() == 1 && self.terms[0].weight == 1.0 && self.terms[0].neuron.is_none()
    }
}

/// Tree node.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Node {
    /// Constant correction, plus — for a linear leaf (Issue #138) — a slope
    /// on one or more inputs.
    Leaf {
        /// Added to the pre-squash sum of the target output neuron.
        correction: f32,
        /// Linear terms (Issue #138): the leaf's value is
        /// `Σ weight · x[feature] + correction`, summed in this order (inputs
        /// ascending, as the `IF` kernel sums its branch synapses). Empty for
        /// an ordinary constant leaf, which then serialises exactly as v1.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        slopes: Vec<Term>,
    },
    /// Binary split.
    Split {
        /// Branch condition.
        condition: Condition,
        /// Taken when the condition is false (`<=`).
        left: Box<Node>,
        /// Taken when the condition is true (`>`).
        right: Box<Node>,
    },
}

/// A constant leaf prints exactly as it did before linear leaves existed, so
/// debug fingerprints of constant-leaf trees are unchanged.
impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Leaf { correction, slopes } => {
                let mut d = f.debug_struct("Leaf");
                d.field("correction", correction);
                if !slopes.is_empty() {
                    d.field("slopes", slopes);
                }
                d.finish()
            }
            Self::Split {
                condition,
                left,
                right,
            } => f
                .debug_struct("Split")
                .field("condition", condition)
                .field("left", left)
                .field("right", right)
                .finish(),
        }
    }
}

impl Node {
    /// Leaf helper.
    pub fn leaf(correction: f32) -> Self {
        Self::Leaf {
            correction,
            slopes: Vec::new(),
        }
    }

    /// Linear leaf (Issue #138): `correction + Σ slope · x[feature]`. Slopes
    /// are kept in ascending feature order — the order the `IF` kernel sums
    /// them in.
    pub fn linear_leaf(correction: f32, mut slopes: Vec<(usize, f32)>) -> Self {
        slopes.sort_by_key(|s| s.0);
        Self::Leaf {
            correction,
            slopes: slopes.into_iter().map(|(f, w)| Term::input(f, w)).collect(),
        }
    }

    /// `true` when any leaf carries a slope (Issue #138).
    pub fn has_slopes(&self) -> bool {
        match self {
            Self::Leaf { slopes, .. } => !slopes.is_empty(),
            Self::Split { left, right, .. } => left.has_slopes() || right.has_slopes(),
        }
    }

    /// Axis-aligned split helper.
    pub fn stump(feature: usize, threshold: f32, left: f32, right: f32) -> Self {
        Self::Split {
            condition: Condition::axis(feature, threshold),
            left: Box::new(Self::leaf(left)),
            right: Box::new(Self::leaf(right)),
        }
    }

    /// Correction for one record.
    pub fn evaluate(&self, inputs: &[f32]) -> f32 {
        self.evaluate_with(inputs, &|_| f32::NAN)
    }

    /// Correction for one record, with `hidden(uuid)` giving each hidden
    /// source's activation (Issue #139).
    pub fn evaluate_with(&self, inputs: &[f32], hidden: &dyn Fn(&str) -> f32) -> f32 {
        match self {
            Self::Leaf { correction, slopes } => {
                // Inputs first, then the bias-1 constant: the order the `IF`
                // kernel sums a branch's synapses in.
                let mut sum = 0.0f32;
                for t in slopes {
                    sum += inputs.get(t.feature).copied().unwrap_or(0.0) * t.weight;
                }
                if slopes.is_empty() {
                    *correction
                } else {
                    sum + 1.0 * *correction
                }
            }
            Self::Split {
                condition,
                left,
                right,
            } => {
                if condition.goes_right_with(inputs, hidden) {
                    right.evaluate_with(inputs, hidden)
                } else {
                    left.evaluate_with(inputs, hidden)
                }
            }
        }
    }

    /// Distinct hidden neurons any condition reads (Issue #139), ascending.
    pub fn hidden_sources(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.visit_terms(&mut |t| {
            if let Some(u) = &t.neuron {
                out.push(u.clone());
            }
        });
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Call `f` on every condition term, pre-order.
    pub fn visit_terms(&self, f: &mut dyn FnMut(&Term)) {
        if let Self::Split {
            condition,
            left,
            right,
        } = self
        {
            condition.terms.iter().for_each(&mut *f);
            left.visit_terms(f);
            right.visit_terms(f);
        }
    }

    /// Call `f` on every condition term, pre-order, mutably.
    pub fn visit_terms_mut(&mut self, f: &mut dyn FnMut(&mut Term)) {
        if let Self::Split {
            condition,
            left,
            right,
        } = self
        {
            condition.terms.iter_mut().for_each(&mut *f);
            left.visit_terms_mut(f);
            right.visit_terms_mut(f);
        }
    }

    /// Maximum depth (a leaf is depth 0, a stump depth 1).
    pub fn depth(&self) -> usize {
        match self {
            Self::Leaf { .. } => 0,
            Self::Split { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }

    /// Number of split nodes.
    pub fn split_count(&self) -> usize {
        match self {
            Self::Leaf { .. } => 0,
            Self::Split { left, right, .. } => 1 + left.split_count() + right.split_count(),
        }
    }

    /// Number of leaves whose correction is exactly zero.
    pub fn zero_leaves(&self) -> usize {
        match self {
            Self::Leaf { correction, slopes } => {
                usize::from(*correction == 0.0 && slopes.iter().all(|t| t.weight == 0.0))
            }
            Self::Split { left, right, .. } => left.zero_leaves() + right.zero_leaves(),
        }
    }

    /// Largest absolute leaf correction.
    pub fn max_abs_correction(&self) -> f32 {
        match self {
            Self::Leaf { correction, slopes } => slopes
                .iter()
                .map(|t| t.weight.abs())
                .fold(correction.abs(), f32::max),
            Self::Split { left, right, .. } => {
                left.max_abs_correction().max(right.max_abs_correction())
            }
        }
    }

    /// Distinct input features referenced by any condition, ascending (hidden
    /// sources are in [`Node::hidden_sources`]).
    pub fn features(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.collect_features(&mut out);
        out.sort_unstable();
        out.dedup();
        out
    }

    fn collect_features(&self, out: &mut Vec<usize>) {
        if let Self::Leaf { slopes, .. } = self {
            out.extend(
                slopes
                    .iter()
                    .filter(|t| t.neuron.is_none())
                    .map(|t| t.feature),
            );
        }
        if let Self::Split {
            condition,
            left,
            right,
        } = self
        {
            out.extend(
                condition
                    .terms
                    .iter()
                    .filter(|t| t.neuron.is_none())
                    .map(|t| t.feature),
            );
            left.collect_features(out);
            right.collect_features(out);
        }
    }

    /// Multiply every leaf by `scale` (used for magnitude variants).
    pub fn scaled(&self, scale: f32) -> Self {
        match self {
            Self::Leaf { correction, slopes } => Self::Leaf {
                correction: correction * scale,
                slopes: slopes
                    .iter()
                    .map(|t| Term {
                        weight: t.weight * scale,
                        ..t.clone()
                    })
                    .collect(),
            },
            Self::Split {
                condition,
                left,
                right,
            } => Self::Split {
                condition: condition.clone(),
                left: Box::new(left.scaled(scale)),
                right: Box::new(right.scaled(scale)),
            },
        }
    }

    /// `true` if every value is finite.
    pub fn is_finite(&self) -> bool {
        match self {
            Self::Leaf { correction, slopes } => {
                correction.is_finite() && slopes.iter().all(|t| t.weight.is_finite())
            }
            Self::Split {
                condition,
                left,
                right,
            } => {
                condition.threshold.is_finite()
                    && condition.terms.iter().all(|t| t.weight.is_finite())
                    && left.is_finite()
                    && right.is_finite()
            }
        }
    }
}

/// Where a patch came from. Every strategy must describe itself honestly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Provenance {
    /// Strategy label, e.g. `histogram-stump`, `random-stump`, `xgboost-import`.
    pub strategy: String,
    /// Search backend that produced the split statistics (`cpu`, `gpu`, `random`, `external`).
    pub backend: String,
    /// Predicted proxy gain (sum-of-squares reduction in correction space). Not a score.
    pub predicted_gain: f64,
    /// Records whose correction is non-zero under the patch. Weighted by the
    /// search set's importance weights, so under `residual-weighted` sampling
    /// it estimates the count over the whole corpus rather than over
    /// `search_records` — the journal's `affectedFraction` is the ratio that
    /// accounts for that (#64).
    pub affected_records: u64,
    /// Rows in the search set (unweighted).
    pub search_records: u64,
    /// Checksum of the incumbent the patch was searched against.
    pub incumbent_checksum: String,
    /// RNG seed that generated this candidate (if any randomness was involved).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Free-form notes (sampling rates, jitter offsets, scale factor …).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
}

/// A complete patch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Patch {
    /// Format version.
    pub version: u32,
    /// Output neuron index the correction feeds.
    pub output: usize,
    /// Tree root.
    pub root: Node,
    /// `Some` for a refinement (Issue #140): the root feeds this branch of an
    /// existing `IF` instead of the output's anchor, so its leaves only apply
    /// to records that already reach that leaf.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<Attach>,
    /// Provenance.
    pub provenance: Provenance,
}

impl Patch {
    /// New v1 patch.
    pub fn new(output: usize, root: Node, provenance: Provenance) -> Self {
        Self {
            version: if root.hidden_sources().is_empty() {
                PATCH_FORMAT_VERSION
            } else {
                HIDDEN_SOURCE_PATCH_FORMAT_VERSION
            },
            output,
            root,
            parent: None,
            provenance,
        }
    }

    /// New refinement of the leaf at `parent` (Issue #140), format v2.
    pub fn refinement(output: usize, parent: Attach, root: Node, provenance: Provenance) -> Self {
        Self {
            version: REFINE_PATCH_FORMAT_VERSION,
            output,
            root,
            parent: Some(parent),
            provenance,
        }
    }

    /// The lowest format version that can carry this patch, read from its
    /// shape: version 2 for a refinement (`parent`) or a hidden-source term
    /// (`Term.neuron`), version 1 otherwise. A reader that knows only
    /// version 1 must refuse anything above it rather than graft it at the
    /// anchor or on an input.
    pub fn required_version(&self) -> u32 {
        if self.parent.is_some() || !self.root.hidden_sources().is_empty() {
            REFINE_PATCH_FORMAT_VERSION.max(HIDDEN_SOURCE_PATCH_FORMAT_VERSION)
        } else {
            PATCH_FORMAT_VERSION
        }
    }

    /// Whether NEAT-AI-Rebase can replay this patch: version-1 shapes only.
    /// It has no `parent` and no `Term.neuron` yet, so it would mirror a
    /// refinement or a hidden-source patch under a different id.
    pub fn rebase_can_replay(&self) -> bool {
        self.required_version() == PATCH_FORMAT_VERSION
    }

    /// Deterministic identity: SHA-256 of `output` + canonical root JSON
    /// (provenance excluded). A refinement also hashes where it attaches, so
    /// one split under two different leaves is two patches; a patch without a
    /// parent keeps the id it always had.
    pub fn id(&self) -> String {
        let canon = match &self.parent {
            None => serde_json::to_string(&(&self.output, &self.root)),
            Some(parent) => serde_json::to_string(&(&self.output, &self.root, parent)),
        }
        .unwrap_or_default();
        crate::incumbent::sha256_hex(canon.as_bytes())[..16].to_string()
    }

    /// Correction for one record.
    pub fn evaluate(&self, inputs: &[f32]) -> f32 {
        self.root.evaluate(inputs)
    }

    /// Correction for one record, with `hidden(uuid)` giving each hidden
    /// source's activation (Issue #139).
    pub fn evaluate_with(&self, inputs: &[f32], hidden: &dyn Fn(&str) -> f32) -> f32 {
        self.root.evaluate_with(inputs, hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stump_evaluates_with_strict_greater_than_and_nan_left() {
        let n = Node::stump(0, 0.5, -1.0, 1.0);
        assert_eq!(n.evaluate(&[0.5]), -1.0);
        assert_eq!(n.evaluate(&[0.5000001]), 1.0);
        assert_eq!(n.evaluate(&[f32::NAN]), -1.0);
        assert_eq!(n.evaluate(&[f32::INFINITY]), 1.0);
        assert_eq!(n.evaluate(&[f32::NEG_INFINITY]), -1.0);
        assert_eq!(n.depth(), 1);
        assert_eq!(n.split_count(), 1);
    }

    #[test]
    fn json_round_trip_and_id_ignore_provenance() {
        let p = Patch::new(0, Node::stump(3, 0.25, 0.0, 0.01), Provenance::default());
        let json = serde_json::to_string(&p).unwrap();
        let back: Patch = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
        let mut q = p.clone();
        q.provenance.strategy = "other".into();
        assert_eq!(p.id(), q.id());
        let r = Patch::new(1, p.root.clone(), Provenance::default());
        assert_ne!(p.id(), r.id());
    }

    #[test]
    fn a_refinement_round_trips_and_its_parent_is_part_of_its_identity() {
        let root = Node::stump(3, 0.25, 0.0, 0.01);
        let plain = Patch::new(0, root.clone(), Provenance::default());
        let attach = |uuid: &str, role| Attach {
            uuid: uuid.into(),
            role,
        };
        let a = Patch::refinement(
            0,
            attach("forest-x-if0", BranchRole::Positive),
            root.clone(),
            Provenance::default(),
        );
        let b = Patch::refinement(
            0,
            attach("forest-x-if0", BranchRole::Negative),
            root,
            Provenance::default(),
        );
        assert_eq!(a.version, REFINE_PATCH_FORMAT_VERSION);
        assert_ne!(a.id(), plain.id());
        assert_ne!(a.id(), b.id());
        let json = serde_json::to_string(&a).unwrap();
        assert!(json.contains(r#""parent":{"uuid":"forest-x-if0","role":"positive"}"#));
        assert_eq!(serde_json::from_str::<Patch>(&json).unwrap(), a);
        // A patch with no parent serialises exactly as v1 always did.
        assert!(!serde_json::to_string(&plain).unwrap().contains("parent"));
    }

    #[test]
    fn a_hidden_source_term_round_trips_and_reads_the_activation_it_is_given() {
        let n = Node::Split {
            condition: Condition::hidden("h-1", 0.5),
            left: Box::new(Node::leaf(-1.0)),
            right: Box::new(Node::leaf(1.0)),
        };
        let p = Patch::new(0, n.clone(), Provenance::default());
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains(r#"{"neuron":"h-1","feature":0,"weight":1.0}"#));
        assert_eq!(serde_json::from_str::<Patch>(&json).unwrap(), p);
        assert_eq!(n.evaluate_with(&[], &|_| 0.75), 1.0);
        assert_eq!(n.evaluate_with(&[], &|_| 0.25), -1.0);
        // Inputs alone cannot evaluate it: NaN goes left.
        assert_eq!(n.evaluate(&[0.9]), -1.0);
        assert_eq!(n.hidden_sources(), vec!["h-1".to_string()]);
        assert!(n.features().is_empty());
        assert!(!Condition::hidden("h-1", 0.5).is_axis_aligned());
        // An input term serialises exactly as it always did.
        let plain = serde_json::to_string(&Term::input(3, 1.0)).unwrap();
        assert_eq!(plain, r#"{"feature":3,"weight":1.0}"#);
    }

    #[test]
    fn a_linear_leaf_round_trips_and_evaluates_like_the_kernel() {
        let n = Node::Split {
            condition: Condition::axis(0, 0.0),
            left: Box::new(Node::linear_leaf(0.1, vec![(1, -2.0)])),
            right: Box::new(Node::linear_leaf(-0.1, vec![(1, 2.0)])),
        };
        assert!(n.has_slopes());
        assert_eq!(n.features(), vec![0, 1]);
        assert_eq!(n.evaluate(&[1.0, 0.5]), 2.0 * 0.5 - 0.1);
        assert_eq!(n.evaluate(&[-1.0, 0.5]), -2.0 * 0.5 + 0.1);
        assert_eq!(n.scaled(0.5).evaluate(&[1.0, 0.5]), 0.5 * (2.0 * 0.5 - 0.1));
        let p = Patch::new(0, n, Provenance::default());
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains(r#""slopes":[{"feature":1,"weight":2.0}]"#));
        assert_eq!(serde_json::from_str::<Patch>(&json).unwrap(), p);
        // A constant leaf serialises and prints as it always did.
        assert_eq!(
            serde_json::to_string(&Node::leaf(0.5)).unwrap(),
            r#"{"kind":"leaf","correction":0.5}"#
        );
        assert_eq!(format!("{:?}", Node::leaf(0.5)), "Leaf { correction: 0.5 }");
        assert!(!Node::stump(0, 0.0, 0.0, 1.0).has_slopes());
    }

    #[test]
    fn oblique_condition_accumulates_in_order() {
        let c = Condition {
            terms: vec![Term::input(0, 0.5), Term::input(1, -1.0)],
            threshold: 0.0,
        };
        assert!(c.goes_right(&[2.0, 0.5]));
        assert!(!c.goes_right(&[2.0, 1.0]));
        assert!(!c.is_axis_aligned());
    }
}

//! Deepen trees already on the creature (Issue #140).
//!
//! Forests otherwise only plants *new* trees at an output's anchor, at most
//! three conditions deep, so a region that needs a fourth condition can never
//! be expressed and restating a path costs a neuron per condition. This module
//! reads the decision trees back **from the creature's structure** — not from
//! the journal, so patches from every host and every earlier run count, at the
//! weights they carry now — and lets the search split one of their leaves.
//!
//! A refinement stays additive. The leaf keeps its constant edge; a new child
//! `IF` is added to the same branch sum of the parent and its own leaves are
//! corrections. Nothing is deleted or rewired, and a child whose corrections
//! are zero leaves every record bit-identical.
//!
//! ## What counts as a tree
//!
//! Walking down from the output's anchor (the same walk every graft does,
//! Issue #58):
//!
//! * a **root** is an `IF` neuron whose only outward edges reach the anchor —
//!   one untyped synapse into a point-wise anchor, or the typed
//!   `positive`/`negative` pair of equal weight into an `IF` anchor (#68);
//! * a **child** is an `IF` neuron in a branch sum of a tree node whose one and
//!   only outward edge is that branch;
//! * a node is **readable** when every condition source is an input or a
//!   bias-only constant, so the condition is a [`Condition`] the abstract
//!   evaluator can route a record through. An unreadable node and everything
//!   below it are left alone;
//! * a **leaf** is a branch of a readable node whose sources are all
//!   constants.
//!
//! The `gain` of a node is `d(output) / d(node activation)`, the product of
//! the outward weights on the way up times the anchor's own gain. A
//! refinement's outward edge carries `1 / gain`, so its leaves live in the
//! same output correction space every other patch is searched in.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use neat_core::{CreatureExport, NeuronExport, SquashType, SynapseExport};

use crate::bins::BinCache;
use crate::candidates::Discovery;
use crate::graft::{GraftError, output_uuid, resolve_anchor};
use crate::histogram::{BinnedChunk, HistogramSet, MemorySource, SearchControls, search_stumps};
use crate::patch::{Attach, BranchRole, Condition, Term};
use crate::strategies::{SearchSet, for_each_set_row};
use crate::tree::{TreeSearchControls, grow_tree};

/// One readable `IF` node of a tree already on the creature.
#[derive(Debug, Clone, PartialEq)]
pub struct TreeNode {
    /// The neuron's UUID.
    pub uuid: String,
    /// Its condition, as the abstract evaluator reads it.
    pub condition: Condition,
    /// The node it hangs under and on which side (`true` = positive/right);
    /// `None` for a root.
    pub parent: Option<(usize, bool)>,
    /// Conditions from the root down to and including this one.
    pub depth: usize,
    /// `d(output) / d(this node's activation)`.
    pub gain: f64,
}

/// A leaf of a tree already on the creature: one branch of a readable node
/// whose sources are all constants.
#[derive(Debug, Clone, PartialEq)]
pub struct ExistingLeaf {
    /// Index of the node in [`ForestView::nodes`].
    pub node: usize,
    /// Where a refinement of this leaf attaches.
    pub attach: Attach,
    /// Depth of the leaf (its node's depth).
    pub depth: usize,
    /// `d(output) / d(branch sum)` — the node's gain.
    pub gain: f64,
    /// What the leaf contributes now, at the node: `Σ bias · weight` over its
    /// constants plus the node's own bias.
    pub value: f64,
}

/// The trees an output's anchor carries, read back from structure.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ForestView {
    /// Readable nodes, parents before children.
    pub nodes: Vec<TreeNode>,
    /// Leaves, in node order, negative branch before positive.
    pub leaves: Vec<ExistingLeaf>,
    /// For each node, the nodes in each branch (`[negative, positive]`).
    children: Vec<[Vec<usize>; 2]>,
    /// For each node, the leaf in each branch, when that branch is a leaf.
    leaf_of: Vec<[Option<usize>; 2]>,
    /// Indices of root nodes.
    roots: Vec<usize>,
}

/// A node waiting to be read: its UUID, the node and side it hangs under,
/// and its gain.
type Visit<'a> = (&'a str, Option<(usize, bool)>, f64);

/// The weight `w` a root grafted with `--if-correction relay` carries into
/// an `IF` anchor: `root → anchor` `positive` at `w`, plus `root → relay` at
/// `a` where the relay is a bias-0 IDENTITY neuron fed by the root alone whose
/// only outward edge is `relay → anchor` `negative` at `w / a`. Both branch
/// sums then see `w · root`, as with the typed pair. `None` for any other
/// shape.
fn relay_root_weight(
    outs: &[&SynapseExport],
    anchor: &str,
    by_uuid: &HashMap<&str, &NeuronExport>,
    inward: &HashMap<&str, Vec<&SynapseExport>>,
    outward: &HashMap<&str, Vec<&SynapseExport>>,
) -> Option<f64> {
    let [a, b] = outs else { return None };
    let (direct, via) = if a.to_uuid == anchor { (a, b) } else { (b, a) };
    if direct.to_uuid != anchor
        || direct.synapse_type.as_deref() != Some("positive")
        || via.to_uuid == anchor
        || via.synapse_type.is_some()
    {
        return None;
    }
    let relay = by_uuid.get(via.to_uuid.as_str())?;
    let identity = relay
        .squash
        .as_deref()
        .is_none_or(|s| matches!(neat_core::parse_squash_name(s), Ok(SquashType::Identity)));
    let fed_by_root_alone = inward
        .get(relay.uuid.as_str())
        .is_some_and(|ins| ins.len() == 1);
    let [out] = outward
        .get(relay.uuid.as_str())
        .map_or(&[][..], Vec::as_slice)
    else {
        return None;
    };
    let w = direct.weight;
    let carried = via.weight * out.weight;
    (relay.neuron_type == "hidden"
        && identity
        && relay.bias == 0.0
        && fed_by_root_alone
        && out.to_uuid == anchor
        && out.synapse_type.as_deref() == Some("negative")
        && (carried - w).abs() <= 1e-9 * w.abs().max(1.0))
    .then_some(w)
}

fn role_of(s: &SynapseExport) -> Option<BranchRole> {
    match s.synapse_type.as_deref() {
        Some("positive") | None => Some(BranchRole::Positive),
        Some("negative") => Some(BranchRole::Negative),
        _ => None,
    }
}

fn is_if(n: &NeuronExport) -> bool {
    n.neuron_type == "hidden"
        && n.squash
            .as_deref()
            .is_some_and(|s| matches!(neat_core::parse_squash_name(s), Ok(SquashType::If)))
}

fn is_plain_constant(n: &NeuronExport) -> bool {
    n.neuron_type == "constant" && n.squash.is_none()
}

/// Parse `input-k` into `k` when `k < width`.
fn input_index(uuid: &str, width: usize) -> Option<usize> {
    let k: usize = uuid.strip_prefix("input-")?.parse().ok()?;
    (k < width).then_some(k)
}

impl ForestView {
    /// Read every tree hanging off `output`'s anchor.
    ///
    /// # Errors
    ///
    /// The [`GraftError`] the anchor walk fails with: an output nothing can be
    /// grafted onto carries no tree a refinement could reach either.
    pub fn read(creature: &CreatureExport, output: usize) -> Result<Self, GraftError> {
        let anchor = resolve_anchor(creature, &output_uuid(creature, output)?)?;
        let by_uuid: HashMap<&str, &NeuronExport> = creature
            .neurons
            .iter()
            .map(|n| (n.uuid.as_str(), n))
            .collect();
        let mut inward: HashMap<&str, Vec<&SynapseExport>> = HashMap::new();
        let mut outward: HashMap<&str, Vec<&SynapseExport>> = HashMap::new();
        for s in &creature.synapses {
            inward.entry(s.to_uuid.as_str()).or_default().push(s);
            outward.entry(s.from_uuid.as_str()).or_default().push(s);
        }
        let anchor_is_if = anchor.squash == SquashType::If;
        let mut view = Self::default();
        let mut seen: HashSet<&str> = HashSet::new();
        // Roots, in the order the anchor lists its sources.
        let mut stack: Vec<Visit<'_>> = Vec::new();
        let mut root_sources: Vec<&str> = Vec::new();
        for s in inward.get(anchor.uuid.as_str()).into_iter().flatten() {
            if !root_sources.contains(&s.from_uuid.as_str()) {
                root_sources.push(s.from_uuid.as_str());
            }
        }
        for uuid in root_sources {
            let Some(n) = by_uuid.get(uuid) else { continue };
            if !is_if(n) {
                continue;
            }
            let outs = outward.get(uuid).map_or(&[][..], Vec::as_slice);
            if anchor_is_if
                && let Some(w) = relay_root_weight(outs, &anchor.uuid, &by_uuid, &inward, &outward)
            {
                stack.push((uuid, None, w * anchor.gain));
                continue;
            }
            let weight = outs.first().map(|s| s.weight);
            let to_anchor = outs.iter().all(|s| s.to_uuid == anchor.uuid);
            let same_weight = outs.iter().all(|s| Some(s.weight) == weight);
            let shape_ok = if anchor_is_if {
                outs.len() == 2
                    && outs
                        .iter()
                        .any(|s| s.synapse_type.as_deref() == Some("positive"))
                    && outs
                        .iter()
                        .any(|s| s.synapse_type.as_deref() == Some("negative"))
            } else {
                outs.len() == 1 && outs[0].synapse_type.is_none()
            };
            if let Some(w) = weight
                && to_anchor
                && same_weight
                && shape_ok
            {
                stack.push((uuid, None, w * anchor.gain));
            }
        }
        // Depth-first, parents before children; `stack` is reversed so the
        // first root is read first.
        stack.reverse();
        while let Some((uuid, parent, gain)) = stack.pop() {
            if !seen.insert(uuid) || !gain.is_finite() || gain == 0.0 {
                continue;
            }
            let Some(neuron) = by_uuid.get(uuid) else {
                continue;
            };
            let ins = inward.get(uuid).map_or(&[][..], Vec::as_slice);
            let Some(condition) = read_condition(ins, &by_uuid, creature.input) else {
                continue;
            };
            let index = view.nodes.len();
            let depth = parent.map_or(1, |(p, _)| view.nodes[p].depth + 1);
            view.nodes.push(TreeNode {
                uuid: uuid.to_string(),
                condition,
                parent,
                depth,
                gain,
            });
            view.children.push([Vec::new(), Vec::new()]);
            view.leaf_of.push([None, None]);
            if let Some((p, right)) = parent {
                view.children[p][usize::from(right)].push(index);
            } else {
                view.roots.push(index);
            }
            let mut pending: Vec<Visit<'_>> = Vec::new();
            for role in [BranchRole::Negative, BranchRole::Positive] {
                let sources: Vec<&SynapseExport> = ins
                    .iter()
                    .copied()
                    .filter(|s| s.synapse_type.as_deref() != Some("condition"))
                    .filter(|s| role_of(s) == Some(role))
                    .collect();
                let right = role == BranchRole::Positive;
                let all_constant = !sources.is_empty()
                    && sources.iter().all(|s| {
                        by_uuid
                            .get(s.from_uuid.as_str())
                            .is_some_and(|n| is_plain_constant(n))
                    });
                if all_constant {
                    let value = sources
                        .iter()
                        .map(|s| by_uuid[s.from_uuid.as_str()].bias * s.weight)
                        .sum::<f64>()
                        + neuron.bias;
                    view.leaf_of[index][usize::from(right)] = Some(view.leaves.len());
                    view.leaves.push(ExistingLeaf {
                        node: index,
                        attach: Attach {
                            uuid: uuid.to_string(),
                            role,
                        },
                        depth,
                        gain,
                        value,
                    });
                    continue;
                }
                for s in sources {
                    let from = s.from_uuid.as_str();
                    let Some(child) = by_uuid.get(from) else {
                        continue;
                    };
                    let outs = outward.get(from).map_or(&[][..], Vec::as_slice);
                    if is_if(child) && outs.len() == 1 {
                        pending.push((from, Some((index, right)), gain * s.weight));
                    }
                }
            }
            pending.reverse();
            stack.extend(pending);
        }
        Ok(view)
    }

    /// The node `uuid`, if it is a readable tree node.
    pub fn node(&self, uuid: &str) -> Option<&TreeNode> {
        self.nodes.iter().find(|n| n.uuid == uuid)
    }

    /// Conditions from the root down to `node`, with the side each takes
    /// (`true` = right) — the path a record must follow to reach `node`'s
    /// branch `right`.
    pub fn path(&self, node: usize, right: bool) -> Vec<(&Condition, bool)> {
        let mut out = vec![(&self.nodes[node].condition, right)];
        let mut at = self.nodes[node].parent;
        while let Some((p, r)) = at {
            out.push((&self.nodes[p].condition, r));
            at = self.nodes[p].parent;
        }
        out.reverse();
        out
    }

    /// `true` when a record with `inputs` reaches `leaf`.
    pub fn reaches(&self, leaf: &ExistingLeaf, inputs: &[f32]) -> bool {
        self.path(leaf.node, leaf.attach.role == BranchRole::Positive)
            .iter()
            .all(|(c, right)| c.goes_right(inputs) == *right)
    }

    /// Every leaf a record with `inputs` reaches, one per tree (more where a
    /// branch carries more than one subtree).
    pub fn route(&self, inputs: &[f32], out: &mut Vec<usize>) {
        out.clear();
        let mut stack: Vec<usize> = self.roots.clone();
        while let Some(n) = stack.pop() {
            let side = usize::from(self.nodes[n].condition.goes_right(inputs));
            if let Some(l) = self.leaf_of[n][side] {
                out.push(l);
            }
            stack.extend(self.children[n][side].iter().copied());
        }
    }
}

/// Read a node's condition role as a [`Condition`]: inputs become terms in
/// synapse order, bias-only constants fold into the threshold. Any other
/// source (a hidden neuron) makes the node unreadable.
fn read_condition(
    ins: &[&SynapseExport],
    by_uuid: &HashMap<&str, &NeuronExport>,
    width: usize,
) -> Option<Condition> {
    let mut terms = Vec::new();
    let mut constant = 0.0f64;
    let mut any = false;
    for s in ins
        .iter()
        .filter(|s| s.synapse_type.as_deref() == Some("condition"))
    {
        any = true;
        if let Some(feature) = input_index(&s.from_uuid, width) {
            terms.push(Term {
                feature,
                weight: s.weight as f32,
            });
        } else if let Some(n) = by_uuid.get(s.from_uuid.as_str())
            && is_plain_constant(n)
        {
            constant += n.bias * s.weight;
        } else {
            return None;
        }
    }
    (any && !terms.is_empty()).then(|| Condition {
        terms,
        threshold: (-constant) as f32,
    })
}

/// Controls for the refinement search.
#[derive(Debug, Clone)]
pub struct RefineControls {
    /// Leaves refined per iteration.
    pub leaves: usize,
    /// Largest total depth a refined path may reach.
    pub max_total_depth: usize,
    /// Depth of the subtree planted under a leaf (capped by what is left).
    pub max_depth: usize,
    /// Stump / tree controls (min records per new leaf, clamp, kinds…).
    pub stump: SearchControls,
    /// Growth policy for subtrees deeper than a stump.
    pub growth: crate::config::GrowthPolicy,
    /// Threads for histogram accumulation.
    pub threads: usize,
}

/// Per-leaf sufficient statistics over the search set.
#[derive(Debug, Clone, Copy, Default)]
struct LeafStats {
    rows: u64,
    weight: f64,
    sum: f64,
    sum_sq: f64,
}

impl LeafStats {
    /// Weighted SSE about the leaf's own mean: what a constant cannot fit.
    fn spread(&self) -> f64 {
        if self.weight <= 0.0 {
            0.0
        } else {
            self.sum_sq - self.sum * self.sum / self.weight
        }
    }
}

/// What the refinement search found, and what it looked at.
#[derive(Debug, Clone, Default)]
pub struct RefineOutcome {
    /// Discoveries, each a subtree under one existing leaf.
    pub discoveries: Vec<Discovery>,
    /// Readable trees on the anchor.
    pub trees: usize,
    /// Leaves on them.
    pub leaves: usize,
    /// Leaves searched.
    pub searched: usize,
}

/// Search for refinements of the leaves of `view` on the search set.
///
/// Two passes over the search set's raw records: the first routes every
/// record through every readable tree and keeps per-leaf residual statistics,
/// so the leaves with enough records and the most residual spread a constant
/// cannot fit are chosen; the second collects the rows reaching each chosen
/// leaf, and the ordinary histogram stump and tree search runs on those rows
/// alone. A subtree's leaves are corrections in output space; the graft
/// divides by the leaf's gain.
///
/// # Errors
///
/// A corpus read failure.
pub fn search_refinements(
    view: &ForestView,
    set: &SearchSet,
    bins: &BinCache,
    training_dir: &Path,
    outputs: usize,
    chunk_records: usize,
    controls: &RefineControls,
) -> Result<RefineOutcome, String> {
    let mut outcome = RefineOutcome {
        trees: view.roots.len(),
        leaves: view.leaves.len(),
        ..RefineOutcome::default()
    };
    if view.leaves.is_empty() || controls.leaves == 0 {
        return Ok(outcome);
    }
    let mut stats = vec![LeafStats::default(); view.leaves.len()];
    let mut reached = Vec::new();
    let (residual, weight) = set_residuals(set);
    for_each_set_row(
        set,
        training_dir,
        bins.features(),
        outputs,
        chunk_records,
        |pos, row| {
            view.route(row, &mut reached);
            let (r, w) = (f64::from(residual[pos]), f64::from(weight[pos]));
            for &l in &reached {
                let s = &mut stats[l];
                s.rows += 1;
                s.weight += w;
                s.sum += w * r;
                s.sum_sq += w * r * r;
            }
            Ok(())
        },
    )?;
    // Leaves worth refining: room left under the depth cap, enough records for
    // two new leaves, and the most spread a constant cannot fit.
    let mut ranked: Vec<usize> = (0..view.leaves.len())
        .filter(|&l| view.leaves[l].depth < controls.max_total_depth)
        .filter(|&l| stats[l].weight >= 2.0 * controls.stump.min_leaf_records)
        .filter(|&l| stats[l].spread() > 0.0)
        .collect();
    ranked.sort_by(|&a, &b| {
        stats[b]
            .spread()
            .total_cmp(&stats[a].spread())
            .then_with(|| view.leaves[a].attach.uuid.cmp(&view.leaves[b].attach.uuid))
            .then_with(|| a.cmp(&b))
    });
    ranked.truncate(controls.leaves);
    if ranked.is_empty() {
        return Ok(outcome);
    }
    outcome.searched = ranked.len();
    // Second pass: which rows reach each chosen leaf.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); ranked.len()];
    for_each_set_row(
        set,
        training_dir,
        bins.features(),
        outputs,
        chunk_records,
        |pos, row| {
            for (k, &l) in ranked.iter().enumerate() {
                if view.reaches(&view.leaves[l], row) {
                    members[k].push(pos);
                }
            }
            Ok(())
        },
    )?;
    let bins_per_feature = set.bins_per_feature(bins);
    let threshold = set.threshold(bins);
    for (k, &l) in ranked.iter().enumerate() {
        let leaf = &view.leaves[l];
        let source = subset(set, &members[k]);
        let notes = [
            format!(
                "refine={}:{} leaf-depth={}",
                leaf.attach.uuid,
                leaf.attach.role.label(),
                leaf.depth
            ),
            format!("leaf-records={}", members[k].len()),
        ];
        let hist = HistogramSet::from_source_threads(&source, &bins_per_feature, controls.threads)?;
        if let Some(best) = search_stumps(&hist, &threshold, &controls.stump, "cpu")
            .into_iter()
            .next()
        {
            let mut d = Discovery::from_stump(&best, set.feature_map[best.feature], "cpu");
            d.strategy = "refine-stump".into();
            d.parent = Some(leaf.attach.clone());
            d.notes.extend(notes.iter().cloned());
            outcome.discoveries.push(d);
        }
        let room = controls
            .max_total_depth
            .saturating_sub(leaf.depth)
            .min(controls.max_depth);
        if room >= 2 {
            let tree_controls = TreeSearchControls {
                stump: controls.stump.clone(),
                max_depth: room.min(3),
                growth: controls.growth,
            };
            let best = grow_tree(
                &source,
                &bins_per_feature,
                &threshold,
                &|f| set.feature_map[f],
                &tree_controls,
                None,
            )?
            .into_iter()
            .filter(|t| t.depth > 1)
            .max_by(|a, b| a.gain.total_cmp(&b.gain));
            if let Some(t) = best {
                let mut d = Discovery::from_tree(&t, "cpu");
                d.strategy = format!("refine-tree-depth{}", t.depth);
                d.parent = Some(leaf.attach.clone());
                d.notes.extend(notes.iter().cloned());
                outcome.discoveries.push(d);
            }
        }
    }
    Ok(outcome)
}

/// Residual and weight of every search-set row, in set order.
fn set_residuals(set: &SearchSet) -> (Vec<f32>, Vec<f32>) {
    let mut residual = Vec::with_capacity(set.records() as usize);
    let mut weight = Vec::with_capacity(set.records() as usize);
    for c in &set.source.chunks {
        residual.extend_from_slice(&c.residual);
        match &c.weight {
            Some(w) => weight.extend_from_slice(w),
            None => weight.extend(std::iter::repeat_n(1.0, c.records)),
        }
    }
    (residual, weight)
}

/// The search set restricted to the rows at `positions` (ascending, set order).
fn subset(set: &SearchSet, positions: &[usize]) -> MemorySource {
    let nf = set.features();
    let mut chunk = BinnedChunk {
        records: 0,
        features: nf,
        bins: Vec::with_capacity(positions.len() * nf),
        residual: Vec::with_capacity(positions.len()),
        weight: None,
        first_index: 0,
    };
    let mut weights: Vec<f32> = Vec::with_capacity(positions.len());
    let mut any_weight = false;
    let mut base = 0usize;
    let mut want = positions.iter().peekable();
    for c in &set.source.chunks {
        while let Some(&&p) = want.peek() {
            if p >= base + c.records {
                break;
            }
            let r = p - base;
            chunk.bins.extend_from_slice(&c.bins[r * nf..(r + 1) * nf]);
            chunk.residual.push(c.residual[r]);
            let w = c.weight.as_ref().map_or(1.0, |w| w[r]);
            any_weight |= c.weight.is_some();
            weights.push(w);
            chunk.records += 1;
            want.next();
        }
        base += c.records;
    }
    if any_weight {
        chunk.weight = Some(weights);
    }
    MemorySource {
        chunks: vec![chunk],
        label: format!("{}+refine-leaf", set.source.label),
    }
}

/// Test and parity fixtures (Issue #140).
pub mod fixtures {
    use neat_core::CreatureExport;

    use super::ForestView;
    use crate::graft::graft_patch;
    use crate::patch::{BranchRole, Condition, Node, Patch, Provenance};

    /// `base` with a depth-2 tree on output 0, one of its leaves refined by a
    /// stump, and that stump's own left leaf refined again — the shape a
    /// refinement of an already-refined leaf leaves behind. Needs three inputs.
    ///
    /// # Panics
    ///
    /// When `base` cannot take the graft; this is a fixture.
    pub fn refined_twice(base: &CreatureExport) -> CreatureExport {
        let planted = Node::Split {
            condition: Condition::axis(0, 0.1),
            left: Box::new(Node::stump(1, -0.5, -0.2, 0.1)),
            right: Box::new(Node::stump(2, 0.0, 0.3, -0.1)),
        };
        let mut c = graft_patch(base, &Patch::new(0, planted, Provenance::default()))
            .expect("plant")
            .creature;
        for (depth, role, stump) in [
            (2, BranchRole::Positive, Node::stump(2, 0.6, 0.05, -0.15)),
            (3, BranchRole::Negative, Node::stump(1, 0.3, 0.0, 0.07)),
        ] {
            let view = ForestView::read(&c, 0).expect("read");
            let leaf = view
                .leaves
                .iter()
                .rfind(|l| l.depth == depth && l.attach.role == role)
                .expect("leaf")
                .attach
                .clone();
            c = graft_patch(
                &c,
                &Patch::refinement(0, leaf, stump, Provenance::default()),
            )
            .expect("refine")
            .creature;
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ForestsConfig, GraftConstants};
    use crate::graft::fixtures::{identity_creature, if_output_creature};
    use crate::graft::{graft_patch, graft_patch_with};
    use crate::patch::{Node, Patch, Provenance};
    use neat_core::compile_creature;

    fn records(n: usize, width: usize) -> Vec<Vec<f32>> {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        (0..n)
            .map(|_| {
                (0..width)
                    .map(|_| {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((seed >> 40) as f32) / (1u64 << 24) as f32
                    })
                    .collect()
            })
            .collect()
    }

    fn depth2() -> Node {
        Node::Split {
            condition: Condition::axis(0, 0.5),
            left: Box::new(Node::stump(1, 0.25, -0.1, 0.2)),
            right: Box::new(Node::stump(2, 0.75, 0.3, -0.4)),
        }
    }

    /// The leaf of `view` a record with `x` reaches in the tree rooted at the
    /// node `root_uuid`.
    fn leaf_named<'a>(view: &'a ForestView, uuid: &str, role: BranchRole) -> &'a ExistingLeaf {
        view.leaves
            .iter()
            .find(|l| l.attach.uuid == uuid && l.attach.role == role)
            .unwrap_or_else(|| panic!("no leaf {uuid}:{}", role.label()))
    }

    #[test]
    fn planted_trees_are_read_back_and_route_like_the_evaluator() {
        for base in [identity_creature(4, 1), if_output_creature(4)] {
            for constants in [GraftConstants::Shared, GraftConstants::PerPatch] {
                let patch = Patch::new(0, depth2(), Provenance::default());
                let grafted = graft_patch_with(&base, &patch, constants).unwrap().creature;
                let view = ForestView::read(&grafted, 0).unwrap();
                assert_eq!(view.roots.len(), 1, "one tree");
                assert_eq!(view.nodes.len(), 3);
                assert_eq!(view.leaves.len(), 4);
                assert!(view.leaves.iter().all(|l| l.depth == 2 && l.gain == 1.0));
                let mut reached = Vec::new();
                for x in records(300, 4) {
                    view.route(&x, &mut reached);
                    assert_eq!(reached.len(), 1);
                    let leaf = &view.leaves[reached[0]];
                    assert!(view.reaches(leaf, &x));
                    assert_eq!(leaf.value as f32, patch.evaluate(&x));
                }
            }
        }
    }

    #[test]
    fn an_unreadable_condition_hides_its_subtree() {
        let base = identity_creature(4, 1);
        let grafted = graft_patch(&base, &Patch::new(0, depth2(), Provenance::default()))
            .unwrap()
            .creature;
        let view = ForestView::read(&grafted, 0).unwrap();
        let root = view.roots[0];
        // Feed the root's condition from the output's own hidden neighbour:
        // it is no longer a function of the inputs alone.
        let mut broken = grafted.clone();
        let root_uuid = view.nodes[root].uuid.clone();
        let hidden = broken
            .neurons
            .iter()
            .find(|n| n.neuron_type == "hidden" && n.uuid != root_uuid)
            .unwrap()
            .uuid
            .clone();
        for s in &mut broken.synapses {
            if s.to_uuid == root_uuid && s.from_uuid.starts_with("input-") {
                s.from_uuid = hidden.clone();
            }
        }
        assert!(ForestView::read(&broken, 0).unwrap().nodes.is_empty());
    }

    /// Issue #140 — a refinement, and a refinement of that refinement, are
    /// grafted into the parent's branch sum and agree record by record with
    /// `incumbent + Σ [record reaches leaf] · subtree(record)`.
    #[test]
    fn refinements_match_the_abstract_evaluator_record_by_record() {
        for base in [identity_creature(4, 1), if_output_creature(4)] {
            for constants in [GraftConstants::Shared, GraftConstants::PerPatch] {
                let planted = Patch::new(0, depth2(), Provenance::default());
                let c1 = graft_patch_with(&base, &planted, constants)
                    .unwrap()
                    .creature;
                let view1 = ForestView::read(&c1, 0).unwrap();
                // Refine the leaf right of the right child (x0 > .5, x2 > .75).
                let right_child = view1
                    .nodes
                    .iter()
                    .find(|n| n.depth == 2 && n.parent == Some((0, true)))
                    .unwrap();
                let l1 = leaf_named(&view1, &right_child.uuid, BranchRole::Positive).clone();
                let r1 = Patch::refinement(
                    0,
                    l1.attach.clone(),
                    Node::stump(3, 0.5, 0.05, -0.15),
                    Provenance::default(),
                );
                let c2 = graft_patch_with(&c1, &r1, constants).unwrap().creature;
                let view2 = ForestView::read(&c2, 0).unwrap();
                assert_eq!(view2.nodes.len(), 4, "the refinement is a tree node now");
                assert_eq!(
                    view2.leaves.len(),
                    5,
                    "its leaf took the refined leaf's place"
                );
                let refined = view2.nodes.iter().find(|n| n.depth == 3).unwrap();
                assert_eq!(
                    refined
                        .parent
                        .map(|(p, r)| (view2.nodes[p].uuid.clone(), r)),
                    Some((right_child.uuid.clone(), true))
                );
                // …and refine the refinement's own left leaf.
                let l2 = leaf_named(&view2, &refined.uuid, BranchRole::Negative).clone();
                let r2 = Patch::refinement(
                    0,
                    l2.attach.clone(),
                    Node::stump(1, 0.6, 0.0, 0.07),
                    Provenance::default(),
                );
                let c3 = graft_patch_with(&c2, &r2, constants).unwrap().creature;
                let view3 = ForestView::read(&c3, 0).unwrap();
                assert!(view3.nodes.iter().any(|n| n.depth == 4));
                let mut net0 = compile_creature(&base).unwrap();
                let mut net3 = compile_creature(&c3).unwrap();
                let (mut hit1, mut hit2) = (0, 0);
                for x in records(2000, 4) {
                    let mut expected = planted.evaluate(&x);
                    if view1.reaches(&l1, &x) {
                        expected += r1.evaluate(&x);
                        hit1 += 1;
                    }
                    if view2.reaches(&l2, &x) {
                        expected += r2.evaluate(&x);
                        hit2 += 1;
                    }
                    let got = net3.activate(&x, 1)[0] - net0.activate(&x, 1)[0];
                    assert!(
                        (got - expected).abs() <= 1e-6,
                        "{got} vs {expected} at {x:?}"
                    );
                }
                assert!(
                    hit1 > 50 && hit2 > 10,
                    "fixture must reach both leaves: {hit1} {hit2}"
                );
            }
        }
    }

    /// Issue #140 — refinements graft at the weights a tree carries *now*.
    /// Retune a planted tree so its root edge and one child edge are no longer
    /// what the graft wrote, making the refined leaf's gain differ from 1 and
    /// from the anchor's (0.5 on the MIN-clamped fixture); the refinement must still move the output by exactly
    /// its own correction on the rows reaching the leaf, and by 0 elsewhere.
    #[test]
    fn refinements_graft_at_the_gain_a_retuned_tree_carries() {
        use crate::graft::fixtures::min_clamped_if_creature;
        for base in [
            identity_creature(4, 1),
            if_output_creature(4),
            min_clamped_if_creature(4),
        ] {
            let anchor_gain = resolve_anchor(&base, &output_uuid(&base, 0).unwrap())
                .unwrap()
                .gain;
            let planted = Patch::new(0, depth2(), Provenance::default());
            let mut c1 = graft_patch(&base, &planted).unwrap().creature;
            let view0 = ForestView::read(&c1, 0).unwrap();
            let root = view0.nodes[view0.roots[0]].uuid.clone();
            let right_child = view0
                .nodes
                .iter()
                .find(|n| n.depth == 2 && n.parent == Some((0, true)))
                .unwrap()
                .uuid
                .clone();
            for s in &mut c1.synapses {
                if s.from_uuid == root {
                    s.weight *= 3.0;
                } else if s.from_uuid == right_child && s.to_uuid == root {
                    s.weight *= 0.5;
                }
            }
            let view1 = ForestView::read(&c1, 0).unwrap();
            let l1 = leaf_named(&view1, &right_child, BranchRole::Positive).clone();
            assert!(
                (l1.gain - 1.5).abs() < 1e-9,
                "root gain 3, child edge 0.5: {}",
                l1.gain
            );
            assert!((l1.gain - anchor_gain).abs() > 0.1, "and not the anchor's");
            let r1 = Patch::refinement(
                0,
                l1.attach.clone(),
                Node::stump(1, 0.6, 0.05, -0.15),
                Provenance::default(),
            );
            let c2 = graft_patch(&c1, &r1).unwrap().creature;
            let mut net1 = compile_creature(&c1).unwrap();
            let mut net2 = compile_creature(&c2).unwrap();
            let (mut hit, mut miss) = (0, 0);
            for mut x in records(2000, 4) {
                // Keep the MIN clamp's other source high, so the clamp passes
                // the anchor through: on the clamped side nothing moves.
                x[3] = 10.0;
                let before = net1.activate(&x, 1)[0];
                let after = net2.activate(&x, 1)[0];
                let expected = if view1.reaches(&l1, &x) {
                    hit += 1;
                    r1.evaluate(&x)
                } else {
                    miss += 1;
                    0.0
                };
                assert!(
                    (after - before - expected).abs() <= 1e-6,
                    "{} vs {expected} at {x:?}",
                    after - before
                );
            }
            assert!(hit > 50 && miss > 50, "fixture reaches both: {hit} {miss}");
        }
    }

    /// Trees grafted with `--if-correction relay` (the only shape before
    /// #68) are read back too: the root feeds the `IF` anchor's positive
    /// branch directly and its negative branch through an IDENTITY relay.
    /// They route like the evaluator, and a refinement of one of their leaves
    /// moves the output by exactly its correction there and 0 elsewhere.
    #[test]
    fn relay_grafted_trees_are_read_back_and_refined() {
        use crate::config::IfCorrection;
        use crate::graft::fixtures::min_clamped_if_creature;
        use crate::graft::{GraftOptions, graft_patch_options};
        for base in [if_output_creature(4), min_clamped_if_creature(4)] {
            let planted = Patch::new(0, depth2(), Provenance::default());
            let c1 = graft_patch_options(
                &base,
                &planted,
                GraftOptions::new(GraftConstants::Shared).with_if_correction(IfCorrection::Relay),
            )
            .unwrap()
            .creature;
            assert!(
                c1.neurons.iter().any(|n| n.uuid.contains("-relay")),
                "the fixture really is relay-shaped"
            );
            let view = ForestView::read(&c1, 0).unwrap();
            assert_eq!(view.roots.len(), 1, "the relay-shaped root is read");
            assert_eq!(view.leaves.len(), 4);
            assert!(view.leaves.iter().all(|l| (l.gain - 1.0).abs() < 1e-9));
            let leaf = view.leaves[1].clone();
            let r = Patch::refinement(
                0,
                leaf.attach.clone(),
                Node::stump(1, 0.6, 0.05, -0.15),
                Provenance::default(),
            );
            let c2 = graft_patch(&c1, &r).unwrap().creature;
            let mut net1 = compile_creature(&c1).unwrap();
            let mut net2 = compile_creature(&c2).unwrap();
            let mut reached = Vec::new();
            let (mut hit, mut miss) = (0, 0);
            for mut x in records(2000, 4) {
                x[3] = 10.0; // keep the MIN clamp passing the anchor through
                view.route(&x, &mut reached);
                assert_eq!(reached.len(), 1);
                assert_eq!(view.leaves[reached[0]].value as f32, planted.evaluate(&x));
                let expected = if view.reaches(&leaf, &x) {
                    hit += 1;
                    r.evaluate(&x)
                } else {
                    miss += 1;
                    0.0
                };
                let moved = net2.activate(&x, 1)[0] - net1.activate(&x, 1)[0];
                assert!((moved - expected).abs() <= 1e-6, "{moved} vs {expected}");
            }
            assert!(hit > 50 && miss > 50, "{hit} {miss}");
        }
    }

    #[test]
    fn a_refinement_whose_parent_is_gone_is_refused() {
        let base = identity_creature(4, 1);
        let r = Patch::refinement(
            0,
            Attach {
                uuid: "forest-gone-if0".into(),
                role: BranchRole::Positive,
            },
            Node::stump(3, 0.5, 0.05, -0.15),
            Provenance::default(),
        );
        assert_eq!(
            graft_patch(&base, &r).unwrap_err(),
            GraftError::NoSuchParent("forest-gone-if0".into())
        );
    }

    /// The target is +1 on the one region four conditions deep
    /// (`x0, x1, x2, x3 > 0.5`) and 0 elsewhere. No depth-3 tree can isolate
    /// it; a depth-3 tree planted first and then refined at its deepest leaf
    /// does.
    #[test]
    fn a_four_condition_region_needs_a_refinement() {
        use crate::bins::{BinBuildOptions, build_bin_cache};
        use crate::corpus::{corpus_info, write_bin_file};
        use crate::incumbent::Incumbent;
        use crate::residuals::compute_residuals;
        use crate::strategies::build_search_set;
        use neat_core::training_data::TrainingDataConfig;

        let tmp = tempfile::tempdir().unwrap();
        let region = |x: &[f32]| x.iter().all(|&v| v > 0.5);
        let recs: Vec<(Vec<f32>, Vec<f32>)> = records(8000, 4)
            .into_iter()
            .map(|x| {
                let y = x[0] + if region(&x) { 1.0 } else { 0.0 };
                (x, vec![y])
            })
            .collect();
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let tcfg = TrainingDataConfig::new(4, 1);
        let corpus = corpus_info(tmp.path(), &tcfg).unwrap();
        let bins = build_bin_cache(
            tmp.path(),
            &tcfg,
            &corpus,
            &BinBuildOptions {
                bins: 16,
                ..Default::default()
            },
        )
        .unwrap();
        let cfg = ForestsConfig {
            search_records: 0,
            min_leaf_records: 20.0,
            ..Default::default()
        };
        let mse = |c: &CreatureExport| {
            let inc = Incumbent::from_creature(c.clone(), "t").unwrap();
            compute_residuals(&inc, tmp.path(), &corpus, 512, 1)
                .unwrap()
                .meta
                .local_mse
        };
        let base = identity_creature(4, 1);
        let inc = Incumbent::from_creature(base.clone(), "t").unwrap();
        let res = compute_residuals(&inc, tmp.path(), &corpus, 512, 1).unwrap();
        let set = build_search_set(&cfg, &bins, &res, tmp.path(), 0, 1).unwrap();
        let bpf = set.bins_per_feature(&bins);
        let threshold = set.threshold(&bins);
        let controls = cfg.search_controls();
        let tree_controls = TreeSearchControls {
            stump: controls.clone(),
            max_depth: 3,
            growth: cfg.growth,
        };
        let best = grow_tree(
            &set.source,
            &bpf,
            &threshold,
            &|f| set.feature_map[f],
            &tree_controls,
            None,
        )
        .unwrap()
        .into_iter()
        .max_by(|a, b| a.gain.total_cmp(&b.gain))
        .unwrap();
        assert_eq!(best.depth, 3);
        let planted = graft_patch(
            &base,
            &Patch::new(0, best.root.clone(), Provenance::default()),
        )
        .unwrap()
        .creature;
        let after_tree = mse(&planted);
        // The best a depth-3 tree can do still leaves the planted region's
        // fourth condition unexplained: half its records are wrong by ~0.5.
        assert!(after_tree > 0.005, "depth 3 alone reached {after_tree}");

        let inc = Incumbent::from_creature(planted.clone(), "t2").unwrap();
        let res = compute_residuals(&inc, tmp.path(), &corpus, 512, 1).unwrap();
        let set = build_search_set(&cfg, &bins, &res, tmp.path(), 0, 1).unwrap();
        let view = ForestView::read(&planted, 0).unwrap();
        let found = search_refinements(
            &view,
            &set,
            &bins,
            tmp.path(),
            1,
            512,
            &RefineControls {
                leaves: 2,
                max_total_depth: 6,
                max_depth: 3,
                stump: controls,
                growth: cfg.growth,
                threads: 1,
            },
        )
        .unwrap();
        assert_eq!(found.trees, 1);
        assert!(found.searched >= 1);
        let best_refinement = found
            .discoveries
            .iter()
            .filter(|d| d.strategy == "refine-stump")
            .max_by(|a, b| a.gain.total_cmp(&b.gain))
            .unwrap();
        assert_eq!(
            best_refinement.root.features(),
            vec![3],
            "splits on the missing condition"
        );
        let patch = Patch::refinement(
            0,
            best_refinement.parent.clone().unwrap(),
            best_refinement.root.clone(),
            Provenance::default(),
        );
        let refined = graft_patch(&planted, &patch).unwrap().creature;
        let after_refine = mse(&refined);
        assert!(
            after_refine < after_tree * 0.25,
            "refining reached {after_refine} from {after_tree}"
        );
    }

    /// Issue #140's two guards take effect at their boundaries: a leaf at the
    /// depth cap is not searched, one level of room offers only a stump, a
    /// subtree never takes a path past the cap, and a leaf with fewer records
    /// than two new leaves need is skipped.
    #[test]
    fn the_depth_cap_and_the_leaf_record_floor_bound_the_search() {
        use crate::bins::{BinBuildOptions, build_bin_cache};
        use crate::corpus::{corpus_info, write_bin_file};
        use crate::incumbent::Incumbent;
        use crate::residuals::compute_residuals;
        use crate::strategies::build_search_set;
        use neat_core::training_data::TrainingDataConfig;

        let tmp = tempfile::tempdir().unwrap();
        // Structure under every depth-2 leaf, so each is worth refining.
        let recs: Vec<(Vec<f32>, Vec<f32>)> = records(8000, 4)
            .into_iter()
            .map(|x| {
                let y = if x[3] > 0.5 { 0.6 } else { -0.4 } + if x[1] > 0.3 { 0.2 } else { 0.0 };
                (x, vec![y])
            })
            .collect();
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let tcfg = TrainingDataConfig::new(4, 1);
        let corpus = corpus_info(tmp.path(), &tcfg).unwrap();
        let bins = build_bin_cache(
            tmp.path(),
            &tcfg,
            &corpus,
            &BinBuildOptions {
                bins: 16,
                ..Default::default()
            },
        )
        .unwrap();
        let cfg = ForestsConfig {
            search_records: 0,
            min_leaf_records: 20.0,
            ..Default::default()
        };
        let planted = graft_patch(
            &identity_creature(4, 1),
            &Patch::new(0, depth2(), Provenance::default()),
        )
        .unwrap()
        .creature;
        let inc = Incumbent::from_creature(planted.clone(), "t").unwrap();
        let res = compute_residuals(&inc, tmp.path(), &corpus, 512, 1).unwrap();
        let set = build_search_set(&cfg, &bins, &res, tmp.path(), 0, 1).unwrap();
        let view = ForestView::read(&planted, 0).unwrap();
        assert!(view.leaves.iter().all(|l| l.depth == 2));
        let search = |max_total_depth: usize, max_depth: usize, min_leaf_records: f64| {
            let mut stump = cfg.search_controls();
            stump.min_leaf_records = min_leaf_records;
            search_refinements(
                &view,
                &set,
                &bins,
                tmp.path(),
                1,
                512,
                &RefineControls {
                    leaves: 4,
                    max_total_depth,
                    max_depth,
                    stump,
                    growth: cfg.growth,
                    threads: 1,
                },
            )
            .unwrap()
        };
        let depth_of = |d: &Discovery| match d.strategy.strip_prefix("refine-tree-depth") {
            Some(n) => n.parse::<usize>().unwrap(),
            None => {
                assert_eq!(d.strategy, "refine-stump");
                1
            }
        };

        let at_cap = search(2, 6, 20.0);
        assert_eq!(at_cap.searched, 0, "a leaf at the cap is not searched");
        assert!(at_cap.discoveries.is_empty());

        let one_level = search(3, 6, 20.0);
        assert_eq!(one_level.searched, 4);
        assert!(!one_level.discoveries.is_empty());
        assert!(
            one_level
                .discoveries
                .iter()
                .all(|d| d.strategy == "refine-stump"),
            "one level of room offers only stumps: {:?}",
            one_level
                .discoveries
                .iter()
                .map(|d| &d.strategy)
                .collect::<Vec<_>>()
        );

        for (cap, max_depth) in [(4, 6), (5, 2), (6, 3)] {
            let found = search(cap, max_depth, 20.0);
            assert!(
                found.discoveries.iter().any(|d| depth_of(d) > 1),
                "cap {cap} leaves room for a subtree"
            );
            for d in &found.discoveries {
                assert!(
                    2 + depth_of(d) <= cap && depth_of(d) <= max_depth,
                    "{} under a depth-2 leaf breaks cap {cap} / max {max_depth}",
                    d.strategy
                );
            }
        }

        // The largest leaves (x0 ≤ .5, x1 > .25 and x0 > .5, x2 ≤ .75) each
        // hold about 3000 of the 8000 rows (½ · ¾), the others about 1000.
        let floor = search(6, 3, 1600.0);
        assert_eq!(floor.searched, 0, "no leaf holds 2 × 1600 records");
        assert!(floor.discoveries.is_empty());
        let enough = search(6, 3, 1400.0);
        assert_eq!(
            enough.searched, 2,
            "only the two large leaves cover 2 × 1400"
        );
    }
}

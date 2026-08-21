//! Graft a [`Patch`] onto a **clone** of an incumbent as ordinary NEAT-AI `IF`
//! structure (Issue #7), described with NEAT-AI-core's canonical
//! [`IfNodeSpec`] and emitted by its canonical helper wherever that helper can
//! place the node (Issue #42).
//!
//! Layout appended for each patch (all UUIDs prefixed `forest-<patch id>-`),
//! chosen so that **no (from, to) synapse pair is ever repeated** — NEAT-AI's
//! TypeScript loader keys synapses by that pair and silently collapses
//! duplicates, which `rust_scorer` does not:
//!
//! ```text
//! shared:     one_c, one_p, one_n   constant, bias 1.0 — one per synapse role, reused
//!                            from the creature when it has bias-1 constants, else
//!                            created once (`forest-one-a/b/c`); never per patch (#43)
//! per split:  ifN     hidden IF, bias 0
//!                 condition:  input-f (weight w_f per term)  and  one_c (weight = −threshold)
//!                 positive:   right child ifN (weight 1)  |  one_p (weight = right leaf)
//!                 negative:   left  child ifN (weight 1)  |  one_n (weight = left leaf)
//! root ifN ──(weight 1, untyped)──▶ output-j            (point-wise output squash)
//! root ifN ──(positive)──▶ output-j  and  root ifN → relayN (IDENTITY) ──(negative)──▶ output-j   (IF output)
//! ```
//!
//! That condition shape — the split point as a **weight** on a shared bias-1
//! constant rather than the bias of a per-split IDENTITY neuron — is
//! NEAT-AI-core's own (`neat_core::decision_tree`, NEAT-AI-core #555), which
//! keeps every threshold and leaf where training can reach it. Because the root
//! feeds the output neuron's *pre-squash* sum with weight 1, a leaf of exactly
//! `0.0` leaves every record in that region bit-identical to the incumbent.
//!
//! The finished creature is emitted in NEAT-AI's canonical order — new
//! constants ahead of the first hidden neuron, synapses ascending by
//! `(from index, to index)` — and gated on `neat_core::creature_validate`
//! before it is returned (Issue #39). A candidate that breaks the shared
//! definition of a valid creature is refused here, at the graft that broke it,
//! rather than surfacing downstream; see `docs/architecture.md`, *Creature
//! validation*, for the reject-and-journal failure policy.
//!
//! ## What NEAT-AI-core emits, and what is still emitted here
//!
//! Every node is described as an [`IfNodeSpec`] — the canonical description of
//! an `IF` node — so the synapse-role strings come from NEAT-AI-core, not from
//! this module. A single-split patch entering a point-wise output is then built
//! by [`neat_core::graft_if_node`] itself. Two shapes the helper cannot yet
//! express are still written out here by `write_spec`:
//!
//! * a **child node feeding its parent's branch** — the helper requires every
//!   outward edge to name a neuron that already exists, so a nested tree cannot
//!   be grafted node by node; and
//! * the **`IF`-output relay**, whose two outward edges must carry the
//!   `positive` / `negative` roles, where the helper only emits untyped ones.
//!
//! `write_spec` is pinned to the helper by
//! `local_emission_matches_the_canonical_helper`, and every grafted creature is
//! pinned against the abstract evaluator record by record.

use std::collections::{HashMap, HashSet};
use std::fmt;

use neat_core::topology_ops::{STRUCTURAL_VALID, validate_structural_integrity};
use neat_core::{
    CompiledNetwork, CreatureExport, IfNodeSpec, NeuronExport, SquashType, SynapseExport,
    SynapseType, ValidateOptions, ValidationFailure, compile_creature, creature_validate,
    graft_if_node, squash_name_from, synapse_type_name_from, validate_no_duplicate_synapses,
};

use crate::patch::{Node, Patch};

/// Graft failure (fail closed — nothing is emitted).
#[derive(Debug, Clone, PartialEq)]
pub enum GraftError {
    /// Patch references an input beyond the creature's width.
    FeatureOutOfRange {
        /// Offending feature.
        feature: usize,
        /// Creature input width.
        input: usize,
    },
    /// Patch targets an output beyond the creature's width.
    OutputOutOfRange {
        /// Offending output.
        output: usize,
        /// Creature output width.
        outputs: usize,
    },
    /// Patch root is a bare leaf (no split) — a no-op that would still add structure.
    RootIsLeaf,
    /// Patch contains non-finite numbers.
    NonFinite,
    /// A generated UUID already exists in the creature.
    UuidCollision(String),
    /// A (from, to) synapse pair would be repeated.
    DuplicateSynapse(String),
    /// Output neurons are not trailing.
    OutputsNotLast,
    /// NEAT-AI-core refused to compile the result.
    Compile(String),
    /// The target output neuron is an aggregate (MINIMUM/MAXIMUM/MEAN/HYPOT…)
    /// whose activation is not additive in an extra synapse.
    UnsupportedOutputSquash(String),
    /// Structural validator rejected the result.
    Structural {
        /// Validator code.
        code: i32,
        /// Neuron/synapse index reported by the validator.
        index: i32,
    },
    /// The shared creature validator (`neat_core::creature_validate`) rejected
    /// the finished candidate (Issue #39). Boxed to keep `GraftError` small.
    Invalid(Box<ValidationFailure>),
    /// NEAT-AI-core's canonical `IF` graft helper refused the node, carrying
    /// its `neat_core::GraftError` text verbatim (Issue #42).
    Canonical(String),
}

impl fmt::Display for GraftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FeatureOutOfRange { feature, input } => {
                write!(f, "feature {feature} >= input width {input}")
            }
            Self::OutputOutOfRange { output, outputs } => {
                write!(f, "output {output} >= output width {outputs}")
            }
            Self::RootIsLeaf => write!(f, "patch root is a leaf; refusing a split-free graft"),
            Self::NonFinite => write!(f, "patch contains non-finite values"),
            Self::UuidCollision(u) => write!(f, "uuid `{u}` already exists in the incumbent"),
            Self::DuplicateSynapse(p) => write!(
                f,
                "duplicate synapse {p}; NEAT-AI collapses repeated (from, to) pairs"
            ),
            Self::OutputsNotLast => write!(
                f,
                "output neurons are not the trailing entries of `neurons`"
            ),
            Self::Compile(m) => write!(f, "grafted creature does not compile: {m}"),
            Self::UnsupportedOutputSquash(s) => write!(
                f,
                "output neuron squash `{s}` is an aggregate whose value is not additive in a new synapse; refusing to graft"
            ),
            Self::Structural { code, index } => {
                write!(f, "structural validation failed: code {code} at {index}")
            }
            Self::Invalid(failure) => {
                write!(
                    f,
                    "grafted creature is invalid: {} ({}): {}",
                    failure.class, failure.reason, failure.message
                )?;
                if let Some(neuron) = failure.neuron_index {
                    write!(f, " [neuron index {neuron}]")?;
                }
                if let Some(synapse) = failure.synapse_index {
                    write!(f, " [synapse index {synapse}]")?;
                }
                Ok(())
            }
            Self::Canonical(m) => {
                write!(f, "NEAT-AI-core refused the canonical IF graft: {m}")
            }
        }
    }
}

impl std::error::Error for GraftError {}

/// The three shared bias-1 constants a graft's `IF` nodes hang off — one per
/// synapse role.
///
/// A creature may not carry two synapses between the same ordered pair of
/// neurons, so one node's three roles need three distinct sources; the same
/// rule NEAT-AI-core's canonical fixtures encode as `const-condition` /
/// `const-positive` / `const-negative`. Thresholds and leaves are therefore
/// synapse *weights*, which is what training adjusts (#43, NEAT-AI-core #555).
#[derive(Debug, Clone)]
struct SharedOnes {
    condition: String,
    positive: String,
    negative: String,
}

/// Describes a patch as a post-order list of canonical [`IfNodeSpec`]s — a
/// child is described before the parent whose branch reads it.
struct Emitter<'a> {
    prefix: String,
    ones: SharedOnes,
    specs: Vec<IfNodeSpec>,
    counter: usize,
    input: usize,
    existing: &'a HashSet<&'a str>,
}

impl Emitter<'_> {
    fn fresh(&mut self, tag: &str) -> Result<String, GraftError> {
        let uuid = format!("{}-{tag}{}", self.prefix, self.counter);
        self.counter += 1;
        if self.existing.contains(uuid.as_str()) {
            return Err(GraftError::UuidCollision(uuid));
        }
        Ok(uuid)
    }

    /// Source feeding a parent's branch: a leaf is `(shared constant, weight =
    /// correction)`, a split is `(its IF neuron, 1.0)`. `positive` selects the
    /// constant so the two leaves of one IF never share a source.
    fn branch_source(&mut self, node: &Node, positive: bool) -> Result<(String, f64), GraftError> {
        match node {
            Node::Leaf { correction } => {
                let one = if positive {
                    self.ones.positive.clone()
                } else {
                    self.ones.negative.clone()
                };
                Ok((one, f64::from(*correction)))
            }
            Node::Split { .. } => Ok((self.emit(node)?, 1.0)),
        }
    }

    /// Describe a split; returns the uuid of its `IF` neuron. Every (from, to)
    /// pair described is unique — NEAT-AI's TypeScript keys synapses by that
    /// pair and collapses duplicates.
    fn emit(&mut self, node: &Node) -> Result<String, GraftError> {
        match node {
            Node::Leaf { .. } => Err(GraftError::RootIsLeaf),
            Node::Split {
                condition,
                left,
                right,
            } => {
                let (left_src, left_w) = self.branch_source(left, false)?;
                let (right_src, right_w) = self.branch_source(right, true)?;
                let uuid = self.fresh("if")?;
                let mut spec = IfNodeSpec::new(uuid.clone(), 0.0);
                let mut seen = HashSet::new();
                for t in &condition.terms {
                    if t.feature >= self.input {
                        return Err(GraftError::FeatureOutOfRange {
                            feature: t.feature,
                            input: self.input,
                        });
                    }
                    if !seen.insert(t.feature) {
                        return Err(GraftError::DuplicateSynapse(format!(
                            "input-{} → {uuid}",
                            t.feature
                        )));
                    }
                    spec = spec.with_condition(format!("input-{}", t.feature), f64::from(t.weight));
                }
                // The split point rides as a weight on the shared condition
                // constant: Σ w·x − threshold > 0 ⇔ right.
                spec = spec
                    .with_condition(self.ones.condition.clone(), f64::from(-condition.threshold))
                    .with_positive(right_src, right_w)
                    .with_negative(left_src, left_w);
                self.specs.push(spec);
                Ok(uuid)
            }
        }
    }
}

/// Write one canonical [`IfNodeSpec`] out as its neuron and typed synapses.
///
/// Only for the two shapes [`neat_core::graft_if_node`] cannot place — a child
/// feeding its parent's branch, and the `IF`-output relay (see the module
/// documentation). The role strings still come from NEAT-AI-core, and
/// `local_emission_matches_the_canonical_helper` pins the result against the
/// helper's own output for a shape both can build.
fn write_spec(spec: &IfNodeSpec) -> (NeuronExport, Vec<SynapseExport>) {
    let neuron = NeuronExport {
        id: None,
        neuron_type: "hidden".into(),
        uuid: spec.uuid.clone(),
        bias: spec.bias,
        squash: Some(squash_name_from(SquashType::If).to_string()),
    };
    let mut synapses = Vec::with_capacity(
        spec.condition.len() + spec.positive.len() + spec.negative.len() + spec.targets.len(),
    );
    for (role, edges) in [
        (SynapseType::Condition, &spec.condition),
        (SynapseType::Positive, &spec.positive),
        (SynapseType::Negative, &spec.negative),
    ] {
        for edge in edges {
            synapses.push(SynapseExport {
                from_uuid: edge.uuid.clone(),
                to_uuid: spec.uuid.clone(),
                weight: edge.weight,
                synapse_type: synapse_type_name_from(role).map(str::to_string),
            });
        }
    }
    for edge in &spec.targets {
        synapses.push(SynapseExport {
            from_uuid: spec.uuid.clone(),
            to_uuid: edge.uuid.clone(),
            weight: edge.weight,
            synapse_type: None,
        });
    }
    (neuron, synapses)
}

/// Resolve every neuron uuid to its index in the compiled ordering: implicit
/// input `i` is index `i`, listed neuron `j` is `input + j`. This is the same
/// derivation `neat_core::compile_creature` and `creature_validate` perform,
/// and it is what "sorted by (from, to)" is measured against.
fn uuid_indices(creature: &CreatureExport) -> HashMap<String, u32> {
    creature
        .neurons
        .iter()
        .enumerate()
        .map(|(i, n)| (n.uuid.clone(), (creature.input + i) as u32))
        .collect()
}

/// Put the synapse list into NEAT-AI's canonical wire order — ascending by
/// `(from index, to index)` — which `creature_validate` rule 25 requires of
/// every valid creature. Appending grafted synapses to the incumbent's list
/// leaves them out of order, so the assembled candidate is re-sorted rather
/// than emitted piecemeal.
///
/// An endpoint naming no neuron sorts last (`u32::MAX`) instead of being
/// dropped: `creature_validate` then reports it as `INVALID_SYNAPSE_REFERENCE`
/// rather than the graft quietly reordering a broken creature.
fn sort_synapses_canonically(creature: &mut CreatureExport) {
    let index = uuid_indices(creature);
    let input = creature.input;
    let resolve = |uuid: &str| -> u32 {
        if let Some(i) = index.get(uuid) {
            return *i;
        }
        uuid.strip_prefix("input-")
            .and_then(|n| n.parse::<usize>().ok())
            .filter(|n| *n < input)
            .map_or(u32::MAX, |n| n as u32)
    };
    // Stable sort: synapses sharing a (from, to) pair keep their relative
    // order so `check_no_duplicate_synapses` still names the later one.
    creature
        .synapses
        .sort_by_key(|s| (resolve(&s.from_uuid), resolve(&s.to_uuid)));
}

/// The [`ValidateOptions`] Forests gates its output with (Issue #39).
///
/// * `neurons` / `connections` stay `None`: the graft *changes* both counts by
///   construction, so pinning them would only restate what it just built.
/// * `feedback_loop` stays `None` — the creature's own `forwardOnly`
///   declaration decides, via `forward_only` below.
/// * `forward_only` follows the creature's declared `forwardOnly`. Forests only
///   ever appends feed-forward structure (`input → threshold → IF → output`),
///   so for the feed-forward creatures it actually optimises this is the
///   strongest gate available: it adds the self-connection, acyclicity and
///   structural-integrity rules on top of the unconditional ones. A creature
///   that declares itself recurrent is not failed for recursion the graft did
///   not introduce.
fn validate_options(creature: &CreatureExport) -> ValidateOptions {
    ValidateOptions {
        neurons: None,
        connections: None,
        feedback_loop: None,
        forward_only: creature.forward_only,
    }
}

/// Gate a finished candidate on `neat_core::creature_validate` — the shared
/// definition of a valid creature — before it can escape the graft.
///
/// The failure is returned, never logged and dropped: [`graft_patch`]'s callers
/// record it against the candidate id (see `docs/architecture.md`, *Creature
/// validation*).
fn assert_valid(creature: &CreatureExport) -> Result<(), GraftError> {
    creature_validate(creature, &validate_options(creature))
        .map(|_stats| ())
        .map_err(|failure| GraftError::Invalid(Box::new(failure)))
}

/// Reject any repeated (from, to) synapse pair — NEAT-AI's TypeScript loader
/// keys synapses by that pair, so a creature with duplicates scores
/// differently there than under `rust_scorer`.
///
/// The rule itself is NEAT-AI-core's
/// [`neat_core::validate_no_duplicate_synapses`] (NEAT-AI-core #556), which
/// `compile_creature` now applies too; this wrapper only restates the failure
/// as a [`GraftError`] so a caller sees one error type.
pub fn check_no_duplicate_synapses(creature: &CreatureExport) -> Result<(), GraftError> {
    validate_no_duplicate_synapses(creature)
        .map_err(|e| GraftError::DuplicateSynapse(e.to_string()))
}

/// One untyped synapse — the additive edge into a point-wise neuron.
fn untyped(from: &str, to: &str, weight: f64) -> SynapseExport {
    SynapseExport {
        from_uuid: from.into(),
        to_uuid: to.into(),
        weight,
        synapse_type: None,
    }
}

/// One synapse carrying a NEAT-AI-core [`SynapseType`] role.
fn typed(from: &str, to: &str, weight: f64, role: SynapseType) -> SynapseExport {
    SynapseExport {
        from_uuid: from.into(),
        to_uuid: to.into(),
        weight,
        synapse_type: synapse_type_name_from(role).map(str::to_string),
    }
}

/// Clone `incumbent` with `constants` listed ahead of its first non-constant
/// neuron, which is where `creature_validate` rule 11 requires them.
///
/// `first_output` is the index of the incumbent's first output neuron, so a
/// creature made entirely of constants and outputs still lands them correctly.
fn with_constants(
    incumbent: &CreatureExport,
    constants: Vec<NeuronExport>,
    first_output: usize,
) -> CreatureExport {
    let first_hidden = incumbent.neurons[..first_output]
        .iter()
        .position(|n| n.neuron_type != "constant")
        .unwrap_or(first_output);
    let mut creature = incumbent.clone();
    creature
        .neurons
        .splice(first_hidden..first_hidden, constants);
    creature
}

/// Result of a graft: the new creature plus what was appended.
#[derive(Debug, Clone)]
pub struct Grafted {
    /// The candidate creature (incumbent clone + appended structure).
    pub creature: CreatureExport,
    /// Neurons appended.
    pub added_neurons: usize,
    /// Synapses appended.
    pub added_synapses: usize,
    /// UUIDs of the neurons appended, in listed order.
    pub added_uuids: Vec<String>,
}

/// Graft `patch` onto a clone of `incumbent`. The incumbent is never modified.
pub fn graft_patch(incumbent: &CreatureExport, patch: &Patch) -> Result<Grafted, GraftError> {
    if !patch.root.is_finite() {
        return Err(GraftError::NonFinite);
    }
    if matches!(patch.root, Node::Leaf { .. }) {
        return Err(GraftError::RootIsLeaf);
    }
    if patch.output >= incumbent.output {
        return Err(GraftError::OutputOutOfRange {
            output: patch.output,
            outputs: incumbent.output,
        });
    }
    let n = incumbent.neurons.len();
    if n < incumbent.output
        || incumbent.neurons[n - incumbent.output..]
            .iter()
            .any(|x| x.neuron_type != "output")
    {
        return Err(GraftError::OutputsNotLast);
    }
    let first_output = n - incumbent.output;
    let target_uuid = incumbent.neurons[first_output + patch.output].uuid.clone();
    let existing: HashSet<&str> = incumbent.neurons.iter().map(|x| x.uuid.as_str()).collect();
    let prefix = format!("forest-{}", patch.id());
    // Shared bias-1 constants (#43): reuse the creature's own where present
    // (constants are never mutated; evolution tunes the synapse weights), else
    // create the rest, named without the patch id so later grafts share them.
    let mut ones: Vec<String> = incumbent
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "constant" && n.bias == 1.0 && n.squash.is_none())
        .map(|n| n.uuid.clone())
        .take(3)
        .collect();
    let mut new_constants = Vec::new();
    for name in ["forest-one-a", "forest-one-b", "forest-one-c"] {
        if ones.len() >= 3 {
            break;
        }
        if existing.contains(name) {
            return Err(GraftError::UuidCollision(name.into()));
        }
        new_constants.push(NeuronExport {
            id: None,
            neuron_type: "constant".into(),
            uuid: name.into(),
            bias: 1.0,
            squash: None,
        });
        ones.push(name.into());
    }
    let mut em = Emitter {
        prefix,
        ones: SharedOnes {
            condition: ones[0].clone(),
            positive: ones[1].clone(),
            negative: ones[2].clone(),
        },
        specs: Vec::new(),
        counter: 0,
        input: incumbent.input,
        existing: &existing,
    };
    let root_uuid = em.emit(&patch.root)?;
    // How the correction enters the output depends on the output's squash:
    // * point-wise squash: one untyped synapse adds to the pre-squash sum;
    // * `IF` output: an untyped synapse would feed only the positive branch,
    //   so the root feeds the positive branch directly and an IDENTITY relay
    //   feeds the negative branch (two distinct (from, to) pairs);
    // * other aggregates (MIN/MAX/MEAN/HYPOT): not additive — fail closed.
    let target_squash = incumbent.neurons[first_output + patch.output]
        .squash
        .as_deref()
        .unwrap_or("IDENTITY");
    let parsed = neat_core::parse_squash_name(target_squash)
        .map_err(|e| GraftError::Compile(e.to_string()))?;
    let target_is_if = parsed == neat_core::SquashType::If;
    if !target_is_if && parsed.is_aggregate() {
        return Err(GraftError::UnsupportedOutputSquash(
            target_squash.to_string(),
        ));
    }
    let mut relay_neuron = None;
    let mut relay_synapses = Vec::new();
    if target_is_if {
        let relay = em.fresh("relay")?;
        relay_synapses.push(untyped(&root_uuid, &relay, 1.0));
        relay_synapses.push(typed(&root_uuid, &target_uuid, 1.0, SynapseType::Positive));
        relay_synapses.push(typed(&relay, &target_uuid, 1.0, SynapseType::Negative));
        relay_neuron = Some(NeuronExport {
            id: None,
            neuron_type: "hidden".into(),
            uuid: relay,
            bias: 0.0,
            squash: Some(squash_name_from(SquashType::Identity).to_string()),
        });
    } else {
        let root = em
            .specs
            .last_mut()
            .expect("a split patch describes at least one node");
        *root = root.clone().with_target(target_uuid.clone(), 1.0);
    }

    // New constants go in front of the first non-constant neuron:
    // `creature_validate` rule 11 rejects a constant that follows a hidden one.
    let base = with_constants(incumbent, new_constants, first_output);
    // A lone split entering a point-wise output is exactly the shape
    // NEAT-AI-core's canonical helper covers — let it build the node (Issue
    // #42). Anything nested, or wired into an `IF` output, still needs the
    // typed edges the helper cannot emit.
    let mut creature = if em.specs.len() == 1 && !target_is_if {
        graft_if_node(&base, &em.specs[0]).map_err(|e| GraftError::Canonical(e.to_string()))?
    } else {
        let mut creature = base;
        let mut new_neurons = Vec::with_capacity(em.specs.len() + 1);
        let mut new_synapses = Vec::new();
        for spec in &em.specs {
            let (neuron, synapses) = write_spec(spec);
            new_neurons.push(neuron);
            new_synapses.extend(synapses);
        }
        new_neurons.extend(relay_neuron);
        new_synapses.extend(relay_synapses);
        // New hidden neurons go before the first output, so listed order stays
        // `constant, hidden, output` and remains topological.
        let at = creature.neurons.len() - creature.output;
        creature.neurons.splice(at..at, new_neurons);
        creature.synapses.extend(new_synapses);
        creature
    };
    sort_synapses_canonically(&mut creature);

    let added_neurons = creature.neurons.len() - incumbent.neurons.len();
    let added_synapses = creature.synapses.len() - incumbent.synapses.len();
    let added_uuids: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| !existing.contains(n.uuid.as_str()))
        .map(|n| n.uuid.clone())
        .collect();

    check_no_duplicate_synapses(&creature)?;
    let network = compile_creature(&creature).map_err(|e| GraftError::Compile(e.to_string()))?;
    validate_compiled(&network, &creature)?;
    // Last gate before the candidate escapes: the shared definition of a valid
    // creature (Issue #39). Anything the graft breaks is attributed here, to
    // the graft that broke it, instead of surfacing downstream.
    assert_valid(&creature)?;
    Ok(Grafted {
        creature,
        added_neurons,
        added_synapses,
        added_uuids,
    })
}

/// Graft several patches in sequence onto one clone (a *combination*
/// candidate). Returns the final creature and, per patch, the neuron uuids it
/// added. Patch ids must be distinct (they prefix the uuids).
pub fn graft_patches(
    incumbent: &CreatureExport,
    patches: &[Patch],
) -> Result<(CreatureExport, Vec<Vec<String>>), GraftError> {
    let mut creature = incumbent.clone();
    let mut added = Vec::with_capacity(patches.len());
    for p in patches {
        let g = graft_patch(&creature, p)?;
        creature = g.creature;
        added.push(g.added_uuids);
    }
    Ok((creature, added))
}

/// Run NEAT-AI-core's structural validator over a compiled creature.
pub fn validate_compiled(
    network: &CompiledNetwork,
    creature: &CreatureExport,
) -> Result<(), GraftError> {
    let num_inputs = network.num_inputs;
    let total = network.num_neurons;
    let mut from = Vec::with_capacity(network.synapses.len());
    let mut to = Vec::with_capacity(network.synapses.len());
    let mut types = Vec::with_capacity(network.synapses.len());
    let mut is_constant = vec![0u8; total];
    let mut squash = vec![0u8; total];
    let mut biases = vec![0f64; total];
    for (i, neuron) in network.neurons.iter().enumerate() {
        let idx = num_inputs + i;
        is_constant[idx] = u8::from(neuron.is_constant);
        squash[idx] = neuron.squash_type;
        biases[idx] = f64::from(neuron.bias);
        let start = neuron.start_synapse as usize;
        for s in &network.synapses[start..start + neuron.num_synapses as usize] {
            from.push(u32::from(s.from_index));
            to.push(idx as u32);
            types.push(s.synapse_type);
        }
    }
    let result = validate_structural_integrity(
        &from,
        &to,
        &is_constant,
        &squash,
        &biases,
        num_inputs as u32,
        creature.output as u32,
        &types,
    );
    match result.as_slice() {
        [code, _] if *code == STRUCTURAL_VALID => Ok(()),
        [code, index] => Err(GraftError::Structural {
            code: *code,
            index: *index,
        }),
        _ => Err(GraftError::Structural { code: -1, index: 0 }),
    }
}

/// Activate a creature over one record and return its outputs.
pub fn activate(network: &mut CompiledNetwork, inputs: &[f32], outputs: usize) -> Vec<f32> {
    network.activate(inputs, outputs)
}

/// Test/bench fixtures shared across the crate.
pub mod fixtures {
    use neat_core::{CreatureExport, NeuronExport, SynapseExport};

    /// Minimal forward-only creature: each output is the identity of `input-j`
    /// (or `input-0` when there are fewer inputs than outputs).
    pub fn identity_creature(inputs: usize, outputs: usize) -> CreatureExport {
        let neurons = (0..outputs)
            .map(|j| NeuronExport {
                id: None,
                neuron_type: "output".into(),
                uuid: format!("output-{j}"),
                bias: 0.0,
                squash: Some("IDENTITY".into()),
            })
            .collect();
        let synapses = (0..outputs)
            .map(|j| SynapseExport {
                from_uuid: format!("input-{}", j.min(inputs - 1)),
                to_uuid: format!("output-{j}"),
                weight: 1.0,
                synapse_type: None,
            })
            .collect();
        CreatureExport {
            input: inputs,
            output: outputs,
            neurons,
            synapses,
            semantic_version: Some("4.0.0".into()),
            forward_only: true,
            memetic: None,
        }
    }

    /// JSON text of [`identity_creature`].
    pub fn identity_creature_json(inputs: usize, outputs: usize) -> String {
        neat_core::creature_to_json_pretty(&identity_creature(inputs, outputs)).unwrap()
    }

    /// A creature whose single output is itself an `IF` aggregate (as the
    /// production champion's is): condition on `input-0`, positive branch
    /// `2·input-1`, negative branch `-input-2`.
    pub fn if_output_creature(inputs: usize) -> CreatureExport {
        assert!(inputs >= 3);
        CreatureExport {
            input: inputs,
            output: 1,
            neurons: vec![NeuronExport {
                id: None,
                neuron_type: "output".into(),
                uuid: "output-0".into(),
                bias: 0.01,
                squash: Some("IF".into()),
            }],
            synapses: vec![
                SynapseExport {
                    from_uuid: "input-0".into(),
                    to_uuid: "output-0".into(),
                    weight: 1.0,
                    synapse_type: Some("condition".into()),
                },
                SynapseExport {
                    from_uuid: "input-1".into(),
                    to_uuid: "output-0".into(),
                    weight: 2.0,
                    synapse_type: Some("positive".into()),
                },
                SynapseExport {
                    from_uuid: "input-2".into(),
                    to_uuid: "output-0".into(),
                    weight: -1.0,
                    synapse_type: Some("negative".into()),
                },
            ],
            semantic_version: Some("4.0.0".into()),
            forward_only: true,
            memetic: None,
        }
    }

    /// A creature whose neurons are listed `hidden, constant, output` — the one
    /// order `creature_validate` rule 11 forbids (Issue #39) and every other
    /// gate in this module accepts: it compiles, clears
    /// `validate_structural_integrity`, and repeats no synapse pair. Grafting
    /// onto it must therefore fail on the shared validator and nothing else.
    pub fn constant_after_hidden_creature() -> CreatureExport {
        CreatureExport {
            input: 2,
            output: 1,
            neurons: vec![
                NeuronExport {
                    id: None,
                    neuron_type: "hidden".into(),
                    uuid: "hidden-0".into(),
                    bias: 0.0,
                    squash: Some("TANH".into()),
                },
                NeuronExport {
                    id: None,
                    neuron_type: "constant".into(),
                    uuid: "const-1".into(),
                    bias: 1.0,
                    squash: None,
                },
                NeuronExport {
                    id: None,
                    neuron_type: "output".into(),
                    uuid: "output-0".into(),
                    bias: 0.0,
                    squash: Some("IDENTITY".into()),
                },
            ],
            synapses: vec![
                SynapseExport {
                    from_uuid: "input-0".into(),
                    to_uuid: "hidden-0".into(),
                    weight: 1.0,
                    synapse_type: None,
                },
                SynapseExport {
                    from_uuid: "input-1".into(),
                    to_uuid: "hidden-0".into(),
                    weight: 1.0,
                    synapse_type: None,
                },
                SynapseExport {
                    from_uuid: "hidden-0".into(),
                    to_uuid: "output-0".into(),
                    weight: 1.0,
                    synapse_type: None,
                },
                SynapseExport {
                    from_uuid: "const-1".into(),
                    to_uuid: "output-0".into(),
                    weight: 0.1,
                    synapse_type: None,
                },
            ],
            semantic_version: Some("4.0.0".into()),
            forward_only: true,
            memetic: None,
        }
    }

    /// A small creature with a hidden TANH layer and a LOGISTIC output, so
    /// tests cover a non-identity output squash.
    pub fn small_mlp(inputs: usize) -> CreatureExport {
        let mut neurons = Vec::new();
        let mut synapses = Vec::new();
        for h in 0..3 {
            neurons.push(NeuronExport {
                id: None,
                neuron_type: "hidden".into(),
                uuid: format!("hidden-{h}"),
                bias: 0.1 * h as f64,
                squash: Some("TANH".into()),
            });
            for i in 0..inputs {
                synapses.push(SynapseExport {
                    from_uuid: format!("input-{i}"),
                    to_uuid: format!("hidden-{h}"),
                    weight: 0.3 * (h as f64 + 1.0) - 0.1 * i as f64,
                    synapse_type: None,
                });
            }
        }
        neurons.push(NeuronExport {
            id: None,
            neuron_type: "output".into(),
            uuid: "output-0".into(),
            bias: 0.05,
            squash: Some("LOGISTIC".into()),
        });
        for h in 0..3 {
            synapses.push(SynapseExport {
                from_uuid: format!("hidden-{h}"),
                to_uuid: "output-0".into(),
                weight: 0.5 - 0.2 * h as f64,
                synapse_type: None,
            });
        }
        CreatureExport {
            input: inputs,
            output: 1,
            neurons,
            synapses,
            semantic_version: Some("4.0.0".into()),
            forward_only: true,
            memetic: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;
    use crate::patch::{Condition, Provenance, Term};

    fn records(n: usize, width: usize) -> Vec<Vec<f32>> {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        (0..n)
            .map(|_| {
                (0..width)
                    .map(|_| {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((seed >> 40) as f32) / (1u64 << 23) as f32 * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    fn assert_graft_matches_evaluator(incumbent: &CreatureExport, patch: &Patch, tol: f32) {
        let grafted = graft_patch(incumbent, patch).unwrap();
        let mut base = compile_creature(incumbent).unwrap();
        let mut cand = compile_creature(&grafted.creature).unwrap();
        // Preservation invariants: nothing pre-existing changed. The graft now
        // emits NEAT-AI's canonical order (Issue #39) — constants ahead of
        // hidden neurons, synapses ascending by `(from, to)` — so preservation
        // is by *content and relative order*, not by list position.
        let n_out = incumbent.output;
        let keep = incumbent.neurons.len() - n_out;
        let kept: Vec<&NeuronExport> = grafted
            .creature
            .neurons
            .iter()
            .filter(|n| incumbent.neurons.contains(n))
            .collect();
        assert_eq!(
            kept,
            incumbent.neurons.iter().collect::<Vec<_>>(),
            "every incumbent neuron survives unchanged, in order"
        );
        assert_eq!(
            &grafted.creature.neurons[grafted.creature.neurons.len() - n_out..],
            &incumbent.neurons[keep..],
            "outputs stay trailing"
        );
        for synapse in &incumbent.synapses {
            assert!(
                grafted.creature.synapses.contains(synapse),
                "incumbent synapse {} → {} was altered by the graft",
                synapse.from_uuid,
                synapse.to_uuid
            );
        }
        assert_eq!(
            grafted.creature.synapses.len(),
            incumbent.synapses.len() + grafted.added_synapses,
            "the graft only adds synapses"
        );
        let mut left = 0;
        let mut right = 0;
        for rec in records(400, incumbent.input) {
            let expected = patch.evaluate(&rec);
            if expected == 0.0 {
                // Leaf zero → behaviourally unchanged. The extra (zero-valued)
                // synapse can alter NEAT-AI-core's SIMD summation grouping by
                // one ulp, so this is "within float tolerance", not bitwise.
                let (b, c) = (base.activate(&rec, n_out), cand.activate(&rec, n_out));
                for (x, y) in b.iter().zip(&c) {
                    assert!((x - y).abs() <= 1e-6, "zero leaf changed output {x} -> {y}");
                }
            }
            // Compare pre-squash: use a trace to read the output hint.
            let tb = base.activate_and_trace(&rec, n_out);
            let tc = cand.activate_and_trace(&rec, n_out);
            let nb = base.num_neurons() - base.num_inputs();
            let nc = cand.num_neurons() - cand.num_inputs();
            let hint_b = tb[n_out + nb + (nb - n_out) + patch.output];
            let hint_c = tc[n_out + nc + (nc - n_out) + patch.output];
            assert!(
                ((hint_c - hint_b) - expected).abs() <= tol,
                "pre-squash delta {} != patch {expected}",
                hint_c - hint_b
            );
            if let Node::Split { condition, .. } = &patch.root {
                if condition.goes_right(&rec) {
                    right += 1
                } else {
                    left += 1
                }
            }
        }
        assert!(left > 0 && right > 0, "fixture must cover both branches");
    }

    #[test]
    fn depth1_identity_output_matches_evaluator() {
        let inc = identity_creature(4, 1);
        let patch = Patch::new(0, Node::stump(2, 0.1, 0.0, 0.013), Provenance::default());
        assert_graft_matches_evaluator(&inc, &patch, 1e-6);
        let g = graft_patch(&inc, &patch).unwrap();
        assert_eq!(g.added_neurons, 4); // three shared constants (first graft only) + if
        assert_eq!(g.added_synapses, 5); // input→if, one_c→if, one_p→if, one_n→if, if→output
        // A second graft reuses the shared constants and adds only its node.
        let again = graft_patch(
            &g.creature,
            &Patch::new(0, Node::stump(1, 0.3, 0.0, 0.02), Provenance::default()),
        )
        .unwrap();
        assert_eq!(again.added_neurons, 1);
        assert_eq!(
            again
                .creature
                .neurons
                .iter()
                .filter(|n| n.neuron_type == "constant")
                .count(),
            3
        );
    }

    #[test]
    fn depth2_and_logistic_output_match_evaluator() {
        let inc = small_mlp(3);
        let root = Node::Split {
            condition: Condition::axis(0, 0.0),
            left: Box::new(Node::stump(1, -0.5, -0.2, 0.0)),
            right: Box::new(Node::stump(2, 0.3, 0.0, 0.4)),
        };
        let patch = Patch::new(0, root, Provenance::default());
        assert_graft_matches_evaluator(&inc, &patch, 1e-6);
    }

    #[test]
    fn oblique_condition_matches_evaluator() {
        let inc = identity_creature(3, 2);
        let root = Node::Split {
            condition: Condition {
                terms: vec![
                    Term {
                        feature: 0,
                        weight: 0.8,
                    },
                    Term {
                        feature: 2,
                        weight: -1.3,
                    },
                ],
                threshold: 0.05,
            },
            left: Box::new(Node::leaf(0.0)),
            right: Box::new(Node::leaf(-0.07)),
        };
        let patch = Patch::new(1, root, Provenance::default());
        assert_graft_matches_evaluator(&inc, &patch, 1e-6);
    }

    #[test]
    fn if_output_neuron_receives_the_correction_on_both_branches() {
        let inc = if_output_creature(4);
        let patch = Patch::new(0, Node::stump(3, 0.0, -0.05, 0.07), Provenance::default());
        let g = graft_patch(&inc, &patch).unwrap();
        // Two typed synapses into the IF output, none untyped.
        let into_out: Vec<_> = g
            .creature
            .synapses
            .iter()
            .filter(|s| s.to_uuid == "output-0" && s.from_uuid.starts_with("forest-"))
            .collect();
        assert_eq!(into_out.len(), 2);
        assert_eq!(
            into_out
                .iter()
                .map(|s| s.synapse_type.as_deref())
                .collect::<Vec<_>>(),
            [Some("positive"), Some("negative")]
        );
        let mut base = compile_creature(&inc).unwrap();
        let mut cand = compile_creature(&g.creature).unwrap();
        let (mut pos, mut neg) = (0, 0);
        for rec in records(300, 4) {
            let delta = cand.activate(&rec, 1)[0] - base.activate(&rec, 1)[0];
            assert!(
                (delta - patch.evaluate(&rec)).abs() < 1e-6,
                "delta {delta} vs {}",
                patch.evaluate(&rec)
            );
            if rec[0] > 0.0 { pos += 1 } else { neg += 1 }
        }
        assert!(pos > 0 && neg > 0);
        // A MAXIMUM output is refused.
        let mut max_out = inc.clone();
        max_out.neurons[0].squash = Some("MAXIMUM".into());
        assert!(matches!(
            graft_patch(&max_out, &patch),
            Err(GraftError::UnsupportedOutputSquash(_))
        ));
    }

    #[test]
    fn combined_patches_stack_additively() {
        let inc = identity_creature(4, 1);
        let a = Patch::new(0, Node::stump(0, 0.0, 0.0, 0.1), Provenance::default());
        let b = Patch::new(0, Node::stump(1, 0.2, -0.05, 0.0), Provenance::default());
        let (creature, added) = graft_patches(&inc, &[a.clone(), b.clone()]).unwrap();
        assert_eq!(added.len(), 2);
        assert_eq!(added[0].len(), 4);
        assert_eq!(added[1].len(), 1); // shared constants already present
        assert_eq!(
            creature
                .neurons
                .iter()
                .filter(|n| n.neuron_type == "constant")
                .count(),
            3
        );
        let mut base = compile_creature(&inc).unwrap();
        let mut cand = compile_creature(&creature).unwrap();
        for rec in records(200, 4) {
            let delta = cand.activate(&rec, 1)[0] - base.activate(&rec, 1)[0];
            assert!((delta - (a.evaluate(&rec) + b.evaluate(&rec))).abs() < 1e-6);
        }
    }

    #[test]
    fn json_round_trip_preserves_if_roles() {
        let inc = identity_creature(2, 1);
        let g = graft_patch(
            &inc,
            &Patch::new(0, Node::stump(0, 0.0, 0.0, 1.0), Provenance::default()),
        )
        .unwrap();
        let json = neat_core::creature_to_json(&g.creature).unwrap();
        let back = neat_core::parse_creature_json(&json).unwrap();
        assert_eq!(back, g.creature);
        // All three roles survive the round trip — two condition edges, since
        // the split point rides on the shared condition constant (Issue #42).
        // Their *listed* order follows the canonical `(from, to)` synapse order
        // (Issue #39), not the order they were described in, so compare the set.
        let mut roles: Vec<_> = back
            .synapses
            .iter()
            .filter_map(|s| s.synapse_type.clone())
            .collect();
        roles.sort();
        assert_eq!(roles, ["condition", "condition", "negative", "positive"]);
        assert!(json.contains("\"IF\""));
    }

    #[test]
    fn invalid_requests_fail_closed() {
        let inc = identity_creature(2, 1);
        let bad =
            |root| graft_patch(&inc, &Patch::new(0, root, Provenance::default())).unwrap_err();
        assert_eq!(
            bad(Node::stump(5, 0.0, 0.0, 1.0)),
            GraftError::FeatureOutOfRange {
                feature: 5,
                input: 2
            }
        );
        assert_eq!(bad(Node::leaf(1.0)), GraftError::RootIsLeaf);
        assert_eq!(
            bad(Node::stump(0, f32::NAN, 0.0, 1.0)),
            GraftError::NonFinite
        );
        let p = Patch::new(3, Node::stump(0, 0.0, 0.0, 1.0), Provenance::default());
        assert_eq!(
            graft_patch(&inc, &p).unwrap_err(),
            GraftError::OutputOutOfRange {
                output: 3,
                outputs: 1
            }
        );
    }

    #[test]
    fn grafts_never_repeat_a_synapse_pair() {
        for inc in [identity_creature(4, 2), small_mlp(3), if_output_creature(4)] {
            let root = Node::Split {
                condition: Condition::axis(0, 0.1),
                left: Box::new(Node::stump(1, -0.5, -0.2, 0.0)),
                right: Box::new(Node::Split {
                    condition: Condition {
                        terms: vec![
                            Term {
                                feature: 1,
                                weight: 0.7,
                            },
                            Term {
                                feature: 2,
                                weight: -0.4,
                            },
                        ],
                        threshold: 0.05,
                    },
                    left: Box::new(Node::leaf(0.3)),
                    right: Box::new(Node::leaf(0.0)),
                }),
            };
            let g = graft_patch(&inc, &Patch::new(0, root, Provenance::default())).unwrap();
            check_no_duplicate_synapses(&g.creature).unwrap();
        }
        let mut dup = identity_creature(2, 1);
        dup.synapses.push(dup.synapses[0].clone());
        assert!(matches!(
            check_no_duplicate_synapses(&dup),
            Err(GraftError::DuplicateSynapse(_))
        ));
        // A condition naming the same feature twice is refused.
        let bad = Node::Split {
            condition: Condition {
                terms: vec![
                    Term {
                        feature: 0,
                        weight: 1.0,
                    },
                    Term {
                        feature: 0,
                        weight: 1.0,
                    },
                ],
                threshold: 0.0,
            },
            left: Box::new(Node::leaf(0.0)),
            right: Box::new(Node::leaf(1.0)),
        };
        assert!(matches!(
            graft_patch(
                &identity_creature(2, 1),
                &Patch::new(0, bad, Provenance::default())
            ),
            Err(GraftError::DuplicateSynapse(_))
        ));
    }

    #[test]
    fn existing_bias_one_constants_are_reused() {
        let mut inc = identity_creature(2, 1);
        inc.neurons.insert(
            0,
            NeuronExport {
                id: None,
                neuron_type: "constant".into(),
                uuid: "const-1".into(),
                bias: 1.0,
                squash: None,
            },
        );
        inc.neurons.insert(
            1,
            NeuronExport {
                id: None,
                neuron_type: "constant".into(),
                uuid: "const-half".into(),
                bias: 0.5,
                squash: None,
            },
        );
        inc.synapses.push(SynapseExport {
            from_uuid: "const-1".into(),
            to_uuid: "output-0".into(),
            weight: 0.1,
            synapse_type: None,
        });
        inc.synapses.push(SynapseExport {
            from_uuid: "const-half".into(),
            to_uuid: "output-0".into(),
            weight: 0.1,
            synapse_type: None,
        });
        let g = graft_patch(
            &inc,
            &Patch::new(0, Node::stump(0, 0.0, -0.2, 0.4), Provenance::default()),
        )
        .unwrap();
        let consts: Vec<_> = g
            .creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "constant")
            .map(|n| n.uuid.as_str())
            .collect();
        // `const-1` is reused for the condition role; two more are created.
        assert_eq!(
            consts,
            ["const-1", "const-half", "forest-one-a", "forest-one-b"]
        );
        // The reused constant takes the first role (condition); the positive
        // leaf is a weight on the next shared constant.
        assert!(
            g.creature
                .synapses
                .iter()
                .any(|s| s.from_uuid == "const-1"
                    && s.synapse_type.as_deref() == Some("condition"))
        );
        assert!(
            g.creature
                .synapses
                .iter()
                .any(|s| s.from_uuid == "forest-one-a"
                    && s.synapse_type.as_deref() == Some("positive")
                    && (s.weight - 0.4).abs() < 1e-6)
        );
        let mut base = compile_creature(&inc).unwrap();
        let mut cand = compile_creature(&g.creature).unwrap();
        let p = Patch::new(0, Node::stump(0, 0.0, -0.2, 0.4), Provenance::default());
        for rec in records(100, 2) {
            let d = cand.activate(&rec, 1)[0] - base.activate(&rec, 1)[0];
            assert!((d - p.evaluate(&rec)).abs() < 1e-6);
        }
    }

    #[test]
    fn structural_validator_accepts_graft() {
        let inc = small_mlp(2);
        let g = graft_patch(
            &inc,
            &Patch::new(0, Node::stump(1, 0.2, 0.1, 0.0), Provenance::default()),
        )
        .unwrap();
        let net = compile_creature(&g.creature).unwrap();
        validate_compiled(&net, &g.creature).unwrap();
    }

    #[test]
    fn constant_after_hidden_fixture_clears_every_other_gate() {
        // Guards the fixture's premise: only `creature_validate` catches it.
        let inc = constant_after_hidden_creature();
        let net = compile_creature(&inc).expect("fixture compiles");
        validate_compiled(&net, &inc).expect("fixture clears structural validation");
        check_no_duplicate_synapses(&inc).expect("fixture has no duplicate pairs");
    }

    /// Issue #39 — a valid graft is unaffected: every creature the graft
    /// returns satisfies the shared definition of a valid creature, singly and
    /// stacked, whatever the incumbent's own listing order was.
    #[test]
    fn every_returned_creature_passes_the_shared_validator() {
        let patches = [
            Patch::new(0, Node::stump(0, 0.1, -0.3, 0.4), Provenance::default()),
            Patch::new(
                0,
                Node::Split {
                    condition: Condition::axis(1, -0.2),
                    left: Box::new(Node::stump(2, 0.0, 0.2, -0.1)),
                    right: Box::new(Node::leaf(0.05)),
                },
                Provenance {
                    strategy: "depth2".into(),
                    ..Default::default()
                },
            ),
        ];
        for inc in [identity_creature(3, 1), small_mlp(3), if_output_creature(3)] {
            for patch in &patches {
                let g = graft_patch(&inc, patch).unwrap();
                creature_validate(&g.creature, &validate_options(&g.creature))
                    .expect("a valid graft must satisfy neat_core::creature_validate");
            }
            // Stacked grafts stay valid too — each one re-validates.
            let (stacked, _) = graft_patches(&inc, &patches).unwrap();
            let stats = creature_validate(&stacked, &validate_options(&stacked))
                .expect("stacked grafts must satisfy neat_core::creature_validate");
            assert_eq!(stats.neurons() as usize, inc.input + stacked.neurons.len());
            assert_eq!(stats.connections as usize, stacked.synapses.len());
        }
    }

    /// Issue #39 — a graft that would return an invalid creature fails loudly:
    /// the `ValidationFailure` is surfaced, never swallowed or downgraded.
    #[test]
    fn invalid_creature_is_reported_not_swallowed() {
        let inc = constant_after_hidden_creature();
        // Every pre-existing gate accepts it, so only the shared validator can
        // catch this one.
        compile_creature(&inc).expect("fixture compiles");
        let err = graft_patch(
            &inc,
            &Patch::new(0, Node::stump(0, 0.0, -0.2, 0.4), Provenance::default()),
        )
        .expect_err("an invalid grafted creature must not be returned");
        let GraftError::Invalid(failure) = &err else {
            panic!("expected a validation failure, got {err}");
        };
        assert_eq!(failure.reason, "NEURON_ORDER");
        // `const-1` — the misplaced constant — is index 5: two implicit inputs,
        // the two shared constants the graft created, then `hidden-0`.
        assert_eq!(failure.neuron_index, Some(5));
        assert!(failure.message.contains("const-1"), "{}", failure.message);
        // The reason, message and offending index all reach the caller's text.
        let text = err.to_string();
        assert!(text.contains("NEURON_ORDER"), "{text}");
        assert!(text.contains(&failure.message), "{text}");
        assert!(text.contains("neuron index 5"), "{text}");
    }

    /// Issue #39 — the acceptance-criteria case: a hidden neuron left with no
    /// outward connection never escapes the graft.
    #[test]
    fn hidden_neuron_without_an_outward_connection_never_escapes() {
        let mut inc = identity_creature(2, 1);
        inc.neurons.insert(
            0,
            NeuronExport {
                id: None,
                neuron_type: "hidden".into(),
                uuid: "dangling".into(),
                bias: 0.0,
                squash: Some("TANH".into()),
            },
        );
        inc.synapses.push(SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "dangling".into(),
            weight: 1.0,
            synapse_type: None,
        });
        let err = graft_patch(
            &inc,
            &Patch::new(0, Node::stump(0, 0.0, -0.2, 0.4), Provenance::default()),
        )
        .expect_err("a hidden neuron nothing reads must fail the graft");
        assert!(
            err.to_string().contains("dangling")
                || matches!(
                    err,
                    GraftError::Structural { .. }
                        | GraftError::Invalid(_)
                        | GraftError::Canonical(_)
                ),
            "{err}"
        );
    }

    /// Every inward synapse of `node`, as `(source uuid, weight)` pairs for one
    /// role, in listed order.
    fn inward<'a>(creature: &'a CreatureExport, node: &str, role: &str) -> Vec<(&'a str, f64)> {
        creature
            .synapses
            .iter()
            .filter(|s| s.to_uuid == node && s.synapse_type.as_deref() == Some(role))
            .map(|s| (s.from_uuid.as_str(), s.weight))
            .collect()
    }

    /// Issue #42 — a grafted split carries NEAT-AI-core's canonical `IF` shape
    /// (`neat_core::decision_tree`): the split point rides as a **weight** on a
    /// shared bias-1 constant, so no per-split IDENTITY threshold neuron is
    /// emitted and every threshold and leaf stays trainable.
    #[test]
    fn grafted_split_uses_the_canonical_condition_shape() {
        let inc = identity_creature(2, 1);
        let patch = Patch::new(0, Node::stump(0, 0.25, -0.1, 0.4), Provenance::default());
        let g = graft_patch(&inc, &patch).unwrap();
        let added: Vec<&NeuronExport> = g
            .creature
            .neurons
            .iter()
            .filter(|n| !inc.neurons.contains(n))
            .collect();
        let constants: Vec<&&NeuronExport> = added
            .iter()
            .filter(|n| n.neuron_type == "constant" && n.bias == 1.0 && n.squash.is_none())
            .collect();
        assert_eq!(
            constants.len(),
            3,
            "one shared bias-1 constant per synapse role"
        );
        let nodes: Vec<&&NeuronExport> = added
            .iter()
            .filter(|n| n.squash.as_deref() == Some("IF"))
            .collect();
        assert_eq!(nodes.len(), 1);
        assert_eq!(
            added.len(),
            4,
            "three constants and the IF node — no IDENTITY threshold neuron"
        );
        let node = nodes[0].uuid.as_str();
        assert_eq!(nodes[0].bias, 0.0);

        let condition = inward(&g.creature, node, "condition");
        assert_eq!(condition.len(), 2, "one term plus the threshold offset");
        assert!(
            condition.contains(&("input-0", 1.0)),
            "the split feature enters the condition sum directly: {condition:?}"
        );
        let offset = condition
            .iter()
            .find(|(from, _)| *from != "input-0")
            .expect("threshold offset edge");
        assert!(
            constants.iter().any(|c| c.uuid == offset.0),
            "the offset hangs off a shared bias-1 constant: {offset:?}"
        );
        assert!(
            (offset.1 - -0.25).abs() < 1e-9,
            "offset weight is -threshold"
        );

        // Leaves are weights on two further shared constants, so no (from, to)
        // pair repeats.
        let positive = inward(&g.creature, node, "positive");
        let negative = inward(&g.creature, node, "negative");
        assert_eq!(positive.len(), 1);
        assert_eq!(negative.len(), 1);
        assert!((positive[0].1 - 0.4).abs() < 1e-6);
        assert!((negative[0].1 - -0.1).abs() < 1e-6);
        let sources = [offset.0, positive[0].0, negative[0].0];
        assert_eq!(
            sources.iter().collect::<HashSet<_>>().len(),
            3,
            "the three roles take three distinct constants: {sources:?}"
        );
    }

    /// A bias-1 constant neuron.
    fn one(uuid: &str) -> NeuronExport {
        NeuronExport {
            id: None,
            neuron_type: "constant".into(),
            uuid: uuid.into(),
            bias: 1.0,
            squash: None,
        }
    }

    /// Issue #42 — a lone split entering a point-wise output is *built by*
    /// NEAT-AI-core's canonical helper, not by this module: the grafted
    /// creature is exactly what `graft_if_node` returns for the same spec.
    #[test]
    fn stump_graft_is_built_by_the_canonical_helper() {
        let inc = small_mlp(3);
        let patch = Patch::new(0, Node::stump(1, 0.2, -0.1, 0.35), Provenance::default());
        let ours = graft_patch(&inc, &patch).unwrap();

        let base = with_constants(
            &inc,
            vec![
                one("forest-one-a"),
                one("forest-one-b"),
                one("forest-one-c"),
            ],
            inc.neurons.len() - inc.output,
        );
        let spec = IfNodeSpec::new(format!("forest-{}-if0", patch.id()), 0.0)
            .with_condition("input-1", 1.0)
            .with_condition("forest-one-a", f64::from(-0.2f32))
            .with_positive("forest-one-b", f64::from(0.35f32))
            .with_negative("forest-one-c", f64::from(-0.1f32))
            .with_target("output-0", 1.0);
        let mut expected = graft_if_node(&base, &spec).expect("canonical helper builds the stump");
        sort_synapses_canonically(&mut expected);
        assert_eq!(ours.creature, expected);
    }

    /// Issue #42 — the local renderer used for the shapes the helper cannot
    /// place emits exactly what the helper emits for a shape both can build,
    /// so the nested and `IF`-output paths cannot drift from the canonical one.
    #[test]
    fn local_emission_matches_the_canonical_helper() {
        let base = with_constants(
            &identity_creature(2, 1),
            vec![one("c-cond"), one("c-pos"), one("c-neg")],
            0,
        );
        let spec = IfNodeSpec::new("node", 0.0)
            .with_condition("input-0", 1.0)
            .with_condition("c-cond", -0.25)
            .with_positive("c-pos", 0.4)
            .with_negative("c-neg", -0.1)
            .with_target("output-0", 1.0);
        let canonical = graft_if_node(&base, &spec).unwrap();

        let (neuron, synapses) = write_spec(&spec);
        let mut local = base.clone();
        let at = local.neurons.len() - local.output;
        local.neurons.insert(at, neuron);
        local.synapses.extend(synapses);

        // The helper places the node as early as the edges allow and this
        // module places it before the first output, so compare by content.
        let sorted = |mut v: Vec<String>| {
            v.sort();
            v
        };
        let describe_neurons =
            |c: &CreatureExport| sorted(c.neurons.iter().map(|n| format!("{n:?}")).collect());
        let describe_synapses =
            |c: &CreatureExport| sorted(c.synapses.iter().map(|s| format!("{s:?}")).collect());
        assert_eq!(describe_neurons(&local), describe_neurons(&canonical));
        assert_eq!(describe_synapses(&local), describe_synapses(&canonical));
    }

    /// Issue #42 — a Forests stump reproduces NEAT-AI-core's own canonical
    /// residual-correction fixture record for record, so the two readings of
    /// `IF` cannot drift apart.
    #[test]
    fn stump_reproduces_the_canonical_residual_fixture() {
        use neat_core::decision_tree::{
            RESIDUAL_CASES, RESIDUAL_THRESHOLD, RESIDUAL_VALUE, residual_correction_creature,
        };
        let base = neat_core::linear_base_creature();
        let patch = Patch::new(
            0,
            Node::stump(0, RESIDUAL_THRESHOLD as f32, 0.0, RESIDUAL_VALUE as f32),
            Provenance::default(),
        );
        let g = graft_patch(&base, &patch).unwrap();
        let mut ours = compile_creature(&g.creature).unwrap();
        let mut canonical = compile_creature(&residual_correction_creature()).unwrap();
        for case in RESIDUAL_CASES {
            let got = ours.activate(case.inputs, 1)[0];
            let want = canonical.activate(case.inputs, 1)[0];
            assert!(
                (got - want).abs() < 1e-6 && (got - case.expected).abs() < 1e-6,
                "{}: forests {got} vs canonical {want} (documented {})",
                case.branch,
                case.expected
            );
        }
    }

    /// Issue #39 — the emitted synapse list is in NEAT-AI's canonical
    /// `(from, to)` order, which is what rule 25 is measured against.
    #[test]
    fn emitted_synapses_are_in_canonical_order() {
        let inc = small_mlp(3);
        let g = graft_patch(
            &inc,
            &Patch::new(0, Node::stump(1, 0.2, 0.1, -0.3), Provenance::default()),
        )
        .unwrap();
        let index = uuid_indices(&g.creature);
        let key = |uuid: &str| -> u32 {
            index.get(uuid).copied().unwrap_or_else(|| {
                uuid.strip_prefix("input-")
                    .and_then(|n| n.parse::<u32>().ok())
                    .expect("endpoint resolves")
            })
        };
        let keys: Vec<(u32, u32)> = g
            .creature
            .synapses
            .iter()
            .map(|s| (key(&s.from_uuid), key(&s.to_uuid)))
            .collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted);
    }
}

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
//!                            created once (`forest-one-a/b/c`, or the next free
//!                            name when one is taken); never per patch (#43, #50)
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
//! `(from index, to index)` — and gated on `assert_valid` before it is
//! returned (Issue #39): the duplicate-pair rule and
//! `neat_core::creature_validate`, one gate rather than two separate checks
//! (Issue #50). A candidate that breaks the shared definition of a valid
//! creature is refused here, at the graft that broke it, rather than surfacing
//! downstream; see `docs/architecture.md`, *Creature validation*, for the
//! reject-and-journal failure policy.
//!
//! ## Every shape is emitted by NEAT-AI-core
//!
//! This module describes nodes; it no longer writes neurons or synapses out
//! itself (Issue #48). Every node is an [`IfNodeSpec`] and the whole post-order
//! batch goes to [`neat_core::graft_if_nodes`], which places each node, emits
//! the role strings and validates the assembled creature once — a child may
//! leave its outward edge to the parent that reads it, which is why a nested
//! tree no longer needs a local renderer. Where the target output is itself an
//! `IF` aggregate, the root's outward edge carries the `positive` role
//! ([`IfNodeSpec::with_target_role`]) and [`neat_core::graft_relay_node`] adds
//! the IDENTITY relay that carries the same value into the `negative` branch.
//!
//! Every grafted creature is still pinned against the abstract evaluator record
//! by record, and against NEAT-AI-core's own helpers shape by shape.

use std::collections::HashSet;
use std::fmt;

use neat_core::topology_ops::{STRUCTURAL_VALID, validate_structural_integrity};
use neat_core::{
    CompiledNetwork, CreatureExport, IfNodeSpec, NeuronExport, RelaySpec, SynapseType,
    ValidateOptions, ValidationFailure, compile_creature, creature_validate, graft_if_nodes,
    graft_relay_node, validate_no_duplicate_synapses,
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

/// Gate a finished candidate on the shared definition of a valid creature
/// before it can escape the graft.
///
/// Two rules, one gate (Issue #50):
///
/// 1. [`check_no_duplicate_synapses`] — no ordered `(from, to)` pair twice.
///    It runs *first* so a repeat is attributed as
///    [`GraftError::DuplicateSynapse`], the error the journal already records,
///    rather than as whichever later rule happens to trip over it.
/// 2. `neat_core::creature_validate` — every other rule.
///
/// Keeping the duplicate rule inside the gate is the point: it is no longer a
/// check a caller has to remember to run alongside validation, so every path
/// that validates a grafted creature enforces it.
///
/// The failure is returned, never logged and dropped: [`graft_patch`]'s callers
/// record it against the candidate id (see `docs/architecture.md`, *Creature
/// validation*).
fn assert_valid(creature: &CreatureExport) -> Result<(), GraftError> {
    check_no_duplicate_synapses(creature)?;
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
/// as a [`GraftError`] so a caller sees one error type. `assert_valid`
/// applies it as part of the validation gate (Issue #50), so a caller that
/// validates a creature need not call this as well.
pub fn check_no_duplicate_synapses(creature: &CreatureExport) -> Result<(), GraftError> {
    validate_no_duplicate_synapses(creature)
        .map_err(|e| GraftError::DuplicateSynapse(e.to_string()))
}

/// `wanted` uuids for shared bias-1 constants that no neuron in the creature
/// already carries.
///
/// The first choices are `forest-one-a/b/c`, so a creature grafted more than
/// once keeps finding and reusing the same three. A creature that already
/// carries one of those names — a constant of some other bias, a hidden
/// neuron, anything this graft must not repurpose — is **not** a reason to
/// refuse the graft (Issue #50): the taken name is skipped and the next free
/// one (`forest-one-a2`, `forest-one-b2`, …) is used instead. Refusing would
/// have made every later graft on that creature fail as well.
fn free_one_names(existing: &HashSet<&str>, wanted: usize) -> Result<Vec<String>, GraftError> {
    let mut out: Vec<String> = Vec::with_capacity(wanted);
    // Every round offers three fresh names, so `existing.len() / 3 + 1` rounds
    // already exhaust the names in use; the bound just keeps the loop finite.
    for round in 0..=existing.len() + 1 {
        if out.len() == wanted {
            break;
        }
        for letter in ['a', 'b', 'c'] {
            let name = match round {
                0 => format!("forest-one-{letter}"),
                r => format!("forest-one-{letter}{}", r + 1),
            };
            if out.len() < wanted && !existing.contains(name.as_str()) {
                out.push(name);
            }
        }
    }
    if out.len() < wanted {
        // Unreachable for any finite creature; fail closed rather than emit a
        // creature with two neurons under one uuid.
        return Err(GraftError::UuidCollision("forest-one-*".into()));
    }
    Ok(out)
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
    for name in free_one_names(&existing, 3 - ones.len())? {
        new_constants.push(NeuronExport {
            id: None,
            neuron_type: "constant".into(),
            uuid: name.clone(),
            bias: 1.0,
            squash: None,
        });
        ones.push(name);
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
    // The root's outward edge: untyped into a point-wise output, `positive`
    // into an `IF` one, where the relay below carries the `negative` half.
    let relay = if target_is_if {
        Some(em.fresh("relay")?)
    } else {
        None
    };
    {
        let root = em
            .specs
            .last_mut()
            .expect("a split patch describes at least one node");
        *root = if target_is_if {
            root.clone()
                .with_target_role(target_uuid.clone(), 1.0, SynapseType::Positive)
        } else {
            root.clone().with_target(target_uuid.clone(), 1.0)
        };
    }

    // New constants go in front of the first non-constant neuron:
    // `creature_validate` rule 11 rejects a constant that follows a hidden one.
    let base = with_constants(incumbent, new_constants, first_output);
    // NEAT-AI-core builds every shape (Issue #48): the post-order batch in one
    // all-or-nothing graft — a child carries no outward edge of its own, the
    // parent that reads it supplies one — and then the relay, whose source is
    // the root the batch has just placed.
    let mut creature =
        graft_if_nodes(&base, &em.specs).map_err(|e| GraftError::Canonical(e.to_string()))?;
    if let Some(relay) = relay {
        let spec = RelaySpec::new(relay, 0.0)
            .with_source(root_uuid.clone(), 1.0)
            .with_target_role(target_uuid.clone(), 1.0, SynapseType::Negative);
        creature =
            graft_relay_node(&creature, &spec).map_err(|e| GraftError::Canonical(e.to_string()))?;
    }

    let added_neurons = creature.neurons.len() - incumbent.neurons.len();
    let added_synapses = creature.synapses.len() - incumbent.synapses.len();
    let added_uuids: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| !existing.contains(n.uuid.as_str()))
        .map(|n| n.uuid.clone())
        .collect();

    // The gate the candidate has to clear before it escapes: the shared
    // definition of a valid creature (Issue #39), duplicate-pair rule included
    // (Issue #50). It runs ahead of compilation so a broken candidate is
    // attributed to the rule it broke — `DuplicateSynapse`, `NEURON_ORDER`,
    // … — at the graft that broke it, rather than surfacing as whatever the
    // compiler says about it downstream.
    assert_valid(&creature)?;
    let network = compile_creature(&creature).map_err(|e| GraftError::Compile(e.to_string()))?;
    validate_compiled(&network, &creature)?;
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
    use neat_core::{SynapseExport, graft_if_node};
    use std::collections::HashMap;

    /// Resolve every neuron uuid to its index in the compiled ordering:
    /// implicit input `i` is index `i`, listed neuron `j` is `input + j`. The
    /// same derivation `neat_core::compile_creature` and `creature_validate`
    /// perform, and what "sorted by (from, to)" is measured against.
    ///
    /// The graft itself no longer orders synapses — NEAT-AI-core emits the
    /// canonical order (Issue #48) — so this is written out here rather than
    /// reused from the module under test, which would put the oracle on the
    /// code path it is checking.
    fn uuid_indices(creature: &CreatureExport) -> HashMap<String, u32> {
        creature
            .neurons
            .iter()
            .enumerate()
            .map(|(i, n)| (n.uuid.clone(), (creature.input + i) as u32))
            .collect()
    }

    /// Put a fixture's synapse list into that canonical order. An endpoint
    /// naming no neuron sorts last (`u32::MAX`) rather than being dropped.
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

    /// The three shared constants this module allocates, in the order it
    /// allocates them, spliced ahead of `inc`'s first non-constant neuron.
    fn base_with_shared_ones(inc: &CreatureExport) -> CreatureExport {
        with_constants(
            inc,
            vec![
                one("forest-one-a"),
                one("forest-one-b"),
                one("forest-one-c"),
            ],
            inc.neurons.len() - inc.output,
        )
    }

    /// Issue #48 — a **nested** tree is built by NEAT-AI-core's batch helper,
    /// not written out here: the grafted creature is exactly what
    /// `graft_if_nodes` returns for the same post-order specs, the child
    /// carrying no outward edge of its own.
    #[test]
    fn nested_graft_is_built_by_the_canonical_batch_helper() {
        let inc = identity_creature(2, 1);
        let patch = Patch::new(
            0,
            Node::Split {
                condition: Condition::axis(0, 0.1),
                left: Box::new(Node::leaf(-0.2)),
                right: Box::new(Node::stump(1, 0.3, 0.0, 0.4)),
            },
            Provenance::default(),
        );
        let ours = graft_patch(&inc, &patch).unwrap();

        let id = patch.id();
        let child = IfNodeSpec::new(format!("forest-{id}-if0"), 0.0)
            .with_condition("input-1", 1.0)
            .with_condition("forest-one-a", f64::from(-0.3f32))
            .with_positive("forest-one-b", f64::from(0.4f32))
            .with_negative("forest-one-c", 0.0);
        let root = IfNodeSpec::new(format!("forest-{id}-if1"), 0.0)
            .with_condition("input-0", 1.0)
            .with_condition("forest-one-a", f64::from(-0.1f32))
            .with_positive(format!("forest-{id}-if0"), 1.0)
            .with_negative("forest-one-c", f64::from(-0.2f32))
            .with_target("output-0", 1.0);
        let expected = graft_if_nodes(&base_with_shared_ones(&inc), &[child, root])
            .expect("the canonical batch helper builds the nested tree");
        assert_eq!(ours.creature, expected);
    }

    /// Issue #48 — the `IF`-output shape is built by NEAT-AI-core too: a typed
    /// `positive` outward edge from the root and `graft_relay_node` for the
    /// IDENTITY relay that carries the same correction into the `negative`
    /// branch.
    #[test]
    fn if_output_graft_is_built_by_the_canonical_helpers() {
        let inc = if_output_creature(4);
        let patch = Patch::new(0, Node::stump(3, 0.25, -0.05, 0.07), Provenance::default());
        let ours = graft_patch(&inc, &patch).unwrap();

        let id = patch.id();
        let root = IfNodeSpec::new(format!("forest-{id}-if0"), 0.0)
            .with_condition("input-3", 1.0)
            .with_condition("forest-one-a", f64::from(-0.25f32))
            .with_positive("forest-one-b", f64::from(0.07f32))
            .with_negative("forest-one-c", f64::from(-0.05f32))
            .with_target_role("output-0", 1.0, SynapseType::Positive);
        let expected = graft_if_nodes(&base_with_shared_ones(&inc), &[root])
            .expect("the canonical helper builds the typed root");
        let relay = RelaySpec::new(format!("forest-{id}-relay1"), 0.0)
            .with_source(format!("forest-{id}-if0"), 1.0)
            .with_target_role("output-0", 1.0, SynapseType::Negative);
        let expected =
            graft_relay_node(&expected, &relay).expect("the canonical helper builds the relay");
        assert_eq!(ours.creature, expected);
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

    // ---- Issue #50 — every graft must emit a *valid* creature -------------
    //
    // NEAT-AI's TypeScript loader keys synapses by their ordered `(from, to)`
    // pair and silently drops the rest, so a creature carrying a repeated pair
    // scores differently there than it does under `rust_scorer`. The tests
    // below try hard to make the emitter produce one.

    /// The two invariants Issue #50 is about, on one finished creature: no
    /// ordered `(from, to)` pair appears twice, and the shared validator
    /// accepts it.
    fn assert_creature_is_valid(creature: &CreatureExport, what: &str) {
        let mut seen: HashSet<(&str, &str)> = HashSet::new();
        for s in &creature.synapses {
            assert!(
                seen.insert((s.from_uuid.as_str(), s.to_uuid.as_str())),
                "{what}: duplicate synapse {} → {}",
                s.from_uuid,
                s.to_uuid
            );
        }
        if let Err(failure) = creature_validate(creature, &validate_options(creature)) {
            panic!(
                "{what}: creature_validate rejected the graft: {}",
                failure.message
            );
        }
    }

    /// Graft `patch` onto `inc` and assert every Issue #50 invariant: no
    /// repeated pair, `creature_validate` passes, and the result still matches
    /// the abstract evaluator. An `IF` output is compared on its activation —
    /// the correction reaches it through two typed branches, not one additive
    /// edge, so there is no single pre-squash sum to read.
    fn assert_graft_is_sound(inc: &CreatureExport, patch: &Patch, what: &str) -> Grafted {
        let g = graft_patch(inc, patch).unwrap_or_else(|e| panic!("{what}: graft refused: {e}"));
        assert_creature_is_valid(&g.creature, what);
        let target = &inc.neurons[inc.neurons.len() - inc.output + patch.output];
        if target.squash.as_deref() == Some("IF") {
            let mut base = compile_creature(inc).unwrap();
            let mut cand = compile_creature(&g.creature).unwrap();
            let (mut left, mut right) = (0, 0);
            for rec in records(300, inc.input) {
                let delta = cand.activate(&rec, inc.output)[patch.output]
                    - base.activate(&rec, inc.output)[patch.output];
                let want = patch.evaluate(&rec);
                assert!(
                    (delta - want).abs() <= 1e-6,
                    "{what}: delta {delta} != {want}"
                );
                if let Node::Split { condition, .. } = &patch.root {
                    if condition.goes_right(&rec) {
                        right += 1
                    } else {
                        left += 1
                    }
                }
            }
            assert!(
                left > 0 && right > 0,
                "{what}: fixture must cover both branches"
            );
        } else {
            assert_graft_matches_evaluator(inc, patch, 1e-6);
        }
        g
    }

    /// Every tree shape up to `depth`: at each level a branch is independently
    /// a leaf or a further split. Values are filled in by [`decorate`].
    fn shapes(depth: usize) -> Vec<Node> {
        if depth == 0 {
            return vec![Node::leaf(0.0)];
        }
        let mut out = vec![Node::leaf(0.0)];
        for l in shapes(depth - 1) {
            for r in shapes(depth - 1) {
                out.push(Node::Split {
                    condition: Condition::axis(0, 0.0),
                    left: Box::new(l.clone()),
                    right: Box::new(r),
                });
            }
        }
        out
    }

    /// Give a shape distinct features, thresholds and leaf corrections; the
    /// root threshold stays inside the record range so both branches are
    /// exercised.
    fn decorate(node: &Node, k: &mut usize, inputs: usize) -> Node {
        *k += 1;
        let step = *k;
        match node {
            Node::Leaf { .. } => Node::leaf(0.02 * (step % 7) as f32 - 0.06),
            Node::Split { left, right, .. } => Node::Split {
                condition: Condition::axis(step % inputs, 0.1 * (step % 5) as f32 - 0.2),
                left: Box::new(decorate(left, k, inputs)),
                right: Box::new(decorate(right, k, inputs)),
            },
        }
    }

    /// `identity_creature(inputs, 1)` carrying `extra` neurons ahead of the
    /// output, each wired so nothing dangles.
    fn creature_carrying(inputs: usize, extra: &[NeuronExport]) -> CreatureExport {
        let mut c = identity_creature(inputs, 1);
        for (i, n) in extra.iter().enumerate() {
            if n.neuron_type == "hidden" {
                c.synapses.push(SynapseExport {
                    from_uuid: "input-0".into(),
                    to_uuid: n.uuid.clone(),
                    weight: 0.5,
                    synapse_type: None,
                });
            }
            c.synapses.push(SynapseExport {
                from_uuid: n.uuid.clone(),
                to_uuid: "output-0".into(),
                weight: 0.01,
                synapse_type: None,
            });
            c.neurons.insert(i, n.clone());
        }
        sort_synapses_canonically(&mut c);
        c
    }

    /// A constant neuron.
    fn constant(uuid: &str, bias: f64) -> NeuronExport {
        NeuronExport {
            id: None,
            neuron_type: "constant".into(),
            uuid: uuid.into(),
            bias,
            squash: None,
        }
    }

    /// Issue #50 — depth-1, depth-2 and depth-3 trees, every combination of
    /// leaf and split branches, over a point-wise output, a squashing output
    /// and an `IF` output.
    #[test]
    fn every_tree_shape_grafts_to_a_valid_duplicate_free_creature() {
        for inc in [identity_creature(4, 2), small_mlp(3), if_output_creature(4)] {
            for (i, shape) in shapes(3).iter().enumerate() {
                if matches!(shape, Node::Leaf { .. }) {
                    continue; // a bare leaf is refused as `RootIsLeaf`
                }
                let mut k = 0;
                let patch =
                    Patch::new(0, decorate(shape, &mut k, inc.input), Provenance::default());
                assert_graft_is_sound(&inc, &patch, &format!("shape {i} on {} inputs", inc.input));
            }
        }
    }

    /// Issue #50 — the three synapse roles of every `IF` node must read three
    /// *different* neurons, whatever the incumbent already carries: none, one,
    /// two, exactly three, or more bias-1 constants.
    #[test]
    fn any_number_of_existing_bias_one_constants_yields_three_distinct_sources() {
        for k in 0..=5usize {
            let extra: Vec<NeuronExport> = (0..k)
                .map(|i| constant(&format!("const-{i}"), 1.0))
                .collect();
            let inc = creature_carrying(3, &extra);
            let what = format!("{k} pre-existing bias-1 constants");
            let patch = Patch::new(0, Node::stump(0, 0.0, -0.03, 0.05), Provenance::default());
            let g = assert_graft_is_sound(&inc, &patch, &what);
            assert_eq!(
                g.creature
                    .neurons
                    .iter()
                    .filter(|n| n.neuron_type == "constant" && n.bias == 1.0)
                    .count(),
                k.max(3),
                "{what}: three bias-1 constants must be available"
            );
            let node = g
                .added_uuids
                .iter()
                .find(|u| u.contains("-if"))
                .expect("the graft added an IF node");
            let sources: HashSet<&str> = g
                .creature
                .synapses
                .iter()
                .filter(|s| &s.to_uuid == node)
                .map(|s| s.from_uuid.as_str())
                .collect();
            // input-0 plus one constant per role — three distinct constants.
            assert_eq!(
                sources.len(),
                4,
                "{what}: roles share a source: {sources:?}"
            );
        }
    }

    /// Issue #50 — a creature that already carries the shared-constant *names*
    /// must still graft, whether or not those neurons are usable bias-1
    /// constants. A name in use is not a reason to refuse the graft; the
    /// emitter has to pick names that are free.
    #[test]
    fn the_shared_constant_names_being_taken_never_blocks_a_graft() {
        let cases: Vec<(&str, Vec<NeuronExport>)> = vec![
            (
                "all three present and usable",
                vec![
                    constant("forest-one-a", 1.0),
                    constant("forest-one-b", 1.0),
                    constant("forest-one-c", 1.0),
                ],
            ),
            (
                "one name taken by a constant of another bias",
                vec![constant("forest-one-a", 0.5)],
            ),
            (
                "one usable, the next name taken by another bias",
                vec![constant("forest-one-a", 1.0), constant("forest-one-b", 0.5)],
            ),
            (
                "all three names taken by constants of another bias",
                vec![
                    constant("forest-one-a", 0.5),
                    constant("forest-one-b", 0.25),
                    constant("forest-one-c", -1.0),
                ],
            ),
            (
                "a name taken by a hidden neuron",
                vec![NeuronExport {
                    id: None,
                    neuron_type: "hidden".into(),
                    uuid: "forest-one-c".into(),
                    bias: 0.0,
                    squash: Some("TANH".into()),
                }],
            ),
        ];
        for (what, extra) in cases {
            let inc = creature_carrying(3, &extra);
            let patch = Patch::new(0, Node::stump(1, 0.0, -0.02, 0.06), Provenance::default());
            let g = assert_graft_is_sound(&inc, &patch, what);
            // Nothing pre-existing was repurposed.
            for n in &extra {
                let after = g
                    .creature
                    .neurons
                    .iter()
                    .find(|x| x.uuid == n.uuid)
                    .unwrap_or_else(|| panic!("{what}: {} was dropped", n.uuid));
                assert_eq!(after, n, "{what}: {} was rewritten", n.uuid);
            }
            // A second graft on the result must work too — the shared
            // constants have to be re-findable, not re-collided with.
            let again = Patch::new(0, Node::stump(2, 0.1, 0.04, 0.0), Provenance::default());
            assert_graft_is_sound(&g.creature, &again, &format!("{what}, second graft"));
        }
    }

    /// Issue #50 — oblique conditions (two and three terms) at every depth.
    #[test]
    fn oblique_multi_term_conditions_stay_duplicate_free() {
        let oblique = |a: usize, b: usize, c: Option<usize>, threshold: f32| Condition {
            terms: [Some((a, 0.8f32)), Some((b, -1.3)), c.map(|f| (f, 0.45))]
                .into_iter()
                .flatten()
                .map(|(feature, weight)| Term { feature, weight })
                .collect(),
            threshold,
        };
        let root = Node::Split {
            condition: oblique(0, 1, Some(2), 0.05),
            left: Box::new(Node::Split {
                condition: oblique(1, 2, None, -0.1),
                left: Box::new(Node::leaf(0.03)),
                right: Box::new(Node::Split {
                    condition: oblique(0, 3, Some(1), 0.2),
                    left: Box::new(Node::leaf(-0.02)),
                    right: Box::new(Node::leaf(0.0)),
                }),
            }),
            right: Box::new(Node::Split {
                condition: oblique(3, 0, None, 0.0),
                left: Box::new(Node::leaf(0.07)),
                right: Box::new(Node::leaf(-0.05)),
            }),
        };
        for inc in [identity_creature(4, 1), small_mlp(4), if_output_creature(4)] {
            let patch = Patch::new(0, root.clone(), Provenance::default());
            assert_graft_is_sound(&inc, &patch, "oblique depth-3 tree");
        }
    }

    /// Issue #50 — several patches grafted onto one creature (the combination
    /// candidates `generate_combos` builds) share the constants, so this is
    /// where a repeated pair would show up if the roles were collapsed.
    #[test]
    fn stacked_patches_share_the_constants_without_repeating_a_pair() {
        let patches: Vec<Patch> = vec![
            Patch::new(0, Node::stump(0, 0.0, -0.04, 0.06), Provenance::default()),
            Patch::new(
                0,
                Node::Split {
                    condition: Condition::axis(1, -0.15),
                    left: Box::new(Node::stump(2, 0.1, 0.02, -0.03)),
                    right: Box::new(Node::leaf(0.05)),
                },
                Provenance::default(),
            ),
            Patch::new(
                0,
                Node::Split {
                    condition: Condition::axis(2, 0.05),
                    left: Box::new(Node::leaf(-0.01)),
                    right: Box::new(Node::stump(3, 0.0, 0.03, -0.02)),
                },
                Provenance::default(),
            ),
        ];
        for inc in [identity_creature(4, 1), small_mlp(4), if_output_creature(4)] {
            let (stacked, added) = graft_patches(&inc, &patches).unwrap();
            assert_eq!(added.len(), 3);
            assert_creature_is_valid(&stacked, "three stacked patches");
            assert_eq!(
                stacked
                    .neurons
                    .iter()
                    .filter(|n| n.neuron_type == "constant" && n.bias == 1.0)
                    .count(),
                3,
                "the shared constants are created once, not per patch"
            );
            if inc.neurons.last().unwrap().squash.as_deref() != Some("LOGISTIC") {
                let mut base = compile_creature(&inc).unwrap();
                let mut cand = compile_creature(&stacked).unwrap();
                for rec in records(200, inc.input) {
                    let delta = cand.activate(&rec, 1)[0] - base.activate(&rec, 1)[0];
                    let want: f32 = patches.iter().map(|p| p.evaluate(&rec)).sum();
                    assert!((delta - want).abs() < 1e-5, "delta {delta} != {want}");
                }
            }
        }
    }

    /// Issue #50 — "the creature validation should have prevented this in the
    /// first place": the duplicate-pair rule is part of the validation gate,
    /// not a check a caller has to remember to run.
    #[test]
    fn the_validation_gate_itself_rejects_a_duplicate_pair() {
        let mut dup = identity_creature(2, 1);
        dup.synapses.push(dup.synapses[0].clone());
        let err = assert_valid(&dup).expect_err("the gate must reject a repeated (from, to) pair");
        assert!(
            matches!(err, GraftError::DuplicateSynapse(_)),
            "expected a duplicate-synapse failure, got {err}"
        );
        assert!(err.to_string().contains("input-0"), "{err}");
        // A creature without duplicates still clears the same gate.
        assert_valid(&identity_creature(2, 1)).expect("a clean creature passes");
    }
}

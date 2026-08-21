//! Graft a [`Patch`] onto a **clone** of an incumbent as ordinary NEAT-AI `IF`
//! structure (Issue #7).
//!
//! Layout appended for each patch (all UUIDs prefixed `forest-<patch id>-`),
//! chosen so that **no (from, to) synapse pair is ever repeated** — NEAT-AI's
//! TypeScript loader keys synapses by that pair and silently collapses
//! duplicates, which `rust_scorer` does not:
//!
//! ```text
//! shared:     one_a, one_b   constant, bias 1.0 — reused from the creature when it has
//!                            bias-1 constants, else created once (`forest-one-a/b`);
//!                            never per patch (#43)
//! per split:  thrN    hidden IDENTITY, bias = -threshold, inward input-f (weight w_f per term)
//!             ifN     hidden IF, bias 0
//!                 condition:  thrN  (weight 1)                  → Σ w·x − threshold
//!                 positive:   right child ifN (weight 1)  |  one_a (weight = right leaf)
//!                 negative:   left  child ifN (weight 1)  |  one_b (weight = left leaf)
//! root ifN ──(weight 1, untyped)──▶ output-j            (point-wise output squash)
//! root ifN ──(positive)──▶ output-j  and  root ifN → relayN (IDENTITY) ──(negative)──▶ output-j   (IF output)
//! ```
//!
//! This relies only on the documented NEAT-AI-core kernel
//! (`condition_sum > 0 ? positive_sum + bias : negative_sum + bias`, no squash on
//! `IF`) and the structural validator (≥3 inward, one of each role). Because the
//! root feeds the output neuron's *pre-squash* sum with weight 1, a leaf of
//! exactly `0.0` leaves every record in that region bit-identical to the
//! incumbent.
//!
//! Until `NEAT-AI-core#555` ships canonical helpers, this module is the single
//! place in Forests that interprets `IF` synapse roles, and its tests pin the
//! grafted creature against the abstract evaluator record by record.

use std::collections::HashSet;
use std::fmt;

use neat_core::topology_ops::{STRUCTURAL_VALID, validate_structural_integrity};
use neat_core::{CompiledNetwork, CreatureExport, NeuronExport, SynapseExport, compile_creature};

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
        }
    }
}

impl std::error::Error for GraftError {}

struct Emitter<'a> {
    prefix: String,
    /// Shared bias-1 constants: positive leaves hang off `one_a`, negative
    /// leaves off `one_b`, so a leaf is a synapse *weight* and no (from, to)
    /// pair repeats. Reused from the creature when it already has them.
    one_a: String,
    one_b: String,
    neurons: Vec<NeuronExport>,
    synapses: Vec<SynapseExport>,
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

    fn neuron(&mut self, uuid: &str, neuron_type: &str, bias: f64, squash: Option<&str>) {
        self.neurons.push(NeuronExport {
            neuron_type: neuron_type.into(),
            uuid: uuid.into(),
            bias,
            squash: squash.map(str::to_string),
        });
    }

    fn synapse(&mut self, from: &str, to: &str, weight: f64, role: Option<&str>) {
        self.synapses.push(SynapseExport {
            from_uuid: from.into(),
            to_uuid: to.into(),
            weight,
            synapse_type: role.map(str::to_string),
        });
    }

    /// Source feeding a parent's branch: a leaf is `(shared constant, weight =
    /// correction)`, a split is `(its IF neuron, 1.0)`. `positive` selects the
    /// constant so the two leaves of one IF never share a source.
    fn branch_source(&mut self, node: &Node, positive: bool) -> Result<(String, f64), GraftError> {
        match node {
            Node::Leaf { correction } => {
                let one = if positive {
                    self.one_a.clone()
                } else {
                    self.one_b.clone()
                };
                Ok((one, f64::from(*correction)))
            }
            Node::Split { .. } => Ok((self.emit(node)?, 1.0)),
        }
    }

    /// Emit a split; returns the uuid of its IF neuron. Every (from, to) pair
    /// emitted is unique — NEAT-AI's TypeScript keys synapses by that pair and
    /// collapses duplicates.
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
                // Condition: thr = Σ w·x − threshold via one IDENTITY neuron.
                let thr = self.fresh("thr")?;
                self.neuron(
                    &thr,
                    "hidden",
                    f64::from(-condition.threshold),
                    Some("IDENTITY"),
                );
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
                            "input-{} → {thr}",
                            t.feature
                        )));
                    }
                    self.synapse(
                        &format!("input-{}", t.feature),
                        &thr,
                        f64::from(t.weight),
                        None,
                    );
                }
                let uuid = self.fresh("if")?;
                self.neuron(&uuid, "hidden", 0.0, Some("IF"));
                self.synapse(&thr, &uuid, 1.0, Some("condition"));
                self.synapse(&right_src, &uuid, right_w, Some("positive"));
                self.synapse(&left_src, &uuid, left_w, Some("negative"));
                Ok(uuid)
            }
        }
    }
}

/// Reject any repeated (from, to) synapse pair — NEAT-AI's TypeScript loader
/// keys synapses by that pair, so a creature with duplicates scores
/// differently there than under `rust_scorer`.
pub fn check_no_duplicate_synapses(creature: &CreatureExport) -> Result<(), GraftError> {
    let mut seen = HashSet::with_capacity(creature.synapses.len());
    for s in &creature.synapses {
        if !seen.insert((s.from_uuid.as_str(), s.to_uuid.as_str())) {
            return Err(GraftError::DuplicateSynapse(format!(
                "{} → {}",
                s.from_uuid, s.to_uuid
            )));
        }
    }
    Ok(())
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
    // create at most two, named without the patch id so later grafts share them.
    let mut ones: Vec<String> = incumbent
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "constant" && n.bias == 1.0 && n.squash.is_none())
        .map(|n| n.uuid.clone())
        .take(2)
        .collect();
    let mut new_constants = Vec::new();
    for name in ["forest-one-a", "forest-one-b"] {
        if ones.len() >= 2 {
            break;
        }
        if existing.contains(name) {
            return Err(GraftError::UuidCollision(name.into()));
        }
        new_constants.push(NeuronExport {
            neuron_type: "constant".into(),
            uuid: name.into(),
            bias: 1.0,
            squash: None,
        });
        ones.push(name.into());
    }
    let mut em = Emitter {
        prefix,
        one_a: ones[0].clone(),
        one_b: ones[1].clone(),
        neurons: new_constants,
        synapses: Vec::new(),
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
    if parsed == neat_core::SquashType::If {
        let relay = em.fresh("relay")?;
        em.neuron(&relay, "hidden", 0.0, Some("IDENTITY"));
        em.synapse(&root_uuid, &relay, 1.0, None);
        em.synapse(&root_uuid, &target_uuid, 1.0, Some("positive"));
        em.synapse(&relay, &target_uuid, 1.0, Some("negative"));
    } else if parsed.is_aggregate() {
        return Err(GraftError::UnsupportedOutputSquash(
            target_squash.to_string(),
        ));
    } else {
        em.synapse(&root_uuid, &target_uuid, 1.0, None);
    }

    let mut creature = incumbent.clone();
    let added_neurons = em.neurons.len();
    let added_synapses = em.synapses.len();
    let added_uuids: Vec<String> = em.neurons.iter().map(|n| n.uuid.clone()).collect();
    // Insert before the first output so listed order stays topological.
    let tail = creature.neurons.split_off(first_output);
    creature.neurons.extend(em.neurons);
    creature.neurons.extend(tail);
    creature.synapses.extend(em.synapses);

    check_no_duplicate_synapses(&creature)?;
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
        }
    }

    /// A small creature with a hidden TANH layer and a LOGISTIC output, so
    /// tests cover a non-identity output squash.
    pub fn small_mlp(inputs: usize) -> CreatureExport {
        let mut neurons = Vec::new();
        let mut synapses = Vec::new();
        for h in 0..3 {
            neurons.push(NeuronExport {
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
        // Prefix invariants: nothing pre-existing changed.
        let n_out = incumbent.output;
        let keep = incumbent.neurons.len() - n_out;
        assert_eq!(
            &grafted.creature.neurons[..keep],
            &incumbent.neurons[..keep]
        );
        assert_eq!(
            &grafted.creature.neurons[grafted.creature.neurons.len() - n_out..],
            &incumbent.neurons[keep..]
        );
        assert_eq!(
            &grafted.creature.synapses[..incumbent.synapses.len()],
            &incumbent.synapses[..]
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
        assert_eq!(g.added_neurons, 4); // one_a, one_b (first graft only), thr, if
        assert_eq!(g.added_synapses, 5); // input→thr, thr→if, one_a→if, one_b→if, if→output
        // A second graft reuses the shared constants.
        let again = graft_patch(
            &g.creature,
            &Patch::new(0, Node::stump(1, 0.3, 0.0, 0.02), Provenance::default()),
        )
        .unwrap();
        assert_eq!(again.added_neurons, 2);
        assert_eq!(
            again
                .creature
                .neurons
                .iter()
                .filter(|n| n.neuron_type == "constant")
                .count(),
            2
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
        assert_eq!(added[1].len(), 2); // shared constants already present
        assert_eq!(
            creature
                .neurons
                .iter()
                .filter(|n| n.neuron_type == "constant")
                .count(),
            2
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
        let roles: Vec<_> = back
            .synapses
            .iter()
            .filter_map(|s| s.synapse_type.clone())
            .collect();
        assert_eq!(roles, ["condition", "positive", "negative"]);
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
                neuron_type: "constant".into(),
                uuid: "const-1".into(),
                bias: 1.0,
                squash: None,
            },
        );
        inc.neurons.insert(
            1,
            NeuronExport {
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
        assert_eq!(consts, ["const-1", "const-half", "forest-one-a"]); // const-1 reused, one extra created
        assert!(g.creature.synapses.iter().any(|s| s.from_uuid == "const-1"
            && s.synapse_type.as_deref() == Some("positive")
            && (s.weight - 0.4).abs() < 1e-6));
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
}

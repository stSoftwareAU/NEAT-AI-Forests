//! Graft a [`Patch`] onto a **clone** of an incumbent as ordinary NEAT-AI `IF`
//! structure (Issue #7).
//!
//! Layout appended for each patch (all UUIDs prefixed `forest-<patch id>-`):
//!
//! ```text
//! const   bias=1.0                       (type "constant", no squash, no inward)
//! IF node per split, children first:
//!     condition:  input-f  (weight w_f) …,  const (weight -threshold)
//!     positive:   right child  (child IF weight 1.0  |  const weight right_leaf)
//!     negative:   left  child  (child IF weight 1.0  |  const weight left_leaf)
//! root IF ──(weight 1.0, untyped)──▶ output-j
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
    /// Output neurons are not trailing.
    OutputsNotLast,
    /// NEAT-AI-core refused to compile the result.
    Compile(String),
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
            Self::OutputsNotLast => write!(
                f,
                "output neurons are not the trailing entries of `neurons`"
            ),
            Self::Compile(m) => write!(f, "grafted creature does not compile: {m}"),
            Self::Structural { code, index } => {
                write!(f, "structural validation failed: code {code} at {index}")
            }
        }
    }
}

impl std::error::Error for GraftError {}

struct Emitter<'a> {
    prefix: String,
    const_uuid: String,
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

    /// Returns `(source uuid, weight)` that yields the node's correction.
    fn emit(&mut self, node: &Node) -> Result<(String, f64), GraftError> {
        match node {
            Node::Leaf { correction } => Ok((self.const_uuid.clone(), f64::from(*correction))),
            Node::Split {
                condition,
                left,
                right,
            } => {
                let (left_src, left_w) = self.emit(left)?;
                let (right_src, right_w) = self.emit(right)?;
                let uuid = self.fresh("if")?;
                for t in &condition.terms {
                    if t.feature >= self.input {
                        return Err(GraftError::FeatureOutOfRange {
                            feature: t.feature,
                            input: self.input,
                        });
                    }
                    self.synapses.push(SynapseExport {
                        from_uuid: format!("input-{}", t.feature),
                        to_uuid: uuid.clone(),
                        weight: f64::from(t.weight),
                        synapse_type: Some("condition".into()),
                    });
                }
                self.synapses.push(SynapseExport {
                    from_uuid: self.const_uuid.clone(),
                    to_uuid: uuid.clone(),
                    weight: f64::from(-condition.threshold),
                    synapse_type: Some("condition".into()),
                });
                self.synapses.push(SynapseExport {
                    from_uuid: right_src,
                    to_uuid: uuid.clone(),
                    weight: right_w,
                    synapse_type: Some("positive".into()),
                });
                self.synapses.push(SynapseExport {
                    from_uuid: left_src,
                    to_uuid: uuid.clone(),
                    weight: left_w,
                    synapse_type: Some("negative".into()),
                });
                self.neurons.push(NeuronExport {
                    neuron_type: "hidden".into(),
                    uuid: uuid.clone(),
                    bias: 0.0,
                    squash: Some("IF".into()),
                });
                Ok((uuid, 1.0))
            }
        }
    }
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
    let const_uuid = format!("{prefix}-one");
    if existing.contains(const_uuid.as_str()) {
        return Err(GraftError::UuidCollision(const_uuid));
    }
    let mut em = Emitter {
        prefix,
        const_uuid: const_uuid.clone(),
        neurons: vec![NeuronExport {
            neuron_type: "constant".into(),
            uuid: const_uuid,
            bias: 1.0,
            squash: None,
        }],
        synapses: Vec::new(),
        counter: 0,
        input: incumbent.input,
        existing: &existing,
    };
    let (root_uuid, root_w) = em.emit(&patch.root)?;
    em.synapses.push(SynapseExport {
        from_uuid: root_uuid,
        to_uuid: target_uuid,
        weight: root_w,
        synapse_type: None,
    });

    let mut creature = incumbent.clone();
    let added_neurons = em.neurons.len();
    let added_synapses = em.synapses.len();
    // Insert before the first output so listed order stays topological.
    let tail = creature.neurons.split_off(first_output);
    creature.neurons.extend(em.neurons);
    creature.neurons.extend(tail);
    creature.synapses.extend(em.synapses);

    let network = compile_creature(&creature).map_err(|e| GraftError::Compile(e.to_string()))?;
    validate_compiled(&network, &creature)?;
    Ok(Grafted {
        creature,
        added_neurons,
        added_synapses,
    })
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
        assert_eq!(g.added_neurons, 2);
        assert_eq!(g.added_synapses, 5);
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
        assert_eq!(roles, ["condition", "condition", "positive", "negative"]);
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

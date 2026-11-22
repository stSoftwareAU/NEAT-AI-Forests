//! Split on hidden neurons: the creature's own learned features as split
//! candidates (Issue #139).
//!
//! Forests otherwise searches splits over the raw inputs only, while a mature
//! creature carries hundreds of evolved hidden neurons — combinations of
//! inputs that evolution kept, already paid for in the complexity penalty, and
//! invisible to the tree search. Splitting on a neuron that already combines
//! dozens of inputs is how a shallow tree reaches interactions that the same
//! depth over raw inputs cannot.
//!
//! With `--hidden-features N` each iteration traces the incumbent on the
//! search set, quantile-bins the activations of up to `N` hidden neurons and
//! appends them to the search set as extra columns. Every search — stumps,
//! trees, refinements — then sees them like inputs. A discovery that splits on
//! one is written into the patch by UUID ([`Term::neuron`]), so the patch stays
//! portable and the graft emits the condition edge from that neuron.
//!
//! Which neurons:
//!
//! * hidden neurons listed before the output's anchor, so the graft stays
//!   forward-only — the anchor itself and everything it feeds, the output
//!   included, would make a graft that feeds itself;
//! * not Forests' own graft neurons (`forest-…`), which are trees of inputs;
//! * not degenerate: at least two distinct quantile edges on the sample;
//! * ranked by the same between-bin variance of the mean residual the
//!   `error-ranked` feature selection uses, on a small stride sample, and
//!   capped at `N`.
//!
//! A hidden feature can go stale — another optimiser may change the weights
//! behind it — exactly as a threshold goes stale after a corpus refresh: the
//! scorer judged the patch when it was accepted and judges every later
//! change too. A patch that names a neuron which no longer exists is refused
//! at graft time.

use std::collections::HashMap;
use std::path::Path;

use neat_core::{CompiledNetwork, CreatureExport, compile_creature};

use crate::bins::{BinCache, bin_of, quantile_edges};
use crate::graft::{output_uuid, resolve_anchor};
use crate::patch::{HIDDEN_SOURCE_PATCH_FORMAT_VERSION, Node, Patch, Term};
use crate::strategies::{SearchSet, for_each_set_row};

/// Rows of the search set used to rank candidate neurons and place their
/// quantile edges.
pub const RANKING_ROWS: usize = 4096;

/// The hidden columns appended to a search set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HiddenColumns {
    /// Input width; column `k` is extended feature index `input + k`.
    pub input: usize,
    /// UUID per column.
    pub uuids: Vec<String>,
    /// Neurons that were eligible before the cap.
    pub eligible: usize,
}

impl HiddenColumns {
    /// The UUID behind extended feature index `feature`, if it is a hidden
    /// column.
    pub fn uuid(&self, feature: usize) -> Option<&str> {
        feature
            .checked_sub(self.input)
            .and_then(|k| self.uuids.get(k))
            .map(String::as_str)
    }

    /// Rewrite every term of `patch` that names a hidden column by extended
    /// index into a term on that neuron's UUID. Returns `true` when anything
    /// changed; such a patch is format v2 and its strategy gains `+hidden`,
    /// so the economics report keeps hidden-source patches apart.
    pub fn resolve(&self, patch: &mut Patch) -> bool {
        let mut changed = false;
        patch.root.visit_terms_mut(&mut |t: &mut Term| {
            if t.neuron.is_none()
                && let Some(uuid) = self.uuid(t.feature)
            {
                *t = Term::hidden(uuid, t.weight);
                changed = true;
            }
        });
        if changed {
            patch.version = patch.version.max(HIDDEN_SOURCE_PATCH_FORMAT_VERSION);
            if !patch.provenance.strategy.ends_with("+hidden") {
                patch.provenance.strategy.push_str("+hidden");
            }
            let sources = patch.root.hidden_sources();
            patch
                .provenance
                .notes
                .push(format!("hidden-sources={}", sources.join(",")));
        }
        changed
    }
}

/// Hidden neurons a graft on `output` could split on, as `(index in
/// creature.neurons, uuid)`: listed before the anchor, not Forests' own, not
/// the anchor or anything after it.
///
/// # Errors
///
/// The anchor walk's failure, as a string.
pub fn eligible_sources(
    creature: &CreatureExport,
    output: usize,
) -> Result<Vec<(usize, String)>, String> {
    let anchor = resolve_anchor(
        creature,
        &output_uuid(creature, output).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let anchor_at = creature
        .neurons
        .iter()
        .position(|n| n.uuid == anchor.uuid)
        .unwrap_or(0);
    Ok(creature.neurons[..anchor_at]
        .iter()
        .enumerate()
        .filter(|(_, n)| n.neuron_type == "hidden" && !n.uuid.starts_with("forest-"))
        .map(|(i, n)| (i, n.uuid.clone()))
        .collect())
}

/// Post-squash activation of every non-input neuron, in `creature.neurons`
/// order.
fn trace(network: &mut CompiledNetwork, row: &[f32], outputs: usize, neurons: usize) -> Vec<f32> {
    let t = network.activate_and_trace(row, outputs);
    t[outputs..outputs + neurons].to_vec()
}

/// Between-bin variance of the mean residual: how much a binned column
/// explains (the `error-ranked` proxy).
fn explained(edges: &[f32], values: &[f32], residual: &[f32], weight: &[f32]) -> f64 {
    let bins = edges.len() + 1;
    let (mut n, mut s) = (vec![0.0f64; bins], vec![0.0f64; bins]);
    for ((&x, &r), &w) in values.iter().zip(residual).zip(weight) {
        let b = usize::from(bin_of(edges, x));
        n[b] += f64::from(w);
        s[b] += f64::from(w) * f64::from(r);
    }
    let total: f64 = n.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mean = s.iter().sum::<f64>() / total;
    n.iter()
        .zip(&s)
        .filter(|(c, _)| **c > 0.0)
        .map(|(c, s)| c * (s / c - mean).powi(2))
        .sum()
}

/// Append up to `max` hidden-neuron columns to `set`, and their edges to
/// `bins` (a per-iteration copy of the bin cache: extended feature `input + k`
/// is column `k`).
///
/// Two streaming passes over the set's raw rows: the first traces a stride
/// sample of [`RANKING_ROWS`] rows to place quantile edges and rank the
/// eligible neurons, the second traces every row and bins the chosen ones.
///
/// # Errors
///
/// A creature that does not compile, an output nothing can be grafted onto,
/// or a corpus read failure.
#[allow(clippy::too_many_arguments)]
pub fn append_hidden_columns(
    set: &mut SearchSet,
    bins: &mut BinCache,
    creature: &CreatureExport,
    output: usize,
    training_dir: &Path,
    chunk_records: usize,
    max: usize,
    requested_bins: usize,
) -> Result<HiddenColumns, String> {
    let input = creature.input;
    let mut columns = HiddenColumns {
        input,
        ..HiddenColumns::default()
    };
    if max == 0 {
        return Ok(columns);
    }
    let eligible = eligible_sources(creature, output)?;
    columns.eligible = eligible.len();
    if eligible.is_empty() {
        return Ok(columns);
    }
    let mut network = compile_creature(creature).map_err(|e| e.to_string())?;
    let outputs = creature.output;
    let neurons = creature.neurons.len();
    let rows = set.records() as usize;
    let stride = rows.div_ceil(RANKING_ROWS).max(1);
    let (residual, weight) = crate::refine::set_residuals(set);
    // Pass 1: rank on a stride sample.
    let mut sample: Vec<Vec<f32>> = vec![Vec::new(); eligible.len()];
    let (mut sample_residual, mut sample_weight) = (Vec::new(), Vec::new());
    for_each_set_row(
        set,
        training_dir,
        input,
        outputs,
        chunk_records,
        |pos, row| {
            if !pos.is_multiple_of(stride) {
                return Ok(());
            }
            let act = trace(&mut network, row, outputs, neurons);
            for (k, &(i, _)) in eligible.iter().enumerate() {
                sample[k].push(act[i]);
            }
            sample_residual.push(residual[pos]);
            sample_weight.push(weight[pos]);
            Ok(())
        },
    )?;
    let mut ranked: Vec<(f64, usize, Vec<f32>)> = eligible
        .iter()
        .enumerate()
        .filter_map(|(k, _)| {
            let edges = quantile_edges(&mut sample[k].clone(), requested_bins);
            (edges.len() >= 2).then(|| {
                (
                    explained(&edges, &sample[k], &sample_residual, &sample_weight),
                    k,
                    edges,
                )
            })
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| eligible[a.1].1.cmp(&eligible[b.1].1))
    });
    ranked.truncate(max);
    if ranked.is_empty() {
        return Ok(columns);
    }
    // Pass 2: bin every row's activations of the chosen neurons.
    let chosen: Vec<usize> = ranked.iter().map(|r| eligible[r.1].0).collect();
    let edges: Vec<Vec<f32>> = ranked.iter().map(|r| r.2.clone()).collect();
    let mut extra: Vec<u8> = Vec::with_capacity(rows * chosen.len());
    for_each_set_row(
        set,
        training_dir,
        input,
        outputs,
        chunk_records,
        |_, row| {
            let act = trace(&mut network, row, outputs, neurons);
            extra.extend(chosen.iter().zip(&edges).map(|(&i, e)| bin_of(e, act[i])));
            Ok(())
        },
    )?;
    let old = set.features();
    let h = chosen.len();
    let mut at = 0usize;
    for chunk in &mut set.source.chunks {
        let mut widened = Vec::with_capacity(chunk.records * (old + h));
        for r in 0..chunk.records {
            widened.extend_from_slice(&chunk.bins[r * old..(r + 1) * old]);
            widened.extend_from_slice(&extra[(at + r) * h..(at + r + 1) * h]);
        }
        at += chunk.records;
        chunk.bins = widened;
        chunk.features = old + h;
    }
    let base = bins.features();
    debug_assert_eq!(base, input, "the bin cache covers exactly the inputs");
    for (k, e) in edges.into_iter().enumerate() {
        bins.edges.push(e);
        set.feature_map.push(base + k);
    }
    columns.uuids = ranked.iter().map(|r| eligible[r.1].1.clone()).collect();
    set.notes.push(format!(
        "hidden-features={}/{}",
        columns.uuids.len(),
        columns.eligible
    ));
    Ok(columns)
}

/// Every hidden source's activation on one record, by UUID — for the abstract
/// evaluator ([`Patch::evaluate_with`]).
pub fn activations(creature: &CreatureExport, row: &[f32]) -> Result<HashMap<String, f32>, String> {
    let mut network = compile_creature(creature).map_err(|e| e.to_string())?;
    let act = trace(&mut network, row, creature.output, creature.neurons.len());
    Ok(creature
        .neurons
        .iter()
        .zip(act)
        .map(|(n, a)| (n.uuid.clone(), a))
        .collect())
}

/// `true` when `node` reads any hidden source.
pub fn uses_hidden(node: &Node) -> bool {
    !node.hidden_sources().is_empty()
}

/// Test and parity fixtures (Issue #139).
pub mod fixtures {
    use neat_core::{CreatureExport, NeuronExport, SynapseExport};

    use crate::graft::graft_patch;
    use crate::patch::{Condition, Node, Patch, Provenance};

    /// `base` with a depth-2 tree whose root splits on the hidden neuron
    /// `hidden` and whose children split on an input and on the same neuron
    /// again at another threshold.
    ///
    /// # Panics
    ///
    /// When `base` cannot take the graft; this is a fixture.
    pub fn hidden_split(base: &CreatureExport, hidden: &str) -> CreatureExport {
        let root = Node::Split {
            condition: Condition::hidden(hidden, 0.1),
            left: Box::new(Node::stump(1, 0.0, -0.1, 0.05)),
            right: Box::new(Node::Split {
                condition: Condition::hidden(hidden, 0.4),
                left: Box::new(Node::leaf(0.2)),
                right: Box::new(Node::leaf(-0.15)),
            }),
        };
        graft_patch(base, &Patch::new(0, root, Provenance::default()))
            .expect("graft")
            .creature
    }

    /// `output = input-0 + 0.001 · sum`, where the hidden neuron `sum` is
    /// `input-1 + input-2` — a learned feature the output barely uses.
    pub fn sum_creature() -> CreatureExport {
        let synapse = |from: &str, to: &str, weight: f64| SynapseExport {
            from_uuid: from.into(),
            to_uuid: to.into(),
            weight,
            synapse_type: None,
        };
        CreatureExport {
            neurons: vec![
                NeuronExport {
                    id: None,
                    neuron_type: "hidden".into(),
                    uuid: "sum".into(),
                    bias: 0.0,
                    squash: Some("IDENTITY".into()),
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
                synapse("input-0", "output-0", 1.0),
                synapse("input-1", "sum", 1.0),
                synapse("input-2", "sum", 1.0),
                synapse("sum", "output-0", 0.001),
            ],
            input: 3,
            output: 1,
            semantic_version: Some("4.0.0".into()),
            forward_only: true,
            memetic: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bins::{BinBuildOptions, build_bin_cache};
    use crate::candidates::Discovery;
    use crate::config::ForestsConfig;
    use crate::corpus::{corpus_info, write_bin_file};
    use crate::graft::fixtures::{min_clamped_if_creature, small_mlp};
    use crate::graft::{GraftError, graft_patch};
    use crate::histogram::{HistogramSet, search_stumps};
    use crate::incumbent::Incumbent;
    use crate::patch::{Condition, Provenance};
    use crate::residuals::compute_residuals;
    use crate::strategies::build_search_set;
    use neat_core::training_data::TrainingDataConfig;

    fn records(n: usize, width: usize) -> Vec<Vec<f32>> {
        let mut seed = 0x3c6e_f372_fe94_f82bu64;
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

    #[test]
    fn traced_activations_line_up_with_the_creature_neurons() {
        let c = small_mlp(3);
        for x in records(20, 3) {
            let act = activations(&c, &x).unwrap();
            for h in 0..2 {
                let uuid = format!("hidden-{h}");
                let pre: f64 = c
                    .synapses
                    .iter()
                    .filter(|s| s.to_uuid == uuid)
                    .map(|s| {
                        let i: usize = s.from_uuid.strip_prefix("input-").unwrap().parse().unwrap();
                        s.weight * f64::from(x[i])
                    })
                    .sum::<f64>()
                    + c.neurons.iter().find(|n| n.uuid == uuid).unwrap().bias;
                assert!((f64::from(act[&uuid]) - pre.tanh()).abs() < 1e-5);
            }
        }
    }

    /// Issue #139 — the hidden neuron `sum = x1 + x2` and a target that steps
    /// on it. No raw-input stump can fit the step; a stump on `sum` does, and
    /// the grafted creature agrees with the abstract evaluator record by
    /// record.
    #[test]
    fn a_step_on_a_learned_feature_needs_the_hidden_column() {
        let tmp = tempfile::tempdir().unwrap();
        let step = |x: &[f32]| if x[1] + x[2] > 1.0 { 0.5 } else { 0.0 };
        let recs: Vec<(Vec<f32>, Vec<f32>)> = records(6000, 3)
            .into_iter()
            .map(|x| {
                let y = x[0] + 0.001 * (x[1] + x[2]) + step(&x);
                (x, vec![y])
            })
            .collect();
        write_bin_file(&tmp.path().join("0.bin"), &recs).unwrap();
        let tcfg = TrainingDataConfig::new(3, 1);
        let corpus = corpus_info(tmp.path(), &tcfg).unwrap();
        let bins = build_bin_cache(
            tmp.path(),
            &tcfg,
            &corpus,
            &BinBuildOptions {
                bins: 64,
                ..Default::default()
            },
        )
        .unwrap();
        let cfg = ForestsConfig {
            search_records: 0,
            min_leaf_records: 20.0,
            ..Default::default()
        };
        let creature = fixtures::sum_creature();
        let mse = |c: &CreatureExport| {
            let inc = Incumbent::from_creature(c.clone(), "t").unwrap();
            compute_residuals(&inc, tmp.path(), &corpus, 512, 1)
                .unwrap()
                .meta
                .local_mse
        };
        let before = mse(&creature);
        let inc = Incumbent::from_creature(creature.clone(), "t").unwrap();
        let res = compute_residuals(&inc, tmp.path(), &corpus, 512, 1).unwrap();
        let best_stump = |set: &SearchSet, b: &BinCache| {
            let hist = HistogramSet::from_source(&set.source, &set.bins_per_feature(b)).unwrap();
            let s = search_stumps(&hist, &set.threshold(b), &cfg.search_controls(), "cpu")
                .into_iter()
                .next()
                .unwrap();
            Discovery::from_stump(&s, set.feature_map[s.feature], "cpu")
        };

        // Raw inputs only.
        let raw_set = build_search_set(&cfg, &bins, &res, tmp.path(), 0, 1).unwrap();
        let raw = best_stump(&raw_set, &bins);
        let raw_patch = Patch::new(0, raw.root, Provenance::default());
        let after_raw = mse(&graft_patch(&creature, &raw_patch).unwrap().creature);
        // The best raw stump splits on x1 or x2 alone and leaves the
        // diagonal step's two off-diagonal triangles unexplained.
        assert!(
            after_raw > 0.3 * before,
            "a raw-input stump fit the step: {after_raw} from {before}"
        );

        // With the hidden column.
        let mut set = build_search_set(&cfg, &bins, &res, tmp.path(), 0, 1).unwrap();
        let mut extended = bins.clone();
        let columns = append_hidden_columns(
            &mut set,
            &mut extended,
            &creature,
            0,
            tmp.path(),
            512,
            4,
            64,
        )
        .unwrap();
        assert_eq!(columns.uuids, vec!["sum".to_string()]);
        assert_eq!(set.features(), 4);
        let found = best_stump(&set, &extended);
        let mut patch = Patch::new(0, found.root, Provenance::default());
        assert!(columns.resolve(&mut patch));
        assert_eq!(patch.root.hidden_sources(), vec!["sum".to_string()]);
        assert_eq!(patch.version, HIDDEN_SOURCE_PATCH_FORMAT_VERSION);
        let grafted = graft_patch(&creature, &patch).unwrap().creature;
        let after_hidden = mse(&grafted);
        assert!(
            after_hidden < 0.02 * before,
            "the hidden stump left {after_hidden} from {before}"
        );

        // Abstract evaluator, record by record.
        let mut base = compile_creature(&creature).unwrap();
        let mut cand = compile_creature(&grafted).unwrap();
        for x in records(500, 3) {
            let act = activations(&creature, &x).unwrap();
            let expected = patch.evaluate_with(&x, &|u| act[u]);
            let got = cand.activate(&x, 1)[0] - base.activate(&x, 1)[0];
            assert!((got - expected).abs() < 1e-6, "{got} vs {expected}");
        }
    }

    #[test]
    fn only_neurons_upstream_of_the_anchor_are_eligible() {
        // Behind the clamps the anchor is `body-if`; `clamp-0` is downstream.
        let c = min_clamped_if_creature(4);
        let eligible: Vec<String> = eligible_sources(&c, 0)
            .unwrap()
            .into_iter()
            .map(|(_, u)| u)
            .collect();
        assert!(!eligible.contains(&"clamp-0".to_string()));
        assert!(!eligible.contains(&"body-if".to_string()));
        let mlp: Vec<String> = eligible_sources(&small_mlp(3), 0)
            .unwrap()
            .into_iter()
            .map(|(_, u)| u)
            .collect();
        assert_eq!(mlp, ["hidden-0", "hidden-1", "hidden-2"]);
    }

    /// Issue #139 — a condition on the anchor or anything downstream of it
    /// would feed itself, and is refused; so is a neuron that does not exist.
    #[test]
    fn a_source_downstream_of_the_anchor_is_refused() {
        let c = min_clamped_if_creature(4);
        let stump = |uuid: &str| {
            Patch::new(
                0,
                Node::Split {
                    condition: Condition::hidden(uuid, 0.0),
                    left: Box::new(Node::leaf(0.0)),
                    right: Box::new(Node::leaf(0.1)),
                },
                Provenance::default(),
            )
        };
        assert_eq!(
            graft_patch(&c, &stump("clamp-0")).unwrap_err(),
            GraftError::DownstreamSource("clamp-0".into())
        );
        assert_eq!(
            graft_patch(&c, &stump("body-if")).unwrap_err(),
            GraftError::DownstreamSource("body-if".into())
        );
        assert_eq!(
            graft_patch(&c, &stump("gone")).unwrap_err(),
            GraftError::NoSuchSource("gone".into())
        );
        assert_eq!(
            graft_patch(&c, &stump("output-0")).unwrap_err(),
            GraftError::NoSuchSource("output-0".into())
        );
    }

    #[test]
    fn the_parity_fixture_matches_the_evaluator() {
        let base = small_mlp(3);
        let grafted = fixtures::hidden_split(&base, "hidden-1");
        let root = Node::Split {
            condition: Condition::hidden("hidden-1", 0.1),
            left: Box::new(Node::stump(1, 0.0, -0.1, 0.05)),
            right: Box::new(Node::Split {
                condition: Condition::hidden("hidden-1", 0.4),
                left: Box::new(Node::leaf(0.2)),
                right: Box::new(Node::leaf(-0.15)),
            }),
        };
        let mut b = compile_creature(&base).unwrap();
        let mut g = compile_creature(&grafted).unwrap();
        let (mut left, mut right) = (0, 0);
        for x in records(400, 3) {
            let x: Vec<f32> = x.iter().map(|v| 2.0 * v - 1.0).collect();
            let act = activations(&base, &x).unwrap();
            let expected = root.evaluate_with(&x, &|u| act[u]);
            if act["hidden-1"] > 0.1 {
                right += 1;
            } else {
                left += 1;
            }
            // Compare pre-squash at the LOGISTIC output.
            let tb = b.activate_and_trace(&x, 1);
            let tg = g.activate_and_trace(&x, 1);
            let nb = b.num_neurons() - b.num_inputs();
            let ng = g.num_neurons() - g.num_inputs();
            let delta = tg[1 + ng + ng - 1] - tb[1 + nb + nb - 1];
            assert!((delta - expected).abs() < 1e-5, "{delta} vs {expected}");
        }
        assert!(left > 0 && right > 0);
    }
}

//! Refit every leaf already on the creature in one solve (Issue #137).
//!
//! Forests fits each leaf once, against the residual of the moment, and never
//! looks at it again — stage-wise boosting. Every later patch moved the
//! residual those leaves were fitted to, the corpus has been regenerated
//! underneath them, and nothing re-fits them **jointly**. This is the
//! totally-corrective step of boosting (LightGBM's `refit`).
//!
//! With every split held fixed, the output's correction-space sum is linear in
//! the leaf values, and each record selects one leaf per tree:
//!
//! ```text
//! r_i ≈ Σ_l  m_i · g_l · [record i reaches leaf l] · δ_l
//! ```
//!
//! where `δ_l` is the change to leaf `l`'s value at its node, `g_l` the node's
//! gain to the output, and `m_i` is `0` on records where an output clamp binds
//! — no leaf can move them (#58) — and `1` elsewhere. The least-squares step
//! is ridge-regularised towards the current values,
//!
//! ```text
//! (AᵀWA + ρ·diag(AᵀWA)) δ = AᵀW r
//! ```
//!
//! so `δ = 0` is the incumbent and `ρ` (scale-free, relative to each leaf's
//! own curvature) and a step multiplier are the knobs. `A` is sparse — one
//! entry per record per tree — and the system is solved by conjugate gradient
//! on a stride sample of the search set held in memory, a few passes over a
//! compressed row list rather than a corpus pass per leaf.
//!
//! The trees are read back from the creature by structure
//! ([`crate::refine::ForestView`]), so leaves other tools have changed and
//! patches from other hosts are refitted at their current values. A leaf's
//! value moves by changing the weight of one of its constant edges; nothing
//! structural changes. Every candidate goes through the usual gate.

use std::path::Path;

use neat_core::{CreatureExport, compile_creature};

use crate::candidates::Candidate;
use crate::graft::{GraftError, gate_candidate, output_uuid, resolve_anchor};
use crate::patch::{BranchRole, Node, Patch, Provenance};
use crate::refine::ForestView;
use crate::strategies::{SearchSet, for_each_set_row};

/// Default records held for the refit solve.
pub const DEFAULT_REFIT_RECORDS: u64 = 20_000;

/// Controls for one refit.
#[derive(Debug, Clone)]
pub struct RefitControls {
    /// Search-set rows held for the solve (stride-sampled; 0 = all).
    pub records: u64,
    /// Step multipliers on the least-squares step.
    pub steps: Vec<f64>,
    /// Ridge strengths, relative to each leaf's own curvature.
    pub ridges: Vec<f64>,
    /// Conjugate-gradient iteration cap.
    pub max_iterations: usize,
}

/// The sparse least-squares problem for one output.
#[derive(Debug, Clone)]
pub struct RefitProblem {
    /// The trees read back from the creature.
    pub view: ForestView,
    /// Leaves in the solve (indices into `view.leaves`) and, for each, the
    /// constant edge that carries its value: `(synapse index, constant bias)`.
    pub leaves: Vec<(usize, usize, f64)>,
    /// CSR row offsets into `cols`.
    offsets: Vec<usize>,
    /// Column (index into `leaves`) per entry; every entry's value is the
    /// leaf's gain.
    cols: Vec<u32>,
    /// `g_l` per solve column.
    gain: Vec<f64>,
    /// Residual per row.
    residual: Vec<f64>,
    /// `w_i · m_i` per row.
    weight: Vec<f64>,
    /// Rows where an output clamp binds (weight forced to zero).
    pub clamped_rows: usize,
}

impl RefitProblem {
    /// Rows held.
    pub fn rows(&self) -> usize {
        self.residual.len()
    }

    /// `y = Aᵀ W A x`.
    fn normal(&self, x: &[f64], y: &mut [f64]) {
        y.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..self.rows() {
            let w = self.weight[i];
            if w == 0.0 {
                continue;
            }
            let cols = &self.cols[self.offsets[i]..self.offsets[i + 1]];
            let ax: f64 = cols
                .iter()
                .map(|&c| self.gain[c as usize] * x[c as usize])
                .sum();
            for &c in cols {
                y[c as usize] += w * self.gain[c as usize] * ax;
            }
        }
    }

    /// `AᵀW r` and `diag(AᵀWA)`.
    fn rhs_and_diagonal(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.leaves.len();
        let (mut b, mut d) = (vec![0.0; n], vec![0.0; n]);
        for i in 0..self.rows() {
            let w = self.weight[i];
            for &c in &self.cols[self.offsets[i]..self.offsets[i + 1]] {
                let g = self.gain[c as usize];
                b[c as usize] += w * g * self.residual[i];
                d[c as usize] += w * g * g;
            }
        }
        (b, d)
    }

    /// Solve `(AᵀWA + ρ·D) δ = AᵀW r` by conjugate gradient. Returns `δ` per
    /// solve column and the SSE reduction a unit step predicts.
    pub fn solve(&self, ridge: f64, max_iterations: usize) -> (Vec<f64>, f64) {
        let n = self.leaves.len();
        let (b, d) = self.rhs_and_diagonal();
        let ridge = ridge.max(1e-9);
        let apply = |x: &[f64], y: &mut [f64]| {
            self.normal(x, y);
            for k in 0..n {
                y[k] += ridge * d[k] * x[k];
            }
        };
        // Jacobi-preconditioned CG; columns no record reaches stay at zero.
        let pre: Vec<f64> = d
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    1.0 / (v * (1.0 + ridge))
                } else {
                    0.0
                }
            })
            .collect();
        let mut x = vec![0.0; n];
        let mut r = b.clone();
        let mut z: Vec<f64> = r.iter().zip(&pre).map(|(a, p)| a * p).collect();
        let mut p = z.clone();
        let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let tolerance = 1e-20 * b.iter().map(|v| v * v).sum::<f64>().max(1e-300);
        let mut ap = vec![0.0; n];
        for _ in 0..max_iterations.max(1) {
            if rz.abs() <= tolerance {
                break;
            }
            apply(&p, &mut ap);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if pap <= 0.0 || !pap.is_finite() {
                break;
            }
            let alpha = rz / pap;
            for k in 0..n {
                x[k] += alpha * p[k];
                r[k] -= alpha * ap[k];
            }
            z = r.iter().zip(&pre).map(|(a, p)| a * p).collect();
            let next: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let beta = next / rz;
            rz = next;
            for k in 0..n {
                p[k] = z[k] + beta * p[k];
            }
        }
        // Predicted SSE reduction of a unit step: 2·δᵀb − δᵀ(AᵀWA)δ.
        let mut ax = vec![0.0; n];
        self.normal(&x, &mut ax);
        let gain = 2.0 * dot(&x, &b) - dot(&x, &ax);
        (x, gain)
    }

    /// Predicted SSE reduction of `step · δ`.
    pub fn predicted_gain(&self, delta: &[f64], step: f64) -> f64 {
        let (b, _) = self.rhs_and_diagonal();
        let mut ax = vec![0.0; delta.len()];
        self.normal(delta, &mut ax);
        2.0 * step * dot(delta, &b) - step * step * dot(delta, &ax)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The constant edge that carries a leaf's value: the first synapse into that
/// branch from a constant with a non-zero bias.
fn value_edge(creature: &CreatureExport, uuid: &str, role: BranchRole) -> Option<(usize, f64)> {
    creature.synapses.iter().enumerate().find_map(|(i, s)| {
        if s.to_uuid != uuid {
            return None;
        }
        let r = match s.synapse_type.as_deref() {
            Some("positive") | None => BranchRole::Positive,
            Some("negative") => BranchRole::Negative,
            _ => return None,
        };
        if r != role {
            return None;
        }
        let from = creature.neurons.iter().find(|n| n.uuid == s.from_uuid)?;
        (from.neuron_type == "constant" && from.squash.is_none() && from.bias != 0.0)
            .then_some((i, from.bias))
    })
}

/// Build the refit problem for `output` on a stride sample of `set`.
///
/// # Errors
///
/// A creature whose output cannot take a graft, or a corpus read failure.
pub fn build_problem(
    creature: &CreatureExport,
    output: usize,
    set: &SearchSet,
    training_dir: &Path,
    chunk_records: usize,
    records: u64,
) -> Result<RefitProblem, String> {
    let view = ForestView::read(creature, output).map_err(|e| e.to_string())?;
    let mut column = vec![None; view.leaves.len()];
    let mut leaves = Vec::new();
    let mut gain = Vec::new();
    for (l, leaf) in view.leaves.iter().enumerate() {
        if let Some((synapse, bias)) = value_edge(creature, &leaf.attach.uuid, leaf.attach.role) {
            column[l] = Some(leaves.len() as u32);
            leaves.push((l, synapse, bias));
            gain.push(leaf.gain);
        }
    }
    // Records where an output clamp binds cannot be moved by any leaf (#58):
    // probe the anchor with a small bias step and see whether the output
    // follows it at the walk's gain.
    let anchor = resolve_anchor(
        creature,
        &output_uuid(creature, output).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let output_neuron = output_uuid(creature, output).map_err(|e| e.to_string())?;
    let mut probes = if anchor.uuid == output_neuron {
        None
    } else {
        let mut bumped = creature.clone();
        if let Some(n) = bumped.neurons.iter_mut().find(|n| n.uuid == anchor.uuid) {
            n.bias += PROBE;
        }
        Some((
            compile_creature(creature).map_err(|e| e.to_string())?,
            compile_creature(&bumped).map_err(|e| e.to_string())?,
        ))
    };
    let total = set.records() as usize;
    let stride = if records == 0 || records as usize >= total {
        1
    } else {
        total.div_ceil(records as usize)
    };
    let (set_residual, set_weight) = crate::refine::set_residuals(set);
    let mut problem = RefitProblem {
        view,
        leaves,
        offsets: vec![0],
        cols: Vec::new(),
        gain,
        residual: Vec::new(),
        weight: Vec::new(),
        clamped_rows: 0,
    };
    let mut reached = Vec::new();
    let outputs = creature.output;
    for_each_set_row(
        set,
        training_dir,
        creature.input,
        outputs,
        chunk_records,
        |pos, row| {
            if !pos.is_multiple_of(stride) {
                return Ok(());
            }
            let mut w = f64::from(set_weight[pos]);
            if let Some((base, bumped)) = probes.as_mut() {
                let a = base.activate(row, outputs)[output];
                let b = bumped.activate(row, outputs)[output];
                let slope = f64::from(b - a) / PROBE;
                if (slope - anchor.gain).abs() > 0.05 * anchor.gain.abs() {
                    w = 0.0;
                    problem.clamped_rows += 1;
                }
            }
            problem.view.route(row, &mut reached);
            problem
                .cols
                .extend(reached.iter().filter_map(|&l| column[l]));
            problem.offsets.push(problem.cols.len());
            problem.residual.push(f64::from(set_residual[pos]));
            problem.weight.push(w);
            Ok(())
        },
    )?;
    Ok(problem)
}

/// Bias step used to find records where an output clamp binds.
const PROBE: f64 = 1e-3;

/// `creature` with every solve column's leaf moved by `step · δ`.
///
/// # Errors
///
/// A refitted creature that fails the validation gate every candidate clears.
pub fn apply(
    creature: &CreatureExport,
    problem: &RefitProblem,
    delta: &[f64],
    step: f64,
) -> Result<CreatureExport, GraftError> {
    let mut out = creature.clone();
    for (k, &(_, synapse, bias)) in problem.leaves.iter().enumerate() {
        let d = step * delta[k];
        if d != 0.0 && d.is_finite() {
            out.synapses[synapse].weight += d / bias;
        }
    }
    gate_candidate(&out)?;
    Ok(out)
}

/// Refit candidates for `output`: one per `(ridge, step)`, each a clone of
/// `creature` with every leaf's value moved by the scaled solve.
///
/// A refit is not a patch: each candidate carries a placeholder patch whose
/// root is a bare leaf (which no graft ever produces) and whose provenance
/// says `refit` and how it was built. It is journalled and judged like every
/// other candidate, but never carried forward, filed in the learnings cache or
/// replayed — it is cheap enough to re-run on whatever creature comes next.
///
/// # Errors
///
/// A corpus read failure. Candidates the gate refuses are returned as discards.
#[allow(clippy::too_many_arguments)]
pub fn refit_candidates(
    creature: &CreatureExport,
    checksum: &str,
    output: usize,
    set: &SearchSet,
    training_dir: &Path,
    chunk_records: usize,
    controls: &RefitControls,
    corpus_identity: &str,
) -> Result<RefitBatch, String> {
    let problem = build_problem(
        creature,
        output,
        set,
        training_dir,
        chunk_records,
        controls.records,
    )?;
    let mut summary = RefitSummary {
        leaves: problem.leaves.len(),
        rows: problem.rows(),
        clamped_rows: problem.clamped_rows,
    };
    let (mut out, mut discarded) = (Vec::new(), Vec::new());
    if problem.leaves.is_empty() {
        return Ok(RefitBatch {
            candidates: out,
            discarded,
            summary,
        });
    }
    for &ridge in &controls.ridges {
        let (delta, _) = problem.solve(ridge, controls.max_iterations);
        let moved = delta.iter().filter(|d| **d != 0.0).count();
        for &step in &controls.steps {
            let label = format!("refit ridge={ridge} step={step}");
            let id = crate::incumbent::sha256_hex(
                format!("{checksum}|{output}|{corpus_identity}|{ridge}|{step}|{delta:?}")
                    .as_bytes(),
            )[..16]
                .to_string();
            match apply(creature, &problem, &delta, step) {
                Ok(c) => out.push(Candidate {
                    id,
                    patch: Patch::new(
                        output,
                        Node::leaf(0.0),
                        Provenance {
                            strategy: "refit".into(),
                            backend: "cpu-cg".into(),
                            predicted_gain: problem.predicted_gain(&delta, step),
                            affected_records: problem.rows() as u64,
                            search_records: problem.rows() as u64,
                            incumbent_checksum: checksum.to_string(),
                            seed: None,
                            notes: vec![
                                format!("ridge={ridge}"),
                                format!("step={step}"),
                                format!("leaves={} moved={moved}", problem.leaves.len()),
                                format!("clamped-rows={}", problem.clamped_rows),
                                format!("corpus={corpus_identity}"),
                            ],
                        },
                    ),
                    combo: Vec::new(),
                    creature: c,
                    added_uuids: vec![Vec::new()],
                }),
                Err(e) => discarded.push((format!("{id} ({label})"), e.to_string())),
            }
        }
    }
    summary.leaves = problem.leaves.len();
    Ok(RefitBatch {
        candidates: out,
        discarded,
        summary,
    })
}

/// What [`refit_candidates`] returns.
#[derive(Debug, Clone)]
pub struct RefitBatch {
    /// One candidate per `(ridge, step)` that cleared the gate.
    pub candidates: Vec<Candidate>,
    /// `(id, reason)` for each one that did not.
    pub discarded: Vec<(String, String)>,
    /// What the solve looked at.
    pub summary: RefitSummary,
}

/// What a refit looked at.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RefitSummary {
    /// Leaves in the solve.
    pub leaves: usize,
    /// Rows held.
    pub rows: usize,
    /// Rows given no weight because an output clamp binds there.
    pub clamped_rows: usize,
}

/// `true` for a candidate [`refit_candidates`] built.
pub fn is_refit(candidate: &Candidate) -> bool {
    is_refit_patch(&candidate.patch)
}

/// `true` for the placeholder patch a refit candidate carries.
pub fn is_refit_patch(patch: &Patch) -> bool {
    matches!(patch.root, Node::Leaf { .. }) && patch.provenance.strategy == "refit"
}

/// Creature tag naming the corpus the incumbent was last refitted on.
pub const REFIT_CORPUS_TAG: &str = "forests-refit-corpus";

/// Test and parity fixtures (Issue #137).
pub mod fixtures {
    use neat_core::CreatureExport;

    use crate::graft::graft_patches;
    use crate::patch::{Condition, Node, Patch, Provenance};

    /// The two trees every refit fixture plants, at full strength.
    pub fn trees() -> [Node; 2] {
        [
            Node::Split {
                condition: Condition::axis(0, 0.5),
                left: Box::new(Node::stump(1, 0.3, -0.2, 0.1)),
                right: Box::new(Node::stump(2, 0.6, 0.3, -0.1)),
            },
            Node::stump(2, 0.25, 0.15, -0.05),
        ]
    }

    /// `base` with [`trees`] planted at `scale` — the "every leaf shrunk to
    /// half its optimum" creature when `scale` is `0.5`.
    ///
    /// # Panics
    ///
    /// When `base` cannot take the graft; this is a fixture.
    pub fn planted(base: &CreatureExport, scale: f32) -> CreatureExport {
        let patches: Vec<Patch> = trees()
            .iter()
            .map(|t| Patch::new(0, t.scaled(scale), Provenance::default()))
            .collect();
        graft_patches(base, &patches).expect("plant").0
    }

    /// `planted(base, 0.5)` with every leaf value moved by hand, through the
    /// same edge a refit moves — a refitted creature for the parity suites.
    ///
    /// # Panics
    ///
    /// When `base` cannot take the graft; this is a fixture.
    pub fn refitted(base: &CreatureExport) -> CreatureExport {
        let mut c = planted(base, 0.5);
        let view = crate::refine::ForestView::read(&c, 0).expect("read");
        for (k, leaf) in view.leaves.iter().enumerate() {
            let (synapse, bias) =
                super::value_edge(&c, &leaf.attach.uuid, leaf.attach.role).expect("edge");
            c.synapses[synapse].weight += (0.01 * (k as f64 + 1.0)) / bias;
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bins::{BinBuildOptions, build_bin_cache};
    use crate::config::ForestsConfig;
    use crate::corpus::{corpus_info, write_bin_file};
    use crate::graft::fixtures::{identity_creature, min_clamped_if_creature};
    use crate::incumbent::Incumbent;
    use crate::residuals::compute_residuals;
    use crate::strategies::build_search_set;
    use neat_core::training_data::TrainingDataConfig;

    fn records(n: usize, width: usize) -> Vec<Vec<f32>> {
        let mut seed = 0x51_7cc1_b727_220au64;
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

    struct World {
        tmp: tempfile::TempDir,
        corpus: crate::corpus::CorpusInfo,
        bins: crate::bins::BinCache,
    }

    /// A corpus whose target is the incumbent's own `x0` plus both fixture
    /// trees at full strength — so the optimum of every planted leaf is
    /// exactly twice its value in `planted(…, 0.5)`.
    fn world(target: impl Fn(&[f32]) -> f32) -> World {
        let tmp = tempfile::tempdir().unwrap();
        let recs: Vec<(Vec<f32>, Vec<f32>)> = records(3000, 4)
            .into_iter()
            .map(|x| {
                let y = target(&x);
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
        World { tmp, corpus, bins }
    }

    impl World {
        fn residuals(&self, c: &CreatureExport) -> crate::residuals::ResidualCache {
            let inc = Incumbent::from_creature(c.clone(), "t").unwrap();
            compute_residuals(&inc, self.tmp.path(), &self.corpus, 512, 1).unwrap()
        }

        fn set(&self, c: &CreatureExport) -> SearchSet {
            let cfg = ForestsConfig {
                search_records: 0,
                ..Default::default()
            };
            build_search_set(&cfg, &self.bins, &self.residuals(c), self.tmp.path(), 0, 1).unwrap()
        }
    }

    fn full_strength(x: &[f32]) -> f32 {
        let [a, b] = fixtures::trees();
        x[0] + a.evaluate(x) + b.evaluate(x)
    }

    /// Issue #137 — every leaf planted at half its optimum: one refit (unit
    /// step, negligible ridge) recovers them, the residual vanishes, and the
    /// abstract evaluator — `x0 + Σ` the refitted leaves the record reaches —
    /// agrees with the refitted creature record by record.
    #[test]
    fn one_refit_recovers_leaves_shrunk_to_half_their_optimum() {
        let w = world(full_strength);
        let base = identity_creature(4, 1);
        let shrunk = fixtures::planted(&base, 0.5);
        let before = w.residuals(&shrunk).meta.local_mse;
        assert!(
            before > 1e-3,
            "the fixture must start off the optimum: {before}"
        );
        let set = w.set(&shrunk);
        let problem = build_problem(&shrunk, 0, &set, w.tmp.path(), 512, 0).unwrap();
        assert_eq!(problem.leaves.len(), 6);
        assert_eq!(problem.clamped_rows, 0);
        let (delta, gain) = problem.solve(1e-9, 200);
        assert!(gain > 0.0);
        let refitted = apply(&shrunk, &problem, &delta, 1.0).unwrap();
        let after = w.residuals(&refitted).meta.local_mse;
        assert!(after < 1e-9, "refit left local MSE {after} (from {before})");

        // Record-by-record agreement with the abstract evaluator.
        let view = ForestView::read(&refitted, 0).unwrap();
        let mut net = compile_creature(&refitted).unwrap();
        let mut reached = Vec::new();
        for x in records(500, 4) {
            view.route(&x, &mut reached);
            let expected: f64 = f64::from(x[0])
                + reached
                    .iter()
                    .map(|&l| view.leaves[l].value * view.leaves[l].gain)
                    .sum::<f64>();
            let got = net.activate(&x, 1)[0];
            assert!(
                (f64::from(got) - expected).abs() < 1e-5,
                "{got} vs {expected}"
            );
            assert!(
                (got - full_strength(&x)).abs() < 1e-4,
                "refit missed the optimum"
            );
        }
    }

    /// The zero step is the incumbent, and a larger ridge shrinks the step
    /// towards it.
    #[test]
    fn ridge_pulls_the_step_towards_the_incumbent() {
        let w = world(full_strength);
        let shrunk = fixtures::planted(&identity_creature(4, 1), 0.5);
        let set = w.set(&shrunk);
        let problem = build_problem(&shrunk, 0, &set, w.tmp.path(), 512, 0).unwrap();
        let norm = |d: &[f64]| d.iter().map(|v| v * v).sum::<f64>().sqrt();
        let (weak, _) = problem.solve(0.01, 200);
        let (strong, _) = problem.solve(10.0, 200);
        assert!(norm(&strong) < norm(&weak));
        let unchanged = apply(&shrunk, &problem, &weak, 0.0).unwrap();
        assert_eq!(unchanged.synapses, shrunk.synapses);
    }

    /// Records where an output clamp binds get no weight: no leaf can move
    /// them (#58).
    #[test]
    fn records_behind_a_binding_clamp_get_no_weight() {
        // `min_clamped_if_creature` is min(x3, min(0.98·x3, 0.5·body)); with
        // inputs in [0, 1) the clamp on x3 binds wherever 0.5·body > 0.98·x3.
        let w = world(|x| x[3]);
        let clamped = fixtures::planted(&min_clamped_if_creature(4), 0.5);
        let set = w.set(&clamped);
        let problem = build_problem(&clamped, 0, &set, w.tmp.path(), 512, 0).unwrap();
        assert!(problem.clamped_rows > 0, "the clamp must bind somewhere");
        assert!(problem.clamped_rows < problem.rows(), "and not everywhere");
        let (delta, _) = problem.solve(1.0, 200);
        assert!(delta.iter().all(|d| d.is_finite()));
        apply(&clamped, &problem, &delta, 0.5).unwrap();
    }

    #[test]
    fn candidates_cover_every_step_and_ridge_and_say_they_are_refits() {
        let w = world(full_strength);
        let shrunk = fixtures::planted(&identity_creature(4, 1), 0.5);
        let set = w.set(&shrunk);
        let RefitBatch {
            candidates: cands,
            discarded,
            summary,
        } = refit_candidates(
            &shrunk,
            "abc",
            0,
            &set,
            w.tmp.path(),
            512,
            &RefitControls {
                records: 1000,
                steps: vec![0.5, 1.0],
                ridges: vec![1.0, 0.1],
                max_iterations: 100,
            },
            "corpus",
        )
        .unwrap();
        assert!(discarded.is_empty(), "{discarded:?}");
        assert_eq!(cands.len(), 4);
        assert_eq!(summary.leaves, 6);
        assert!(summary.rows <= 1000);
        let ids: std::collections::HashSet<_> = cands.iter().map(|c| c.id.clone()).collect();
        assert_eq!(ids.len(), 4, "every candidate has its own id");
        assert!(cands.iter().all(is_refit));
        assert!(cands.iter().all(|c| c.creature.neurons == shrunk.neurons));
        assert!(
            cands
                .iter()
                .all(|c| c.patch.provenance.predicted_gain > 0.0)
        );
    }
}

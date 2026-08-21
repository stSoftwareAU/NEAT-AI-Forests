//! The iterative Forest evolution loop (Issue #10).
//!
//! ```text
//! load + copy immutable incumbent → bin cache → authoritative baseline
//!   ┌─▶ residuals → search set → histogram (GPU/CPU) → stumps / trees / oblique
//!   │     → candidate population → screen → full-corpus promotion
//!   │     → accepted?  yes: promote clone to experimental incumbent, recompute
//!   └─────────────────  residuals, journal;  no: journal and continue
//! until timeout / max iterations / cancellation / repeated scorer failure
//! ```
//!
//! `best.json` starts as a byte-for-byte copy of the source creature and is
//! only ever replaced by a creature the scorer verified on the full corpus in
//! the same call as its parent.

use std::path::{Path, PathBuf};
use std::time::Instant;

use neat_core::training_data::TrainingDataConfig;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use crate::baseline::{AuthoritativeBaseline, establish_baseline};
use crate::bins::{BinBuildOptions, BinCache, ensure_bin_cache};
use crate::cancel::CancelToken;
use crate::candidates::{
    Candidate, CandidateConfig, Discovery, expand_discoveries, generate_candidates, random_stumps,
};
use crate::config::ForestsConfig;
use crate::corpus::{CorpusInfo, corpus_info};
use crate::histogram::search_stumps;
use crate::incumbent::{Incumbent, load_incumbent, now_unix};
use crate::journal::{
    CandidateRecord, ExperimentRecord, FullSummary, JournalLine, ScreenSummary, SummaryRecord,
    append_journal_line,
};
use crate::log;
use crate::meta::{CreatureMeta, Tag};
use crate::oblique::{ObliqueControls, search_oblique};
use crate::patch::Patch;
use crate::promote::{PromoteConfig, screen_and_promote};
use crate::residuals::{ResidualCache, ensure_residual_cache};
use crate::scorer::DirectoryScorer;
use crate::strategies::{build_search_set, raw_sample};
use crate::tree::{TreeSearchControls, grow_tree};

/// Why the loop stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Wall-clock budget exhausted.
    Timeout,
    /// `--max-iterations` reached.
    MaxIterations,
    /// SIGINT/SIGTERM.
    Cancelled,
    /// Too many consecutive scorer failures.
    ScorerFailures,
}

impl StopReason {
    /// Journal label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::MaxIterations => "max-iterations",
            Self::Cancelled => "cancelled",
            Self::ScorerFailures => "scorer-failures",
        }
    }
}

/// Outcome of a run.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// `best.json`.
    pub best_path: PathBuf,
    /// `experiments.jsonl`.
    pub journal_path: PathBuf,
    /// Opening authoritative score.
    pub opening_score: f64,
    /// Final authoritative score.
    pub best_score: f64,
    /// Iterations run.
    pub iterations: u64,
    /// Accepted winners.
    pub acceptances: u64,
    /// Stop reason.
    pub stop_reason: StopReason,
    /// Seed used.
    pub seed: u64,
    /// Wall ms.
    pub wall_ms: u64,
    /// Final incumbent checksum.
    pub final_checksum: String,
    /// Iterations that failed on the scorer.
    pub scorer_failures: u64,
}

struct State {
    incumbent: Incumbent,
    baseline: AuthoritativeBaseline,
    residuals: ResidualCache,
    meta: CreatureMeta,
    /// Strategy of the last accepted candidate.
    last_strategy: String,
    /// Output neuron uuid of the last accepted candidate.
    last_target: String,
    /// Full-scored non-winners with a positive authoritative Δ from the last
    /// iteration, carried forward as combination material.
    runner_ups: Vec<Patch>,
}

/// Lamarck-style run summary used as the GRQ commit subject.
fn forests_tag(
    acceptances: u64,
    iterations: u64,
    last_strategy: &str,
    last_target: &str,
    opening: f64,
    score: f64,
) -> String {
    let mut s = format!(
        "🌳 Forests · {acceptances} accepts / {iterations} iters · last: {} · 🎯 {last_target} · score: {score:.6}",
        if last_strategy.is_empty() {
            "none"
        } else {
            last_strategy
        }
    );
    if score > opening {
        s.push_str(&format!(" improved by {:.2e}", score - opening));
    }
    s
}

fn write_best(
    path: &Path,
    state: &State,
    opening: f64,
    acceptances: u64,
    iterations: u64,
) -> Result<(), String> {
    let mut meta = state.meta.clone();
    meta.upsert("score", format!("{}", state.baseline.score));
    meta.upsert("error", format!("{}", state.baseline.error));
    meta.upsert(
        "forests",
        forests_tag(
            acceptances,
            iterations,
            &state.last_strategy,
            &state.last_target,
            opening,
            state.baseline.score,
        ),
    );
    let text = meta.serialize_with(&state.incumbent.creature, true)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Run the complete optimiser.
pub fn run_forests(
    cfg: &ForestsConfig,
    scorer: &dyn DirectoryScorer,
    cancel: &CancelToken,
) -> Result<RunResult, String> {
    cfg.validate()?;
    let started = Instant::now();
    // Load before anything is written: a bad creature produces no output dir.
    let incumbent = load_incumbent(&cfg.creature).map_err(|e| e.to_string())?;
    let creature_cfg = TrainingDataConfig::new(incumbent.creature.input, incumbent.creature.output);
    let corpus: CorpusInfo = corpus_info(&cfg.training_data, &creature_cfg)?;
    log::info(&format!(
        "incumbent {} ({} in / {} out, {} neurons, {} synapses); corpus {} records in {} files",
        incumbent.short_checksum(),
        incumbent.creature.input,
        incumbent.creature.output,
        incumbent.creature.neurons.len(),
        incumbent.creature.synapses.len(),
        corpus.record_count,
        corpus.file_count
    ));

    std::fs::create_dir_all(&cfg.output_dir)
        .map_err(|e| format!("{}: {e}", cfg.output_dir.display()))?;
    let workspace = cfg.output_dir.join("workspace");
    incumbent
        .write_workspace(&workspace)
        .map_err(|e| e.to_string())?;
    let best_path = cfg.output_dir.join("best.json");
    let journal_path = cfg.output_dir.join("experiments.jsonl");
    let winners_dir = cfg.output_dir.join("winners");
    std::fs::write(&best_path, &incumbent.text)
        .map_err(|e| format!("{}: {e}", best_path.display()))?;

    let (seed, seed_source) = match cfg.seed {
        Some(s) => (s, "supplied"),
        None => (rand::rng().next_u64(), "drawn"),
    };
    log::info(&format!(
        "seed {seed} ({seed_source}); replay with --seed {seed}"
    ));
    let mut rng = StdRng::seed_from_u64(seed);

    let cache_dir = cfg.cache_dir();
    let bin_opts = BinBuildOptions {
        bins: cfg.bins,
        sample_records: cfg.bin_sample_records,
        memory_budget_bytes: cfg.bin_memory_budget_bytes,
        chunk_records: cfg.chunk_records,
    };
    let bins: BinCache = ensure_bin_cache(
        &cfg.training_data,
        &cache_dir,
        &creature_cfg,
        &corpus,
        &bin_opts,
    )
    .map_err(|e| e.to_string())?;

    append_journal_line(
        &journal_path,
        &JournalLine::RunHeader {
            timestamp_unix: now_unix(),
            seed,
            seed_source: seed_source.into(),
            version: env!("CARGO_PKG_VERSION").into(),
            source_creature: cfg.creature.display().to_string(),
            incumbent_checksum: incumbent.checksum.clone(),
            corpus_identity: corpus.identity.clone(),
            bin_cache: format!("{}:{}", bins.meta.corpus_identity, bins.meta.requested_bins),
            config: serde_json::to_value(cfg).map_err(|e| e.to_string())?,
        },
    )?;

    let residuals = ensure_residual_cache(
        &incumbent,
        &cfg.training_data,
        &cache_dir,
        &corpus,
        cfg.chunk_records,
        cfg.analysis_threads,
    )
    .map_err(|e| e.to_string())?;
    log::info(&format!(
        "local MSE {:.6e}; establishing authoritative baseline…",
        residuals.meta.local_mse
    ));
    let baseline = establish_baseline(
        &incumbent,
        &cfg.training_data,
        &corpus,
        scorer,
        &workspace,
        cfg.parity_policy(),
        Some(residuals.meta.local_mse),
    )?;
    log::ok(&format!(
        "baseline score {:.9} error {:.6e} ({}) — parity {}",
        baseline.score,
        baseline.error,
        baseline.scorer_backend.clone().unwrap_or_default(),
        baseline.parity
    ));
    append_journal_line(&journal_path, &JournalLine::Baseline(baseline.clone()))?;
    let opening_score = baseline.score;
    let meta = CreatureMeta::from_json(&incumbent.text);
    let mut state = State {
        incumbent,
        baseline,
        residuals,
        meta,
        last_strategy: String::new(),
        last_target: String::new(),
        runner_ups: Vec::new(),
    };

    let mut iterations = 0u64;
    let mut acceptances = 0u64;
    let mut scorer_failures = 0u64;
    let mut consecutive_failures = 0u32;
    let stop_reason;
    loop {
        if cancel.is_cancelled() {
            stop_reason = StopReason::Cancelled;
            break;
        }
        if started.elapsed() >= cfg.timeout {
            stop_reason = StopReason::Timeout;
            break;
        }
        if cfg.max_iterations.is_some_and(|m| iterations >= m) {
            stop_reason = StopReason::MaxIterations;
            break;
        }
        if consecutive_failures >= cfg.max_consecutive_scorer_failures {
            stop_reason = StopReason::ScorerFailures;
            break;
        }
        iterations += 1;
        let iteration_seed = seed ^ iterations.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let output = ((iterations - 1) % state.incumbent.creature.output as u64) as usize;
        log::info(&format!(
            "iteration {iterations}: incumbent {} output {output} (score {:.9})",
            state.incumbent.short_checksum(),
            state.baseline.score
        ));

        // ---- search -------------------------------------------------------
        let search_started = Instant::now();
        let set = build_search_set(
            cfg,
            &bins,
            &state.residuals,
            &cfg.training_data,
            output,
            iteration_seed,
        )?;
        let bins_per_feature = set.bins_per_feature(&bins);
        let (hist, backend) = crate::gpu::accumulate(
            cfg.gpu,
            &set.source,
            &bins_per_feature,
            cfg.analysis_threads,
        )?;
        let backend_label = backend.label();
        let controls = cfg.search_controls();
        let threshold = set.threshold(&bins);
        let stumps = search_stumps(&hist, &threshold, &controls, &backend_label);
        let mut discoveries: Vec<Discovery> = stumps
            .iter()
            .map(|s| Discovery::from_stump(s, set.feature_map[s.feature], &backend_label))
            .collect();
        let mut strategies: Vec<String> = vec!["histogram-stump".into()];
        if cfg.max_depth > 1 {
            let tree_controls = TreeSearchControls {
                stump: controls.clone(),
                max_depth: cfg.max_depth,
                growth: cfg.growth,
            };
            let mut roots: Vec<Option<(usize, usize)>> = vec![None];
            let mut seen = Vec::new();
            for s in &stumps {
                if seen.len() >= 3 {
                    break;
                }
                if !seen.contains(&s.feature) {
                    seen.push(s.feature);
                    roots.push(Some((s.feature, s.bin)));
                }
            }
            for root in roots {
                for t in grow_tree(
                    &set.source,
                    &bins_per_feature,
                    &threshold,
                    &|f| set.feature_map[f],
                    &tree_controls,
                    root,
                )? {
                    if t.depth > 1 {
                        discoveries.push(Discovery::from_tree(&t, &backend_label));
                    }
                }
            }
            strategies.push(format!("histogram-tree-depth{}", cfg.max_depth));
        }
        if cfg.oblique_candidates > 0 {
            let mut feats: Vec<usize> = Vec::new();
            for s in &stumps {
                let f = set.feature_map[s.feature];
                if !feats.contains(&f) {
                    feats.push(f);
                }
                if feats.len() >= 6 {
                    break;
                }
            }
            if feats.len() >= 2 {
                let raw = raw_sample(
                    &set,
                    &feats,
                    &cfg.training_data,
                    &bins,
                    state.incumbent.creature.output,
                    cfg.chunk_records,
                )?;
                let scales: Vec<f32> = feats.iter().map(|&f| bins.scale(f)).collect();
                let ob = ObliqueControls {
                    stump: controls.clone(),
                    count: cfg.oblique_candidates,
                    random_combos: cfg.oblique_candidates * 4,
                    jitter_rounds: 2,
                    max_terms: 3,
                };
                for o in search_oblique(&raw, &scales, &ob, Some(0), &mut rng) {
                    discoveries.push(Discovery::from_oblique(&o));
                }
                strategies.push("oblique-split".into());
            }
        }
        let search_ms = search_started.elapsed().as_millis() as u64;

        // ---- candidates ---------------------------------------------------
        let graft_started = Instant::now();
        let sigma = state.residuals.meta.stats[output].correction_mse.sqrt() as f32;
        let cand_cfg = CandidateConfig {
            magnitude_scales: cfg.magnitude_scales.clone(),
            random_candidates: cfg.random_candidates,
            max_candidates: cfg.candidates,
            backend: backend_label.clone(),
            search_records: set.records(),
            max_correction: cfg.max_correction as f32,
            random_scale: if sigma.is_finite() && sigma > 0.0 {
                sigma
            } else {
                0.01
            },
            threshold_jitter: cfg.threshold_jitter,
            notes: set.notes.clone(),
        };
        let mut patches = expand_discoveries(
            &discoveries,
            &bins,
            output,
            &state.incumbent.checksum,
            iteration_seed,
            &cand_cfg,
        );
        if cfg.random_candidates > 0 {
            patches.extend(random_stumps(
                &bins,
                output,
                &state.incumbent.checksum,
                iteration_seed,
                &mut rng,
                &cand_cfg,
            ));
            strategies.push("random-stump".into());
        }
        // Combinations: stack the top-k distinct discoveries, and carry the
        // previous iteration's near-winners forward onto the new incumbent
        // (alone, together, and together with this iteration's best).
        let mut combo_groups: Vec<Vec<Patch>> = Vec::new();
        // Boosting rounds (#40): subtract the best patch from the sample
        // residuals, search again, and verify the bundle's prefixes together.
        if cfg.boost_rounds > 1
            && let Some(first) = patches.first().cloned()
        {
            let boost_started = Instant::now();
            let mut boosted = set.clone();
            let mut bundle = vec![first];
            for round in 2..=cfg.boost_rounds {
                let Some(last) = bundle.last() else { break };
                if crate::strategies::apply_patch_residuals(&mut boosted, &bins, &last.root)
                    .is_err()
                {
                    break;
                }
                let (h, _) = crate::gpu::accumulate(
                    cfg.gpu,
                    &boosted.source,
                    &bins_per_feature,
                    cfg.analysis_threads,
                )?;
                let Some(best) = search_stumps(&h, &threshold, &controls, &backend_label)
                    .into_iter()
                    .next()
                else {
                    break;
                };
                let d =
                    Discovery::from_stump(&best, boosted.feature_map[best.feature], &backend_label);
                let mut p = Patch::new(
                    output,
                    d.root.clone(),
                    crate::patch::Provenance {
                        strategy: "boost-round".into(),
                        backend: d.backend.clone(),
                        predicted_gain: d.gain,
                        affected_records: d.affected as u64,
                        search_records: boosted.records(),
                        incumbent_checksum: state.incumbent.checksum.clone(),
                        seed: Some(iteration_seed),
                        notes: vec![format!("boost-round={round}")],
                    },
                );
                p.provenance.notes.extend(boosted.notes.iter().cloned());
                bundle.push(p);
            }
            for k in 2..=bundle.len() {
                combo_groups.push(bundle[..k].to_vec());
            }
            if bundle.len() > 1 {
                strategies.push(format!("boost-rounds/{}", bundle.len()));
                log::detail(&format!(
                    "boosting: {} rounds in {} ms",
                    bundle.len(),
                    boost_started.elapsed().as_millis()
                ));
            }
        }
        if cfg.combo_candidates > 0 {
            let ranked: Vec<Patch> = patches.iter().take(discoveries.len()).cloned().collect();
            combo_groups.extend(crate::candidates::top_k_groups(
                &ranked,
                cfg.combo_candidates.max(2),
            ));
            if !state.runner_ups.is_empty() {
                let carried: Vec<Patch> = state
                    .runner_ups
                    .iter()
                    .map(|p| {
                        let mut p = p.clone();
                        p.provenance.strategy = format!("carry-forward:{}", p.provenance.strategy);
                        p.provenance
                            .notes
                            .push(format!("carried from iteration {}", iterations - 1));
                        p
                    })
                    .collect();
                // Each runner-up alone on the new incumbent.
                patches.extend(carried.iter().cloned());
                if carried.len() >= 2 {
                    combo_groups.push(carried.clone());
                }
                if let Some(best) = ranked.first() {
                    let mut g = vec![best.clone()];
                    g.extend(carried.iter().cloned());
                    combo_groups.push(g);
                }
                strategies.push("carry-forward".into());
            }
            strategies.push("combo".into());
        }
        let generated = (patches.len() + combo_groups.len()) as u64;
        let (mut candidates, mut discarded): (Vec<Candidate>, Vec<(String, String)>) =
            crate::candidates::generate_combos(&state.incumbent, combo_groups, "combination");
        let combos_kept = candidates.len();
        let room = cfg.candidates.saturating_sub(candidates.len());
        let single_cfg = CandidateConfig {
            max_candidates: room,
            ..cand_cfg.clone()
        };
        let (singles, single_discarded) =
            generate_candidates(&state.incumbent, patches, &single_cfg);
        // Singles first (rank order), then combinations.
        let combos: Vec<Candidate> = std::mem::take(&mut candidates);
        candidates = singles;
        candidates.extend(combos);
        discarded.extend(single_discarded);
        for (id, why) in &discarded {
            log::detail(&format!("discarded {id}: {why}"));
        }
        log::detail(&format!("{combos_kept} combination candidate(s)"));
        let graft_ms = graft_started.elapsed().as_millis() as u64;
        log::detail(&format!(
            "search {search_ms} ms on {} records × {} features via {backend_label}: {} stumps, {} discoveries, {} candidates ({} discarded)",
            set.records(),
            set.features(),
            stumps.len(),
            discoveries.len(),
            candidates.len(),
            discarded.len()
        ));

        // ---- score --------------------------------------------------------
        let mut record = ExperimentRecord {
            iteration: iterations,
            timestamp_unix: now_unix(),
            incumbent_checksum: state.incumbent.checksum.clone(),
            baseline_score: state.baseline.score,
            search_backend: backend_label.clone(),
            search_set: set.label(),
            search_records: set.records(),
            search_features: set.features() as u64,
            strategies,
            search_ms,
            graft_ms,
            candidates_generated: generated,
            candidates_discarded: discarded.len() as u64,
            discarded: discarded
                .iter()
                .map(|(id, why)| crate::journal::DiscardRecord {
                    id: id.clone(),
                    reason: why.clone(),
                })
                .collect(),
            candidates: candidates
                .iter()
                .map(|c| CandidateRecord {
                    id: c.id.clone(),
                    strategy: c.strategy(),
                    backend: c.patch.provenance.backend.clone(),
                    depth: c.depth(),
                    features: c.features(),
                    predicted_gain: c.predicted_gain(),
                    affected_records: c.affected_records(),
                    affected_fraction: if set.records() > 0 {
                        c.affected_records() as f64 / set.records() as f64
                    } else {
                        0.0
                    },
                    screen_score: None,
                    screen_delta: None,
                    promoted: false,
                    bypass: false,
                    full_score: None,
                    full_delta: None,
                    patch: c.patch.clone(),
                    combo: c.combo.clone(),
                })
                .collect(),
            screen: None,
            full: None,
            winner: None,
            improvement: None,
            accepted: false,
            new_incumbent_checksum: None,
            screen_false_positives: 0,
            screen_false_negatives: 0,
            scorer_error: None,
        };
        if candidates.is_empty() {
            log::warn("no valid candidates this iteration");
            append_journal_line(&journal_path, &JournalLine::Experiment(Box::new(record)))?;
            continue;
        }
        if cancel.is_cancelled() {
            // Abandon before any scorer work; the iteration is not counted.
            iterations -= 1;
            stop_reason = StopReason::Cancelled;
            break;
        }
        // Frugal screen (#40): the screen exists to choose `promote_count`
        // candidates; when the cohort already fits, go straight to the full call.
        let screen_rate = if candidates.len() <= cfg.promote_count {
            None
        } else {
            cfg.screen_sample_rate
        };
        let promote_cfg = PromoteConfig {
            screen_sample_rate: screen_rate,
            screen_phase: iterations,
            screen_threshold: cfg.screen_threshold,
            promote_count: cfg.promote_count,
            explore_quota: cfg.explore_quota,
            min_improvement: cfg.min_improvement,
            authoritative_baseline: state.baseline.score,
            baseline_drift_epsilon: cfg.baseline_drift_epsilon,
            preserve_dirs: cfg.preserve_candidates,
        };
        let work_dir = workspace.join(format!("iteration-{iterations:04}"));
        let outcome = match screen_and_promote(
            &candidates,
            &state.incumbent.creature,
            &cfg.training_data,
            scorer,
            &work_dir,
            &promote_cfg,
            &mut rng,
        ) {
            Ok(o) => o,
            Err(e) => {
                scorer_failures += 1;
                consecutive_failures += 1;
                log::warn(&format!("scorer failure: {e}"));
                record.scorer_error = Some(e.to_string());
                append_journal_line(&journal_path, &JournalLine::Experiment(Box::new(record)))?;
                continue;
            }
        };
        if !cfg.preserve_candidates {
            let _ = std::fs::remove_dir_all(&work_dir);
        }
        consecutive_failures = 0;
        if let Some(s) = &outcome.screen {
            record.screen = Some(ScreenSummary {
                sample_rate: s.rate,
                sample_phase: s.phase,
                baseline_score: s.baseline_score,
                screened: (s.scores.len() - 1) as u64,
                promoted: s.promoted.len() as u64,
                bypass: s.bypass.len() as u64,
                scorer_ms: s.scorer_ms,
            });
            for c in &mut record.candidates {
                if let Some(r) = s.scores.get(&c.id) {
                    c.screen_score = Some(r.score);
                    c.screen_delta = Some(r.score - s.baseline_score);
                }
                c.promoted = s.promoted.contains(&c.id);
                c.bypass = s.bypass.contains(&c.id);
            }
        } else {
            for c in &mut record.candidates {
                c.promoted = true;
            }
        }
        if let Some(f) = &outcome.full {
            record.full = Some(FullSummary {
                baseline_score: f.baseline_score,
                baseline_drift: f.baseline_drift,
                scored: (f.scores.len() - 1) as u64,
                scorer_ms: f.scorer_ms,
            });
            for c in &mut record.candidates {
                if let Some(r) = f.scores.get(&c.id) {
                    c.full_score = Some(r.score);
                    c.full_delta = Some(r.score - f.baseline_score);
                }
            }
        }
        record.screen_false_positives = outcome.false_positives;
        record.screen_false_negatives = outcome.false_negatives;
        if let Some(v) = &outcome.veto {
            log::warn(v);
            record.scorer_error = Some(v.clone());
        }
        match (&outcome.winner, &outcome.full) {
            (Some((id, improvement)), Some(full)) => {
                let winner = candidates
                    .iter()
                    .find(|c| &c.id == id)
                    .ok_or("winner id not in cohort")?;
                let result = &full.scores[id];
                let new_incumbent = Incumbent::from_creature(
                    winner.creature.clone(),
                    &format!("winner-{acceptances:04}"),
                )
                .map_err(|e| e.to_string())?;
                acceptances += 1;
                log::ok(&format!(
                    "iteration {iterations}: accepted {id} ({}) Δscore {improvement:+.3e} → {:.9}; affected {} of {} search records",
                    winner.strategy(),
                    result.score,
                    winner.patch.provenance.affected_records,
                    set.records()
                ));
                record.winner = Some(id.clone());
                record.improvement = Some(*improvement);
                record.accepted = true;
                record.new_incumbent_checksum = Some(new_incumbent.checksum.clone());
                let new_baseline = AuthoritativeBaseline {
                    incumbent_checksum: new_incumbent.checksum.clone(),
                    score: result.score,
                    error: result.error,
                    complexity_penalty: result.complexity_penalty,
                    record_count: result.record_count,
                    scorer_identity: scorer.identity(),
                    cost_name: result.cost_name.clone(),
                    scorer_backend: result.gpu_backend.clone(),
                    corpus_identity: corpus.identity.clone(),
                    corpus_record_count: corpus.record_count,
                    local_mse: None,
                    parity: "inherited: scored in the same authoritative call as its parent".into(),
                    scorer_ms: full.scorer_ms,
                    created_at_unix: now_unix(),
                };
                append_journal_line(&journal_path, &JournalLine::Experiment(Box::new(record)))?;
                append_journal_line(&journal_path, &JournalLine::Baseline(new_baseline.clone()))?;
                // Promote atomically: write winner file, then best.json.
                std::fs::create_dir_all(&winners_dir).map_err(|e| e.to_string())?;
                // Tag every neuron this winner appended with its provenance.
                let target_uuid = {
                    let n = state.incumbent.creature.neurons.len();
                    state.incumbent.creature.neurons
                        [n - state.incumbent.creature.output + winner.patch.output]
                        .uuid
                        .clone()
                };
                for (p, uuids) in winner.patches().zip(&winner.added_uuids) {
                    let describe = match &p.root {
                        crate::patch::Node::Split {
                            condition,
                            left,
                            right,
                        } if condition.is_axis_aligned() => format!(
                            "input-{} > {} ? {} : {}",
                            condition.terms[0].feature,
                            condition.threshold,
                            match &**right {
                                crate::patch::Node::Leaf { correction } => correction.to_string(),
                                _ => "subtree".into(),
                            },
                            match &**left {
                                crate::patch::Node::Leaf { correction } => correction.to_string(),
                                _ => "subtree".into(),
                            },
                        ),
                        _ => format!(
                            "depth {} tree over inputs {:?}",
                            p.root.depth(),
                            p.root.features()
                        ),
                    };
                    for uuid in uuids {
                        state.meta.tag_neuron(
                            uuid,
                            vec![
                                Tag::new(
                                    "forests",
                                    format!(
                                        "neat_ai_forests v{} iteration {iterations} patch {} ({}, {}) → {target_uuid}: {describe}; Δscore {improvement:+.3e} verified by NEAT-AI-scorer",
                                        env!("CARGO_PKG_VERSION"),
                                        p.id(),
                                        p.provenance.strategy,
                                        p.provenance.backend
                                    ),
                                ),
                                Tag::new("forests-patch", p.id()),
                            ],
                        );
                    }
                }
                state.last_strategy = winner.strategy();
                state.last_target = target_uuid;
                // Near-winners: fully scored, positive Δ, not the winner → carry forward.
                state.runner_ups = candidates
                    .iter()
                    .filter(|c| &c.id != id && c.combo.is_empty())
                    .filter(|c| {
                        full.scores
                            .get(&c.id)
                            .is_some_and(|r| r.score - full.baseline_score > 0.0)
                    })
                    .take(cfg.combo_candidates)
                    .map(|c| c.patch.clone())
                    .collect();
                state.incumbent = new_incumbent;
                state.baseline = new_baseline;
                let winner_path = winners_dir.join(format!("winner-{acceptances:04}.json"));
                write_best(&winner_path, &state, opening_score, acceptances, iterations)?;
                write_best(&best_path, &state, opening_score, acceptances, iterations)?;
                // Incumbent changed: residual caches keyed by checksum are now stale by construction.
                state.residuals = ensure_residual_cache(
                    &state.incumbent,
                    &cfg.training_data,
                    &cache_dir,
                    &corpus,
                    cfg.chunk_records,
                    cfg.analysis_threads,
                )
                .map_err(|e| e.to_string())?;
            }
            _ => {
                state.runner_ups = outcome
                    .full
                    .as_ref()
                    .map(|f| {
                        candidates
                            .iter()
                            .filter(|c| c.combo.is_empty())
                            .filter(|c| {
                                f.scores
                                    .get(&c.id)
                                    .is_some_and(|r| r.score - f.baseline_score > 0.0)
                            })
                            .take(cfg.combo_candidates)
                            .map(|c| c.patch.clone())
                            .collect()
                    })
                    .unwrap_or_default();
                let best_delta = outcome.full.as_ref().and_then(|f| {
                    f.scores
                        .iter()
                        .filter(|(k, _)| k.as_str() != "baseline")
                        .map(|(_, v)| v.score - f.baseline_score)
                        .fold(None, |m: Option<f64>, d| Some(m.map_or(d, |m| m.max(d))))
                });
                log::detail(&format!(
                    "iteration {iterations}: no winner (best full Δ {}, screen FP {} FN {})",
                    best_delta.map_or("n/a".to_string(), |d| format!("{d:+.3e}")),
                    outcome.false_positives,
                    outcome.false_negatives
                ));
                append_journal_line(&journal_path, &JournalLine::Experiment(Box::new(record)))?;
            }
        }
    }
    let wall_ms = started.elapsed().as_millis() as u64;
    append_journal_line(
        &journal_path,
        &JournalLine::Summary(SummaryRecord {
            timestamp_unix: now_unix(),
            stop_reason: stop_reason.label().into(),
            iterations,
            acceptances,
            opening_score,
            final_score: state.baseline.score,
            wall_ms,
            final_checksum: state.incumbent.checksum.clone(),
        }),
    )?;
    Ok(RunResult {
        best_path,
        journal_path,
        opening_score,
        best_score: state.baseline.score,
        iterations,
        acceptances,
        stop_reason,
        seed,
        wall_ms,
        final_checksum: state.incumbent.checksum,
        scorer_failures,
    })
}

/// Print the end-of-run summary to stderr.
pub fn print_run_summary(r: &RunResult) {
    log::info(&format!(
        "done: {} iteration(s), {} accepted, {} scorer failure(s), stopped on {} after {:.1} s",
        r.iterations,
        r.acceptances,
        r.scorer_failures,
        r.stop_reason.label(),
        r.wall_ms as f64 / 1000.0
    ));
    log::info(&format!(
        "score {:.9} → {:.9} (Δ {:+.3e}); seed {}",
        r.opening_score,
        r.best_score,
        r.best_score - r.opening_score,
        r.seed
    ));
    log::info(&format!(
        "best: {}  journal: {}",
        r.best_path.display(),
        r.journal_path.display()
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::fake::LocalMseScorer;
    use crate::corpus::write_bin_file;
    use crate::graft::fixtures::identity_creature_json;
    use crate::journal::read_journal;
    use std::time::Duration;

    /// target = x0 + 0.3·[x1 > 0] + 0.2·[x2 > 0.5]; identity incumbent leaves two
    /// stump-shaped residual regions for sequential boosting to find.
    fn fixture() -> (tempfile::TempDir, ForestsConfig) {
        let tmp = tempfile::tempdir().unwrap();
        let train = tmp.path().join("train");
        std::fs::create_dir_all(&train).unwrap();
        let mut recs = Vec::new();
        for i in 0..600u32 {
            let x0 = (i % 17) as f32 / 17.0;
            let x1 = if (i * 7) % 5 < 2 { 1.0 } else { -1.0 };
            let x2 = ((i * 13) % 11) as f32 / 11.0;
            recs.push((
                vec![x0, x1, x2],
                vec![x0 + if x1 > 0.0 { 0.3 } else { 0.0 } + if x2 > 0.5 { 0.2 } else { 0.0 }],
            ));
        }
        write_bin_file(&train.join("0.bin"), &recs[..300]).unwrap();
        write_bin_file(&train.join("1.bin"), &recs[300..]).unwrap();
        let creature = tmp.path().join("creature.json");
        std::fs::write(&creature, identity_creature_json(3, 1)).unwrap();
        let cfg = ForestsConfig {
            creature,
            training_data: train,
            output_dir: tmp.path().join("out"),
            seed: Some(5),
            max_iterations: Some(4),
            timeout: Duration::from_secs(600),
            search_records: 0,
            min_leaf_records: 10.0,
            bins: 32,
            screen_sample_rate: Some(0.5),
            promote_count: 4,
            candidates: 24,
            random_candidates: 2,
            ..Default::default()
        };
        (tmp, cfg)
    }

    #[test]
    fn loop_accepts_sequentially_and_keeps_source_untouched() {
        let (tmp, cfg) = fixture();
        let original = std::fs::read_to_string(&cfg.creature).unwrap();
        let scorer = LocalMseScorer::new();
        let r = run_forests(&cfg, &scorer, &CancelToken::new()).unwrap();
        assert_eq!(std::fs::read_to_string(&cfg.creature).unwrap(), original);
        assert_eq!(r.iterations, 4);
        assert!(r.acceptances >= 2, "acceptances {}", r.acceptances);
        assert!(r.best_score > r.opening_score);
        assert_eq!(r.stop_reason, StopReason::MaxIterations);
        // best.json is the final incumbent, loadable, and carries tags.
        let best = std::fs::read_to_string(&r.best_path).unwrap();
        let c = neat_core::parse_creature_json(&best).unwrap();
        assert!(c.neurons.iter().any(|n| n.squash.as_deref() == Some("IF")));
        assert!(best.contains("\"forests\""));
        let meta = crate::meta::CreatureMeta::from_json(&best);
        assert!(meta.tags.iter().any(|t| t.name == "forests"
            && t.value.starts_with("🌳 Forests · ")
            && t.value.contains("improved by")));
        assert!(
            meta.tagged_neurons() >= 2,
            "grafted neurons must carry provenance tags"
        );
        assert!(
            meta.neuron_tags
                .values()
                .flatten()
                .any(|t| t.name == "forests" && t.value.contains("verified by NEAT-AI-scorer"))
        );
        assert!(tmp.path().join("out/winners/winner-0001.json").exists());
        // Journal: header, baseline, experiments, baselines after accepts, summary.
        let lines = read_journal(&r.journal_path).unwrap();
        assert!(matches!(lines[0], JournalLine::RunHeader { .. }));
        assert!(matches!(lines[1], JournalLine::Baseline(_)));
        assert!(matches!(lines.last().unwrap(), JournalLine::Summary(_)));
        let exps: Vec<&ExperimentRecord> = lines
            .iter()
            .filter_map(|l| {
                if let JournalLine::Experiment(e) = l {
                    Some(&**e)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(exps.len(), 4);
        // An accepted winner becomes the parent of the next search.
        let first_accept = exps.iter().position(|e| e.accepted).unwrap();
        let next = exps.get(first_accept + 1).unwrap();
        assert_eq!(
            Some(&next.incumbent_checksum),
            exps[first_accept].new_incumbent_checksum.as_ref()
        );
        assert!(next.baseline_score > exps[first_accept].baseline_score);
        // Screen and full results are recorded separately.
        let a = exps[first_accept];
        assert!(a.screen.is_some() && a.full.is_some());
        let w = a
            .candidates
            .iter()
            .find(|c| Some(&c.id) == a.winner.as_ref())
            .unwrap();
        assert!(w.screen_score.is_some() && w.full_score.is_some());
        assert!(w.full_delta.unwrap() > cfg.min_improvement);
        // Full-corpus verification: the fake scorer recorded a "full" call containing the winner.
        assert!(
            scorer
                .calls
                .borrow()
                .iter()
                .any(|(m, stems)| m == "full" && stems.contains(&w.id))
        );
        // Report consumes the journal.
        let rep = crate::report::report_from_journal(&r.journal_path).unwrap();
        assert_eq!(rep.acceptances, r.acceptances);
        assert!(rep.improvement_per_wall_hour.is_some());
    }

    #[test]
    fn no_winner_keeps_incumbent_and_scorer_failures_stop_the_run() {
        let (_tmp, mut cfg) = fixture();
        // Impossible threshold: nothing can ever be accepted.
        cfg.min_improvement = 10.0;
        cfg.max_iterations = Some(2);
        let scorer = LocalMseScorer::new();
        let r = run_forests(&cfg, &scorer, &CancelToken::new()).unwrap();
        assert_eq!(r.acceptances, 0);
        assert_eq!(r.best_score, r.opening_score);
        assert_eq!(
            std::fs::read_to_string(&r.best_path).unwrap(),
            std::fs::read_to_string(&cfg.creature).unwrap()
        );
        let lines = read_journal(&r.journal_path).unwrap();
        assert!(
            lines
                .iter()
                .all(|l| !matches!(l, JournalLine::Experiment(e) if e.accepted))
        );

        // A scorer that fails after the baseline stops the run after N failures.
        let mut cfg2 = cfg.clone();
        cfg2.output_dir = cfg.output_dir.join("fail");
        cfg2.max_iterations = None;
        cfg2.min_improvement = 1e-6;
        let flaky = FailAfterBaseline::default();
        let r = run_forests(&cfg2, &flaky, &CancelToken::new()).unwrap();
        assert_eq!(r.stop_reason, StopReason::ScorerFailures);
        assert_eq!(r.scorer_failures, 3);
        assert_eq!(r.acceptances, 0);
    }

    #[test]
    fn cancellation_is_honoured_and_journal_stays_valid() {
        let (_tmp, cfg) = fixture();
        let cancel = CancelToken::new();
        cancel.cancel();
        let r = run_forests(&cfg, &LocalMseScorer::new(), &cancel).unwrap();
        assert_eq!(r.stop_reason, StopReason::Cancelled);
        assert_eq!(r.iterations, 0);
        assert!(matches!(
            read_journal(&r.journal_path).unwrap().last().unwrap(),
            JournalLine::Summary(_)
        ));
    }

    #[test]
    fn boost_rounds_produce_verified_bundles() {
        let (_tmp, mut cfg) = fixture();
        cfg.max_iterations = Some(1);
        cfg.boost_rounds = 3;
        cfg.screen_sample_rate = None;
        let r = run_forests(&cfg, &LocalMseScorer::new(), &CancelToken::new()).unwrap();
        let lines = read_journal(&r.journal_path).unwrap();
        let exp = lines
            .iter()
            .find_map(|l| {
                if let JournalLine::Experiment(e) = l {
                    Some(e)
                } else {
                    None
                }
            })
            .unwrap();
        assert!(
            exp.strategies
                .iter()
                .any(|s| s.starts_with("boost-rounds/")),
            "{:?}",
            exp.strategies
        );
        let bundles: Vec<_> = exp
            .candidates
            .iter()
            .filter(|c| {
                c.combo
                    .iter()
                    .any(|p| p.provenance.strategy == "boost-round")
            })
            .collect();
        assert!(!bundles.is_empty());
        assert!(
            bundles.iter().all(|c| c.full_score.is_some()),
            "bundles must be fully scored"
        );
        // The two planted stumps are found by successive rounds.
        assert!(bundles.iter().any(|c| c.features.len() >= 2));
        assert!(r.acceptances >= 1);
    }

    #[test]
    fn same_seed_reproduces_the_candidate_set() {
        let (_tmp, mut cfg) = fixture();
        cfg.max_iterations = Some(1);
        cfg.row_sampling = crate::config::RowSampling::Uniform;
        cfg.search_records = 300;
        let a = run_forests(&cfg, &LocalMseScorer::new(), &CancelToken::new()).unwrap();
        cfg.output_dir = cfg.output_dir.join("again");
        let b = run_forests(&cfg, &LocalMseScorer::new(), &CancelToken::new()).unwrap();
        let ids = |p: &Path| {
            read_journal(p)
                .unwrap()
                .into_iter()
                .filter_map(|l| {
                    if let JournalLine::Experiment(e) = l {
                        Some(
                            e.candidates
                                .iter()
                                .map(|c| c.id.clone())
                                .collect::<Vec<_>>(),
                        )
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(&a.journal_path), ids(&b.journal_path));
    }

    /// Scores the baseline call, then fails every later call.
    #[derive(Default)]
    struct FailAfterBaseline {
        calls: std::cell::Cell<u32>,
    }

    impl DirectoryScorer for FailAfterBaseline {
        fn score_directory(
            &self,
            dir: &Path,
            training: &Path,
            mode: crate::scorer::ScorerMode,
        ) -> Result<
            std::collections::BTreeMap<String, crate::scorer::ScoreResult>,
            crate::scorer::ScorerError,
        > {
            let n = self.calls.get();
            self.calls.set(n + 1);
            if n == 0 {
                LocalMseScorer::new().score_directory(dir, training, mode)
            } else {
                Err(crate::scorer::ScorerError::Failed {
                    status: "exit 1".into(),
                    stderr: "simulated".into(),
                })
            }
        }
        fn identity(&self) -> String {
            "fake:fail-after-baseline".into()
        }
    }
}

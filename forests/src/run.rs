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
use crate::enhancements::EnhancementLog;
use crate::histogram::{HistogramSet, search_stumps};
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
    /// The enhancement bundle written for Rebase, when `--enhancements` is on
    /// and the run accepted something. `None` means there is nothing to rebase
    /// and Rebase must not be invoked.
    pub enhancements_path: Option<PathBuf>,
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

/// Commit **subject**: one score and one signed delta, nothing else (#98).
///
/// The delta is Rust's scientific rendering (`+1.27e-4`) — the format the whole
/// fleet is standardising on, so the same run reads the same way whichever tool
/// published it. Everything the old subject also carried lives in
/// [`forests_detail`], which belongs in the commit body.
fn forests_tag(opening: f64, score: f64) -> String {
    format!("🌳 Forests · score: {score:.6} ({:+.2e})", score - opening)
}

/// Commit **body** detail: what the subject no longer says (#98).
fn forests_detail(
    acceptances: u64,
    iterations: u64,
    last_strategy: &str,
    last_target: &str,
) -> String {
    let or_none = |s: &str| if s.is_empty() { "none" } else { s }.to_string();
    format!(
        "{acceptances} accepts / {iterations} iters · last: {} · 🎯 {}",
        or_none(last_strategy),
        or_none(last_target)
    )
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
    meta.upsert("forests", forests_tag(opening, state.baseline.score));
    meta.upsert(
        "forests-detail",
        forests_detail(
            acceptances,
            iterations,
            &state.last_strategy,
            &state.last_target,
        ),
    );
    let text = meta.serialize_with(&state.incumbent.creature, true)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// File this iteration's full-corpus verdicts in the shared cache (Issue #60).
///
/// Only candidates a full scorer call actually judged are filed: a screen
/// opinion is a ranking, not a verdict, and a graft refusal belongs to the
/// creature rather than to the patch (see `crate::learnings`). `known` is
/// extended with whatever was filed, so a later iteration of the same run does
/// not file it again.
fn file_learnings(
    store: Option<&crate::learnings::LearningsStore>,
    known: &mut Vec<crate::learnings::Learning>,
    record: &ExperimentRecord,
    incumbent: &Incumbent,
    corpus: &CorpusInfo,
) {
    let Some(store) = store else { return };
    let verdicts: Vec<crate::learnings::Verdict<'_>> = record
        .candidates
        .iter()
        .filter_map(|c| {
            let delta = c.full_delta?;
            let accepted = record.winner.as_deref() == Some(c.id.as_str());
            // `patch` is the first of the stack and `combo` the rest of it,
            // empty for a single-patch candidate.
            let mut patches = vec![c.patch.clone()];
            patches.extend(c.combo.iter().cloned());
            Some(crate::learnings::Verdict {
                id: c.id.as_str(),
                patches,
                outcome: if accepted {
                    crate::learnings::Outcome::Accepted
                } else {
                    crate::learnings::Outcome::Rejected
                },
                delta,
            })
        })
        .collect();
    if verdicts.is_empty() {
        return;
    }
    let ctx = crate::learnings::Context {
        corpus: corpus.identity.clone(),
        inputs: incumbent.creature.input,
        outputs: incumbent.creature.output,
        incumbent: record.incumbent_checksum.clone(),
        incumbent_score: record.baseline_score,
        host: String::new(),
        at_unix: now_unix(),
    };
    let mut ctx = ctx;
    ctx.host = store.host().to_string();
    let filed = crate::learnings::file_verdicts(&verdicts, &ctx, known);
    if filed.is_empty() {
        return;
    }
    match store.append(&filed) {
        Ok(()) => {
            log::detail(&format!(
                "learnings: filed {} verdict(s) to {}",
                filed.len(),
                store.file().display()
            ));
            known.extend(filed);
        }
        // Never fail a run over the cache: the creature and the journal are the
        // deliverables, the cache is an optimisation.
        Err(e) => log::warn(&format!("learnings not written: {e}")),
    }
}

/// Label journalled as the search backend that produced a split (Issue #67).
///
/// Kept as a field rather than dropped: every discovery records how it was
/// found, so a second search path added later can be told apart from this one
/// in journals written before it existed.
const SEARCH_BACKEND: &str = "cpu";

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

    // The fleet's shared cache of what has been tried (Issue #60). Loaded once
    // — the directory is a git checkout the caller pulls between runs — and
    // extended in memory as this run reaches its own verdicts, so the dedupe
    // sees them too.
    let learnings_store = cfg.learnings_dir.as_ref().map(|dir| {
        crate::learnings::LearningsStore::new(
            dir.clone(),
            corpus.identity.clone(),
            cfg.learnings_host
                .clone()
                .unwrap_or_else(crate::learnings::default_host),
        )
    });
    let mut known: Vec<crate::learnings::Learning> = match &learnings_store {
        Some(store) => match store.load() {
            Ok(all) => {
                log::info(&format!(
                    "learnings: {} record(s) for this corpus in {}",
                    all.len(),
                    store.corpus_dir().display()
                ));
                all
            }
            Err(e) => {
                // A cache that cannot be read is a cache miss, never a reason
                // to abandon a run that would otherwise improve the creature.
                log::warn(&format!("learnings unreadable ({e}); continuing without"));
                Vec::new()
            }
        },
        None => Vec::new(),
    };

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
    // Producer side of population re-entry (stSoftwareAU/NEAT-AI-Rebase#65).
    // Opened on the creature the run starts from and on the score just
    // established, because those are the facts every filed patch is stamped
    // with — rebuilding them later would name a creature nobody else has.
    let bundle_path = cfg.output_dir.join("enhancements.json");
    let mut enhancements = if cfg.enhancements {
        let log = EnhancementLog::open(&incumbent.creature, opening_score, &corpus.identity)?;
        log::info(&format!(
            "re-entry: filing accepted patches for Rebase (--enhancements) → {}",
            bundle_path.display()
        ));
        log
    } else {
        log::info("re-entry: direct path; --enhancements is off, no bundle will be written");
        EnhancementLog::off()
    };
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
        // Where this iteration's corrections can enter (Issue #58). On a
        // creature whose output is clamped by `MINIMUM`/`MAXIMUM` aggregates
        // that is a hidden neuron behind the clamps, not the output itself; an
        // output nothing can be grafted onto is worth saying before a whole
        // iteration of search is spent finding candidates that cannot be used.
        match crate::graft::graft_anchor(&state.incumbent.creature, output) {
            Ok((uuid, gain)) if gain != 1.0 => log::detail(&format!(
                "corrections enter at `{uuid}` behind the output's clamps (gain {gain:.6})"
            )),
            Ok(_) => {}
            Err(e) => log::warn(&format!("output {output} cannot take a graft: {e}")),
        }

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
        let hist = HistogramSet::from_source_threads(
            &set.source,
            &bins_per_feature,
            cfg.analysis_threads,
        )?;
        // Journalled against every discovery, so a future second search path
        // can be told apart from this one after the fact (Issue #67).
        let backend_label = SEARCH_BACKEND.to_string();
        // Σ of the search set's importance weights — the denominator any
        // weighted count has to be compared against (Issue #64). Equal to the
        // row count when every weight is 1, which is every stride-sampled set.
        let searched_weight = hist.total_count;
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
            // `None` is the unconstrained best-first tree; the rest fix the
            // first split to one of the best-ranked stumps, one per distinct
            // feature (Issue #63).
            let mut roots: Vec<Option<(usize, usize)>> = vec![None];
            roots.extend(
                crate::tree::root_features(&stumps, cfg.tree_roots)
                    .into_iter()
                    .map(Some),
            );
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
            graft_constants: cfg.graft_constants,
            if_correction: cfg.if_correction,
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
        // What the fleet already knows (Issue #60), ahead of this iteration's
        // own discoveries: a patch another host got past the full scorer is
        // better evidence than a proxy gain, and it is the only way a win
        // survives the fittest creature moving on before we could re-apply it.
        if !known.is_empty() {
            let chosen = crate::learnings::choose(
                &known,
                &state.incumbent.checksum,
                state.incumbent.creature.input,
                state.incumbent.creature.output,
                &crate::learnings::grafted_patch_ids(&state.incumbent.creature),
                &crate::learnings::ReplayConfig {
                    max: cfg.learnings_replay,
                    retry_after_secs: cfg.learnings_retry_after_secs,
                    now_unix: now_unix(),
                },
            );
            if !chosen.is_empty() {
                let (mut wins, mut retries) = (0, 0);
                let mut replayed = Vec::new();
                for learning in &chosen {
                    match learning.outcome {
                        crate::learnings::Outcome::Accepted => wins += 1,
                        crate::learnings::Outcome::Rejected => retries += 1,
                    }
                    let group = learning.replay(&state.incumbent.checksum);
                    if group.len() == 1 {
                        replayed.extend(group);
                    } else {
                        combo_groups.push(group);
                    }
                }
                log::detail(&format!(
                    "replaying {} cached candidate(s): {wins} the fleet accepted elsewhere, {retries} due another try",
                    chosen.len()
                ));
                // Ahead of this iteration's own discoveries: the cohort is
                // capped, and known-good beats predicted-good.
                replayed.append(&mut patches);
                patches = replayed;
                strategies.push("replay".into());
            }
            // The other half of the cheat: drop what the fleet has already
            // proved does not work, so the slot goes to the next discovery
            // rather than to a scorer call whose answer is on file. A patch
            // that ever cleared the full scorer is never dropped, and after
            // the retry window the mistake is worth making again.
            let avoid = crate::learnings::known_failures(
                &known,
                &crate::learnings::ReplayConfig {
                    max: cfg.learnings_replay,
                    retry_after_secs: cfg.learnings_retry_after_secs,
                    now_unix: now_unix(),
                },
            );
            if !avoid.is_empty() {
                let before = patches.len() + combo_groups.len();
                patches.retain(|p| !avoid.contains(&p.id()));
                combo_groups.retain(|g| !avoid.contains(&crate::candidates::combo_id(g)));
                let dropped = before - (patches.len() + combo_groups.len());
                if dropped > 0 {
                    log::detail(&format!(
                        "skipping {dropped} candidate(s) the fleet has already scored and turned down"
                    ));
                    strategies.push("avoid-known-failures".into());
                }
            }
        }
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
                let h = HistogramSet::from_source_threads(
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
            crate::candidates::generate_combos(
                &state.incumbent,
                combo_groups,
                "combination",
                cfg.graft_constants,
            );
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
                    // The weighted total, not the row count: under
                    // residual-weighted sampling the numerator estimates a
                    // corpus-wide count and the two must be on one scale
                    // (Issue #64).
                    affected_fraction: crate::journal::affected_fraction(
                        c.affected_records() as f64,
                        searched_weight,
                        set.records(),
                    ),
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
                // File what was authoritatively accepted, in the order the
                // winner applies it: `patch` then `combo`. A combo that grew
                // from an already-filed single appends only its new members, so
                // the bundle's prefix of that length reproduces exactly the
                // creature `result.score` was measured on.
                let accepted_patches: Vec<Patch> = winner.patches().cloned().collect();
                match enhancements.accept(&accepted_patches, result.score) {
                    Ok(filed) if filed > 0 => log::detail(&format!(
                        "enhancements: filed {filed} patch(es) ({} total)",
                        enhancements.filed()
                    )),
                    Ok(_) => {}
                    // Loud here, and fatal at the end: a bundle missing an
                    // accepted patch would claim scores its prefixes do not
                    // reproduce, so no bundle is written at all.
                    Err(e) => log::warn(&format!("enhancement not filed: {e}")),
                }
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
                file_learnings(
                    learnings_store.as_ref(),
                    &mut known,
                    &record,
                    &state.incumbent,
                    &corpus,
                );
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
                file_learnings(
                    learnings_store.as_ref(),
                    &mut known,
                    &record,
                    &state.incumbent,
                    &corpus,
                );
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
    // After the journal, so a filing failure cannot cost the run its record of
    // what happened — and before the result, so it cannot be reported as a
    // success the caller then rebases nothing from.
    let enhancements_path = if enhancements.write_bundle(&bundle_path)? {
        log::ok(&format!(
            "enhancements: {} patch(es) filed to {}; rebase them onto a freshly fetched champion",
            enhancements.filed(),
            bundle_path.display()
        ));
        Some(bundle_path)
    } else {
        if enhancements.is_on() {
            log::info("enhancements: nothing was accepted, so no bundle and nothing to rebase");
        }
        None
    };
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
        enhancements_path,
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
    if let Some(bundle) = &r.enhancements_path {
        log::info(&format!("enhancements: {}", bundle.display()));
    }
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

    /// Issue #56 — end to end under `--graft-constants per-patch`: the accepted
    /// creature's bias-1 constants are all named for the patch that made them,
    /// and every constant is read only by `IF` nodes of that same patch. That
    /// is the blast-radius property on a creature the loop actually produced,
    /// not a hand-built fixture.
    #[test]
    fn per_patch_constants_survive_the_loop_and_stay_within_their_patch() {
        let (_tmp, mut cfg) = fixture();
        cfg.graft_constants = crate::config::GraftConstants::PerPatch;
        let scorer = LocalMseScorer::new();
        let r = run_forests(&cfg, &scorer, &CancelToken::new()).unwrap();
        assert!(r.acceptances >= 2, "acceptances {}", r.acceptances);
        let best = std::fs::read_to_string(&r.best_path).unwrap();
        let c = neat_core::parse_creature_json(&best).unwrap();
        let ones: Vec<&str> = c
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "constant" && n.bias == 1.0)
            .map(|n| n.uuid.as_str())
            .collect();
        assert!(ones.len() >= 6, "expected constants per patch: {ones:?}");
        // `forest-<patch id>-one-<letter>` — the id is what bounds the radius.
        let patch_of = |uuid: &str, sep: &str| -> Option<String> {
            uuid.strip_prefix("forest-")
                .and_then(|rest| rest.split_once(sep))
                .map(|(id, _)| id.to_string())
        };
        for one in &ones {
            let owner = patch_of(one, "-one-")
                .unwrap_or_else(|| panic!("constant {one} is not named for a patch"));
            for s in c.synapses.iter().filter(|s| s.from_uuid == **one) {
                let reader = patch_of(&s.to_uuid, "-if").unwrap_or_else(|| {
                    panic!("{one} feeds {}, which is not a grafted IF node", s.to_uuid)
                });
                assert_eq!(
                    reader, owner,
                    "{one} is read by a node of another patch ({})",
                    s.to_uuid
                );
            }
        }
    }

    /// Issue #60 — end to end: one run files what the full scorer decided, and
    /// a second run on a *different* creature replays it. That second half is
    /// the whole point: by the time the fleet gets back to a win, the fittest
    /// creature has usually moved on, and the patch is portable precisely
    /// because it names feature indices rather than neuron uuids.
    #[test]
    fn the_fleet_cache_records_verdicts_and_replays_them_onto_another_creature() {
        let (tmp, mut cfg) = fixture();
        let shared = tmp.path().join("shared/learnings");
        cfg.learnings_dir = Some(shared.clone());
        cfg.learnings_host = Some("host-a".into());
        let scorer = LocalMseScorer::new();
        let first = run_forests(&cfg, &scorer, &CancelToken::new()).unwrap();
        assert!(first.acceptances >= 1);

        let store =
            crate::learnings::LearningsStore::new(&shared, String::new(), "host-a".to_string());
        let corpus_dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&shared)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .collect();
        assert_eq!(
            corpus_dirs.len(),
            1,
            "one directory per corpus: {corpus_dirs:?}"
        );
        let _ = store;
        let files: Vec<String> = std::fs::read_dir(&corpus_dirs[0])
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(files, vec!["host-a.jsonl"], "one file per host");

        let filed: Vec<crate::learnings::Learning> =
            std::fs::read_to_string(corpus_dirs[0].join("host-a.jsonl"))
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect();
        assert!(
            filed
                .iter()
                .any(|l| l.outcome == crate::learnings::Outcome::Accepted),
            "the winner is filed"
        );
        assert!(
            filed
                .iter()
                .any(|l| l.outcome == crate::learnings::Outcome::Rejected),
            "so is what the full scorer turned down"
        );
        assert!(
            filed.iter().all(|l| l.host == "host-a" && l.inputs == 3),
            "every line says who filed it and how wide the creature was"
        );

        // A second host on a *sibling* creature — same widths, same corpus, no
        // ancestry, so it carries none of the first run's patches. That is the
        // case the cache exists for: the win would otherwise be lost with the
        // creature it was found on. Its own search is switched off, so anything
        // it grafts can only have come from the cache.
        let sibling = tmp.path().join("sibling.json");
        let mut c = neat_core::parse_creature_json(&identity_creature_json(3, 1)).unwrap();
        c.neurons.last_mut().unwrap().bias = 1e-4;
        std::fs::write(&sibling, neat_core::creature_to_json_pretty(&c).unwrap()).unwrap();
        let mut second = cfg.clone();
        second.output_dir = tmp.path().join("out2");
        second.learnings_host = Some("host-b".into());
        second.creature = sibling;
        second.max_iterations = Some(1);
        second.top_k = 0;
        second.random_candidates = 0;
        second.combo_candidates = 0;
        let r2 = run_forests(&second, &scorer, &CancelToken::new()).unwrap();
        let journal = read_journal(&r2.journal_path).unwrap();
        let replayed: Vec<&crate::journal::CandidateRecord> = journal
            .iter()
            .filter_map(|l| match l {
                JournalLine::Experiment(e) => Some(&e.candidates),
                _ => None,
            })
            .flatten()
            .filter(|c| c.patch.provenance.strategy == "replay")
            .collect();
        assert!(
            !replayed.is_empty(),
            "nothing was replayed onto the sibling creature"
        );
        assert!(
            replayed.iter().any(|c| filed
                .iter()
                .any(|l| l.id == c.id && l.outcome == crate::learnings::Outcome::Accepted)),
            "the win the other host found is among them"
        );
        assert!(
            replayed
                .iter()
                .all(|c| c.patch.provenance.backend == "learnings"),
            "a replayed candidate says where it came from"
        );
        // The mistakes the first host paid for are not made again: nothing the
        // full scorer turned down there is scored here.
        let turned_down: std::collections::HashSet<&str> = filed
            .iter()
            .filter(|l| l.outcome == crate::learnings::Outcome::Rejected)
            .map(|l| l.id.as_str())
            .collect();
        assert!(!turned_down.is_empty(), "the fixture must reject something");
        let scored_again: Vec<&str> = journal
            .iter()
            .filter_map(|l| match l {
                JournalLine::Experiment(e) => Some(&e.candidates),
                _ => None,
            })
            .flatten()
            .filter(|c| c.full_score.is_some())
            .map(|c| c.id.as_str())
            .filter(|id| turned_down.contains(id))
            .collect();
        assert!(
            scored_again.is_empty(),
            "spent a scorer call on what the fleet already turned down: {scored_again:?}"
        );

        // And the second host wrote its own file rather than touching the first's.
        let files: Vec<String> = std::fs::read_dir(&corpus_dirs[0])
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(files.contains(&"host-b.jsonl".to_string()), "{files:?}");
    }

    /// The cache is opt-in, and switched off it must leave the optimiser
    /// exactly as it was: nothing written anywhere, and — with the cache
    /// writing but neither replaying nor avoiding — the same candidates in the
    /// same order. Recording a verdict must never perturb the search that
    /// produced it; only *acting* on one may.
    #[test]
    fn with_no_learnings_directory_nothing_changes_and_nothing_is_written() {
        let ids = |journal: &std::path::Path| -> Vec<String> {
            read_journal(journal)
                .unwrap()
                .iter()
                .filter_map(|l| match l {
                    JournalLine::Experiment(e) => Some(e.candidates.clone()),
                    _ => None,
                })
                .flatten()
                .map(|c| c.id)
                .collect()
        };
        let (tmp, cfg) = fixture();
        let scorer = LocalMseScorer::new();
        let off = run_forests(&cfg, &scorer, &CancelToken::new()).unwrap();

        let mut writing = cfg.clone();
        writing.output_dir = tmp.path().join("out-writing");
        writing.learnings_dir = Some(tmp.path().join("shared/learnings"));
        writing.learnings_host = Some("host-a".into());
        writing.learnings_replay = 0;
        // Nothing is old enough to avoid or to retry, so the cache is
        // write-only for the length of this run.
        writing.learnings_retry_after_secs = 0;
        let on = run_forests(&writing, &scorer, &CancelToken::new()).unwrap();

        assert_eq!(
            ids(&off.journal_path),
            ids(&on.journal_path),
            "writing verdicts changed which candidates were tried"
        );
        assert_eq!(off.best_score, on.best_score);
        assert!(
            !tmp.path().join("shared").exists() || tmp.path().join("shared/learnings").exists(),
            "the writing run owns that directory"
        );
        // The run with the option off touched nothing outside its output dir.
        let stray: Vec<std::path::PathBuf> = std::fs::read_dir(cfg.output_dir.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.file_name().is_some_and(|n| n == "learnings"))
            .collect();
        assert!(stray.is_empty(), "{stray:?}");
    }

    /// The ids of every patch the run authoritatively accepted, in acceptance
    /// order, read back out of the journal — the independent account of what
    /// the bundle is supposed to contain.
    fn accepted_patch_ids(journal: &std::path::Path) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        for line in read_journal(journal).unwrap() {
            let JournalLine::Experiment(e) = line else {
                continue;
            };
            let Some(winner) = e.winner.as_deref() else {
                continue;
            };
            let Some(c) = e.candidates.iter().find(|c| c.id == winner) else {
                continue;
            };
            for p in std::iter::once(&c.patch).chain(c.combo.iter()) {
                // A combo that grew from an already-filed single files only
                // its new members, so first acceptance wins.
                if !ids.contains(&p.id()) {
                    ids.push(p.id());
                }
            }
        }
        ids
    }

    /// stSoftwareAU/NEAT-AI-Rebase#65 — the call site: every patch the full
    /// scorer accepted is filed, in acceptance order, stamped with the facts of
    /// the creature the run opened on, and the bundle grafts onto a champion
    /// this run never produced. That last half is the whole point: by the time
    /// a 45-minute run ends the fleet's champion has moved on, and publishing
    /// this run's own descendant would throw away everybody else's work.
    #[test]
    fn accepted_patches_are_filed_and_the_bundle_rebases_onto_a_champion_the_run_never_saw() {
        let (tmp, mut cfg) = fixture();
        cfg.enhancements = true;
        // Boosting verifies prefixes of a bundle in one scorer call, so a
        // winner here can be a combo — the case `accept_combo` exists for.
        cfg.boost_rounds = 3;
        let scorer = LocalMseScorer::new();
        let r = run_forests(&cfg, &scorer, &CancelToken::new()).unwrap();
        assert!(r.acceptances >= 1, "the fixture must accept something");

        let bundle_path = r
            .enhancements_path
            .clone()
            .expect("a run that accepted a patch files a bundle");
        assert_eq!(bundle_path, cfg.output_dir.join("enhancements.json"));
        let bundle = neat_ai_rebase::enhancement::EnhancementBundle::parse_json(
            &std::fs::read_to_string(&bundle_path).unwrap(),
        )
        .unwrap();

        let expected = accepted_patch_ids(&r.journal_path);
        assert!(!expected.is_empty());
        let filed: Vec<String> = bundle
            .enhancements
            .iter()
            .map(|e| e.meta.id.clone())
            .collect();
        assert_eq!(filed, expected, "acceptance order, and nothing else");

        // Every enhancement names the creature the run opened on — not the
        // descendant it reached — and the corpus both scores were measured on.
        let opening =
            neat_core::parse_creature_json(&std::fs::read_to_string(&cfg.creature).unwrap())
                .unwrap();
        let opening_checksum = neat_ai_rebase::creature::creature_checksum(&opening).unwrap();
        let identity = corpus_info(
            &cfg.training_data,
            &TrainingDataConfig::new(opening.input, opening.output),
        )
        .unwrap()
        .identity;
        for e in &bundle.enhancements {
            assert_eq!(e.meta.base_checksum, opening_checksum);
            assert!((e.meta.base_score - r.opening_score).abs() < 1e-12);
            assert_eq!(e.meta.corpus_identity, identity);
            assert_eq!(e.meta.input_count, opening.input);
            assert!(e.meta.producer.starts_with("neat-ai-forests/"));
            assert!(e.id_is_consistent(), "a filed id must match its payload");
        }

        // A champion the fleet reached independently: same widths, no ancestry
        // with this run's descendant, so it carries none of these patches.
        let mut champion = opening.clone();
        champion.neurons.last_mut().unwrap().bias = 1e-4;
        let outcome = neat_ai_rebase::engine::rebase(&neat_ai_rebase::engine::RebaseRequest {
            champion: &champion,
            enhancements: &bundle.enhancements,
            corpus_identity: &identity,
            max_candidates: 0,
        })
        .unwrap();
        for report in &outcome.reports {
            assert_eq!(
                report.outcome,
                neat_ai_rebase::engine::EnhancementOutcome::Applied,
                "{report:?}"
            );
        }
        let full = outcome
            .cohort
            .iter()
            .find(|c| c.label == "bundle")
            .expect("every patch applied, so the full bundle is a candidate");
        assert_eq!(full.applied_ids.len(), bundle.enhancements.len());
        let _ = tmp;
    }

    /// The switch is the contract: with it off the optimiser must be exactly
    /// what it was — same candidates, same score — and nothing is written.
    /// Recording what was accepted must never perturb what gets accepted.
    #[test]
    fn with_enhancements_off_nothing_is_written_and_the_run_is_unchanged() {
        let ids = |journal: &std::path::Path| -> Vec<String> {
            read_journal(journal)
                .unwrap()
                .iter()
                .filter_map(|l| match l {
                    JournalLine::Experiment(e) => Some(e.candidates.clone()),
                    _ => None,
                })
                .flatten()
                .map(|c| c.id)
                .collect()
        };
        let (tmp, cfg) = fixture();
        let scorer = LocalMseScorer::new();
        let off = run_forests(&cfg, &scorer, &CancelToken::new()).unwrap();
        assert!(off.enhancements_path.is_none());
        assert!(
            !cfg.output_dir.join("enhancements.json").exists(),
            "the switch is off: no bundle"
        );

        let mut on = cfg.clone();
        on.output_dir = tmp.path().join("out-filing");
        on.enhancements = true;
        let on = run_forests(&on, &scorer, &CancelToken::new()).unwrap();
        assert!(on.enhancements_path.is_some());
        assert_eq!(
            ids(&off.journal_path),
            ids(&on.journal_path),
            "filing changed which candidates were tried"
        );
        assert_eq!(off.best_score, on.best_score);
        assert_eq!(off.acceptances, on.acceptances);
        assert_eq!(off.final_checksum, on.final_checksum);
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
        // #98: the subject is one score and one signed scientific delta; the
        // run detail moved to `forests-detail` for the commit body.
        assert!(meta.tags.iter().any(|t| t.name == "forests"
            && t.value.starts_with("🌳 Forests · score: ")
            && t.value.contains(" (+")));
        assert!(
            meta.tags
                .iter()
                .any(|t| t.name == "forests-detail" && t.value.contains(" accepts / "))
        );
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

    /// The fleet refuses a check-in that lost the source champion's provenance
    /// (GRQ #4216). Its three rules, asserted on the creature a whole run
    /// publishes rather than on `serialize_with` alone:
    ///
    /// 1. every creature-level tag **name** on the source is still there
    ///    (values move — `score` and `error` are re-stamped every run);
    /// 2. every source neuron that carried tags still carries them, and the
    ///    neurons the graft appended carry their own on top;
    /// 3. the creature-level `uuid` and `memetic` are **gone** — both describe
    ///    a structure that no longer exists once a patch is grafted in.
    #[test]
    fn published_creature_keeps_source_provenance_and_drops_stale_identity() {
        let (_tmp, cfg) = fixture();
        // Re-write the source with the metadata a mature champion carries.
        let mut source: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cfg.creature).unwrap()).unwrap();
        source["uuid"] = serde_json::json!("6ec3d0b6-0f4c-4a0e-9f4a-2a5f9f7c1b23");
        source["memetic"] = serde_json::json!({"biases": {"1": 0.25}});
        source["tags"] = serde_json::json!([
            {"name": "score", "value": "0.100000"},
            {"name": "error", "value": "0.900000"},
            {"name": "lamarck", "value": "🧬 Lamarck · 3 accepts / 9 iters"},
        ]);
        source["neurons"][0]["tags"] = serde_json::json!([
            {"name": "discovered", "value": "ReLU6"},
            {"name": "intelligentDesign", "value": "STEP -> ELU"},
        ]);
        std::fs::write(
            &cfg.creature,
            serde_json::to_string_pretty(&source).unwrap(),
        )
        .unwrap();

        let r = run_forests(&cfg, &LocalMseScorer::new(), &CancelToken::new()).unwrap();
        assert!(
            r.acceptances >= 1,
            "the run must publish a grafted creature"
        );
        let published = std::fs::read_to_string(&r.best_path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&published).unwrap();
        neat_core::parse_creature_json(&published).unwrap();

        // Rule 1 — no creature-level tag name lost.
        let meta = CreatureMeta::from_json(&published);
        for name in ["score", "error", "lamarck"] {
            assert!(
                meta.tags.iter().any(|t| t.name == name),
                "creature tag `{name}` lost: {:?}",
                meta.tags
            );
        }
        assert_ne!(
            meta.tags
                .iter()
                .find(|t| t.name == "score")
                .map(|t| t.value.as_str()),
            Some("0.100000"),
            "the run must re-stamp its own score"
        );

        // Rule 2 — no per-neuron tag set lost, and the graft adds its own.
        let output_tags = &meta.neuron_tags["output-0"];
        assert_eq!(
            output_tags
                .iter()
                .find(|t| t.name == "intelligentDesign")
                .map(|t| t.value.as_str()),
            Some("STEP -> ELU")
        );
        assert!(output_tags.iter().any(|t| t.name == "discovered"));
        assert!(meta.neuron_tags.iter().any(
            |(uuid, tags)| uuid != "output-0" && tags.iter().any(|t| t.name == "forests-patch")
        ));

        // Rule 3 — the stale identity of a structure that changed is gone.
        assert!(
            value.get("uuid").is_none(),
            "a grafted creature must not carry the source `uuid`"
        );
        assert!(
            value.get("memetic").is_none(),
            "a grafted creature must not carry the source `memetic`: the graft \
             inserts neurons, so id-keyed memetic entries now name other neurons"
        );
    }

    /// The commit subject is one score and one delta, in Rust's scientific
    /// rendering with an explicit sign, and nothing else (#98).
    #[test]
    fn commit_subject_is_one_score_and_one_signed_scientific_delta() {
        assert_eq!(
            forests_tag(0.4069110716468155, 0.407038264798196),
            "🌳 Forests · score: 0.407038 (+1.27e-4)"
        );
        // A run that accepted nothing still reads the same way.
        assert_eq!(
            forests_tag(0.25, 0.25),
            "🌳 Forests · score: 0.250000 (+0.00e0)"
        );
        // A smaller final score is reported, never hidden behind prose.
        assert_eq!(
            forests_tag(0.25, 0.2499),
            "🌳 Forests · score: 0.249900 (-1.00e-4)"
        );
        // No detail leaks back into the subject.
        let subject = forests_tag(0.1, 0.2);
        for noise in ["accepts", "iters", "last:", "🎯", "improved by"] {
            assert!(
                !subject.contains(noise),
                "subject must stay short, found `{noise}` in `{subject}`"
            );
        }
    }

    /// What the subject dropped is not lost: it moves to the body detail (#98).
    #[test]
    fn commit_body_detail_carries_accepts_iterations_strategy_and_target() {
        assert_eq!(
            forests_detail(7, 9, "histogram-tree-depth3/scale", "output-0"),
            "7 accepts / 9 iters · last: histogram-tree-depth3/scale · 🎯 output-0"
        );
        // Nothing accepted yet: no empty fields, no dangling separators.
        assert_eq!(
            forests_detail(0, 3, "", ""),
            "0 accepts / 3 iters · last: none · 🎯 none"
        );
    }

    /// A published creature carries both halves: the short subject on
    /// `forests`, the detail on `forests-detail` (#98).
    #[test]
    fn published_creature_carries_subject_and_detail_tags() {
        let (_tmp, cfg) = fixture();
        let r = run_forests(&cfg, &LocalMseScorer::new(), &CancelToken::new()).unwrap();
        assert!(
            r.acceptances >= 1,
            "the run must publish a grafted creature"
        );
        let meta = CreatureMeta::from_json(&std::fs::read_to_string(&r.best_path).unwrap());
        let tag = |name: &str| {
            meta.tags
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("creature tag `{name}` missing: {:?}", meta.tags))
                .value
                .clone()
        };
        assert_eq!(
            tag("forests"),
            forests_tag(r.opening_score, r.best_score),
            "the subject must describe the run that was published"
        );
        // `best.json` is written at each acceptance, so the detail counts the
        // run as at the last one: every acceptance, and the iteration it
        // happened on.
        let detail = tag("forests-detail");
        assert!(
            detail.starts_with(&format!("{} accepts / ", r.acceptances)),
            "detail must count the acceptances that were published: {detail}"
        );
        let iters: u64 = detail
            .split(" / ")
            .nth(1)
            .and_then(|s| s.split(' ').next())
            .unwrap()
            .parse()
            .unwrap();
        assert!(iters >= 1 && iters <= r.iterations, "{detail}");
        // After an acceptance the strategy and the target are both real.
        assert!(!detail.contains("last: none"), "{detail}");
        assert!(detail.ends_with("· 🎯 output-0"), "{detail}");
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

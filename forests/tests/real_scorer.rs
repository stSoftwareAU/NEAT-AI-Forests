//! Integration with the real NEAT-AI-scorer binary (Issues #2, #7, #9, #10).
//!
//! These tests locate `rust_scorer` via `NEAT_SCORER_BIN`, then the sibling
//! `../../NEAT-AI-scorer/target/release/rust_scorer`, then `$PATH`. When no
//! binary is available they print a skip notice and pass — CI for this repo
//! checks out NEAT-AI-core but not a built scorer.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use neat_ai_forests::baseline::ParityPolicy;
use neat_ai_forests::config::{ForestsConfig, GraftConstants, RankingMode};
use neat_ai_forests::corpus::{corpus_info, write_bin_file};
use neat_ai_forests::graft::fixtures::{
    identity_creature, identity_creature_json, if_output_creature, min_clamped_if_creature, small_mlp,
};
use neat_ai_forests::graft::{graft_patch, graft_patches_with};
use neat_ai_forests::incumbent::Incumbent;
use neat_ai_forests::journal::{JournalLine, read_journal};
use neat_ai_forests::patch::{Node, Patch, Provenance};
use neat_ai_forests::scorer::{CohortExecution, DirectoryScorer, ExternalScorer, ScorerMode};
use neat_ai_forests::{CancelToken, establish_baseline, run_forests};
use neat_core::training_data::TrainingDataConfig;

fn scorer_binary() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("NEAT_SCORER_BIN") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let sibling = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../NEAT-AI-scorer/target/release/rust_scorer");
    if sibling.is_file() {
        return Some(sibling);
    }
    Command::new("rust_scorer")
        .arg("--help")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| PathBuf::from("rust_scorer"))
}

macro_rules! require_scorer {
    () => {
        match scorer_binary() {
            Some(p) => p,
            None => {
                eprintln!("skipping: no rust_scorer binary (set NEAT_SCORER_BIN)");
                return;
            }
        }
    };
}

/// target = x0 + 0.3·[x1 > 0]; identity incumbent leaves a stump-shaped residual.
fn fixture(tmp: &Path, n: u32) -> (PathBuf, PathBuf) {
    let train = tmp.join("train");
    std::fs::create_dir_all(&train).unwrap();
    let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..n)
        .map(|i| {
            let x0 = (i % 17) as f32 / 17.0;
            let x1 = if (i * 7) % 5 < 2 { 1.0 } else { -1.0 };
            let x2 = ((i * 13) % 11) as f32 / 11.0;
            (
                vec![x0, x1, x2],
                vec![x0 + if x1 > 0.0 { 0.3 } else { 0.0 }],
            )
        })
        .collect();
    write_bin_file(&train.join("0.bin"), &recs).unwrap();
    let creature = tmp.join("creature.json");
    std::fs::write(&creature, identity_creature_json(3, 1)).unwrap();
    (creature, train)
}

#[test]
fn real_scorer_baseline_parity_and_if_graft_agree_with_local_forward_pass() {
    let bin = require_scorer!();
    let tmp = tempfile::tempdir().unwrap();
    let (creature_path, train) = fixture(tmp.path(), 500);
    let scorer = ExternalScorer {
        binary: bin,
        extra_args: vec!["--gpu".into(), "off".into()],
    };
    let incumbent = neat_ai_forests::load_incumbent(&creature_path).unwrap();
    let cfg = TrainingDataConfig::new(3, 1);
    let corpus = corpus_info(&train, &cfg).unwrap();
    let residuals =
        neat_ai_forests::residuals::compute_residuals(&incumbent, &train, &corpus, 128, 2).unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let baseline = establish_baseline(
        &incumbent,
        &train,
        &corpus,
        &scorer,
        &ws,
        ParityPolicy::default(),
        Some(residuals.meta.local_mse),
    )
    .unwrap();
    assert!(
        baseline.parity.starts_with("verified"),
        "{}",
        baseline.parity
    );
    assert_eq!(baseline.record_count, 500);
    assert_eq!(baseline.cost_name.as_deref(), Some("MSE"));

    // A grafted IF creature scores under the real scorer exactly like the
    // local forward pass predicts (same kernel in NEAT-AI-core).
    let patch = Patch::new(0, Node::stump(1, 0.0, 0.0, 0.3), Provenance::default());
    let grafted = graft_patch(&incumbent.creature, &patch).unwrap().creature;
    let mlp = small_mlp(3);
    let deep_patch = Patch::new(
        0,
        Node::Split {
            condition: neat_ai_forests::patch::Condition::axis(0, 0.5),
            left: Box::new(Node::stump(1, 0.0, -0.1, 0.0)),
            right: Box::new(Node::stump(2, 0.4, 0.0, 0.2)),
        },
        Provenance::default(),
    );
    // Two stacked patches, so the constant policy actually changes the neuron
    // count: shared creates three for the creature, per-patch three per patch.
    let deep_patches = [
        deep_patch,
        Patch::new(0, Node::stump(1, 0.25, 0.05, 0.0), Provenance::default()),
    ];
    let deep = graft_patches_with(&mlp, &deep_patches, GraftConstants::Shared)
        .unwrap()
        .0;
    // Issue #56 — the same two patches with per-patch constants: three more
    // constant neurons, not one more synapse.
    let deep_per_patch = graft_patches_with(&mlp, &deep_patches, GraftConstants::PerPatch)
        .unwrap()
        .0;
    assert_eq!(deep_per_patch.neurons.len(), deep.neurons.len() + 3);
    assert_eq!(deep_per_patch.synapses.len(), deep.synapses.len());
    // Issue #140 — a refinement of an already-refined leaf on the MLP.
    let refined = neat_ai_forests::refine::fixtures::refined_twice(&small_mlp(3));
    // Issue #137 — leaves moved through their constant edges, as a refit does.
    let refitted = neat_ai_forests::refit::fixtures::refitted(&small_mlp(3));
    // Issue #139 — a tree whose conditions read an evolved hidden neuron.
    let hidden_split = neat_ai_forests::hidden::fixtures::hidden_split(&small_mlp(3), "hidden-1");
    // Issue #138 — linear leaves: one input in both branch roles with
    // opposite signs, and one input in all three roles of one `IF`.
    let linear_leaves = neat_ai_forests::linear::fixtures::linear_stumps(&small_mlp(3));
    let dir = tmp.path().join("cohort");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, c) in [
        ("baseline", &incumbent.creature),
        ("stump", &grafted),
        ("mlp", &mlp),
        ("deep", &deep),
        ("deep-per-patch", &deep_per_patch),
        ("refined", &refined),
        ("refitted", &refitted),
        ("hidden-split", &hidden_split),
        ("linear-leaves", &linear_leaves),
    ] {
        std::fs::write(
            dir.join(format!("{name}.json")),
            neat_core::creature_to_json(c).unwrap(),
        )
        .unwrap();
    }
    let results = scorer
        .score_directory(&dir, &train, ScorerMode::Full)
        .unwrap();
    for (name, c) in [
        ("baseline", &incumbent.creature),
        ("stump", &grafted),
        ("mlp", &mlp),
        ("deep", &deep),
        ("deep-per-patch", &deep_per_patch),
        ("refined", &refined),
        ("refitted", &refitted),
        ("hidden-split", &hidden_split),
        ("linear-leaves", &linear_leaves),
    ] {
        let inc = Incumbent::from_creature(c.clone(), name).unwrap();
        let local = neat_ai_forests::residuals::compute_residuals(&inc, &train, &corpus, 128, 1)
            .unwrap()
            .meta
            .local_mse;
        let scored = results[name].error;
        assert!(
            (local - scored).abs() <= 1e-7 + 1e-4 * scored.abs(),
            "{name}: local {local} vs scorer {scored}"
        );
    }
    // Issue #56 — per-patch constants are numerically free: the authoritative
    // scorer reports the identical error. Only the complexity term sees the
    // three extra constant neurons, so the score is at most a hair lower.
    assert_eq!(
        results["deep-per-patch"].error, results["deep"].error,
        "per-patch constants changed the authoritative error"
    );
    assert!(
        results["deep"].score - results["deep-per-patch"].score >= 0.0
            && results["deep"].score - results["deep-per-patch"].score < 1e-6,
        "deep {} vs per-patch {}",
        results["deep"].score,
        results["deep-per-patch"].score
    );
    // The perfect stump removes the residual entirely.
    assert!(results["stump"].error < 1e-9, "{}", results["stump"].error);
    assert!(results["stump"].score > results["baseline"].score);
    // Sample mode is accepted and reports its rate.
    let sampled = scorer
        .score_directory(
            &dir,
            &train,
            ScorerMode::Sample {
                rate: 0.2,
                phase: 1,
            },
        )
        .unwrap();
    assert_eq!(sampled["baseline"].sample_rate, Some(0.2));
    assert!(sampled["baseline"].record_count < 500);
}

#[test]
fn real_scorer_end_to_end_loop_accepts_a_verified_stump() {
    let bin = require_scorer!();
    let tmp = tempfile::tempdir().unwrap();
    let (creature, train) = fixture(tmp.path(), 800);
    let original = std::fs::read_to_string(&creature).unwrap();
    let cfg = ForestsConfig {
        creature: creature.clone(),
        training_data: train,
        output_dir: tmp.path().join("out"),
        scorer_path: bin.clone(),
        scorer_args: vec!["--gpu".into(), "off".into()],
        seed: Some(11),
        max_iterations: Some(2),
        timeout: Duration::from_secs(600),
        search_records: 0,
        min_leaf_records: 10.0,
        bins: 32,
        screen_sample_rate: Some(0.5),
        promote_count: 4,
        candidates: 16,
        random_candidates: 2,
        ..Default::default()
    };
    let scorer = ExternalScorer {
        binary: bin,
        extra_args: cfg.scorer_args.clone(),
    };
    let r = run_forests(&cfg, &scorer, &CancelToken::new()).unwrap();
    assert_eq!(
        std::fs::read_to_string(&creature).unwrap(),
        original,
        "source creature must be untouched"
    );
    assert!(
        r.acceptances >= 1,
        "expected the real scorer to verify the planted stump"
    );
    assert!(r.best_score > r.opening_score);
    let best =
        neat_core::parse_creature_json(&std::fs::read_to_string(&r.best_path).unwrap()).unwrap();
    assert!(
        best.neurons
            .iter()
            .any(|n| n.squash.as_deref() == Some("IF"))
    );
    // best.json is provably scorer-verified: re-score it alone on the full corpus.
    let dir = tmp.path().join("verify");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("baseline.json"),
        neat_core::creature_to_json(&best).unwrap(),
    )
    .unwrap();
    let again = scorer
        .score_directory(&dir, &cfg.training_data, ScorerMode::Full)
        .unwrap();
    assert!(
        (again["baseline"].score - r.best_score).abs() < 1e-9,
        "{} vs {}",
        again["baseline"].score,
        r.best_score
    );
    let lines = read_journal(&r.journal_path).unwrap();
    assert!(
        lines
            .iter()
            .any(|l| matches!(l, JournalLine::Experiment(e) if e.accepted && e.full.is_some()))
    );
}

#[test]
fn real_cohort_matches_standalone_scores_and_ordering_for_generated_grafts() {
    let bin = require_scorer!();
    let tmp = tempfile::tempdir().unwrap();
    let (_, train) = fixture(tmp.path(), 517);
    let scorer = ExternalScorer { binary: bin, extra_args: vec!["--gpu=off".into()] };
    assert!(scorer.supports_cohort().unwrap(), "parity requires a scorer with --cohort");
    for (shape, parent) in [
        ("identity", identity_creature(3, 1)), ("mlp", small_mlp(3)),
        ("if", if_output_creature(3)), ("clamp", min_clamped_if_creature(3)),
    ] {
        let dir = tmp.path().join(shape);
        std::fs::create_dir_all(&dir).unwrap();
        let mut creatures = vec![("baseline".to_string(), parent.clone())];
        for (i, correction) in [-0.3, 0.05, 0.1, 0.3].into_iter().enumerate() {
            let patch = Patch::new(0, Node::stump(1, 0.0, 0.0, correction), Provenance::default());
            creatures.push((format!("candidate-{i}"), graft_patch(&parent, &patch).unwrap().creature));
        }
        for (id, creature) in &creatures {
            std::fs::write(dir.join(format!("{id}.json")), neat_core::creature_to_json(creature).unwrap()).unwrap();
        }
        let cohort = scorer.score_cohort(&dir, &train).unwrap();
        assert_eq!(cohort.execution, CohortExecution::Hosted);
        assert_eq!(cohort.scores.len(), creatures.len());
        let standalone_dir = tmp.path().join(format!("{shape}-standalone"));
        std::fs::create_dir_all(&standalone_dir).unwrap();
        let mut standalone = std::collections::BTreeMap::new();
        for (id, creature) in &creatures {
            std::fs::write(standalone_dir.join("baseline.json"), neat_core::creature_to_json(creature).unwrap()).unwrap();
            let result = scorer.score_directory(&standalone_dir, &train, ScorerMode::Full).unwrap();
            let result = &result["baseline"];
            assert_eq!(cohort.scores[id].score.to_bits(), result.score.to_bits(), "{shape}/{id} score");
            assert_eq!(cohort.scores[id].error.to_bits(), result.error.to_bits(), "{shape}/{id} error");
            assert_eq!(cohort.scores[id].record_count, 517);
            standalone.insert(id.clone(), result.score);
        }
        let order = |scores: &std::collections::BTreeMap<String, f64>| {
            let mut ids: Vec<_> = scores.keys().cloned().collect();
            ids.sort_by(|a, b| scores[b].total_cmp(&scores[a]).then(a.cmp(b)));
            ids
        };
        let exact = cohort.scores.iter().map(|(id, result)| (id.clone(), result.score)).collect();
        assert_eq!(order(&exact), order(&standalone));
    }
}

#[test]
fn real_cohort_directory_fallback_is_reported_and_matches_plain_directory_scoring() {
    let bin = require_scorer!();
    let tmp = tempfile::tempdir().unwrap();
    let (creature, train) = fixture(tmp.path(), 37);
    let scorer = ExternalScorer { binary: bin, extra_args: vec!["--gpu=off".into(), "--cost=MAE".into()] };
    let dir = tmp.path().join("fallback");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(creature, dir.join("baseline.json")).unwrap();
    std::fs::write(dir.join("other.json"), identity_creature_json(3, 1)).unwrap();
    let cohort = scorer.score_cohort(&dir, &train).unwrap();
    assert!(matches!(cohort.execution, CohortExecution::DirectoryFallback { ref reason } if reason.contains("MAE")));
    let plain = scorer.score_directory(&dir, &train, ScorerMode::Full).unwrap();
    assert_eq!(cohort.scores.len(), plain.len());
    for (id, result) in &plain {
        assert_eq!(cohort.scores[id].score, result.score);
        assert_eq!(cohort.scores[id].error, result.error);
        assert_eq!(cohort.scores[id].record_count, 37);
    }
}

#[test]
fn real_exact_loop_accepts_only_a_standalone_verified_winner() {
    let bin = require_scorer!();
    let tmp = tempfile::tempdir().unwrap();
    let (creature, train) = fixture(tmp.path(), 517);
    let cfg = ForestsConfig {
        creature: creature.clone(), training_data: train, output_dir: tmp.path().join("out"),
        ranking_mode: RankingMode::Exact, max_iterations: Some(2), seed: Some(11),
        search_records: 0, min_leaf_records: 10.0, bins: 32, candidates: 48,
        promote_count: 4, screen_sample_rate: Some(0.05), ..Default::default()
    };
    let original = std::fs::read(&creature).unwrap();
    let scorer = ExternalScorer { binary: bin, extra_args: vec!["--gpu=off".into()] };
    let result = run_forests(&cfg, &scorer, &CancelToken::new()).unwrap();
    assert!(result.acceptances > 0);
    assert!(result.best_score > result.opening_score);
    assert_eq!(std::fs::read(creature).unwrap(), original);
    for line in read_journal(&result.journal_path).unwrap() {
        if let JournalLine::Experiment(e) = line {
            assert_eq!(e.ranking.as_ref().unwrap().scored, e.candidates.len() as u64);
            assert_eq!(e.ranking.as_ref().unwrap().execution, CohortExecution::Hosted);
            if e.accepted {
                assert_eq!(e.full.as_ref().unwrap().scored, 1);
                let winner = e.candidates.iter().find(|c| Some(&c.id) == e.winner.as_ref()).unwrap();
                assert_eq!(winner.full_delta, e.improvement);
                assert_eq!(winner.exact_rank, Some(1));
            }
        }
    }
    let verify = tmp.path().join("verify");
    std::fs::create_dir_all(&verify).unwrap();
    std::fs::copy(&result.best_path, verify.join("baseline.json")).unwrap();
    let score = scorer.score_directory(&verify, &cfg.training_data, ScorerMode::Full).unwrap();
    assert!((score["baseline"].score - result.best_score).abs() < 1e-9);
}

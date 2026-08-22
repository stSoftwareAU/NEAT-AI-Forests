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
use neat_ai_forests::config::ForestsConfig;
use neat_ai_forests::config::GraftConstants;
use neat_ai_forests::corpus::{corpus_info, write_bin_file};
use neat_ai_forests::graft::fixtures::{identity_creature_json, small_mlp};
use neat_ai_forests::graft::{graft_patch, graft_patches_with};
use neat_ai_forests::incumbent::Incumbent;
use neat_ai_forests::journal::{JournalLine, read_journal};
use neat_ai_forests::patch::{Node, Patch, Provenance};
use neat_ai_forests::scorer::{DirectoryScorer, ExternalScorer, ScorerMode};
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
    let dir = tmp.path().join("cohort");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, c) in [
        ("baseline", &incumbent.creature),
        ("stump", &grafted),
        ("mlp", &mlp),
        ("deep", &deep),
        ("deep-per-patch", &deep_per_patch),
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

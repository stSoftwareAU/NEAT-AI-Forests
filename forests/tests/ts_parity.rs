//! `rust_scorer` vs NEAT-AI TypeScript `Creature.scoreDir` parity on grafted
//! fixtures (Issue #35).
//!
//! NEAT-AI's TypeScript loader keys synapses by `(from, to)` and collapses
//! duplicates; `rust_scorer` does not. This test grafts stump / depth-2 /
//! oblique patches onto `IF`-output and MLP fixtures, scores each with both
//! engines on a small synthetic corpus, and asserts agreement within `1e-6`.
//!
//! Both constant-ownership policies are covered (Issue #56): a per-patch graft
//! must load and score exactly like the shared one it is otherwise identical to.
//!
//! It needs `deno`, a `rust_scorer` binary (`NEAT_SCORER_BIN` or the sibling
//! build) and `NEAT_AI_TS_ROOT` — a directory whose Deno import map resolves
//! `@stsoftware/neat-ai` (a GRQ or NEAT-AI checkout). Without them it prints
//! a skip notice and passes.

use std::path::{Path, PathBuf};
use std::process::Command;

use neat_ai_forests::config::GraftConstants;
use neat_ai_forests::corpus::write_bin_file;
use neat_ai_forests::graft::fixtures::{if_output_creature, min_clamped_if_creature, small_mlp};
use neat_ai_forests::graft::{graft_patch, graft_patch_with, graft_patches_with};
use neat_ai_forests::patch::{Condition, Node, Patch, Provenance, Term};

fn scorer_binary() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("NEAT_SCORER_BIN") {
        let p = PathBuf::from(p);
        return p.is_file().then_some(p);
    }
    let sibling = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../NEAT-AI-scorer/target/release/rust_scorer");
    sibling.is_file().then_some(sibling)
}

fn deno_ok() -> bool {
    Command::new("deno")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

const PROBE: &str = r#"
import { Creature } from "@stsoftware/neat-ai";
const [dataDir, ...paths] = Deno.args;
for (const path of paths) {
  const raw = JSON.parse(await Deno.readTextFile(path));
  const c = Creature.fromJSON(raw);
  c.validate();
  const loaded = c.exportJSON().synapses.length;
  const r = await c.scoreDir(dataDir, {});
  console.log(JSON.stringify({ path, score: r.score, error: r.error, jsonSynapses: raw.synapses.length, loadedSynapses: loaded }));
}
"#;

#[test]
fn grafted_fixtures_score_identically_under_rust_and_typescript() {
    let (Some(scorer), Some(ts_root)) = (scorer_binary(), std::env::var_os("NEAT_AI_TS_ROOT"))
    else {
        eprintln!(
            "skipping: needs rust_scorer and NEAT_AI_TS_ROOT (a checkout whose import map provides @stsoftware/neat-ai)"
        );
        return;
    };
    if !deno_ok() {
        eprintln!("skipping: deno not installed");
        return;
    }
    let ts_root = PathBuf::from(ts_root);
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let mut seed = 7u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    // Four inputs so the same corpus covers the clamped fixture (Issue #58),
    // whose clamps read `input-3`.
    let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..2000)
        .map(|_| {
            let x = vec![next(), next(), next(), next()];
            let base = if x[0] > 0.0 {
                2.0 * x[1] + 0.01
            } else {
                -x[2] + 0.01
            };
            (x.clone(), vec![base + if x[1] > 0.2 { 0.3 } else { 0.0 }])
        })
        .collect();
    write_bin_file(&data.join("0.bin"), &recs).unwrap();

    let deep = Node::Split {
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
    let stump = Node::stump(1, 0.2, 0.0, 0.3);
    let cases = [
        ("if-base", if_output_creature(4)),
        (
            "if-stump",
            graft_patch(
                &if_output_creature(4),
                &Patch::new(0, stump.clone(), Provenance::default()),
            )
            .unwrap()
            .creature,
        ),
        (
            "if-deep",
            graft_patch(
                &if_output_creature(4),
                &Patch::new(0, deep.clone(), Provenance::default()),
            )
            .unwrap()
            .creature,
        ),
        ("mlp-base", small_mlp(4)),
        (
            "mlp-deep",
            graft_patch(
                &small_mlp(4),
                &Patch::new(0, deep.clone(), Provenance::default()),
            )
            .unwrap()
            .creature,
        ),
        // Issue #56 — the same two patches with per-patch constants. The
        // creatures differ only in which bias-1 constants the `IF` nodes read,
        // so both engines must return the shared variant's score exactly.
        (
            "if-stump-per-patch",
            graft_patch_with(
                &if_output_creature(4),
                &Patch::new(0, stump.clone(), Provenance::default()),
                GraftConstants::PerPatch,
            )
            .unwrap()
            .creature,
        ),
        // Two stacked patches, where the two policies really do build
        // different creatures: three bias-1 constants against six.
        (
            "mlp-two",
            graft_patches_with(
                &small_mlp(4),
                &[
                    Patch::new(0, deep.clone(), Provenance::default()),
                    Patch::new(0, stump.clone(), Provenance::default()),
                ],
                GraftConstants::Shared,
            )
            .unwrap()
            .0,
        ),
        // Issue #68 — the default graft's typed pair, scored rather than only
        // counted: one source into the `IF` output as both roles must give the
        // same score in both engines, not merely load with every synapse.
        (
            "if-typed-pair",
            graft_patch(
                &if_output_creature(4),
                &Patch::new(0, Node::stump(2, -0.1, 0.2, 0.0), Provenance::default()),
            )
            .unwrap()
            .creature,
        ),
        // Issue #58 — a `MINIMUM`-clamped output: the graft attaches behind the
        // clamps and scales its outward edge, a shape neither engine had seen.
        ("clamp-base", min_clamped_if_creature(4)),
        (
            "clamp-stump",
            graft_patch(
                &min_clamped_if_creature(4),
                &Patch::new(0, stump.clone(), Provenance::default()),
            )
            .unwrap()
            .creature,
        ),
        (
            "clamp-deep",
            graft_patch(
                &min_clamped_if_creature(4),
                &Patch::new(0, deep.clone(), Provenance::default()),
            )
            .unwrap()
            .creature,
        ),
        (
            "mlp-two-per-patch",
            graft_patches_with(
                &small_mlp(4),
                &[
                    Patch::new(0, deep, Provenance::default()),
                    Patch::new(0, stump, Provenance::default()),
                ],
                GraftConstants::PerPatch,
            )
            .unwrap()
            .0,
        ),
    ];
    let mut paths = Vec::new();
    let mut rust_scores = Vec::new();
    for (name, creature) in &cases {
        let dir = tmp.path().join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("baseline.json");
        std::fs::write(&path, neat_core::creature_to_json_pretty(creature).unwrap()).unwrap();
        let out = Command::new(&scorer)
            .args(["--gpu", "off"])
            .arg(&dir)
            .arg(&data)
            .output()
            .expect("run rust_scorer");
        assert!(
            out.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        rust_scores.push(v["baseline"]["score"].as_f64().unwrap());
        paths.push(path);
    }

    let probe = ts_root.join(".forests-ts-parity-probe.ts");
    std::fs::write(&probe, PROBE).unwrap();
    let out = Command::new("deno")
        .current_dir(&ts_root)
        .args([
            "run",
            "--no-prompt",
            "--allow-read",
            "--allow-env",
            "--allow-import",
            // NEAT-AI fetches its WASM activation bundle at import time; the
            // module graph itself already comes over the network via
            // `--allow-import`.
            "--allow-net",
        ])
        .arg(&probe)
        .arg(&data)
        .args(&paths)
        .output()
        .expect("run deno");
    let _ = std::fs::remove_file(&probe);
    assert!(
        out.status.success(),
        "deno failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let lines: Vec<serde_json::Value> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.starts_with('{'))
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(
        lines.len(),
        cases.len(),
        "stdout: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    let mut by_name = std::collections::HashMap::new();
    for ((name, _), (rust, ts)) in cases.iter().zip(rust_scores.iter().zip(&lines)) {
        assert_eq!(
            ts["jsonSynapses"], ts["loadedSynapses"],
            "{name}: TypeScript dropped synapses on load"
        );
        let ts_score = ts["score"].as_f64().unwrap();
        assert!(
            (rust - ts_score).abs() < 1e-6,
            "{name}: rust {rust} vs typescript {ts_score}"
        );
        by_name.insert(*name, (*rust, ts_score, ts["error"].as_f64().unwrap()));
    }
    // Issue #56 — per-patch constants do not change what the creature computes:
    // the TypeScript error is bit-identical to the shared variant's, and the
    // score moves only by NEAT-AI's complexity term for the extra constant
    // neurons — by the same amount in both engines.
    for (shared, per_patch) in [
        ("if-stump", "if-stump-per-patch"),
        ("mlp-two", "mlp-two-per-patch"),
    ] {
        let (a_rust, a_ts, a_err) = by_name[shared];
        let (b_rust, b_ts, b_err) = by_name[per_patch];
        assert_eq!(
            a_err, b_err,
            "{per_patch}: per-patch constants changed the error"
        );
        let (d_rust, d_ts) = (a_rust - b_rust, a_ts - b_ts);
        assert!(
            (d_rust - d_ts).abs() < 1e-9,
            "{per_patch}: rust penalty {d_rust} vs typescript {d_ts}"
        );
        assert!(
            (0.0..1e-6).contains(&d_rust),
            "{per_patch}: score moved by {d_rust}, more than a complexity term"
        );
    }
}

/// Issue #68 — the default graft emits one source into an `IF` target as both
/// `positive` and `negative`. `neat_core` 0.10.6 accepts it, `rust_scorer` sums
/// both roles, and @stsoftware/neat-ai **6.6.40** keeps both on load
/// (NEAT-AI#3873).
///
/// Anything older silently keeps one — a six-synapse creature loading as five,
/// with no error raised — so the same JSON would mean two different things in
/// the two engines. This test is what stops that regressing: it asserts
/// TypeScript loads every synapse of a default graft, so a fleet that drops
/// back to an older pin fails here rather than in a check-in nobody can explain.
#[test]
fn typescript_keeps_both_roles_of_a_typed_pair() {
    let (Some(_scorer), Some(ts_root)) = (scorer_binary(), std::env::var_os("NEAT_AI_TS_ROOT"))
    else {
        eprintln!("skipping: needs rust_scorer and NEAT_AI_TS_ROOT");
        return;
    };
    if !deno_ok() {
        eprintln!("skipping: deno not installed");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let creature = graft_patch(
        &if_output_creature(4),
        &Patch::new(0, Node::stump(1, 0.2, 0.0, 0.3), Provenance::default()),
    )
    .expect("the default graft emits a typed pair into an IF target")
    .creature;
    let path = tmp.path().join("typed-pair.json");
    std::fs::write(
        &path,
        neat_core::creature_to_json_pretty(&creature).unwrap(),
    )
    .unwrap();

    let ts_root = PathBuf::from(ts_root);
    let probe = ts_root.join(".forests-typed-pair-probe.ts");
    std::fs::write(
        &probe,
        r#"
import { Creature } from "@stsoftware/neat-ai";
const raw = JSON.parse(await Deno.readTextFile(Deno.args[0]));
const c = Creature.fromJSON(raw);
c.validate();
console.log(JSON.stringify({ json: raw.synapses.length, loaded: c.exportJSON().synapses.length }));
"#,
    )
    .unwrap();
    let out = Command::new("deno")
        .current_dir(&ts_root)
        .args([
            "run",
            "--no-prompt",
            "--allow-read",
            "--allow-env",
            "--allow-import",
            "--allow-net",
        ])
        .arg(&probe)
        .arg(&path)
        .output()
        .expect("run deno");
    let _ = std::fs::remove_file(&probe);
    assert!(
        out.status.success(),
        "deno failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let line = String::from_utf8_lossy(&out.stdout)
        .lines()
        .find(|l| l.starts_with('{'))
        .expect("probe output")
        .to_string();
    let v: serde_json::Value = serde_json::from_str(&line).unwrap();
    let (json, loaded) = (v["json"].as_u64().unwrap(), v["loaded"].as_u64().unwrap());
    assert_eq!(
        loaded,
        json,
        "TypeScript dropped {} of {json} synapses. A source feeding an `IF` \
         target as two roles needs @stsoftware/neat-ai 6.6.40 or newer \
         (NEAT-AI#3873); on an older pin the two engines disagree about what \
         this creature computes, and every check-in is gated on them agreeing.",
        json - loaded
    );
}

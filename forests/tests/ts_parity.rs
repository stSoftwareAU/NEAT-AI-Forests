//! `rust_scorer` vs NEAT-AI TypeScript `Creature.scoreDir` parity on grafted
//! fixtures (Issue #35).
//!
//! NEAT-AI's TypeScript loader keys synapses by `(from, to)` and collapses
//! duplicates; `rust_scorer` does not. This test grafts stump / depth-2 /
//! oblique patches onto `IF`-output and MLP fixtures, scores each with both
//! engines on a small synthetic corpus, and asserts agreement within `1e-6`.
//!
//! It needs `deno`, a `rust_scorer` binary (`NEAT_SCORER_BIN` or the sibling
//! build) and `NEAT_AI_TS_ROOT` — a directory whose Deno import map resolves
//! `@stsoftware/neat-ai` (a GRQ or NEAT-AI checkout). Without them it prints
//! a skip notice and passes.

use std::path::{Path, PathBuf};
use std::process::Command;

use neat_ai_forests::corpus::write_bin_file;
use neat_ai_forests::graft::fixtures::{if_output_creature, small_mlp};
use neat_ai_forests::graft::graft_patch;
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
  console.log(JSON.stringify({ path, score: r.score, jsonSynapses: raw.synapses.length, loadedSynapses: loaded }));
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
    let recs: Vec<(Vec<f32>, Vec<f32>)> = (0..2000)
        .map(|_| {
            let x = vec![next(), next(), next()];
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
        ("if-base", if_output_creature(3)),
        (
            "if-stump",
            graft_patch(
                &if_output_creature(3),
                &Patch::new(0, stump.clone(), Provenance::default()),
            )
            .unwrap()
            .creature,
        ),
        (
            "if-deep",
            graft_patch(
                &if_output_creature(3),
                &Patch::new(0, deep.clone(), Provenance::default()),
            )
            .unwrap()
            .creature,
        ),
        ("mlp-base", small_mlp(3)),
        (
            "mlp-deep",
            graft_patch(&small_mlp(3), &Patch::new(0, deep, Provenance::default()))
                .unwrap()
                .creature,
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
    }
}

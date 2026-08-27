//! README-as-contract tests.
//!
//! The README documents the tool as built: every long flag the binary accepts
//! must appear in it, the README must not advertise flags the binary lacks,
//! the charter sections the project was founded on must survive, and the
//! repository-layout tree must list every source file.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

fn readme() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../README.md");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Top-level `--help` plus each subcommand's, so a flag that only exists under
/// a subcommand still has to be documented — and cannot be documented if it
/// does not exist (Issue #61 added `prune-learnings`, whose flags live there).
fn help() -> String {
    let mut all = help_for(&[]);
    for sub in [
        "report",
        "export-matrix",
        "import-xgboost",
        "prune-learnings",
    ] {
        all.push('\n');
        all.push_str(&help_for(&[sub]));
    }
    all
}

fn help_for(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_neat_ai_forests"))
        .args(args)
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(out.status.success(), "{args:?} --help failed");
    String::from_utf8(out.stdout).unwrap()
}

/// The body of a `## `-level section, up to the next `## ` heading.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let start = doc
        .find(heading)
        .unwrap_or_else(|| panic!("README has no {heading:?} section"));
    let rest = &doc[start..];
    let end = rest[3..].find("\n## ").map_or(rest.len(), |i| i + 3);
    &rest[..end]
}

fn long_flags(text: &str) -> BTreeSet<String> {
    let bytes = text.as_bytes();
    let mut out = BTreeSet::new();
    let mut i = 0;
    while i + 2 < bytes.len() {
        let boundary = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'-');
        if boundary && bytes[i] == b'-' && bytes[i + 1] == b'-' && bytes[i + 2].is_ascii_lowercase()
        {
            let mut j = i + 2;
            while j < bytes.len()
                && (bytes[j].is_ascii_lowercase() || bytes[j].is_ascii_digit() || bytes[j] == b'-')
            {
                j += 1;
            }
            let flag = text[i..j].trim_end_matches('-').to_string();
            out.insert(flag);
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Flags of other tools the README legitimately quotes.
const FOREIGN_FLAGS: &[&str] = &[
    "--sample-rate",
    "--sample-phase",
    "--cost",
    "--release",
    "--features",
    "--check",
    "--all-features",
    "--all-targets",
    "--workspace",
    "--example",
    "--no-deps",
    "--locked",
];

#[test]
fn readme_documents_every_cli_flag() {
    let documented = long_flags(&readme());
    let missing: Vec<String> = long_flags(&help())
        .into_iter()
        .filter(|f| !matches!(f.as_str(), "--help" | "--version"))
        .filter(|f| !documented.contains(f))
        .collect();
    assert!(
        missing.is_empty(),
        "README.md does not document these CLI flags: {missing:?}"
    );
}

#[test]
fn readme_mentions_no_unknown_flags() {
    let known = long_flags(&help());
    let unknown: Vec<String> = long_flags(&readme())
        .into_iter()
        .filter(|f| !known.contains(f))
        .filter(|f| !FOREIGN_FLAGS.contains(&f.as_str()))
        .collect();
    assert!(
        unknown.is_empty(),
        "README.md documents flags the binary does not accept: {unknown:?}"
    );
}

#[test]
fn charter_sections_survive() {
    let r = readme();
    for needle in [
        "Find all the dirty tricks that uncover real improvements — but trust only the scorer.",
        "## Safety invariants",
        "The supplied incumbent is immutable.",
        "Full-corpus NEAT-AI-scorer is the final authority.",
        "`best.json` may never be worse than the opening scorer-verified baseline.",
        "experimental Rust optimiser for already-fit",
        "is **not** an attempt to replace that creature",
        "## Non-goals",
        "scorer-verified improvement per wall-clock hour",
        "## Planned phases",
    ] {
        assert!(r.contains(needle), "README lost charter text: {needle:?}");
    }
    assert!(r.contains("neat_ai_forests report"));
}

/// Issue #93: every mechanism in the pipeline has a name in the research
/// literature, and the README has to name it — in a section of its own, with a
/// citation per mechanism.
#[test]
fn literature_section_cites_the_research() {
    let r = readme();
    let lit = section(&r, "## Where this sits in the literature");
    for citation in [
        // residual loop → gradient boosting, with a network as the base model
        "Friedman 2001",
        "Badirli",
        "GrowNet",
        "Kontschieder",
        "Popov",
        // the XGBoost control is the incumbent method for this problem shape
        "Chen & Guestrin",
        // the graft → automated software transplantation / genetic improvement
        "Barr et al. 2015",
        "Automated Software Transplantation",
        "Petke",
        // two-phase screening → racing
        "Maron & Moore",
        "Birattari",
        "Jamieson",
        "Li et al. 2017",
        "Hyperband",
        // learnings cache → memory-based search
        "Glover 1986",
        "Fialho",
        // splits
        "Murthy",
        "OC1",
        "Breiman 2001",
    ] {
        assert!(
            lit.contains(citation),
            "'Where this sits in the literature' omits {citation:?}"
        );
    }
}

/// Issue #93: naming the research must not cost the house terminology.
#[test]
fn house_terminology_survives_the_literature_section() {
    let r = readme();
    for needle in ["dirty tricks", "trust only the scorer", "🌳"] {
        assert!(r.contains(needle), "README lost house term: {needle:?}");
    }
}

/// Issue #93: the exposure that thousands of acceptances against one corpus
/// creates is adaptive data analysis, and the README must say so — next to the
/// invariants that promise the scorer is the final authority — and answer the
/// holdout question outright.
#[test]
fn adaptive_data_analysis_exposure_is_stated_with_the_holdout_answer() {
    let r = readme();
    let invariants = section(&r, "## Safety invariants");
    for needle in [
        "adaptive data analysis",
        "Dwork",
        "Blum",
        "held back from every optimiser",
        "no holdout",
    ] {
        assert!(
            invariants.contains(needle),
            "the safety invariants do not state the adaptive-overfitting exposure: {needle:?}"
        );
    }
}

#[test]
fn section_body_stops_at_the_next_heading() {
    let doc = "# t\n\n## A\n\nbody a\n\n## B\n\nbody b\n";
    assert!(section(doc, "## A").contains("body a"));
    assert!(!section(doc, "## A").contains("body b"));
    assert!(section(doc, "## B").contains("body b"));
}

#[test]
fn repository_layout_lists_every_source_file() {
    let r = readme();
    let tree = section(&r, "## Repository layout");
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for entry in std::fs::read_dir(&src).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().to_string();
        assert!(
            tree.contains(&name),
            "README repository layout omits forests/src/{name}"
        );
    }
    let docs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs");
    for entry in std::fs::read_dir(&docs).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().to_string();
        assert!(
            r.contains(&format!("docs/{name}")),
            "README does not link docs/{name}"
        );
    }
}

#[test]
fn long_flags_extracts_flags_and_ignores_prose_dashes() {
    let f = long_flags("use --seed 1 and --output-dir x -- not a—flag; `--scorer-arg=--gpu=off`");
    assert!(
        f.contains("--seed")
            && f.contains("--output-dir")
            && f.contains("--scorer-arg")
            && f.contains("--gpu")
    );
    assert_eq!(f.len(), 4);
}

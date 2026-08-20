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

fn help() -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_neat_ai_forests"))
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(out.status.success());
    String::from_utf8(out.stdout).unwrap()
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
    "--depth",
    "--rounds",
    "--out",
    "--dump",
    "--max-records",
    "--output",
    "--allow-missing-divergence",
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

#[test]
fn repository_layout_lists_every_source_file() {
    let r = readme();
    let start = r.find("## Repository layout").expect("layout section");
    let section = &r[start..];
    let end = section[3..].find("\n## ").map_or(section.len(), |i| i + 3);
    let tree = &section[..end];
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

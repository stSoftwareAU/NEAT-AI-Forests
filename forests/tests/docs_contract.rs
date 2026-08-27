//! Docs-as-contract tests for `docs/benchmarks.md` (Issue #93).
//!
//! The measured economics are described in house terms; the mechanisms behind
//! them have names in the research literature. The screening description in
//! particular must cite the racing literature and state the effect size the
//! sampled screen is actually powered to resolve, so nobody reads a screen
//! rejection as a verdict.

use std::path::Path;

fn benchmarks() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/benchmarks.md");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The body of a `### `-level section, up to the next heading of any level.
fn subsection<'a>(doc: &'a str, heading: &'a str) -> &'a str {
    let start = doc
        .find(heading)
        .unwrap_or_else(|| panic!("docs/benchmarks.md has no {heading:?} section"));
    let rest = &doc[start..];
    let end = rest[4..].find("\n## ").map_or(rest.len(), |i| i + 4);
    &rest[..end]
}

#[test]
fn screen_description_cites_the_racing_literature() {
    let doc = benchmarks();
    let screen = subsection(&doc, "### Where the screen sits");
    for citation in [
        "Maron & Moore",
        "Birattari",
        "Jamieson",
        "Hyperband",
        "racing",
    ] {
        assert!(
            screen.contains(citation),
            "the screening description omits {citation:?}"
        );
    }
}

#[test]
fn screen_description_states_the_effect_size_it_is_powered_for() {
    let doc = benchmarks();
    let screen = subsection(&doc, "### Where the screen sits");
    for needle in [
        // the sample the screen actually sees, and the effects it must resolve
        "5 %",
        "1e-4",
        "powered",
        // the measured false-negative evidence that the power is not there
        "exploratory bypass",
    ] {
        assert!(
            screen.contains(needle),
            "the screening description does not state its power: {needle:?}"
        );
    }
}

#[test]
fn shrinkage_result_is_attributed() {
    let doc = benchmarks();
    let shrinkage = subsection(&doc, "### Shrinkage");
    assert!(
        shrinkage.contains("Friedman"),
        "the shrinkage result is not attributed to the boosting literature"
    );
}

#[test]
fn subsection_body_stops_at_the_next_top_level_heading() {
    let doc = "## A\n\n### one\n\nbody one\n\n## B\n\nbody b\n";
    let one = subsection(doc, "### one");
    assert!(one.contains("body one"));
    assert!(!one.contains("body b"));
}

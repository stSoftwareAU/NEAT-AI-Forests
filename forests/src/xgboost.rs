//! XGBoost as an external scientific control (Issue #13).
//!
//! XGBoost is **not** a runtime dependency. Forests exports a reproducible
//! training matrix (`export-matrix`), `scripts/xgboost-control.py` trains
//! shallow boosted trees on the incumbent's correction-space residuals, and
//! `import-xgboost` converts the dumped trees into ordinary Forest patches
//! that go through the very same graft → screen → full-scorer path as native
//! candidates. An XGBoost training metric never proves anything.
//!
//! ## Semantics mapping
//!
//! XGBoost routes `x < split_condition` to `yes`, `x >= split_condition` to
//! `no`, and `NaN` to `missing`. The `IF` kernel routes `x > t` right. With
//! `t = prev_f32(split_condition)` we have `x > t ⇔ x >= split_condition` for
//! finite `f32` `x`, so `yes → left`, `no → right` **exactly**. `NaN` always
//! goes left in a creature, so a node whose `missing` child is not its `yes`
//! child is rejected (with the reason recorded) unless the caller accepts the
//! documented divergence.

use std::io::Write;
use std::path::Path;

use neat_core::training_data::TrainingDataConfig;
use serde::Deserialize;

use crate::corpus::for_each_chunk;
use crate::patch::{Condition, Node, Patch, Provenance};
use crate::residuals::ResidualCache;

/// Largest `f32` strictly below `x` (for finite `x`).
pub fn prev_f32(x: f32) -> f32 {
    if x.is_nan() || x == f32::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f32::from_bits(1);
    }
    let bits = x.to_bits();
    f32::from_bits(if x > 0.0 { bits - 1 } else { bits + 1 })
}

/// One node of an XGBoost JSON dump (`booster.get_dump(dump_format="json")`).
#[derive(Debug, Clone, Deserialize)]
pub struct DumpNode {
    /// Node id.
    pub nodeid: u32,
    /// Split feature (`f12` or a column name `f12` from the export header).
    #[serde(default)]
    pub split: Option<String>,
    /// Threshold.
    #[serde(default)]
    pub split_condition: Option<f64>,
    /// `yes` child id (`x < split_condition`).
    #[serde(default)]
    pub yes: Option<u32>,
    /// `no` child id.
    #[serde(default)]
    pub no: Option<u32>,
    /// `missing` child id.
    #[serde(default)]
    pub missing: Option<u32>,
    /// Leaf value (already scaled by the learning rate).
    #[serde(default)]
    pub leaf: Option<f64>,
    /// Children.
    #[serde(default)]
    pub children: Vec<DumpNode>,
}

/// Why a tree could not be converted.
#[derive(Debug, Clone, PartialEq)]
pub struct Rejected {
    /// Tree index in the dump.
    pub tree: usize,
    /// Reason.
    pub reason: String,
}

fn child(node: &DumpNode, id: u32) -> Option<&DumpNode> {
    node.children.iter().find(|c| c.nodeid == id)
}

fn convert_node(
    node: &DumpNode,
    allow_missing_divergence: bool,
    input: usize,
) -> Result<Node, String> {
    if let Some(v) = node.leaf {
        return Ok(Node::leaf(v as f32));
    }
    let split = node.split.as_deref().ok_or("split node without `split`")?;
    let feature: usize = split
        .strip_prefix('f')
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("cannot parse feature name `{split}` (expected fN)"))?;
    if feature >= input {
        return Err(format!("feature {feature} >= input width {input}"));
    }
    let t = node
        .split_condition
        .ok_or("split node without `split_condition`")? as f32;
    let yes = node.yes.ok_or("split node without `yes`")?;
    let no = node.no.ok_or("split node without `no`")?;
    if let Some(m) = node.missing
        && m != yes
        && !allow_missing_divergence
    {
        return Err(format!(
            "node {} routes missing values to the `no` branch; the IF kernel routes NaN left (yes)",
            node.nodeid
        ));
    }
    let left = child(node, yes).ok_or("missing yes child")?;
    let right = child(node, no).ok_or("missing no child")?;
    Ok(Node::Split {
        condition: Condition::axis(feature, prev_f32(t)),
        left: Box::new(convert_node(left, allow_missing_divergence, input)?),
        right: Box::new(convert_node(right, allow_missing_divergence, input)?),
    })
}

/// Convert every tree of a dump into patches for `output`.
pub fn convert_dump(
    dump: &[DumpNode],
    input: usize,
    output: usize,
    incumbent_checksum: &str,
    allow_missing_divergence: bool,
    notes: Vec<String>,
) -> (Vec<Patch>, Vec<Rejected>) {
    let mut patches = Vec::new();
    let mut rejected = Vec::new();
    for (i, tree) in dump.iter().enumerate() {
        match convert_node(tree, allow_missing_divergence, input) {
            Ok(root) if matches!(root, Node::Split { .. }) => {
                let mut n = notes.clone();
                n.push(format!("xgboost-tree={i}"));
                if allow_missing_divergence {
                    n.push("missing-divergence-allowed".into());
                }
                patches.push(Patch::new(
                    output,
                    root,
                    Provenance {
                        strategy: "xgboost-import".into(),
                        backend: "external:xgboost".into(),
                        predicted_gain: 0.0,
                        affected_records: 0,
                        search_records: 0,
                        incumbent_checksum: incumbent_checksum.to_string(),
                        seed: None,
                        notes: n,
                    },
                ));
            }
            Ok(_) => rejected.push(Rejected {
                tree: i,
                reason: "tree is a single leaf".into(),
            }),
            Err(reason) => rejected.push(Rejected { tree: i, reason }),
        }
    }
    (patches, rejected)
}

/// Parse a dump file (a JSON array of trees).
pub fn parse_dump(text: &str) -> Result<Vec<DumpNode>, String> {
    serde_json::from_str(text).map_err(|e| format!("xgboost dump: {e}"))
}

/// Export `f0..fN, residual, correction` as CSV (deterministic stride sample).
///
/// Writes `<out>` and `<out>.meta.json` describing the dataset identity.
pub fn export_matrix(
    training_dir: &Path,
    residuals: &ResidualCache,
    input: usize,
    output: usize,
    max_records: u64,
    out: &Path,
) -> Result<u64, String> {
    let outputs = residuals.meta.output_count;
    let config = TrainingDataConfig::new(input, outputs);
    let total = residuals.meta.record_count;
    let stride = if max_records == 0 {
        1
    } else {
        total.div_ceil(max_records).max(1)
    };
    let mut w = std::io::BufWriter::new(
        std::fs::File::create(out).map_err(|e| format!("{}: {e}", out.display()))?,
    );
    let header: Vec<String> = (0..input)
        .map(|i| format!("f{i}"))
        .chain(["residual".into(), "correction".into()])
        .collect();
    writeln!(w, "{}", header.join(",")).map_err(|e| e.to_string())?;
    let mut written = 0u64;
    for_each_chunk(training_dir, &config, 2048, |chunk| {
        for r in 0..chunk.records {
            let idx = chunk.first_index + r as u64;
            if !idx.is_multiple_of(stride) {
                continue;
            }
            let row = &chunk.inputs[r * input..(r + 1) * input];
            let mut line = String::with_capacity(input * 8);
            for (i, v) in row.iter().enumerate() {
                if i > 0 {
                    line.push(',');
                }
                line.push_str(&format!("{v}"));
            }
            line.push_str(&format!(
                ",{},{}",
                residuals.residual_at(idx as usize, output),
                residuals.correction_at(idx as usize, output)
            ));
            writeln!(w, "{line}").map_err(|e| e.to_string())?;
            written += 1;
        }
        Ok(())
    })?;
    w.flush().map_err(|e| e.to_string())?;
    let meta = serde_json::json!({
        "incumbentChecksum": residuals.meta.incumbent_checksum,
        "corpusIdentity": residuals.meta.corpus_identity,
        "corpusRecords": total,
        "exportedRecords": written,
        "stride": stride,
        "output": output,
        "outputSquash": residuals.meta.output_squashes.get(output),
        "target": "correction (pre-squash residual; train XGBoost on this column with base_score=0)",
        "createdAtUnix": crate::incumbent::now_unix(),
    });
    let meta_path = out.with_extension("meta.json");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
        .map_err(|e| format!("{}: {e}", meta_path.display()))?;
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DUMP: &str = r#"[
      {"nodeid":0,"depth":0,"split":"f2","split_condition":0.5,"yes":1,"no":2,"missing":1,"children":[
        {"nodeid":1,"depth":1,"split":"f0","split_condition":-1.25,"yes":3,"no":4,"missing":3,"children":[
          {"nodeid":3,"leaf":0.01},{"nodeid":4,"leaf":-0.02}]},
        {"nodeid":2,"leaf":0.03}]},
      {"nodeid":0,"leaf":0.5},
      {"nodeid":0,"depth":0,"split":"f1","split_condition":0.0,"yes":1,"no":2,"missing":2,"children":[
        {"nodeid":1,"leaf":1.0},{"nodeid":2,"leaf":-1.0}]}
    ]"#;

    #[test]
    fn prev_f32_makes_ge_exact() {
        for t in [0.5f32, -1.25, 0.0, 1e-30, 123456.0] {
            let p = prev_f32(t);
            assert!(p < t);
            assert!(t > p); // t itself goes right
            let below = prev_f32(p);
            assert!(below <= p); // anything below stays left
        }
    }

    #[test]
    fn dump_converts_with_exact_routing_and_rejects_divergent_missing() {
        let dump = parse_dump(DUMP).unwrap();
        let (patches, rejected) = convert_dump(&dump, 3, 0, "abc", false, vec![]);
        assert_eq!(patches.len(), 1);
        assert_eq!(rejected.len(), 2);
        assert!(rejected[0].reason.contains("single leaf"));
        assert!(rejected[1].reason.contains("missing"));
        let p = &patches[0];
        assert_eq!(p.provenance.strategy, "xgboost-import");
        // x2 = 0.5 → xgboost "no" (>=) → right → 0.03
        assert_eq!(p.evaluate(&[0.0, 0.0, 0.5]), 0.03);
        assert_eq!(p.evaluate(&[0.0, 0.0, 0.49]), -0.02); // yes → f0 < -1.25? no → -0.02
        assert_eq!(p.evaluate(&[-2.0, 0.0, 0.49]), 0.01); // yes → yes
        assert_eq!(p.evaluate(&[-1.25, 0.0, 0.49]), -0.02); // x0 == split → xgboost "no"
        assert_eq!(p.evaluate(&[f32::NAN, 0.0, 0.49]), 0.01); // missing → yes
        let (allowed, rej) = convert_dump(&dump, 3, 0, "abc", true, vec![]);
        assert_eq!(allowed.len(), 2);
        assert_eq!(rej.len(), 1);
        assert!(
            allowed[1]
                .provenance
                .notes
                .iter()
                .any(|n| n.contains("missing-divergence"))
        );
        let (_, rej) = convert_dump(&dump, 1, 0, "abc", true, vec![]);
        assert!(rej.iter().any(|r| r.reason.contains("input width")));
    }
}

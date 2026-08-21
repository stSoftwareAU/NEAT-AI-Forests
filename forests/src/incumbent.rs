//! Immutable incumbent creature (Issue #2).
//!
//! The supplied creature is read once, checksummed, round-tripped through
//! NEAT-AI-core and copied byte-for-byte into the run workspace. Nothing in
//! Forests ever writes to the source path.

use std::fmt;
use std::path::{Path, PathBuf};

use neat_core::{CreatureExport, compile_creature, creature_to_json, parse_creature_json};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// SHA-256 hex digest of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

/// Errors establishing the incumbent.
#[derive(Debug)]
pub enum IncumbentError {
    /// Filesystem failure.
    Io(PathBuf, std::io::Error),
    /// NEAT-AI-core rejected the creature.
    Creature(String),
    /// Serialise→parse round trip did not reproduce the creature.
    RoundTrip(String),
    /// Output neurons are not the last `output` entries of `neurons`.
    OutputsNotLast,
    /// Checksum of the workspace copy differs from the source.
    CopyDrift {
        /// Expected (source) checksum.
        expected: String,
        /// Observed checksum of the copy.
        observed: String,
    },
}

impl fmt::Display for IncumbentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(p, e) => write!(f, "{}: {e}", p.display()),
            Self::Creature(m) => write!(f, "creature rejected by NEAT-AI-core: {m}"),
            Self::RoundTrip(m) => {
                write!(f, "creature does not round-trip through NEAT-AI-core: {m}")
            }
            Self::OutputsNotLast => write!(
                f,
                "creature output neurons are not the trailing entries of `neurons`; refusing to graft"
            ),
            Self::CopyDrift { expected, observed } => {
                write!(f, "workspace copy checksum {observed} != source {expected}")
            }
        }
    }
}

impl std::error::Error for IncumbentError {}

/// The immutable starting creature.
#[derive(Debug, Clone)]
pub struct Incumbent {
    /// Where it was read from (never written).
    pub source_path: PathBuf,
    /// Exact source bytes.
    pub text: String,
    /// SHA-256 of `text`.
    pub checksum: String,
    /// Parsed creature.
    pub creature: CreatureExport,
}

/// Metadata written beside the workspace copy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct IncumbentMeta {
    /// SHA-256 of the source bytes.
    pub checksum: String,
    /// Source path as supplied.
    pub source_path: String,
    /// Input width.
    pub input: usize,
    /// Output width.
    pub output: usize,
    /// Listed (non-input) neuron count.
    pub neurons: usize,
    /// Synapse count.
    pub synapses: usize,
    /// `forwardOnly` flag.
    pub forward_only: bool,
    /// Unix seconds when the workspace copy was made.
    pub created_at_unix: u64,
    /// Version of Forests that made the copy.
    pub forests_version: String,
}

/// Validate a parsed creature: compiles, round-trips, outputs are trailing.
pub fn validate_creature(creature: &CreatureExport) -> Result<(), IncumbentError> {
    compile_creature(creature).map_err(|e| IncumbentError::Creature(e.to_string()))?;
    // NEAT-AI's TypeScript loader silently drops repeated (from, to) synapses,
    // so a creature carrying them means two different creatures to the two
    // judges. Refuse rather than optimise something the fleet cannot score.
    crate::graft::check_no_duplicate_synapses(creature)
        .map_err(|e| IncumbentError::Creature(e.to_string()))?;
    // serde_json's default float parser (used by NEAT-AI-core) is not
    // correctly rounded, so a round trip may move a weight by one ulp. The
    // contract is therefore structural equality with a 1e-12 relative
    // tolerance on weights/biases — anything larger is a real defect.
    let json = creature_to_json(creature).map_err(|e| IncumbentError::RoundTrip(e.to_string()))?;
    let again = parse_creature_json(&json).map_err(|e| IncumbentError::RoundTrip(e.to_string()))?;
    if let Err(m) = creatures_equivalent(creature, &again) {
        return Err(IncumbentError::RoundTrip(m));
    }
    let n = creature.neurons.len();
    if n < creature.output
        || creature.neurons[n - creature.output..]
            .iter()
            .any(|x| x.neuron_type != "output")
        || creature.neurons[..n - creature.output]
            .iter()
            .any(|x| x.neuron_type == "output")
    {
        return Err(IncumbentError::OutputsNotLast);
    }
    Ok(())
}

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

/// Structural equality with a 1e-12 relative tolerance on weights and biases.
pub fn creatures_equivalent(a: &CreatureExport, b: &CreatureExport) -> Result<(), String> {
    if a.input != b.input
        || a.output != b.output
        || a.forward_only != b.forward_only
        || a.semantic_version != b.semantic_version
    {
        return Err("header differs".into());
    }
    if a.neurons.len() != b.neurons.len() || a.synapses.len() != b.synapses.len() {
        return Err("neuron/synapse counts differ".into());
    }
    for (x, y) in a.neurons.iter().zip(&b.neurons) {
        if x.uuid != y.uuid
            || x.neuron_type != y.neuron_type
            || x.squash != y.squash
            || !close(x.bias, y.bias)
        {
            return Err(format!("neuron `{}` differs", x.uuid));
        }
    }
    for (x, y) in a.synapses.iter().zip(&b.synapses) {
        if x.from_uuid != y.from_uuid
            || x.to_uuid != y.to_uuid
            || x.synapse_type != y.synapse_type
            || !close(x.weight, y.weight)
        {
            return Err(format!("synapse `{}`→`{}` differs", x.from_uuid, x.to_uuid));
        }
    }
    Ok(())
}

/// Load and validate the incumbent from `path` without modifying it.
pub fn load_incumbent(path: &Path) -> Result<Incumbent, IncumbentError> {
    let text =
        std::fs::read_to_string(path).map_err(|e| IncumbentError::Io(path.to_path_buf(), e))?;
    let creature =
        parse_creature_json(&text).map_err(|e| IncumbentError::Creature(e.to_string()))?;
    validate_creature(&creature)?;
    Ok(Incumbent {
        source_path: path.to_path_buf(),
        checksum: sha256_hex(text.as_bytes()),
        text,
        creature,
    })
}

impl Incumbent {
    /// Build an incumbent from an in-memory creature (used after acceptance).
    pub fn from_creature(creature: CreatureExport, label: &str) -> Result<Self, IncumbentError> {
        validate_creature(&creature)?;
        let text =
            creature_to_json(&creature).map_err(|e| IncumbentError::Creature(e.to_string()))?;
        Ok(Self {
            source_path: PathBuf::from(label),
            checksum: sha256_hex(text.as_bytes()),
            text,
            creature,
        })
    }

    /// Short checksum prefix for file names and logs.
    pub fn short_checksum(&self) -> &str {
        &self.checksum[..12]
    }

    /// Copy the incumbent byte-for-byte into `workspace/incumbent.json` and
    /// write `incumbent.meta.json`. Verifies the copy's checksum.
    pub fn write_workspace(&self, workspace: &Path) -> Result<IncumbentMeta, IncumbentError> {
        std::fs::create_dir_all(workspace)
            .map_err(|e| IncumbentError::Io(workspace.to_path_buf(), e))?;
        let copy = workspace.join("incumbent.json");
        std::fs::write(&copy, &self.text).map_err(|e| IncumbentError::Io(copy.clone(), e))?;
        let observed =
            sha256_hex(&std::fs::read(&copy).map_err(|e| IncumbentError::Io(copy.clone(), e))?);
        if observed != self.checksum {
            return Err(IncumbentError::CopyDrift {
                expected: self.checksum.clone(),
                observed,
            });
        }
        let meta = IncumbentMeta {
            checksum: self.checksum.clone(),
            source_path: self.source_path.display().to_string(),
            input: self.creature.input,
            output: self.creature.output,
            neurons: self.creature.neurons.len(),
            synapses: self.creature.synapses.len(),
            forward_only: self.creature.forward_only,
            created_at_unix: now_unix(),
            forests_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        let meta_path = workspace.join("incumbent.meta.json");
        let json = serde_json::to_string_pretty(&meta)
            .map_err(|e| IncumbentError::Creature(e.to_string()))?;
        std::fs::write(&meta_path, json).map_err(|e| IncumbentError::Io(meta_path, e))?;
        Ok(meta)
    }
}

/// Current unix time in seconds.
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graft::fixtures::identity_creature_json;

    #[test]
    fn source_is_untouched_and_copy_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("creature.json");
        let text = identity_creature_json(2, 1);
        std::fs::write(&src, &text).unwrap();
        let before = std::fs::metadata(&src).unwrap().modified().unwrap();
        let inc = load_incumbent(&src).unwrap();
        let ws = tmp.path().join("ws");
        let meta = inc.write_workspace(&ws).unwrap();
        assert_eq!(std::fs::read_to_string(&src).unwrap(), text);
        assert_eq!(std::fs::metadata(&src).unwrap().modified().unwrap(), before);
        assert_eq!(
            std::fs::read_to_string(ws.join("incumbent.json")).unwrap(),
            text
        );
        assert_eq!(meta.checksum, sha256_hex(text.as_bytes()));
        assert_eq!(meta.input, 2);
    }

    #[test]
    fn malformed_creature_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("bad.json");
        std::fs::write(&src, "{\"input\": 1}").unwrap();
        assert!(matches!(
            load_incumbent(&src),
            Err(IncumbentError::Creature(_))
        ));
    }

    #[test]
    fn duplicate_synapses_are_refused() {
        let mut c = parse_creature_json(&identity_creature_json(2, 1)).unwrap();
        c.synapses.push(c.synapses[0].clone());
        let err = validate_creature(&c).unwrap_err().to_string();
        assert!(err.contains("duplicate synapse"), "{err}");
    }

    #[test]
    fn outputs_must_be_trailing() {
        let mut c = parse_creature_json(&identity_creature_json(1, 1)).unwrap();
        c.neurons.push(neat_core::NeuronExport {
            neuron_type: "hidden".into(),
            uuid: "h".into(),
            bias: 0.0,
            squash: Some("IDENTITY".into()),
        });
        c.synapses.push(neat_core::SynapseExport {
            from_uuid: "input-0".into(),
            to_uuid: "h".into(),
            weight: 1.0,
            synapse_type: None,
        });
        assert!(matches!(
            validate_creature(&c),
            Err(IncumbentError::OutputsNotLast)
        ));
    }
}

//! Creature metadata preservation.
//!
//! `neat_core::CreatureExport` carries only structure. When Forests writes a
//! creature it re-attaches from the source JSON everything that is still
//! meaningful:
//!
//! - creature-level `tags` (with `score`, `error`, `forests` — the commit
//!   subject — and `forests-detail` — the commit body — upserted);
//! - **per-neuron `tags`**, keyed by neuron uuid (discovery / intelligent-design
//!   provenance on mature neurons must not be lost);
//! - a `forests` tag on every neuron Forests itself appended, saying which
//!   run/iteration/patch created it.
//!
//! Deliberately dropped: the creature `uuid` (NEAT-AI derives it from content,
//! which changed) and `memetic` (lineage of a structure that no longer exists).

use std::collections::BTreeMap;

use neat_core::CreatureExport;
use serde_json::{Map, Value};

/// Name/value tag.
#[derive(Debug, Clone, PartialEq)]
pub struct Tag {
    /// Tag name.
    pub name: String,
    /// Tag value (stringified).
    pub value: String,
}

impl Tag {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

fn parse_tags(v: Option<&Value>) -> Vec<Tag> {
    let mut out = Vec::new();
    if let Some(Value::Array(tags)) = v {
        for t in tags {
            if let Value::Object(o) = t
                && let Some(Value::String(name)) = o.get("name")
            {
                let value = match o.get("value") {
                    Some(Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => String::new(),
                };
                out.push(Tag {
                    name: name.clone(),
                    value,
                });
            }
        }
    }
    out
}

fn tags_value(tags: &[Tag]) -> Value {
    Value::Array(
        tags.iter()
            .map(|t| serde_json::json!({"name": t.name, "value": t.value}))
            .collect(),
    )
}

/// Metadata carried alongside a creature.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CreatureMeta {
    /// Creature-level tags in original order.
    pub tags: Vec<Tag>,
    /// Per-neuron tags keyed by neuron uuid.
    pub neuron_tags: BTreeMap<String, Vec<Tag>>,
}

/// Backwards-compatible alias.
pub type CreatureTags = CreatureMeta;

impl CreatureMeta {
    /// Parse creature-level and per-neuron tags from raw creature JSON.
    pub fn from_json(text: &str) -> Self {
        let mut out = Self::default();
        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(text) {
            out.tags = parse_tags(map.get("tags"));
            if let Some(Value::Array(neurons)) = map.get("neurons") {
                for n in neurons {
                    if let Value::Object(o) = n
                        && let Some(Value::String(uuid)) = o.get("uuid")
                    {
                        let tags = parse_tags(o.get("tags"));
                        if !tags.is_empty() {
                            out.neuron_tags.insert(uuid.clone(), tags);
                        }
                    }
                }
            }
        }
        out
    }

    /// Replace or append a creature-level tag.
    pub fn upsert(&mut self, name: &str, value: String) {
        if let Some(t) = self.tags.iter_mut().find(|t| t.name == name) {
            t.value = value;
        } else {
            self.tags.push(Tag {
                name: name.into(),
                value,
            });
        }
    }

    /// Attach tags to a neuron (appending to any it already has).
    pub fn tag_neuron(&mut self, uuid: &str, tags: Vec<Tag>) {
        self.neuron_tags
            .entry(uuid.to_string())
            .or_default()
            .extend(tags);
    }

    /// Count of neurons carrying tags.
    pub fn tagged_neurons(&self) -> usize {
        self.neuron_tags.len()
    }

    /// Serialise `creature` with creature-level and per-neuron tags attached.
    /// Neuron tags whose uuid is not in the creature are dropped silently.
    ///
    /// `memetic` is removed here rather than left to whoever built `creature`.
    /// It is a field `CreatureExport` models, so a source that carries one
    /// round-trips it back out, and it describes a structure a graft has
    /// changed: its bias and weight keys resolve by runtime neuron id, and a
    /// graft shifts every id after the constants it inserts ahead of the first
    /// hidden neuron. A stale record therefore does not merely go out of date,
    /// it silently names *other* neurons. The creature-level `uuid` needs no
    /// removal — `CreatureExport` does not model it, so it is already gone.
    pub fn serialize_with(
        &self,
        creature: &CreatureExport,
        pretty: bool,
    ) -> Result<String, String> {
        let text = neat_core::creature_to_json(creature).map_err(|e| e.to_string())?;
        let mut value: Map<String, Value> =
            serde_json::from_str(&text).map_err(|e| e.to_string())?;
        value.remove("memetic");
        if !self.tags.is_empty() {
            value.insert("tags".into(), tags_value(&self.tags));
        }
        if !self.neuron_tags.is_empty()
            && let Some(Value::Array(neurons)) = value.get_mut("neurons")
        {
            for n in neurons.iter_mut() {
                if let Value::Object(o) = n
                    && let Some(Value::String(uuid)) = o.get("uuid")
                    && let Some(tags) = self.neuron_tags.get(uuid)
                {
                    o.insert("tags".into(), tags_value(tags));
                }
            }
        }
        let v = Value::Object(value);
        if pretty {
            serde_json::to_string_pretty(&v)
        } else {
            serde_json::to_string(&v)
        }
        .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graft::fixtures::identity_creature;

    const SRC: &str = r#"{"input":1,"output":1,"uuid":"abc","memetic":{"x":1},
        "tags":[{"name":"score","value":"0.5"},{"name":"x","value":3}],
        "neurons":[{"type":"output","uuid":"output-0","bias":0,"squash":"IDENTITY",
                    "tags":[{"name":"discovered","value":"ReLU6"},{"name":"intelligentDesign","value":"SELU -> BENT"}]}],
        "synapses":[{"fromUUID":"input-0","toUUID":"output-0","weight":1}]}"#;

    #[test]
    fn creature_and_neuron_tags_survive_uuid_and_memetic_do_not() {
        let mut meta = CreatureMeta::from_json(SRC);
        assert_eq!(meta.tags.len(), 2);
        assert_eq!(meta.tags[1].value, "3");
        assert_eq!(meta.neuron_tags["output-0"].len(), 2);
        meta.upsert("score", "0.6".into());
        meta.upsert("forests", "hi".into());
        meta.tag_neuron("output-0", vec![Tag::new("forests", "touched")]);
        meta.tag_neuron("ghost", vec![Tag::new("forests", "dropped")]);
        let json = meta.serialize_with(&identity_creature(1, 1), true).unwrap();
        let v: Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("uuid").is_none() && v.get("memetic").is_none());
        let back = CreatureMeta::from_json(&json);
        assert_eq!(
            back.tags
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            ["score", "x", "forests"]
        );
        assert_eq!(back.tags[0].value, "0.6");
        let nt = &back.neuron_tags["output-0"];
        assert_eq!(
            nt.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            ["discovered", "intelligentDesign", "forests"]
        );
        assert!(!back.neuron_tags.contains_key("ghost"));
        neat_core::parse_creature_json(&json).unwrap();
    }

    /// `memetic` is a field `CreatureExport` models, so it survives a parse and
    /// would be written straight back out. The creature it describes no longer
    /// exists once a patch is grafted in, and its keys are read positionally by
    /// runtime neuron id — which the graft shifts by inserting constants ahead
    /// of the first hidden neuron. Dropping it is this module's contract, and
    /// the test above cannot see it: its fixture creature carries none.
    #[test]
    fn memetic_is_dropped_from_a_creature_that_carries_one() {
        let mut creature = identity_creature(1, 1);
        creature.memetic = Some(neat_core::MemeticExport {
            biases: [("1".to_string(), 0.25)].into_iter().collect(),
            ..Default::default()
        });
        assert!(
            neat_core::creature_to_json(&creature)
                .unwrap()
                .contains("memetic"),
            "precondition: NEAT-AI-core round-trips memetic"
        );
        let json = CreatureMeta::default()
            .serialize_with(&creature, true)
            .unwrap();
        let v: Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("memetic").is_none(), "{json}");
        neat_core::parse_creature_json(&json).unwrap();
    }
}

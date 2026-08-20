//! Creature metadata (`tags`) preservation.
//!
//! `neat_core::CreatureExport` drops `uuid` and `tags`. When Forests writes a
//! new `best.json` it re-attaches the original tags and upserts `score`,
//! `error` and a `forests` progress tag. The `uuid` is dropped on an accepted
//! candidate because the content changed (NEAT-AI derives it from content).

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

/// Tags parsed from creature JSON.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CreatureTags {
    /// Tags in original order.
    pub tags: Vec<Tag>,
}

impl CreatureTags {
    /// Parse `tags` from raw creature JSON (tolerant of missing/odd shapes).
    pub fn from_json(text: &str) -> Self {
        let mut out = Self::default();
        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(text)
            && let Some(Value::Array(tags)) = map.get("tags")
        {
            for t in tags {
                if let Value::Object(o) = t
                    && let Some(Value::String(name)) = o.get("name")
                {
                    let value = match o.get("value") {
                        Some(Value::String(s)) => s.clone(),
                        Some(v) => v.to_string(),
                        None => String::new(),
                    };
                    out.tags.push(Tag {
                        name: name.clone(),
                        value,
                    });
                }
            }
        }
        out
    }

    /// Replace or append a tag.
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

    /// Serialise `creature` with these tags attached.
    pub fn serialize_with(
        &self,
        creature: &CreatureExport,
        pretty: bool,
    ) -> Result<String, String> {
        let text = neat_core::creature_to_json(creature).map_err(|e| e.to_string())?;
        let mut value: Map<String, Value> =
            serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if !self.tags.is_empty() {
            value.insert(
                "tags".into(),
                Value::Array(
                    self.tags
                        .iter()
                        .map(|t| serde_json::json!({"name": t.name, "value": t.value}))
                        .collect(),
                ),
            );
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

    #[test]
    fn tags_survive_and_upsert_replaces() {
        let mut tags = CreatureTags::from_json(
            r#"{"input":1,"tags":[{"name":"score","value":"0.5"},{"name":"x","value":3}]}"#,
        );
        assert_eq!(tags.tags.len(), 2);
        assert_eq!(tags.tags[1].value, "3");
        tags.upsert("score", "0.6".into());
        tags.upsert("forests", "hi".into());
        let json = tags.serialize_with(&identity_creature(1, 1), true).unwrap();
        let back = CreatureTags::from_json(&json);
        assert_eq!(
            back.tags
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            ["score", "x", "forests"]
        );
        assert_eq!(back.tags[0].value, "0.6");
        neat_core::parse_creature_json(&json).unwrap();
    }
}

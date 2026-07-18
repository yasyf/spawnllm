use std::sync::LazyLock;

use regex_lite::Regex;
use serde::Deserialize;
use serde_json::Value;

use crate::{OpResult, from_input};

static JSON_FENCE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)```(?:json)?\s*\n?(.*?)\n?```").unwrap());

#[derive(Debug, Clone, Deserialize)]
pub struct ExtractInput {
    pub text: String,
}

fn first_json_value(source: &str) -> Option<Value> {
    source
        .char_indices()
        .filter(|&(_, c)| c == '{' || c == '[')
        .find_map(|(index, _)| {
            Value::deserialize(&mut serde_json::Deserializer::from_str(&source[index..])).ok()
        })
}

pub fn extract_json(text: &str) -> Option<Value> {
    JSON_FENCE
        .captures(text)
        .and_then(|caps| caps.get(1))
        .and_then(|fence| first_json_value(fence.as_str()))
        .or_else(|| first_json_value(text))
}

pub(crate) fn dispatch(input: Value) -> OpResult {
    let input = from_input::<ExtractInput>(input)?;
    Ok(serde_json::json!({ "value": extract_json(&input.text) }))
}

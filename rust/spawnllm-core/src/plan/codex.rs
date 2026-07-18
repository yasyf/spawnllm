use std::collections::BTreeMap;
use std::fmt::Write;

use serde_json::{Map, Value};

use crate::wire::{ExecPlan, FileId, InvocationPlan, PlanFile, ReadResultFrom, RunSpec};

pub(super) fn plan(spec: &RunSpec) -> InvocationPlan {
    let config = spec.codex.as_ref();
    let (model, effort) = spec
        .model
        .split_once(':')
        .map_or((spec.model.as_str(), None), |(model, effort)| {
            (model, (!effort.is_empty()).then_some(effort))
        });
    let mut argv = vec![
        "codex".into(),
        "exec".into(),
        "--ephemeral".into(),
        "--sandbox".into(),
        config
            .and_then(|config| config.sandbox.clone())
            .unwrap_or_else(|| "read-only".into()),
        "--skip-git-repo-check".into(),
        "--color".into(),
        "never".into(),
        "--model".into(),
        model.into(),
    ];

    if let Some(effort) = effort {
        push_config(&mut argv, format!("model_reasoning_effort={effort}"));
    }
    match config {
        Some(config) => {
            if let Some(service_tier) = &config.service_tier {
                push_config(&mut argv, format!("service_tier={service_tier}"));
            }
        }
        None => push_config(&mut argv, "service_tier=fast".into()),
    }
    if let Some(instructions) = config.and_then(|config| config.developer_instructions.as_deref()) {
        push_config(
            &mut argv,
            format!(
                "developer_instructions={}",
                python_json_string(instructions, false)
            ),
        );
    }
    if spec.isolated {
        argv.push("--ignore-user-config".into());
    }
    if !spec.agent {
        if !config.is_some_and(|config| config.enable_hooks) {
            push_config(&mut argv, "features.hooks=false".into());
        }
        if !config.is_some_and(|config| config.enable_mcp) {
            push_config(&mut argv, "features.mcp_servers=false".into());
        }
    }

    let mut files = Vec::with_capacity(2);
    if let Some(schema) = &spec.schema {
        argv.extend(["--output-schema".into(), "${file:schema}".into()]);
        files.push(PlanFile {
            id: FileId::Schema,
            suffix: ".json".into(),
            content: Some(match schema {
                Value::String(schema) => schema.clone(),
                schema => python_json_dumps(schema),
            }),
        });
    }
    argv.extend(["-o".into(), "${file:result}".into()]);
    files.push(PlanFile {
        id: FileId::Result,
        suffix: ".json".into(),
        content: None,
    });

    InvocationPlan::Exec(ExecPlan {
        argv,
        stdin: spec.prompt.clone(),
        files,
        stdout_to_file: false,
        read_result_from: ReadResultFrom::FileResult,
        env: BTreeMap::new(),
        needs_claude_isolation: false,
    })
}

fn push_config(argv: &mut Vec<String>, value: String) {
    argv.extend(["-c".into(), value]);
}

fn python_json_dumps(value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => python_json_string(value, true),
        Value::Array(values) => {
            let values = values
                .iter()
                .map(python_json_dumps)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{values}]")
        }
        Value::Object(values) => python_json_object(values),
    }
}

fn python_json_object(values: &Map<String, Value>) -> String {
    let mut entries = values.iter().collect::<Vec<_>>();
    entries.sort_by_key(|(key, _)| schema_key_rank(key));
    let entries = entries
        .into_iter()
        .map(|(key, value)| {
            format!(
                "{}: {}",
                python_json_string(key, true),
                python_json_dumps(value)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{entries}}}")
}

fn schema_key_rank(key: &str) -> u8 {
    match key {
        "type" => 0,
        "properties" => 1,
        "required" => 2,
        _ => 3,
    }
}

fn python_json_string(value: &str, ensure_ascii: bool) -> String {
    let mut encoded = String::with_capacity(value.len() + 2);
    encoded.push('"');
    for character in value.chars() {
        match character {
            '"' => encoded.push_str("\\\""),
            '\\' => encoded.push_str("\\\\"),
            '\u{0008}' => encoded.push_str("\\b"),
            '\u{0009}' => encoded.push_str("\\t"),
            '\u{000a}' => encoded.push_str("\\n"),
            '\u{000c}' => encoded.push_str("\\f"),
            '\u{000d}' => encoded.push_str("\\r"),
            '\u{0000}'..='\u{001f}' => write!(encoded, "\\u{:04x}", character as u32).unwrap(),
            '\u{0080}'.. if ensure_ascii => {
                let codepoint = character as u32;
                if codepoint <= 0xffff {
                    write!(encoded, "\\u{codepoint:04x}").unwrap();
                } else {
                    let codepoint = codepoint - 0x10000;
                    let high = 0xd800 + (codepoint >> 10);
                    let low = 0xdc00 + (codepoint & 0x3ff);
                    write!(encoded, "\\u{high:04x}\\u{low:04x}").unwrap();
                }
            }
            _ => encoded.push(character),
        }
    }
    encoded.push('"');
    encoded
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn developer_instructions_use_toml_safe_json_string_encoding() {
        assert_eq!(
            python_json_string("a\u{007f}b\n\\\" café ☕ 你好", false),
            "\"a\u{007f}b\\n\\\\\\\" café ☕ 你好\""
        );
    }

    #[test]
    fn schema_uses_python_json_dumps_separators_and_order() {
        assert_eq!(
            python_json_dumps(&json!({
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "type": "object"
            })),
            r#"{"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}"#
        );
    }
}

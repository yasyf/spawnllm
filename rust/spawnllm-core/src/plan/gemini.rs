use std::collections::BTreeMap;

use serde_json::Value;

use crate::wire::{ExecPlan, InvocationPlan, ReadResultFrom, RunSpec};

const SCHEMA_PROMPT: &str = "Respond with ONLY a single JSON object that conforms to this JSON Schema. No prose, no explanation, no markdown code fences.\nJSON Schema:";

pub(super) fn plan(spec: &RunSpec) -> InvocationPlan {
    let config = spec.gemini.as_ref();
    let approval_mode = config
        .and_then(|config| config.approval_mode.as_deref())
        .unwrap_or(if spec.agent { "yolo" } else { "default" });
    let mut argv = vec![
        "gemini".into(),
        "--model".into(),
        spec.model.clone(),
        "-o".into(),
        "json".into(),
        "--approval-mode".into(),
        approval_mode.into(),
    ];

    match config.and_then(|config| config.extensions.as_ref()) {
        None if spec.agent => {}
        None => argv.extend(["-e".into(), "none".into()]),
        Some(extensions) => {
            for extension in extensions {
                argv.extend(["-e".into(), extension.clone()]);
            }
        }
    }

    let prompt = match &spec.schema {
        Some(schema) => format!(
            "{}\n\n{SCHEMA_PROMPT}\n{}",
            spec.prompt,
            schema_json(schema)
        ),
        None => spec.prompt.clone(),
    };
    argv.extend(["-p".into(), prompt]);

    InvocationPlan::Exec(ExecPlan {
        argv,
        stdin: String::new(),
        files: Vec::new(),
        stdout_to_file: false,
        read_result_from: ReadResultFrom::Stdout,
        env: BTreeMap::new(),
        needs_claude_isolation: false,
    })
}

fn schema_json(value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => {
            serde_json::to_string(value).expect("string serialization cannot fail")
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(schema_json)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(object) => {
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| {
                schema_key_rank(left)
                    .cmp(&schema_key_rank(right))
                    .then_with(|| left.cmp(right))
            });
            format!(
                "{{{}}}",
                entries
                    .into_iter()
                    .map(|(key, value)| {
                        format!(
                            "{}: {}",
                            serde_json::to_string(key)
                                .expect("object key serialization cannot fail"),
                            schema_json(value)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

fn schema_key_rank(key: &str) -> u8 {
    match key {
        "type" => 0,
        "properties" => 1,
        "required" => 2,
        _ => 3,
    }
}

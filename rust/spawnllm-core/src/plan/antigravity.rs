use std::collections::BTreeMap;

use serde_json::Value;

use crate::wire::{ExecPlan, InvocationPlan, ReadResultFrom, RunSpec};

const SCHEMA_PROMPT: &str = "Respond with ONLY a single JSON object that conforms to this JSON Schema. No prose, no explanation, no markdown code fences.\nJSON Schema:";

pub(super) fn plan(spec: &RunSpec) -> InvocationPlan {
    let mut argv = vec!["agy".into(), "--model".into(), spec.model.clone()];
    if spec.agent {
        argv.push("--dangerously-skip-permissions".into());
    }
    argv.extend(["--print-timeout".into(), "120s".into(), "-p".into()]);

    let prompt = spec.schema.as_ref().map_or_else(
        || spec.prompt.clone(),
        |schema| {
            let schema = match schema {
                Value::String(schema) => schema.clone(),
                schema => format_schema_json(schema),
            };
            format!("{}\n\n{SCHEMA_PROMPT}\n{schema}", spec.prompt)
        },
    );
    argv.push(prompt);

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

fn format_schema_json(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            serde_json::to_string(value).expect("JSON values are serializable")
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(format_schema_json)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => {
            let mut fields = values.iter().collect::<Vec<_>>();
            fields.sort_by_key(|(key, _)| match key.as_str() {
                "type" => 0,
                "properties" => 1,
                "required" => 2,
                _ => 3,
            });
            format!(
                "{{{}}}",
                fields
                    .into_iter()
                    .map(|(key, value)| format!(
                        "{}: {}",
                        serde_json::to_string(key).expect("JSON object keys are serializable"),
                        format_schema_json(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

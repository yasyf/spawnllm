use std::collections::BTreeMap;

use serde_json::Value;

use crate::auth::CLAUDE_API_KEY_VARS;
use crate::wire::{
    ClaudeConfig, ExecPlan, FileId, InvocationPlan, PlanFile, ReadResultFrom, RunSpec,
};

pub(super) fn plan(spec: &RunSpec) -> InvocationPlan {
    let mut argv = vec![
        "claude".into(),
        "-p".into(),
        "--no-session-persistence".into(),
        "--model".into(),
        spec.model.clone(),
    ];
    let mut env = BTreeMap::new();

    if spec.isolated {
        argv.extend(["--setting-sources".into(), String::new()]);
        env.insert("CLAUDE_CONFIG_DIR".into(), "${isolated_config_dir}".into());
    }

    let config = spec.claude.as_ref();
    if spec.isolated || config.is_some_and(|config| config.strict_mcp) {
        argv.push("--strict-mcp-config".into());
    }

    match config {
        Some(config) => append_config(&mut argv, spec, config),
        None if spec.agent => argv.extend([
            "--permission-mode".into(),
            "auto".into(),
            "--max-budget-usd".into(),
            "1".into(),
        ]),
        None => argv.extend(["--system-prompt".into(), String::new()]),
    }

    let schema = spec.schema.as_ref().map(schema_arg);
    if let Some(schema) = schema.filter(|schema| !schema.is_empty()) {
        argv.extend([
            "--json-schema".into(),
            schema,
            "--output-format".into(),
            "json".into(),
        ]);
    } else if let Some(output_format) = config
        .and_then(|config| config.output_format.as_ref())
        .filter(|output_format| !output_format.is_empty())
    {
        argv.extend(["--output-format".into(), output_format.clone()]);
    }

    if config.is_some_and(|config| config.verbose) {
        argv.push("--verbose".into());
    }

    InvocationPlan::Exec(ExecPlan {
        argv,
        stdin: spec.prompt.clone(),
        files: vec![PlanFile {
            id: FileId::Stdout,
            suffix: ".json".into(),
            content: None,
        }],
        stdout_to_file: true,
        read_result_from: ReadResultFrom::Stdout,
        env,
        env_unset: if spec.api_auth {
            Vec::new()
        } else {
            CLAUDE_API_KEY_VARS
                .iter()
                .map(|var| (*var).to_owned())
                .collect()
        },
        needs_claude_isolation: spec.isolated,
    })
}

fn append_config(argv: &mut Vec<String>, spec: &RunSpec, config: &ClaudeConfig) {
    let explicit = config.permission_mode.is_some()
        || config.mcp_config.is_some()
        || config.append_system_prompt.is_some()
        || config.system_prompt.is_some()
        || config.settings.is_some()
        || config
            .disallowed_tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
        || config.strict_mcp;

    if explicit {
        append_value(argv, "--permission-mode", config.permission_mode.as_deref());
        append_value(argv, "--mcp-config", config.mcp_config.as_deref());
        append_values(
            argv,
            "--disallowedTools",
            config.disallowed_tools.as_deref(),
        );
        append_value(
            argv,
            "--append-system-prompt",
            config.append_system_prompt.as_deref(),
        );
        append_value(argv, "--settings", config.settings.as_deref());
        if let Some(max_budget_usd) = config.max_budget_usd {
            argv.extend(["--max-budget-usd".into(), format!("{max_budget_usd:?}")]);
        }
    } else if spec.agent {
        argv.extend([
            "--permission-mode".into(),
            "auto".into(),
            "--max-budget-usd".into(),
            "1".into(),
        ]);
    } else {
        argv.extend(["--system-prompt".into(), String::new()]);
    }

    append_value(argv, "--system-prompt", config.system_prompt.as_deref());
    if let Some(max_turns) = config.max_turns {
        argv.extend(["--max-turns".into(), max_turns.to_string()]);
    }
    if let Some(tools) = &config.tools {
        argv.push("--tools".into());
        if tools.is_empty() {
            argv.push(String::new());
        } else {
            argv.extend(tools.iter().cloned());
        }
    }
    if config.disable_slash_commands {
        argv.push("--disable-slash-commands".into());
    }
}

fn append_value(argv: &mut Vec<String>, flag: &str, value: Option<&str>) {
    if let Some(value) = value {
        argv.extend([flag.into(), value.into()]);
    }
}

fn append_values(argv: &mut Vec<String>, flag: &str, values: Option<&[String]>) {
    if let Some(values) = values.filter(|values| !values.is_empty()) {
        argv.push(flag.into());
        argv.extend(values.iter().cloned());
    }
}

fn schema_arg(schema: &Value) -> String {
    match schema {
        Value::String(schema) => schema.clone(),
        _ => schema_json(schema),
    }
}

fn schema_json(schema: &Value) -> String {
    match schema {
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
        Value::Object(values) => {
            let mut entries = values.iter().collect::<Vec<_>>();
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

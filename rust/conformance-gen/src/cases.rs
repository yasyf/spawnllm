use serde_json::{Value, json};
use spawnllm_core::wire::{ClaudeConfig, CodexConfig, GeminiConfig, OpenAiEndpoint, RunSpec};

pub struct Case {
    pub op: &'static str,
    pub name: String,
    pub input: Value,
}

const HOME_DARWIN: &str = "/Users/testuser";
const HOME_LINUX: &str = "/home/testuser";

fn strs(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

fn schema_value() -> Value {
    json!({"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]})
}

fn spec(prompt: &str, model: &str) -> RunSpec {
    RunSpec {
        prompt: prompt.to_owned(),
        model: model.to_owned(),
        agent: false,
        isolated: true,
        timeout: 180,
        max_attempts: 5,
        schema: None,
        claude: None,
        codex: None,
        gemini: None,
        openai_endpoint: None,
    }
}

fn claude() -> ClaudeConfig {
    ClaudeConfig {
        append_system_prompt: None,
        disable_slash_commands: false,
        disallowed_tools: Some(vec![]),
        max_budget_usd: None,
        max_turns: None,
        mcp_config: None,
        output_format: None,
        permission_mode: None,
        settings: None,
        strict_mcp: false,
        system_prompt: None,
        tools: None,
        verbose: false,
    }
}

fn codex() -> CodexConfig {
    CodexConfig {
        developer_instructions: None,
        enable_hooks: false,
        enable_mcp: false,
        sandbox: None,
        service_tier: Some("fast".to_owned()),
    }
}

fn gemini() -> GeminiConfig {
    GeminiConfig {
        approval_mode: None,
        extensions: None,
    }
}

fn plan_case(name: &str, provider: &'static str, spec: RunSpec) -> Case {
    Case {
        op: "plan",
        name: name.to_owned(),
        input: json!({
            "provider": provider,
            "spec": serde_json::to_value(&spec).unwrap(),
            "host": {"platform": "darwin"},
        }),
    }
}

fn claude_cfg_case(name: &str, cfg: ClaudeConfig) -> Case {
    plan_case(
        name,
        "claude",
        RunSpec {
            claude: Some(cfg),
            ..spec("hi", "haiku")
        },
    )
}

fn codex_cfg_case(name: &str, cfg: CodexConfig) -> Case {
    plan_case(
        name,
        "codex",
        RunSpec {
            codex: Some(cfg),
            ..spec("hi", "gpt-5.5")
        },
    )
}

fn gemini_cfg_case(name: &str, cfg: GeminiConfig) -> Case {
    plan_case(
        name,
        "gemini",
        RunSpec {
            gemini: Some(cfg),
            ..spec("hi", "gemini-2.5-flash")
        },
    )
}

fn endpoint_case(name: &str, schema: Option<Value>) -> Case {
    plan_case(
        name,
        "openai_endpoint",
        RunSpec {
            schema,
            openai_endpoint: Some(OpenAiEndpoint {
                api_key: "sk-test".to_owned(),
                base_url: "http://local.test/v1".to_owned(),
                model: "qwen3".to_owned(),
            }),
            ..spec("ping", "qwen3")
        },
    )
}

fn plan_cases() -> Vec<Case> {
    vec![
        plan_case("claude-default", "claude", spec("hi", "haiku")),
        plan_case(
            "claude-isolated-false",
            "claude",
            RunSpec {
                isolated: false,
                ..spec("hi", "haiku")
            },
        ),
        plan_case(
            "claude-agent",
            "claude",
            RunSpec {
                agent: true,
                ..spec("hi", "opus")
            },
        ),
        claude_cfg_case(
            "claude-permission-mode",
            ClaudeConfig {
                permission_mode: Some("bypassPermissions".to_owned()),
                ..claude()
            },
        ),
        claude_cfg_case(
            "claude-mcp-config",
            ClaudeConfig {
                mcp_config: Some(r#"{"mcpServers":{}}"#.to_owned()),
                ..claude()
            },
        ),
        claude_cfg_case(
            "claude-append-system-prompt",
            ClaudeConfig {
                append_system_prompt: Some("extra instructions".to_owned()),
                ..claude()
            },
        ),
        claude_cfg_case(
            "claude-system-prompt",
            ClaudeConfig {
                system_prompt: Some("You are terse.".to_owned()),
                ..claude()
            },
        ),
        claude_cfg_case(
            "claude-settings",
            ClaudeConfig {
                settings: Some(r#"{"model":"opus"}"#.to_owned()),
                ..claude()
            },
        ),
        claude_cfg_case(
            "claude-disallowed-tools",
            ClaudeConfig {
                disallowed_tools: Some(strs(&["Bash", "Write"])),
                ..claude()
            },
        ),
        plan_case(
            "claude-strict-mcp",
            "claude",
            RunSpec {
                isolated: false,
                claude: Some(ClaudeConfig {
                    strict_mcp: true,
                    ..claude()
                }),
                ..spec("hi", "haiku")
            },
        ),
        claude_cfg_case(
            "claude-max-turns",
            ClaudeConfig {
                max_turns: Some(3),
                ..claude()
            },
        ),
        claude_cfg_case(
            "claude-explicit-max-budget",
            ClaudeConfig {
                permission_mode: Some("acceptEdits".to_owned()),
                max_budget_usd: Some(2.5),
                ..claude()
            },
        ),
        claude_cfg_case(
            "claude-tools-empty",
            ClaudeConfig {
                tools: Some(vec![]),
                ..claude()
            },
        ),
        claude_cfg_case(
            "claude-tools-list",
            ClaudeConfig {
                tools: Some(strs(&["Bash", "Read"])),
                ..claude()
            },
        ),
        claude_cfg_case(
            "claude-disable-slash-commands",
            ClaudeConfig {
                disable_slash_commands: true,
                ..claude()
            },
        ),
        claude_cfg_case(
            "claude-output-format",
            ClaudeConfig {
                output_format: Some("stream-json".to_owned()),
                ..claude()
            },
        ),
        claude_cfg_case(
            "claude-verbose",
            ClaudeConfig {
                verbose: true,
                ..claude()
            },
        ),
        plan_case(
            "claude-schema",
            "claude",
            RunSpec {
                schema: Some(schema_value()),
                ..spec("hi", "haiku")
            },
        ),
        plan_case(
            "claude-full-explicit-agent",
            "claude",
            RunSpec {
                agent: true,
                schema: Some(schema_value()),
                claude: Some(ClaudeConfig {
                    permission_mode: Some("bypassPermissions".to_owned()),
                    mcp_config: Some(r#"{"mcpServers":{}}"#.to_owned()),
                    strict_mcp: true,
                    disallowed_tools: Some(strs(&["Bash"])),
                    append_system_prompt: Some("extra".to_owned()),
                    settings: Some(r#"{"model":"opus"}"#.to_owned()),
                    max_budget_usd: Some(5.0),
                    system_prompt: Some("terse".to_owned()),
                    max_turns: Some(2),
                    tools: Some(strs(&["Read"])),
                    disable_slash_commands: true,
                    verbose: true,
                    ..claude()
                }),
                ..spec("hi", "opus")
            },
        ),
        plan_case("codex-default", "codex", spec("hi", "gpt-5.5")),
        plan_case(
            "codex-isolated-false",
            "codex",
            RunSpec {
                isolated: false,
                ..spec("hi", "gpt-5.5")
            },
        ),
        plan_case(
            "codex-agent",
            "codex",
            RunSpec {
                agent: true,
                ..spec("hi", "gpt-5.4-mini")
            },
        ),
        plan_case(
            "codex-effort-suffix",
            "codex",
            spec("hi", "gpt-5.4-mini:medium"),
        ),
        codex_cfg_case(
            "codex-service-tier-none",
            CodexConfig {
                service_tier: None,
                ..codex()
            },
        ),
        codex_cfg_case(
            "codex-sandbox-override",
            CodexConfig {
                sandbox: Some("workspace-write".to_owned()),
                ..codex()
            },
        ),
        codex_cfg_case(
            "codex-enable-hooks",
            CodexConfig {
                enable_hooks: true,
                ..codex()
            },
        ),
        codex_cfg_case(
            "codex-enable-mcp",
            CodexConfig {
                enable_mcp: true,
                ..codex()
            },
        ),
        plan_case(
            "codex-schema",
            "codex",
            RunSpec {
                schema: Some(schema_value()),
                ..spec("hi", "gpt-5.5")
            },
        ),
        codex_cfg_case(
            "codex-dev-instructions-multiline",
            CodexConfig {
                developer_instructions: Some("Be terse.\nCite sources.".to_owned()),
                ..codex()
            },
        ),
        codex_cfg_case(
            "codex-dev-instructions-literal-true",
            CodexConfig {
                developer_instructions: Some("true".to_owned()),
                ..codex()
            },
        ),
        codex_cfg_case(
            "codex-dev-instructions-unicode",
            CodexConfig {
                developer_instructions: Some("café ☕ 你好".to_owned()),
                ..codex()
            },
        ),
        codex_cfg_case(
            "codex-dev-instructions-control-char",
            CodexConfig {
                developer_instructions: Some("a\u{7f}b".to_owned()),
                ..codex()
            },
        ),
        plan_case("gemini-default", "gemini", spec("hi", "gemini-2.5-flash")),
        plan_case(
            "gemini-agent",
            "gemini",
            RunSpec {
                agent: true,
                ..spec("hi", "gemini-2.5-flash")
            },
        ),
        gemini_cfg_case(
            "gemini-approval-mode",
            GeminiConfig {
                approval_mode: Some("auto".to_owned()),
                ..gemini()
            },
        ),
        gemini_cfg_case(
            "gemini-extensions-list",
            GeminiConfig {
                extensions: Some(strs(&["search", "fs"])),
                ..gemini()
            },
        ),
        gemini_cfg_case(
            "gemini-extensions-empty",
            GeminiConfig {
                extensions: Some(vec![]),
                ..gemini()
            },
        ),
        plan_case(
            "gemini-schema",
            "gemini",
            RunSpec {
                schema: Some(schema_value()),
                ..spec("hi", "gemini-2.5-flash")
            },
        ),
        plan_case(
            "antigravity-default",
            "antigravity",
            spec("hi", "gemini-3.5"),
        ),
        plan_case(
            "antigravity-agent",
            "antigravity",
            RunSpec {
                agent: true,
                ..spec("hi", "gemini-3.5")
            },
        ),
        plan_case(
            "antigravity-schema",
            "antigravity",
            RunSpec {
                schema: Some(schema_value()),
                ..spec("hi", "gemini-3.5")
            },
        ),
        endpoint_case("openai-endpoint-plain", None),
        endpoint_case("openai-endpoint-schema", Some(schema_value())),
    ]
}

fn resolve_case(
    name: &str,
    provider: &str,
    raw: &str,
    returncode: i64,
    stderr: &str,
    wants_value: bool,
) -> Case {
    Case {
        op: "resolve",
        name: name.to_owned(),
        input: json!({
            "provider": provider,
            "raw": raw,
            "returncode": returncode,
            "stderr": stderr,
            "wants_value": wants_value,
        }),
    }
}

fn resolve_cases() -> Vec<Case> {
    vec![
        resolve_case(
            "claude-ok-dict",
            "claude",
            r#"{"type": "result", "is_error": false, "result": "hello world"}"#,
            0,
            "",
            false,
        ),
        resolve_case(
            "claude-ok-stream-list",
            "claude",
            r#"[{"type": "system"}, {"type": "result", "is_error": false, "result": "answer", "structured_output": {"answer": "42"}}]"#,
            0,
            "",
            true,
        ),
        resolve_case(
            "claude-is-error",
            "claude",
            r#"{"type": "result", "is_error": true, "result": "Overloaded"}"#,
            0,
            "",
            false,
        ),
        resolve_case(
            "claude-truncated-garbage",
            "claude",
            r#"{"type": "result", "resu"#,
            0,
            "",
            false,
        ),
        resolve_case(
            "claude-exit-nonzero",
            "claude",
            "",
            1,
            "claude: fatal boom",
            false,
        ),
        resolve_case(
            "claude-float-cost",
            "claude",
            r#"{"type": "result", "is_error": false, "result": "hi", "total_cost_usd": 0.0123, "usage": {"input_tokens": 12, "output_tokens": 7}}"#,
            0,
            "",
            false,
        ),
        resolve_case(
            "claude-huge-int-usage",
            "claude",
            r#"{"type": "result", "is_error": false, "result": "hi", "usage": {"input_tokens": 999999999999999999999, "output_tokens": 7}}"#,
            0,
            "",
            false,
        ),
        resolve_case("codex-ok-text", "codex", "plain answer text", 0, "", false),
        resolve_case(
            "codex-ok-value",
            "codex",
            r#"{"block": true, "reason": "policy"}"#,
            0,
            "",
            true,
        ),
        resolve_case("codex-empty", "codex", "", 0, "", false),
        resolve_case(
            "codex-exit-nonzero",
            "codex",
            "",
            42,
            "codex exec failed",
            false,
        ),
        resolve_case(
            "gemini-ok",
            "gemini",
            r#"{"response": "hi there", "stats": {"models": {"g": {"api": {"totalErrors": 0}}}}}"#,
            0,
            "",
            false,
        ),
        resolve_case(
            "gemini-total-errors",
            "gemini",
            r#"{"response": "", "stats": {"models": {"g": {"api": {"totalErrors": 1}}}}}"#,
            0,
            "",
            false,
        ),
        resolve_case(
            "gemini-ok-value",
            "gemini",
            r#"{"response": "```json\n{\"answer\": \"7\"}\n```", "stats": {"models": {"g": {"api": {"totalErrors": 0}}}}}"#,
            0,
            "",
            true,
        ),
        resolve_case(
            "openai-ok",
            "openai_endpoint",
            r#"{"choices": [{"message": {"role": "assistant", "content": "pong"}}]}"#,
            0,
            "",
            false,
        ),
        resolve_case(
            "openai-http-error",
            "openai_endpoint",
            "service overloaded",
            503,
            "service overloaded",
            false,
        ),
        resolve_case(
            "openai-2xx-error-body",
            "openai_endpoint",
            r#"{"error": {"message": "no such model"}}"#,
            0,
            "",
            false,
        ),
        resolve_case(
            "openai-ok-value",
            "openai_endpoint",
            r#"{"choices": [{"message": {"role": "assistant", "content": "{\"answer\": \"9\"}"}}]}"#,
            0,
            "",
            true,
        ),
        resolve_case(
            "antigravity-ok-text",
            "antigravity",
            "  the answer  \n",
            0,
            "",
            false,
        ),
        resolve_case(
            "antigravity-ok-value",
            "antigravity",
            "```json\n{\"answer\": \"7\"}\n```",
            0,
            "",
            true,
        ),
        resolve_case(
            "antigravity-exit-nonzero",
            "antigravity",
            "",
            1,
            "agy print failed",
            false,
        ),
    ]
}

const STRICT_SCHEMA_FIXTURES: [(&str, &str); 9] = [
    ("flat", include_str!("../fixtures/strict_schema/flat.json")),
    (
        "optional-field",
        include_str!("../fixtures/strict_schema/optional-field.json"),
    ),
    (
        "nested-refs",
        include_str!("../fixtures/strict_schema/nested-refs.json"),
    ),
    (
        "arrays",
        include_str!("../fixtures/strict_schema/arrays.json"),
    ),
    (
        "enums",
        include_str!("../fixtures/strict_schema/enums.json"),
    ),
    (
        "defaults",
        include_str!("../fixtures/strict_schema/defaults.json"),
    ),
    (
        "list-of-models",
        include_str!("../fixtures/strict_schema/list-of-models.json"),
    ),
    (
        "union",
        include_str!("../fixtures/strict_schema/union.json"),
    ),
    (
        "constrained",
        include_str!("../fixtures/strict_schema/constrained.json"),
    ),
];

fn strict_schema_cases() -> Vec<Case> {
    STRICT_SCHEMA_FIXTURES
        .iter()
        .flat_map(|(case, src)| {
            let schema: Value = serde_json::from_str(src).unwrap();
            ["anthropic", "openai"]
                .into_iter()
                .map(move |dialect| Case {
                    op: "strict_schema",
                    name: format!("{case}-{dialect}"),
                    input: json!({"dialect": dialect, "schema": schema}),
                })
        })
        .collect()
}

fn extract_case(name: &str, text: &str) -> Case {
    Case {
        op: "extract_json",
        name: name.to_owned(),
        input: json!({"text": text}),
    }
}

fn extract_cases() -> Vec<Case> {
    vec![
        extract_case("fenced-json", "```json\n{\"x\": 1}\n```"),
        extract_case("fenced-no-tag", "```\n{\"x\": 2}\n```"),
        extract_case("bare-object", r#"{"x": 3}"#),
        extract_case("bare-array", "[1, 2, 3]"),
        extract_case("leading-prose", "Here is the result: {\"x\": 4} — done."),
        extract_case("trailing-prose", "{\"x\": 5}\n\nHope that helps!"),
        extract_case("first-value-wins", r#"{"first": 1} and then {"second": 2}"#),
        extract_case("nested-braces-in-string", r#"{"path": "a{b}c", "n": 6}"#),
        extract_case("no-json", "just some plain text with no json at all"),
    ]
}

fn retry_case(name: &str, attempt: u32, max_attempts: u32, error_msg: Option<&str>) -> Case {
    Case {
        op: "retry_decision",
        name: name.to_owned(),
        input: json!({"attempt": attempt, "max_attempts": max_attempts, "error_msg": error_msg}),
    }
}

fn retry_cases() -> Vec<Case> {
    vec![
        retry_case(
            "transient-529-attempt-0",
            0,
            5,
            Some("Error 529 overloaded"),
        ),
        retry_case(
            "transient-rate-limit-attempt-1",
            1,
            5,
            Some("hit rate limit, retry later"),
        ),
        retry_case(
            "transient-503-attempt-2",
            2,
            5,
            Some("upstream returned 503"),
        ),
        retry_case(
            "transient-500-attempt-3-caps-at-60",
            3,
            5,
            Some("internal 500 error"),
        ),
        retry_case(
            "transient-overloaded-attempt-0",
            0,
            5,
            Some("the service is overloaded"),
        ),
        retry_case(
            "transient-last-attempt-no-retry",
            4,
            5,
            Some("529 overloaded"),
        ),
        retry_case("transient-max-attempts-one", 0, 1, Some("529 overloaded")),
        retry_case(
            "non-transient-attempt-0",
            0,
            5,
            Some("invalid request: bad schema"),
        ),
        retry_case("no-error-msg", 0, 5, None),
    ]
}

fn auth_probe_case(name: &str, provider: &str, platform: &str, home: &str) -> Case {
    Case {
        op: "auth_probes",
        name: name.to_owned(),
        input: json!({"provider": provider, "host": {"platform": platform, "home": home}}),
    }
}

fn auth_probe_cases() -> Vec<Case> {
    vec![
        auth_probe_case("claude-darwin", "claude", "darwin", HOME_DARWIN),
        auth_probe_case("codex-darwin", "codex", "darwin", HOME_DARWIN),
        auth_probe_case("gemini-darwin", "gemini", "darwin", HOME_DARWIN),
        auth_probe_case("gemini-linux", "gemini", "linux", HOME_LINUX),
        auth_probe_case("antigravity-darwin", "antigravity", "darwin", HOME_DARWIN),
        auth_probe_case("antigravity-linux", "antigravity", "linux", HOME_LINUX),
        auth_probe_case("openai-endpoint", "openai_endpoint", "darwin", HOME_DARWIN),
    ]
}

fn capabilities_cases() -> Vec<Case> {
    vec![Case {
        op: "capabilities",
        name: "capabilities".to_owned(),
        input: json!({}),
    }]
}

fn iso_sources_case(
    name: &str,
    platform: &str,
    home: &str,
    claude_config_dir_env: Option<&str>,
) -> Case {
    Case {
        op: "claude_isolation_sources",
        name: name.to_owned(),
        input: json!({
            "host": {
                "platform": platform,
                "home": home,
                "claude_config_dir_env": claude_config_dir_env,
            }
        }),
    }
}

fn iso_sources_cases() -> Vec<Case> {
    vec![
        iso_sources_case("default-home-darwin", "darwin", HOME_DARWIN, None),
        iso_sources_case("default-home-linux", "linux", HOME_LINUX, None),
        iso_sources_case(
            "config-dir-env-darwin",
            "darwin",
            HOME_DARWIN,
            Some("/Users/testuser/.acct"),
        ),
        iso_sources_case(
            "config-dir-env-trailing-slash-darwin",
            "darwin",
            HOME_DARWIN,
            Some("/Users/testuser/.acct/"),
        ),
        iso_sources_case(
            "config-dir-env-linux",
            "linux",
            HOME_LINUX,
            Some("/home/testuser/.acct"),
        ),
    ]
}

fn iso_seed_case(name: &str, account_json: Option<&str>, credentials_json: Option<&str>) -> Case {
    Case {
        op: "claude_isolation_seed",
        name: name.to_owned(),
        input: json!({"account_json": account_json, "credentials_json": credentials_json}),
    }
}

fn iso_seed_cases() -> Vec<Case> {
    vec![
        iso_seed_case(
            "both-files-mcp-popped",
            Some(
                r#"{"oauthAccount": {"accountUuid": "a"}, "mcpServers": {"semble": {"command": "x"}}}"#,
            ),
            Some(r#"{"claudeAiOauth": {"accessToken": "tok"}}"#),
        ),
        iso_seed_case(
            "account-only",
            Some(r#"{"oauthAccount": {"accountUuid": "b"}, "mcpServers": {"s": {}}}"#),
            None,
        ),
        iso_seed_case(
            "credentials-only",
            None,
            Some(r#"{"claudeAiOauth": {"accessToken": "kc-tok"}}"#),
        ),
        iso_seed_case("both-null", None, None),
        iso_seed_case(
            "account-without-mcp-servers",
            Some(r#"{"oauthAccount": {"accountUuid": "c"}}"#),
            None,
        ),
    ]
}

pub fn all_cases() -> Vec<Case> {
    let mut cases = plan_cases();
    cases.extend(resolve_cases());
    cases.extend(strict_schema_cases());
    cases.extend(extract_cases());
    cases.extend(retry_cases());
    cases.extend(auth_probe_cases());
    cases.extend(capabilities_cases());
    cases.extend(iso_sources_cases());
    cases.extend(iso_seed_cases());
    cases
}

pub fn contract_schemas() -> [(&'static str, &'static str); 3] {
    [
        (
            "run_spec",
            include_str!("../fixtures/schema/run_spec.schema.json"),
        ),
        (
            "invocation_plan",
            include_str!("../fixtures/schema/invocation_plan.schema.json"),
        ),
        (
            "resolved",
            include_str!("../fixtures/schema/resolved.schema.json"),
        ),
    ]
}

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSpec {
    pub prompt: String,
    pub model: String,
    pub agent: bool,
    pub isolated: bool,
    pub timeout: i64,
    pub max_attempts: i64,
    #[serde(default)]
    pub api_auth: bool,
    #[serde(default)]
    pub schema: Option<Value>,
    #[serde(default)]
    pub claude: Option<ClaudeConfig>,
    #[serde(default)]
    pub codex: Option<CodexConfig>,
    #[serde(default)]
    pub gemini: Option<GeminiConfig>,
    #[serde(default)]
    pub openai_endpoint: Option<OpenAiEndpoint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaudeConfig {
    pub append_system_prompt: Option<String>,
    pub disable_slash_commands: bool,
    pub disallowed_tools: Option<Vec<String>>,
    pub max_budget_usd: Option<f64>,
    pub max_turns: Option<i64>,
    pub mcp_config: Option<String>,
    pub output_format: Option<String>,
    pub permission_mode: Option<String>,
    pub settings: Option<String>,
    pub strict_mcp: bool,
    pub system_prompt: Option<String>,
    pub tools: Option<Vec<String>>,
    pub verbose: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodexConfig {
    pub developer_instructions: Option<String>,
    pub enable_hooks: bool,
    pub enable_mcp: bool,
    pub sandbox: Option<String>,
    pub service_tier: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeminiConfig {
    pub approval_mode: Option<String>,
    pub extensions: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpenAiEndpoint {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InvocationPlan {
    Exec(ExecPlan),
    Http(HttpPlan),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecPlan {
    pub argv: Vec<String>,
    pub stdin: String,
    pub files: Vec<PlanFile>,
    pub stdout_to_file: bool,
    pub read_result_from: ReadResultFrom,
    pub env: BTreeMap<String, String>,
    pub env_unset: Vec<String>,
    pub needs_claude_isolation: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HttpPlan {
    pub method: String,
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanFile {
    pub id: FileId,
    pub suffix: String,
    pub content: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileId {
    Schema,
    Result,
    Stdout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadResultFrom {
    #[serde(rename = "stdout")]
    Stdout,
    #[serde(rename = "file:result")]
    FileResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Resolved {
    Ok(ResolvedOk),
    Error(ResolvedError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedOk {
    pub text: String,
    pub value: Value,
    pub cost_usd: Option<f64>,
    pub usage: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedError {
    pub kind: ResolveErrorKind,
    pub msg: String,
    pub transient: bool,
    pub cost_usd: Option<f64>,
    pub usage: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolveErrorKind {
    Exit,
    Envelope,
    Parse,
}

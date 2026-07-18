use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::backend::Backend;
use crate::error::{Error, RunError};

pub(crate) const DEFAULT_TIMEOUT: Duration = Duration::from_secs(180);
pub(crate) const DEFAULT_MAX_ATTEMPTS: u32 = 5;
pub(crate) const DEFAULT_SELECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Claude CLI flag passthrough applied only when the Claude backend runs a spec.
///
/// Fields mirror `spawnllm-core`'s wire `ClaudeConfig` one-for-one, so a value
/// serializes straight into the portable run configuration the core `plan` op reads.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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

/// Codex CLI knobs applied only when the Codex backend runs a spec.
///
/// `service_tier` defaults to `"fast"`, matching the Python reference: an isolated
/// run passes `--ignore-user-config`, dropping any user-level tier pin, and the
/// standard tier turns long prompts into multi-minute runs. Set it to `None` to
/// drop the flag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodexConfig {
    pub developer_instructions: Option<String>,
    pub enable_hooks: bool,
    pub enable_mcp: bool,
    pub sandbox: Option<String>,
    pub service_tier: Option<String>,
}

impl Default for CodexConfig {
    fn default() -> Self {
        Self {
            developer_instructions: None,
            enable_hooks: false,
            enable_mcp: false,
            sandbox: None,
            service_tier: Some("fast".to_owned()),
        }
    }
}

/// Gemini/Antigravity CLI knobs applied only when a Gemini-family backend runs a spec.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GeminiConfig {
    pub approval_mode: Option<String>,
    pub extensions: Option<Vec<String>>,
}

/// Abstract model tier resolved to a concrete provider model id per backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ModelTier {
    #[default]
    Small,
    Medium,
    Large,
}

/// Task specialty that scopes backend auto-selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Specialty {
    Debugging,
    Review,
    General,
}

impl Specialty {
    pub(crate) fn key(self) -> &'static str {
        match self {
            Specialty::Debugging => "debugging",
            Specialty::Review => "review",
            Specialty::General => "general",
        }
    }
}

/// A single configured run, translated to a provider invocation at execution time.
///
/// Build with [`RunSpec::new`] and chain the consuming setters. `isolated` defaults
/// to `true` (a fresh, host-free config home), `timeout` is per attempt (default
/// 180s), and `max_attempts` bounds the transient-retry loop (default 5).
///
/// Example:
///     let spec = spawnllm::RunSpec::new("ping", "haiku").agent(false);
#[derive(Debug, Clone)]
pub struct RunSpec {
    pub prompt: String,
    pub model: String,
    pub schema: Option<Value>,
    pub agent: bool,
    pub isolated: bool,
    pub cwd: Option<PathBuf>,
    pub env: Option<HashMap<String, String>>,
    pub timeout: Duration,
    pub max_attempts: u32,
    pub claude: Option<ClaudeConfig>,
    pub codex: Option<CodexConfig>,
    pub gemini: Option<GeminiConfig>,
}

impl RunSpec {
    pub fn new(prompt: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            model: model.into(),
            schema: None,
            agent: false,
            isolated: true,
            cwd: None,
            env: None,
            timeout: DEFAULT_TIMEOUT,
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            claude: None,
            codex: None,
            gemini: None,
        }
    }

    pub fn schema(mut self, schema: Value) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn agent(mut self, agent: bool) -> Self {
        self.agent = agent;
        self
    }

    pub fn isolated(mut self, isolated: bool) -> Self {
        self.isolated = isolated;
        self
    }

    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    pub fn env(mut self, env: HashMap<String, String>) -> Self {
        self.env = Some(env);
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    pub fn claude(mut self, config: ClaudeConfig) -> Self {
        self.claude = Some(config);
        self
    }

    pub fn codex(mut self, config: CodexConfig) -> Self {
        self.codex = Some(config);
        self
    }

    pub fn gemini(mut self, config: GeminiConfig) -> Self {
        self.gemini = Some(config);
        self
    }
}

/// Options for the prompt-ergonomic [`call`](crate::call)/[`extract`](crate::extract) entries.
#[derive(Debug, Clone, Default)]
pub struct CallOpts {
    pub backend: Option<Backend>,
    pub specialty: Option<Specialty>,
    pub model: ModelTier,
    pub agent: bool,
    pub cwd: Option<PathBuf>,
    pub timeout: Option<Duration>,
}

/// A backend's fully resolved outcome: the spec, the raw output, and exactly one of result/error.
///
/// `outcome` is `Ok` on success and `Err` on every provider failure — a nonzero
/// exit, an error envelope, a timeout, or backend unavailability — so [`run`](crate::run)
/// never returns an `Err` at its own boundary. `discarded_attempts` carries the
/// transient failures the retry loop threw away before this one.
#[derive(Debug)]
pub struct Response {
    pub spec: RunSpec,
    pub output: String,
    pub outcome: Result<RunResult, RunError>,
    pub discarded_attempts: Vec<DiscardedAttempt>,
}

/// A successful run: the extracted final text and the optional parsed structured value.
#[derive(Debug)]
pub struct RunResult {
    pub raw: String,
    pub parsed: Option<Value>,
}

/// A transient failure the retry loop discarded, summarized for spend accounting.
#[derive(Debug)]
pub struct DiscardedAttempt {
    pub attempt: u32,
    pub error: String,
    pub cost_usd: Option<f64>,
    pub usage: Option<Map<String, Value>>,
    pub raw_bytes: usize,
}

impl RunSpec {
    pub(crate) fn call_spec(prompt: impl Into<String>, model: String, opts: &CallOpts) -> Self {
        let mut spec = RunSpec::new(prompt, model).agent(opts.agent);
        if let Some(cwd) = &opts.cwd {
            spec = spec.cwd(cwd.clone());
        }
        if let Some(timeout) = opts.timeout {
            spec = spec.timeout(timeout);
        }
        spec
    }
}

pub(crate) fn unavailable_response(spec: RunSpec, error: Error) -> Response {
    let msg = error.to_string();
    Response {
        spec,
        output: String::new(),
        outcome: Err(RunError { msg, source: error }),
        discarded_attempts: Vec::new(),
    }
}

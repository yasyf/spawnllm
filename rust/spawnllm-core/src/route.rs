use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

use crate::auth::{
    ANTIGRAVITY_API_KEY_VARS, APPLE_API_KEY_VARS, CLAUDE_API_KEY_VARS, CODEX_API_KEY_VARS,
    GEMINI_API_KEY_VARS,
};
use crate::{OpError, OpResult};

const PROVIDERS: [&str; 5] = ["claude", "codex", "antigravity", "gemini", "apple"];

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ModelTiers {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub large: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub medium: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub small: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Capabilities {
    pub providers: Vec<&'static str>,
    pub priority: Vec<&'static str>,
    pub auto_select_excludes: Vec<&'static str>,
    pub auto_select_tiers: BTreeMap<&'static str, Vec<&'static str>>,
    pub api_key_vars: BTreeMap<&'static str, Vec<&'static str>>,
    pub specialties: BTreeMap<&'static str, &'static str>,
    pub models: BTreeMap<&'static str, ModelTiers>,
    pub binaries: BTreeMap<&'static str, &'static str>,
    pub install_hints: BTreeMap<&'static str, &'static str>,
}

pub fn capabilities() -> Capabilities {
    Capabilities {
        providers: PROVIDERS.to_vec(),
        priority: PROVIDERS.to_vec(),
        auto_select_excludes: vec!["gemini"],
        auto_select_tiers: BTreeMap::from([("apple", vec!["small"])]),
        api_key_vars: BTreeMap::from([
            ("claude", CLAUDE_API_KEY_VARS.to_vec()),
            ("codex", CODEX_API_KEY_VARS.to_vec()),
            ("gemini", GEMINI_API_KEY_VARS.to_vec()),
            ("antigravity", ANTIGRAVITY_API_KEY_VARS.to_vec()),
            ("apple", APPLE_API_KEY_VARS.to_vec()),
        ]),
        specialties: BTreeMap::from([
            ("debugging", "codex"),
            ("review", "codex"),
            ("general", "claude"),
        ]),
        models: BTreeMap::from([
            (
                "claude",
                ModelTiers {
                    large: Some("opus"),
                    medium: Some("sonnet"),
                    small: Some("haiku"),
                },
            ),
            (
                "codex",
                ModelTiers {
                    large: Some("gpt-5.5:medium"),
                    medium: Some("gpt-5.4-mini:medium"),
                    small: Some("gpt-5.4-mini:low"),
                },
            ),
            (
                "antigravity",
                ModelTiers {
                    large: Some("gemini-3.5-pro"),
                    medium: Some("gemini-3.5"),
                    small: Some("gemini-3.5-flash"),
                },
            ),
            (
                "gemini",
                ModelTiers {
                    large: Some("gemini-3-pro-preview"),
                    medium: Some("gemini-2.5-flash"),
                    small: Some("gemini-2.5-flash-lite"),
                },
            ),
            ("apple", ModelTiers::default()),
        ]),
        binaries: BTreeMap::from([
            ("claude", "claude"),
            ("codex", "codex"),
            ("antigravity", "agy"),
            ("gemini", "gemini"),
            ("apple", "spawnllm-apple"),
        ]),
        install_hints: BTreeMap::from([
            ("claude", "curl -fsSL https://claude.ai/install.sh | bash"),
            ("codex", "npm install -g @openai/codex"),
            (
                "antigravity",
                "curl -fsSL https://antigravity.google/cli/install.sh | bash",
            ),
            ("gemini", "npm install -g @google/gemini-cli"),
            (
                "apple",
                "swift build -c release --package-path swift/spawnllm-apple",
            ),
        ]),
    }
}

pub(crate) fn dispatch(_input: Value) -> OpResult {
    serde_json::to_value(capabilities()).map_err(OpError::internal)
}

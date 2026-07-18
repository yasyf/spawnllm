use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

use crate::{OpError, OpResult};

const PROVIDERS: [&str; 4] = ["claude", "codex", "antigravity", "gemini"];

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ModelTiers {
    pub large: &'static str,
    pub medium: &'static str,
    pub small: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct Capabilities {
    pub providers: Vec<&'static str>,
    pub priority: Vec<&'static str>,
    pub auto_select_excludes: Vec<&'static str>,
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
        specialties: BTreeMap::from([
            ("debugging", "codex"),
            ("review", "codex"),
            ("general", "claude"),
        ]),
        models: BTreeMap::from([
            (
                "claude",
                ModelTiers {
                    large: "opus",
                    medium: "sonnet",
                    small: "haiku",
                },
            ),
            (
                "codex",
                ModelTiers {
                    large: "gpt-5.5:medium",
                    medium: "gpt-5.4-mini:medium",
                    small: "gpt-5.4-mini:low",
                },
            ),
            (
                "antigravity",
                ModelTiers {
                    large: "gemini-3.5-pro",
                    medium: "gemini-3.5",
                    small: "gemini-3.5-flash",
                },
            ),
            (
                "gemini",
                ModelTiers {
                    large: "gemini-3-pro-preview",
                    medium: "gemini-2.5-flash",
                    small: "gemini-2.5-flash-lite",
                },
            ),
        ]),
        binaries: BTreeMap::from([
            ("claude", "claude"),
            ("codex", "codex"),
            ("antigravity", "agy"),
            ("gemini", "gemini"),
        ]),
        install_hints: BTreeMap::from([
            ("claude", "curl -fsSL https://claude.ai/install.sh | bash"),
            ("codex", "npm install -g @openai/codex"),
            (
                "antigravity",
                "curl -fsSL https://antigravity.google/cli/install.sh | bash",
            ),
            ("gemini", "npm install -g @google/gemini-cli"),
        ]),
    }
}

pub(crate) fn dispatch(_input: Value) -> OpResult {
    serde_json::to_value(capabilities()).map_err(OpError::internal)
}

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::auth::{ANTIGRAVITY_API_KEY_VARS, GEMINI_API_KEY_VARS};
use crate::{OpError, OpResult, from_input};

#[derive(Debug, Deserialize)]
struct AuthProbesInput {
    provider: Provider,
    host: Host,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Provider {
    Claude,
    Codex,
    Gemini,
    Antigravity,
    Apple,
    OpenaiEndpoint,
}

#[derive(Debug, Deserialize)]
struct Host {
    platform: String,
    home: String,
}

#[derive(Debug, Serialize)]
struct AuthProbes {
    binary: &'static str,
    install_hint: Option<&'static str>,
    probes: Vec<Probe>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Probe {
    ExecExit0 {
        argv: Vec<&'static str>,
    },
    KeychainExists {
        service: &'static str,
        account: &'static str,
    },
    EnvAny {
        vars: Vec<&'static str>,
    },
    FileExists {
        path: String,
    },
}

pub(crate) fn dispatch(input: Value) -> OpResult {
    let input = from_input::<AuthProbesInput>(input)?;
    let probes = match input.provider {
        Provider::Claude => AuthProbes {
            binary: "claude",
            install_hint: Some("curl -fsSL https://claude.ai/install.sh | bash"),
            probes: vec![Probe::ExecExit0 {
                argv: vec!["claude", "auth", "status"],
            }],
        },
        Provider::Codex => AuthProbes {
            binary: "codex",
            install_hint: Some("npm install -g @openai/codex"),
            probes: vec![Probe::ExecExit0 {
                argv: vec!["codex", "login", "status"],
            }],
        },
        Provider::Gemini => AuthProbes {
            binary: "gemini",
            install_hint: Some("npm install -g @google/gemini-cli"),
            probes: vec![
                Probe::FileExists {
                    path: format!("{}/.gemini/oauth_creds.json", input.host.home),
                },
                Probe::EnvAny {
                    vars: GEMINI_API_KEY_VARS.to_vec(),
                },
            ],
        },
        Provider::Antigravity => {
            let mut probes = Vec::new();
            if input.host.platform == "darwin" {
                probes.push(Probe::KeychainExists {
                    service: "gemini",
                    account: "antigravity",
                });
            }
            probes.push(Probe::EnvAny {
                vars: ANTIGRAVITY_API_KEY_VARS.to_vec(),
            });
            AuthProbes {
                binary: "agy",
                install_hint: Some("curl -fsSL https://antigravity.google/cli/install.sh | bash"),
                probes,
            }
        }
        Provider::Apple => AuthProbes {
            binary: "spawnllm-apple",
            install_hint: Some("swift build -c release --package-path swift/spawnllm-apple"),
            probes: vec![Probe::ExecExit0 {
                argv: vec!["spawnllm-apple", "--probe"],
            }],
        },
        Provider::OpenaiEndpoint => AuthProbes {
            binary: "openai_endpoint",
            install_hint: None,
            probes: Vec::new(),
        },
    };
    serde_json::to_value(probes).map_err(OpError::internal)
}

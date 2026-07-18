use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::core_io::core_op;
use crate::error::Error;
use crate::host::{home, platform};
use crate::spec::{ModelTier, Specialty};

#[cfg(feature = "openai")]
use serde_json::Map;

/// An OpenAI-compatible `/chat/completions` endpoint.
#[cfg(feature = "openai")]
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiEndpoint {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

/// A concrete LLM backend: one of the four CLIs, or an OpenAI-compatible endpoint.
#[derive(Debug, Clone)]
pub enum Backend {
    Claude,
    Codex,
    Gemini,
    Antigravity,
    #[cfg(feature = "openai")]
    OpenAiEndpoint(OpenAiEndpoint),
}

/// The install/auth readiness of a backend.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendStatus {
    Ready {
        binary: String,
    },
    NotInstalled {
        binary: String,
        install_hint: String,
    },
    NotAuthenticated {
        binary: String,
    },
}

#[derive(Debug, Deserialize)]
struct AuthProbes {
    binary: String,
    install_hint: Option<String>,
    probes: Vec<Probe>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Probe {
    ExecExit0 { argv: Vec<String> },
    KeychainExists { service: String, account: String },
    EnvAny { vars: Vec<String> },
    FileExists { path: String },
}

impl Backend {
    pub(crate) fn provider(&self) -> &'static str {
        match self {
            Backend::Claude => "claude",
            Backend::Codex => "codex",
            Backend::Gemini => "gemini",
            Backend::Antigravity => "antigravity",
            #[cfg(feature = "openai")]
            Backend::OpenAiEndpoint(_) => "openai_endpoint",
        }
    }

    pub(crate) fn strict_dialect(&self) -> Option<&'static str> {
        match self {
            Backend::Claude => Some("anthropic"),
            Backend::Codex => Some("openai"),
            Backend::Gemini | Backend::Antigravity => None,
            #[cfg(feature = "openai")]
            Backend::OpenAiEndpoint(_) => Some("openai"),
        }
    }

    pub(crate) fn endpoint_value(&self) -> Option<Value> {
        match self {
            #[cfg(feature = "openai")]
            Backend::OpenAiEndpoint(endpoint) => Some(Value::Object(Map::from_iter([
                ("api_key".to_owned(), json!(endpoint.api_key)),
                ("base_url".to_owned(), json!(endpoint.base_url)),
                ("model".to_owned(), json!(endpoint.model)),
            ]))),
            _ => None,
        }
    }

    pub(crate) fn model_for(&self, tier: ModelTier) -> String {
        #[cfg(feature = "openai")]
        if let Backend::OpenAiEndpoint(endpoint) = self {
            return endpoint.model.clone();
        }
        let caps = spawnllm_core::capabilities();
        let tiers = caps
            .models
            .get(self.provider())
            .expect("cli backend has model tiers");
        match tier {
            ModelTier::Small => tiers.small,
            ModelTier::Medium => tiers.medium,
            ModelTier::Large => tiers.large,
        }
        .to_owned()
    }

    /// Report whether this backend's CLI is installed and authenticated.
    pub async fn check_status(&self, timeout: Duration) -> BackendStatus {
        #[cfg(feature = "openai")]
        if let Backend::OpenAiEndpoint(_) = self {
            return BackendStatus::Ready {
                binary: "openai_endpoint".to_owned(),
            };
        }
        let probes: AuthProbes = core_op(
            "auth_probes",
            json!({ "provider": self.provider(), "host": { "platform": platform(), "home": home() } }),
        );
        if which(&probes.binary).is_none() {
            return BackendStatus::NotInstalled {
                binary: probes.binary,
                install_hint: probes.install_hint.unwrap_or_default(),
            };
        }
        if authenticated(&probes.probes, timeout).await {
            BackendStatus::Ready {
                binary: probes.binary,
            }
        } else {
            BackendStatus::NotAuthenticated {
                binary: probes.binary,
            }
        }
    }

    /// Report whether this backend holds valid credentials for its provider.
    pub async fn is_authenticated(&self, timeout: Duration) -> bool {
        matches!(
            self.check_status(timeout).await,
            BackendStatus::Ready { .. }
        )
    }
}

async fn authenticated(probes: &[Probe], timeout: Duration) -> bool {
    if probes.is_empty() {
        return true;
    }
    for probe in probes {
        if run_probe(probe, timeout).await {
            return true;
        }
    }
    false
}

async fn run_probe(probe: &Probe, timeout: Duration) -> bool {
    match probe {
        Probe::ExecExit0 { argv } => exec_exit0(argv, timeout).await,
        Probe::KeychainExists { service, account } => {
            exec_exit0(
                &[
                    "security".to_owned(),
                    "find-generic-password".to_owned(),
                    "-s".to_owned(),
                    service.clone(),
                    "-a".to_owned(),
                    account.clone(),
                ],
                timeout,
            )
            .await
        }
        Probe::EnvAny { vars } => vars
            .iter()
            .any(|var| std::env::var(var).is_ok_and(|value| !value.is_empty())),
        Probe::FileExists { path } => Path::new(path).exists(),
    }
}

async fn exec_exit0(argv: &[String], timeout: Duration) -> bool {
    let Ok(mut child) = Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    match tokio::time::timeout(timeout, child.wait()).await {
        Ok(Ok(status)) => status.success(),
        Ok(Err(_)) => false,
        Err(_) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            false
        }
    }
}

/// Return the first installed, authenticated backend in priority order.
///
/// A `specialty` promotes its registered backend to the front of the chain; the
/// chain otherwise follows core's priority order minus the auto-select excludes
/// (Gemini). Returns [`Error::BackendUnavailable`] when none is ready.
pub async fn select_backend(
    specialty: Option<Specialty>,
    timeout: Duration,
) -> Result<Backend, Error> {
    let caps = spawnllm_core::capabilities();
    let mut names: Vec<&str> = Vec::new();
    if let Some(specialty) = specialty
        && let Some(provider) = caps.specialties.get(specialty.key())
    {
        names.push(provider);
    }
    for &name in &caps.priority {
        if caps.auto_select_excludes.contains(&name) || names.contains(&name) {
            continue;
        }
        names.push(name);
    }

    let mut statuses = Vec::new();
    for name in names {
        let backend = backend_from_name(name);
        match backend.check_status(timeout).await {
            BackendStatus::Ready { .. } => return Ok(backend),
            other => statuses.push((backend, other)),
        }
    }
    Err(Error::BackendUnavailable {
        specialty,
        statuses,
    })
}

fn backend_from_name(name: &str) -> Backend {
    match name {
        "claude" => Backend::Claude,
        "codex" => Backend::Codex,
        "gemini" => Backend::Gemini,
        "antigravity" => Backend::Antigravity,
        other => panic!("core capabilities named an unknown backend: {other}"),
    }
}

fn which(binary: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| {
        let candidate = dir.join(binary);
        is_executable(&candidate).then_some(candidate)
    })
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    std::fs::metadata(path)
        .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

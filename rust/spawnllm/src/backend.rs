use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::core_io::core_op;
use crate::error::Error;
use crate::host::{home, platform, which};
use crate::sidecar;
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

/// A concrete LLM backend: one of the five CLIs, or an OpenAI-compatible endpoint.
#[derive(Debug, Clone)]
pub enum Backend {
    Claude,
    Codex,
    Gemini,
    Antigravity,
    Apple,
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
            Backend::Apple => "apple",
            #[cfg(feature = "openai")]
            Backend::OpenAiEndpoint(_) => "openai_endpoint",
        }
    }

    pub(crate) fn strict_dialect(&self) -> Option<&'static str> {
        match self {
            Backend::Claude => Some("anthropic"),
            Backend::Codex => Some("openai"),
            Backend::Apple => Some("apple"),
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
        if matches!(self, Backend::Apple) {
            return String::new();
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
        .expect("cli backend names a model for every tier")
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
        let probes: AuthProbes = match core_op(
            "auth_probes",
            json!({ "provider": self.provider(), "host": { "platform": platform(), "home": home() } }),
        ) {
            Ok(probes) => probes,
            Err(_) => {
                return BackendStatus::NotAuthenticated {
                    binary: String::new(),
                };
            }
        };
        let Ok(launch) = resolve_binary(&probes.binary) else {
            let install_hint = install_hint(&probes);
            return BackendStatus::NotInstalled {
                binary: probes.binary,
                install_hint,
            };
        };
        if authenticated(&probes.probes, &launch, timeout).await {
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

/// The argv prefix that runs `binary`, or [`std::io::ErrorKind::NotFound`] when it
/// is not installed. Every CLI but the Apple sidecar must be on `PATH`; the
/// sidecar falls back to binrun against its pinned descriptor.
pub(crate) fn resolve_binary(binary: &str) -> std::io::Result<Vec<String>> {
    if binary == sidecar::BINARY {
        return sidecar::launch();
    }
    which(binary)
        .map(|path| vec![path.to_string_lossy().into_owned()])
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{binary} is not on PATH"),
            )
        })
}

fn install_hint(probes: &AuthProbes) -> String {
    if probes.binary == sidecar::BINARY {
        return sidecar::INSTALL_HINT.to_owned();
    }
    probes.install_hint.clone().unwrap_or_default()
}

async fn authenticated(probes: &[Probe], launch: &[String], timeout: Duration) -> bool {
    if probes.is_empty() {
        return true;
    }
    for probe in probes {
        if run_probe(probe, launch, timeout).await {
            return true;
        }
    }
    false
}

async fn run_probe(probe: &Probe, launch: &[String], timeout: Duration) -> bool {
    match probe {
        Probe::ExecExit0 { argv } => exec_exit0(&[launch, &argv[1..]].concat(), timeout).await,
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
/// (Gemini) and minus every backend core restricts to tiers `model` is not among
/// (Apple, which hosts one small on-device model). `model` is `None` when the run
/// names a concrete provider model id rather than a tier, which excludes every
/// tier-restricted backend. Returns [`Error::BackendUnavailable`] when none is ready.
pub async fn select_backend(
    specialty: Option<Specialty>,
    model: Option<ModelTier>,
    timeout: Duration,
) -> Result<Backend, Error> {
    let mut statuses = Vec::new();
    for name in selection_order(&spawnllm_core::capabilities(), specialty, model) {
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

fn selection_order(
    caps: &spawnllm_core::Capabilities,
    specialty: Option<Specialty>,
    model: Option<ModelTier>,
) -> Vec<&'static str> {
    let mut names: Vec<&'static str> = Vec::new();
    if let Some(specialty) = specialty
        && let Some(provider) = caps.specialties.get(specialty.key())
    {
        names.push(*provider);
    }
    for &name in &caps.priority {
        if caps.auto_select_excludes.contains(&name) || names.contains(&name) {
            continue;
        }
        if let Some(tiers) = caps.auto_select_tiers.get(name)
            && !model.is_some_and(|tier| tiers.contains(&tier.key()))
        {
            continue;
        }
        names.push(name);
    }
    names
}

fn backend_from_name(name: &str) -> Backend {
    match name {
        "claude" => Backend::Claude,
        "codex" => Backend::Codex,
        "gemini" => Backend::Gemini,
        "antigravity" => Backend::Antigravity,
        "apple" => Backend::Apple,
        other => panic!("core capabilities named an unknown backend: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(specialty: Option<Specialty>, model: Option<ModelTier>) -> Vec<&'static str> {
        selection_order(&spawnllm_core::capabilities(), specialty, model)
    }

    #[test]
    fn apple_is_auto_selectable_only_for_the_small_tier() {
        assert!(order(None, Some(ModelTier::Small)).contains(&"apple"));
        for model in [None, Some(ModelTier::Medium), Some(ModelTier::Large)] {
            assert!(
                !order(None, model).contains(&"apple"),
                "apple reachable for {model:?}"
            );
        }
    }

    #[test]
    fn the_tier_gate_leaves_the_unrestricted_backends_alone() {
        assert_eq!(
            order(None, Some(ModelTier::Large)),
            vec!["claude", "codex", "antigravity"]
        );
        assert_eq!(
            order(Some(Specialty::Debugging), Some(ModelTier::Large)),
            vec!["codex", "claude", "antigravity"]
        );
    }
}

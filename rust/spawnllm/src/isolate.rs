use std::io::Write;
use std::path::Path;
#[cfg(any(target_os = "macos", test))]
use std::process::Output;
#[cfg(any(target_os = "macos", test))]
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;
use tempfile::TempDir;

use crate::core_io::core_op;
use crate::error::Error;
use crate::host::{home, platform};

#[derive(Debug, Deserialize)]
struct Sources {
    account_path: String,
    credentials_path: String,
    keychain_service: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Seed {
    files: Vec<SeedFile>,
}

#[derive(Debug, Deserialize)]
struct SeedFile {
    name: String,
    content: String,
    mode: String,
}

pub(crate) async fn seed_isolation() -> Result<TempDir, Error> {
    let sources: Sources = core_op(
        "claude_isolation_sources",
        json!({ "host": {
            "platform": platform(),
            "home": home(),
            "claude_config_dir_env": std::env::var("CLAUDE_CONFIG_DIR").ok().filter(|value| !value.is_empty()),
        } }),
    )?;

    let account_json = std::fs::read_to_string(&sources.account_path).ok();
    let credentials_json = match std::fs::read_to_string(&sources.credentials_path) {
        Ok(text) => Some(text),
        Err(_) => match &sources.keychain_service {
            Some(service) => keychain_credentials(service).await,
            None => None,
        },
    };

    let seed: Seed = core_op(
        "claude_isolation_seed",
        json!({ "account_json": account_json, "credentials_json": credentials_json }),
    )?;

    let dir = tempfile::Builder::new()
        .prefix("spawnllm-claude-config-")
        .tempdir()?;
    for file in &seed.files {
        let path = dir.path().join(&file.name);
        let mut handle = std::fs::File::create(&path)?;
        handle.write_all(file.content.as_bytes())?;
        handle.flush()?;
        set_mode(&path, &file.mode)?;
    }
    Ok(dir)
}

async fn keychain_credentials(service: &str) -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        let mut command = tokio::process::Command::new("security");
        command.args(["find-generic-password", "-s", service, "-w"]);
        let output = timed_command_output(&mut command, Duration::from_secs(10)).await?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = service;
        None
    }
}

#[cfg(any(target_os = "macos", test))]
async fn timed_command_output(
    command: &mut tokio::process::Command,
    timeout: Duration,
) -> Option<Output> {
    command.kill_on_drop(true);
    tokio::time::timeout(timeout, command.output())
        .await
        .ok()?
        .ok()
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: &str) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let bits = u32::from_str_radix(mode, 8).expect("core emits octal file modes");
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(bits))
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: &str) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[tokio::test]
    async fn timed_command_output_returns_none_on_spawn_failure() {
        let mut command = tokio::process::Command::new("spawnllm-command-that-does-not-exist");

        assert!(
            timed_command_output(&mut command, Duration::from_secs(1))
                .await
                .is_none()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timed_command_output_returns_none_at_the_deadline() {
        let mut command = tokio::process::Command::new("sh");
        command.args(["-c", "sleep 5"]);
        let started = Instant::now();

        assert!(
            timed_command_output(&mut command, Duration::from_millis(50))
                .await
                .is_none()
        );
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}

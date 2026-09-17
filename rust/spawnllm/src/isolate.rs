use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
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
    credentials_path: Option<String>,
    keychain_service: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Seed {
    files: Vec<SeedFile>,
    env: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct SeedFile {
    name: String,
    content: String,
    mode: String,
}

pub(crate) struct Isolation {
    pub(crate) dir: TempDir,
    pub(crate) env: BTreeMap<String, String>,
}

pub(crate) async fn seed_isolation() -> Result<Isolation, Error> {
    let sources: Sources = core_op(
        "claude_isolation_sources",
        json!({ "host": {
            "platform": platform(),
            "home": home(),
            "claude_config_dir_env": std::env::var("CLAUDE_CONFIG_DIR").ok().filter(|value| !value.is_empty()),
            "claude_securestorage_config_dir_env": std::env::var("CLAUDE_SECURESTORAGE_CONFIG_DIR").ok(),
            "claude_code_custom_oauth_url_env": std::env::var("CLAUDE_CODE_CUSTOM_OAUTH_URL").ok(),
            "claude_code_oauth_token_env": std::env::var("CLAUDE_CODE_OAUTH_TOKEN").ok(),
        } }),
    )?;

    let account_json = std::fs::read_to_string(&sources.account_path).ok();
    let credentials_json = match sources
        .credentials_path
        .as_deref()
        .and_then(|path| std::fs::read_to_string(path).ok())
    {
        Some(text) => Some(text),
        None => match &sources.keychain_service {
            Some(service) => keychain_credentials(service).await,
            None => None,
        },
    };

    let seed: Seed = core_op(
        "claude_isolation_seed",
        json!({ "account_json": account_json, "credentials_json": credentials_json }),
    )?;

    let dir = private_tempdir()?;
    for file in &seed.files {
        let mut handle = create_with_mode(&dir.path().join(&file.name), &file.mode)?;
        handle.write_all(file.content.as_bytes())?;
        handle.flush()?;
    }
    Ok(Isolation { dir, env: seed.env })
}

fn private_tempdir() -> std::io::Result<TempDir> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("spawnllm-claude-config-");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(std::fs::Permissions::from_mode(0o700));
    }
    builder.tempdir()
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
fn create_with_mode(path: &Path, mode: &str) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    let bits = u32::from_str_radix(mode, 8).expect("core emits octal file modes");
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(bits)
        .open(path)
}

#[cfg(not(unix))]
fn create_with_mode(path: &Path, _mode: &str) -> std::io::Result<File> {
    OpenOptions::new().write(true).create_new(true).open(path)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[cfg(unix)]
    #[test]
    fn private_tempdir_is_owner_only_at_creation() {
        use std::os::unix::fs::PermissionsExt;

        let dir = private_tempdir().unwrap();

        assert_eq!(
            dir.path().metadata().unwrap().permissions().mode() & 0o777,
            0o700
        );
    }

    #[cfg(unix)]
    #[test]
    fn create_with_mode_applies_the_mode_before_any_byte_is_written() {
        use std::os::unix::fs::PermissionsExt;

        let dir = private_tempdir().unwrap();
        let path = dir.path().join(".credentials.json");
        let mut handle = create_with_mode(&path, "0600").unwrap();
        let created = path.metadata().unwrap();

        assert_eq!(created.len(), 0);
        assert_eq!(created.permissions().mode() & 0o777, 0o600);
        handle.write_all(b"{}").unwrap();
        assert!(create_with_mode(&path, "0600").is_err());
    }

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

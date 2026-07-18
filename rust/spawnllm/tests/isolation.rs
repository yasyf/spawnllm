mod common;

use std::collections::HashMap;
use std::path::Path;
use std::sync::PoisonError;

use spawnllm::{Backend, RunSpec};

fn set_config_dir(dir: &Path) {
    // SAFETY: gated by ENV_LOCK; this process runs only the isolation tests serially.
    unsafe { std::env::set_var("CLAUDE_CONFIG_DIR", dir) };
}

fn clear_config_dir() {
    // SAFETY: gated by ENV_LOCK; see set_config_dir.
    unsafe { std::env::remove_var("CLAUDE_CONFIG_DIR") };
}

fn env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

// The env-lock guard spans the awaited run so CLAUDE_CONFIG_DIR stays set; on this
// single-threaded test runtime that only serializes the two isolation tests.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn isolation_seeds_stripped_account_and_credentials_from_files() {
    common::fixtures();
    let _guard = common::ENV_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    let source = tempfile::tempdir().unwrap();
    std::fs::write(
        source.path().join(".claude.json"),
        r#"{"mcpServers": {"x": 1}, "account": "me"}"#,
    )
    .unwrap();
    std::fs::write(
        source.path().join(".credentials.json"),
        r#"{"token": "abc"}"#,
    )
    .unwrap();

    let cred_out = tempfile::NamedTempFile::new().unwrap();
    let account_out = tempfile::NamedTempFile::new().unwrap();
    let cred_path = cred_out.path().to_str().unwrap().to_owned();
    let account_path = account_out.path().to_str().unwrap().to_owned();

    set_config_dir(source.path());
    let spec = RunSpec::new("hi", "haiku").env(env(&[
        ("SPAWNLLM_FAKE_CRED_OUT", &cred_path),
        ("SPAWNLLM_FAKE_ACCOUNT_OUT", &account_path),
    ]));
    let response = spawnllm::run_on(&Backend::Claude, spec).await;
    clear_config_dir();

    response.outcome.expect("isolated claude run succeeds");
    assert_eq!(
        std::fs::read_to_string(&cred_path).unwrap(),
        r#"{"token": "abc"}"#
    );
    let account: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&account_path).unwrap()).unwrap();
    assert!(
        account.get("mcpServers").is_none(),
        "mcpServers must be stripped from the seeded account"
    );
    assert_eq!(
        account.get("account").and_then(serde_json::Value::as_str),
        Some("me")
    );
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn empty_claude_config_dir_uses_the_default_home() {
    common::fixtures();
    let _guard = common::ENV_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    let source = tempfile::tempdir().unwrap();
    std::fs::create_dir(source.path().join(".claude")).unwrap();
    std::fs::write(
        source.path().join(".claude/.credentials.json"),
        r#"{"token": "home-token"}"#,
    )
    .unwrap();
    let cred_out = tempfile::NamedTempFile::new().unwrap();
    let cred_path = cred_out.path().to_str().unwrap().to_owned();
    let original_home = std::env::var_os("HOME");

    unsafe {
        std::env::set_var("HOME", source.path());
        std::env::set_var("CLAUDE_CONFIG_DIR", "");
    }
    let spec = RunSpec::new("hi", "haiku").env(env(&[("SPAWNLLM_FAKE_CRED_OUT", &cred_path)]));
    let response = spawnllm::run_on(&Backend::Claude, spec).await;
    clear_config_dir();
    match original_home {
        Some(home) => unsafe { std::env::set_var("HOME", home) },
        None => unsafe { std::env::remove_var("HOME") },
    }

    response.outcome.expect("isolated claude run succeeds");
    assert_eq!(
        std::fs::read_to_string(&cred_path).unwrap(),
        r#"{"token": "home-token"}"#
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn isolation_falls_back_to_the_keychain_for_credentials() {
    common::fixtures();
    let _guard = common::ENV_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    let source = tempfile::tempdir().unwrap();
    std::fs::write(source.path().join(".claude.json"), r#"{"account": "me"}"#).unwrap();

    let cred_out = tempfile::NamedTempFile::new().unwrap();
    let cred_path = cred_out.path().to_str().unwrap().to_owned();

    set_config_dir(source.path());
    let spec = RunSpec::new("hi", "haiku").env(env(&[("SPAWNLLM_FAKE_CRED_OUT", &cred_path)]));
    let response = spawnllm::run_on(&Backend::Claude, spec).await;
    clear_config_dir();

    response.outcome.expect("isolated claude run succeeds");
    assert_eq!(
        std::fs::read_to_string(&cred_path).unwrap(),
        "keychain-token-xyz"
    );
}

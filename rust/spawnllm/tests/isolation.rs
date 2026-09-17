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
    let modes_out = tempfile::NamedTempFile::new().unwrap();
    let cred_path = cred_out.path().to_str().unwrap().to_owned();
    let account_path = account_out.path().to_str().unwrap().to_owned();
    let modes_path = modes_out.path().to_str().unwrap().to_owned();

    set_config_dir(source.path());
    let spec = RunSpec::new("hi", "haiku").env(env(&[
        ("SPAWNLLM_FAKE_CRED_OUT", &cred_path),
        ("SPAWNLLM_FAKE_ACCOUNT_OUT", &account_path),
        ("SPAWNLLM_FAKE_MODES_OUT", &modes_path),
    ]));
    let response = spawnllm::run_on(&Backend::Claude, spec).await;
    clear_config_dir();

    response.outcome.expect("isolated claude run succeeds");
    assert_eq!(
        std::fs::read_to_string(&cred_path).unwrap(),
        r#"{"token": "abc"}"#
    );
    assert_eq!(
        std::fs::read_to_string(&modes_path).unwrap(),
        "drwx------\n-rw-------\n",
        "config dir mode then credentials file mode"
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
fn suffixed_keychain_service(config_dir_env: &str) -> String {
    use sha2::{Digest, Sha256};

    let digest = format!("{:x}", Sha256::digest(config_dir_env.as_bytes()));
    format!("Claude Code-credentials-{}", &digest[..8])
}

#[cfg(target_os = "macos")]
const HOST_KEYCHAIN_VARS: [&str; 3] = [
    "CLAUDE_CONFIG_DIR",
    "CLAUDE_SECURESTORAGE_CONFIG_DIR",
    "CLAUDE_CODE_CUSTOM_OAUTH_URL",
];

#[cfg(target_os = "macos")]
async fn keychain_seeded_run(
    host_env: &[(&str, &str)],
    keychain_service: Option<&str>,
) -> (String, String) {
    let cred_out = tempfile::NamedTempFile::new().unwrap();
    let argv_out = tempfile::NamedTempFile::new().unwrap();
    let cred_path = cred_out.path().to_str().unwrap().to_owned();
    let argv_path = argv_out.path().to_str().unwrap().to_owned();

    // SAFETY: gated by ENV_LOCK, held by the caller; see set_config_dir.
    unsafe {
        for var in HOST_KEYCHAIN_VARS {
            std::env::remove_var(var);
        }
        for (var, value) in host_env {
            std::env::set_var(var, value);
        }
        match keychain_service {
            Some(service) => std::env::set_var("SPAWNLLM_FAKE_KEYCHAIN_SERVICE", service),
            None => std::env::remove_var("SPAWNLLM_FAKE_KEYCHAIN_SERVICE"),
        }
        std::env::set_var("SPAWNLLM_FAKE_SECURITY_ARGV_OUT", &argv_path);
    }
    let spec = RunSpec::new("hi", "haiku").env(env(&[("SPAWNLLM_FAKE_CRED_OUT", &cred_path)]));
    let response = spawnllm::run_on(&Backend::Claude, spec).await;
    unsafe {
        for var in HOST_KEYCHAIN_VARS {
            std::env::remove_var(var);
        }
        std::env::remove_var("SPAWNLLM_FAKE_KEYCHAIN_SERVICE");
        std::env::remove_var("SPAWNLLM_FAKE_SECURITY_ARGV_OUT");
    }

    response.outcome.expect("isolated claude run succeeds");
    (
        std::fs::read_to_string(&cred_path).unwrap(),
        std::fs::read_to_string(&argv_path).unwrap(),
    )
}

#[cfg(target_os = "macos")]
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn config_dir_env_falls_back_to_the_suffixed_keychain_item() {
    common::fixtures();
    let _guard = common::ENV_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    let source = tempfile::tempdir().unwrap();
    std::fs::write(source.path().join(".claude.json"), r#"{"account": "me"}"#).unwrap();
    let config_dir_env = source.path().to_str().unwrap();
    let service = suffixed_keychain_service(config_dir_env);

    let (credentials, argv) =
        keychain_seeded_run(&[("CLAUDE_CONFIG_DIR", config_dir_env)], Some(&service)).await;

    assert_eq!(credentials, "keychain-token-xyz");
    assert_eq!(argv, format!("find-generic-password\n-s\n{service}\n-w\n"));
}

#[cfg(target_os = "macos")]
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn empty_securestorage_env_reads_the_bare_item_over_config_dir_env() {
    common::fixtures();
    let _guard = common::ENV_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    let source = tempfile::tempdir().unwrap();
    std::fs::write(source.path().join(".claude.json"), r#"{"account": "me"}"#).unwrap();
    let config_dir_env = source.path().to_str().unwrap();

    let (credentials, argv) = keychain_seeded_run(
        &[
            ("CLAUDE_CONFIG_DIR", config_dir_env),
            ("CLAUDE_SECURESTORAGE_CONFIG_DIR", ""),
        ],
        Some("Claude Code-credentials"),
    )
    .await;

    assert_eq!(credentials, "keychain-token-xyz");
    assert_eq!(
        argv,
        "find-generic-password\n-s\nClaude Code-credentials\n-w\n"
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn securestorage_env_names_the_hashed_item_over_config_dir_env() {
    common::fixtures();
    let _guard = common::ENV_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    let source = tempfile::tempdir().unwrap();
    std::fs::write(source.path().join(".claude.json"), r#"{"account": "me"}"#).unwrap();
    let config_dir_env = source.path().to_str().unwrap();
    let secure = tempfile::tempdir().unwrap();
    let securestorage_env = secure.path().to_str().unwrap();
    let service = suffixed_keychain_service(securestorage_env);

    let (credentials, argv) = keychain_seeded_run(
        &[
            ("CLAUDE_CONFIG_DIR", config_dir_env),
            ("CLAUDE_SECURESTORAGE_CONFIG_DIR", securestorage_env),
        ],
        Some(&service),
    )
    .await;

    assert_eq!(credentials, "keychain-token-xyz");
    assert_eq!(argv, format!("find-generic-password\n-s\n{service}\n-w\n"));
}

#[cfg(target_os = "macos")]
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn custom_oauth_url_env_names_the_custom_oauth_item() {
    common::fixtures();
    let _guard = common::ENV_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    let source = tempfile::tempdir().unwrap();
    std::fs::write(source.path().join(".claude.json"), r#"{"account": "me"}"#).unwrap();
    let config_dir_env = source.path().to_str().unwrap();
    let service = suffixed_keychain_service(config_dir_env).replace(
        "Claude Code-credentials",
        "Claude Code-custom-oauth-credentials",
    );

    let (credentials, argv) = keychain_seeded_run(
        &[
            ("CLAUDE_CONFIG_DIR", config_dir_env),
            ("CLAUDE_CODE_CUSTOM_OAUTH_URL", "https://oauth.example.test"),
        ],
        Some(&service),
    )
    .await;

    assert_eq!(credentials, "keychain-token-xyz");
    assert_eq!(argv, format!("find-generic-password\n-s\n{service}\n-w\n"));
}

#[cfg(target_os = "macos")]
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn trailing_slash_config_dir_env_is_hashed_as_set() {
    common::fixtures();
    let _guard = common::ENV_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    let source = tempfile::tempdir().unwrap();
    std::fs::write(source.path().join(".claude.json"), r#"{"account": "me"}"#).unwrap();
    let config_dir_env = format!("{}/", source.path().to_str().unwrap());
    let service = suffixed_keychain_service(&config_dir_env);
    assert_ne!(
        service,
        suffixed_keychain_service(source.path().to_str().unwrap())
    );

    let (credentials, argv) =
        keychain_seeded_run(&[("CLAUDE_CONFIG_DIR", &config_dir_env)], Some(&service)).await;

    assert_eq!(credentials, "keychain-token-xyz");
    assert_eq!(argv, format!("find-generic-password\n-s\n{service}\n-w\n"));
}

#[cfg(target_os = "macos")]
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn default_home_falls_back_to_the_bare_keychain_item() {
    common::fixtures();
    let _guard = common::ENV_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    let home = tempfile::tempdir().unwrap();
    std::fs::create_dir(home.path().join(".claude")).unwrap();
    std::fs::write(home.path().join(".claude.json"), r#"{"account": "me"}"#).unwrap();
    let original_home = std::env::var_os("HOME");
    unsafe { std::env::set_var("HOME", home.path()) };

    let (credentials, argv) = keychain_seeded_run(&[], Some("Claude Code-credentials")).await;

    match original_home {
        Some(value) => unsafe { std::env::set_var("HOME", value) },
        None => unsafe { std::env::remove_var("HOME") },
    }
    assert_eq!(credentials, "keychain-token-xyz");
    assert_eq!(
        argv,
        "find-generic-password\n-s\nClaude Code-credentials\n-w\n"
    );
}

#[cfg(target_os = "macos")]
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn keychain_miss_seeds_no_credentials() {
    common::fixtures();
    let _guard = common::ENV_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    let source = tempfile::tempdir().unwrap();
    std::fs::write(source.path().join(".claude.json"), r#"{"account": "me"}"#).unwrap();
    let config_dir_env = source.path().to_str().unwrap();

    let (credentials, argv) =
        keychain_seeded_run(&[("CLAUDE_CONFIG_DIR", config_dir_env)], None).await;

    assert_eq!(credentials, "");
    assert_eq!(
        argv,
        format!(
            "find-generic-password\n-s\n{}\n-w\n",
            suffixed_keychain_service(config_dir_env)
        )
    );
}

use std::time::Duration;

use spawnllm::{Backend, BackendStatus};

#[tokio::test]
async fn check_status_reports_not_installed_when_the_binary_is_absent() {
    // This test owns its process; pointing PATH at an empty dir makes `which` miss.
    // SAFETY: the only env mutation in this single-test binary.
    unsafe { std::env::set_var("PATH", "/spawnllm-nonexistent-dir") };
    match Backend::Claude.check_status(Duration::from_secs(1)).await {
        BackendStatus::NotInstalled {
            binary,
            install_hint,
        } => {
            assert_eq!(binary, "claude");
            assert!(!install_hint.is_empty());
        }
        other => panic!("expected NotInstalled, got {other:?}"),
    }
}

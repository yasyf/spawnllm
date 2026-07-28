use std::path::{Path, PathBuf};

pub(crate) fn platform() -> &'static str {
    if cfg!(target_os = "macos") {
        "darwin"
    } else {
        std::env::consts::OS
    }
}

pub(crate) fn home() -> String {
    std::env::var("HOME").unwrap_or_default()
}

pub(crate) fn cache_dir() -> PathBuf {
    let home = PathBuf::from(home());
    if cfg!(target_os = "macos") {
        return home.join("Library/Caches");
    }
    match std::env::var_os("XDG_CACHE_HOME") {
        Some(dir) => PathBuf::from(dir),
        None => home.join(".cache"),
    }
}

pub(crate) fn which(binary: &str) -> Option<PathBuf> {
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

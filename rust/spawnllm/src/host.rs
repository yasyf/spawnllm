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

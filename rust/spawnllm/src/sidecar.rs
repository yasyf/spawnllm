use std::io::{Error, ErrorKind, Result, Write};
use std::path::{Path, PathBuf};

use tempfile::NamedTempFile;

use crate::host::{cache_dir, which};

pub(crate) const BINARY: &str = "spawnllm-apple";

pub(crate) const INSTALL_HINT: &str = "brew install yasyf/tap/binrun";

const DESCRIPTOR: &str = include_str!("../spawnllm-apple.binrun");

pub(crate) fn launch() -> Result<Vec<String>> {
    if !cfg!(target_os = "macos") {
        return Err(Error::new(
            ErrorKind::Unsupported,
            format!("{BINARY} runs only on macOS"),
        ));
    }
    if let Some(path) = which(BINARY) {
        return Ok(vec![path.to_string_lossy().into_owned()]);
    }
    let binrun = which("binrun").ok_or_else(|| {
        Error::new(
            ErrorKind::NotFound,
            format!("neither {BINARY} nor binrun is on PATH"),
        )
    })?;
    Ok(vec![
        binrun.to_string_lossy().into_owned(),
        descriptor_path()?.to_string_lossy().into_owned(),
    ])
}

fn descriptor_path() -> Result<PathBuf> {
    let dir = cache_dir().join("spawnllm");
    let path = dir.join(format!("{BINARY}.binrun"));
    if std::fs::read_to_string(&path).is_ok_and(|current| current == DESCRIPTOR) {
        return Ok(path);
    }
    std::fs::create_dir_all(&dir)?;
    materialize(&dir, &path)?;
    Ok(path)
}

// Renaming into place keeps a concurrent binrun from reading a half-written descriptor.
fn materialize(dir: &Path, path: &Path) -> Result<()> {
    let mut handle = NamedTempFile::new_in(dir)?;
    handle.write_all(DESCRIPTOR.as_bytes())?;
    handle.flush()?;
    handle.persist(path).map_err(|error| error.error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn the_platform_gate_lets_macos_through() {
        assert!(
            !matches!(launch(), Err(ref error) if error.kind() == ErrorKind::Unsupported),
            "the platform gate rejected macOS"
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn the_sidecar_refuses_to_resolve_off_macos() {
        assert_eq!(launch().unwrap_err().kind(), ErrorKind::Unsupported);
    }
}

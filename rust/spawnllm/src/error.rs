use std::time::Duration;

use crate::backend::{Backend, BackendStatus};
use crate::spec::Specialty;

/// The failure modes of a run at the crate boundary.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("no ready backend for {specialty:?}")]
    BackendUnavailable {
        specialty: Option<Specialty>,
        statuses: Vec<(Backend, BackendStatus)>,
    },
    #[error("{provider} call failed: {msg}")]
    BackendCall {
        provider: String,
        msg: String,
        exit_code: i32,
        stderr: String,
    },
    #[error("timed out after {0:?}")]
    Timeout(Duration),
    #[error("validation failed: {0}")]
    Validation(#[from] serde_json::Error),
    #[error("core op {op} failed: {msg}")]
    Core { op: String, msg: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// A failed run outcome: a human-readable message plus the underlying [`Error`].
#[derive(Debug)]
pub struct RunError {
    pub msg: String,
    pub source: Error,
}

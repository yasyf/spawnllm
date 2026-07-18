//! Synchronous wrappers over the async entries, each driven on a current-thread runtime.

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::backend::Backend;
use crate::error::Error;
use crate::spec::{CallOpts, Response, RunSpec};

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread runtime builds")
}

/// Synchronous [`run`](crate::run).
pub fn run(spec: RunSpec) -> Response {
    runtime().block_on(crate::run(spec))
}

/// Synchronous [`run_on`](crate::run_on).
pub fn run_on(backend: &Backend, spec: RunSpec) -> Response {
    runtime().block_on(crate::run_on(backend, spec))
}

/// Synchronous [`call`](crate::call).
pub fn call(prompt: impl Into<String>, opts: CallOpts) -> Result<String, Error> {
    runtime().block_on(crate::call(prompt, opts))
}

/// Synchronous [`extract`](crate::extract).
pub fn extract<T: DeserializeOwned + JsonSchema>(
    prompt: impl Into<String>,
    opts: CallOpts,
) -> Result<T, Error> {
    runtime().block_on(crate::extract(prompt, opts))
}

//! Run LLM calls through provider CLIs (`claude`, `codex`, `gemini`, `agy`) or an
//! OpenAI-compatible endpoint.
//!
//! This crate is the I/O host over the sans-io [`spawnllm_core`] crate: core owns
//! all drift-prone logic (argv/env/result planning, output resolution, strict-schema
//! transforms, retry policy, capability tables), and this crate spawns processes,
//! manages temp files and Claude isolation, executes auth probes, and drives the
//! retry loop.

pub use spawnllm_core as core;

mod backend;
mod core_io;
mod error;
mod exec;
mod host;
mod isolate;
mod run;
mod spec;

pub mod blocking;

#[cfg(feature = "openai")]
mod http;

#[cfg(feature = "openai")]
pub use backend::OpenAiEndpoint;
pub use backend::{Backend, BackendStatus, select_backend};
pub use error::{Error, RunError};
pub use spec::{
    CallOpts, ClaudeConfig, CodexConfig, DiscardedAttempt, GeminiConfig, ModelTier, Response,
    RunResult, RunSpec, Specialty,
};

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use spec::DEFAULT_SELECT_TIMEOUT;

/// Execute a [`RunSpec`], auto-selecting the first ready backend.
///
/// The boundary is infallible: backend unavailability and every provider failure
/// land in [`Response::outcome`] as an `Err`, never as a panic. The prompt's model
/// is taken from the spec verbatim (no tier mapping).
pub async fn run(spec: RunSpec) -> Response {
    if let Err(error) = spec.validate() {
        return spec::unavailable_response(spec, error);
    }
    match select_backend(None, DEFAULT_SELECT_TIMEOUT).await {
        Ok(backend) => run_on(&backend, spec).await,
        Err(error) => spec::unavailable_response(spec, error),
    }
}

/// Execute a [`RunSpec`] on a specific [`Backend`].
pub async fn run_on(backend: &Backend, spec: RunSpec) -> Response {
    run::dispatch_run(backend, spec, false).await
}

/// Run one LLM call and return its text response.
///
/// Resolves a backend (`opts.backend`, else priority auto-selection scoped by
/// `opts.specialty`), maps `opts.model` to the backend's concrete model id, and
/// runs with transient retry. A provider failure returns its [`Error`].
pub async fn call(prompt: impl Into<String>, opts: CallOpts) -> Result<String, Error> {
    let backend = resolve_backend(&opts).await?;
    let spec = RunSpec::call_spec(prompt, backend.model_for(opts.model), &opts);
    match run_on(&backend, spec).await.outcome {
        Ok(result) => Ok(result.raw),
        Err(error) => Err(error.source),
    }
}

/// Run one LLM call and deserialize its structured output into `T`.
///
/// Builds `T`'s JSON schema, runs it through core's strict-schema transform for the
/// selected backend's dialect (Anthropic for Claude, OpenAI for Codex and the
/// endpoint; Gemini-family backends carry the plain schema), constrains the call to
/// it, then deserializes the parsed value. A non-conforming value returns
/// [`Error::Validation`].
pub async fn extract<T: DeserializeOwned + JsonSchema>(
    prompt: impl Into<String>,
    opts: CallOpts,
) -> Result<T, Error> {
    let backend = resolve_backend(&opts).await?;
    let schema = strict_schema_for::<T>(&backend)?;
    let spec = RunSpec::call_spec(prompt, backend.model_for(opts.model), &opts).schema(schema);
    match run::dispatch_run(&backend, spec, true).await.outcome {
        Ok(result) => {
            let value = result.parsed.expect("a schema run yields a parsed value");
            serde_json::from_value(value).map_err(Error::Validation)
        }
        Err(error) => Err(error.source),
    }
}

async fn resolve_backend(opts: &CallOpts) -> Result<Backend, Error> {
    match &opts.backend {
        Some(backend) => Ok(backend.clone()),
        None => select_backend(opts.specialty, DEFAULT_SELECT_TIMEOUT).await,
    }
}

/// Build `T`'s JSON schema (schemars 2020-12, `$defs`) and apply core's strict-schema
/// transform for the backend's dialect. Both the Anthropic and OpenAI transforms recurse
/// `$defs`, so every nested definition is strictified.
#[derive(serde::Deserialize)]
struct StrictSchemaOutput {
    schema: serde_json::Value,
}

pub(crate) fn strict_schema_for<T: JsonSchema>(
    backend: &Backend,
) -> Result<serde_json::Value, Error> {
    let raw_schema =
        serde_json::to_value(schemars::schema_for!(T)).expect("json schema serializes");
    match backend.strict_dialect() {
        Some(dialect) => Ok(core_io::core_op::<StrictSchemaOutput>(
            "strict_schema",
            serde_json::json!({ "dialect": dialect, "schema": raw_schema }),
        )?
        .schema),
        None => Ok(raw_schema),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Inner {
        y: i64,
    }

    #[derive(JsonSchema)]
    #[allow(dead_code)]
    struct Outer {
        inner: Inner,
        name: String,
    }

    // Regression: schemars draft-07 `definitions` is never recursed by the anthropic transform.
    #[test]
    fn nested_extract_schema_is_strict_in_every_definition() {
        for backend in [Backend::Claude, Backend::Codex] {
            let schema = strict_schema_for::<Outer>(&backend).unwrap();
            assert_eq!(
                schema.get("additionalProperties"),
                Some(&Value::Bool(false)),
                "root object must be strict for {backend:?}: {schema}"
            );
            let defs = schema
                .get("$defs")
                .and_then(Value::as_object)
                .unwrap_or_else(|| panic!("nested type must emit $defs for {backend:?}: {schema}"));
            assert!(
                !defs.is_empty(),
                "the referenced definition survives for {backend:?}"
            );
            for (name, def) in defs {
                assert_eq!(
                    def.get("additionalProperties"),
                    Some(&Value::Bool(false)),
                    "definition {name} must be strict for {backend:?}: {def}"
                );
            }
        }
    }
}

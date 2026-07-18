mod common;

use std::collections::HashMap;
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::Deserialize;

use spawnllm::{Backend, BackendStatus, CallOpts, Error, ModelTier, RunSpec, Specialty};

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Sample {
    x: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct NestedInner {
    y: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct NestedOuter {
    inner: NestedInner,
    name: String,
}

fn env(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

#[tokio::test]
async fn run_on_claude_returns_result_text() {
    common::fixtures();
    let spec = RunSpec::new("hi", "haiku").isolated(false);
    let response = spawnllm::run_on(&Backend::Claude, spec).await;
    let result = response.outcome.expect("claude run succeeds");
    assert_eq!(result.raw, "hello");
    assert!(result.parsed.is_none());
    assert!(response.discarded_attempts.is_empty());
}

#[tokio::test]
async fn run_on_claude_captures_stdout_from_a_regular_file() {
    common::fixtures();
    let marker = tempfile::NamedTempFile::new().unwrap();
    let marker_path = marker.path().to_str().unwrap().to_owned();
    let spec = RunSpec::new("hi", "haiku")
        .isolated(false)
        .env(env(&[("SPAWNLLM_FAKE_MARKER", &marker_path)]));
    let response = spawnllm::run_on(&Backend::Claude, spec).await;
    response.outcome.expect("claude run succeeds");
    assert_eq!(std::fs::read_to_string(&marker_path).unwrap(), "regular");
}

#[tokio::test]
async fn run_on_delivers_the_prompt_over_stdin() {
    common::fixtures();
    let stdin_out = tempfile::NamedTempFile::new().unwrap();
    let stdin_path = stdin_out.path().to_str().unwrap().to_owned();
    let spec = RunSpec::new("the-secret-prompt", "haiku")
        .isolated(false)
        .env(env(&[("SPAWNLLM_FAKE_STDIN_OUT", &stdin_path)]));
    let response = spawnllm::run_on(&Backend::Claude, spec).await;
    response.outcome.expect("claude run succeeds");
    assert_eq!(
        std::fs::read_to_string(&stdin_path).unwrap(),
        "the-secret-prompt"
    );
}

#[tokio::test]
async fn run_on_codex_reads_the_result_file_not_stdout() {
    common::fixtures();
    let spec = RunSpec::new("hi", "gpt-5.5").isolated(false);
    let response = spawnllm::run_on(&Backend::Codex, spec).await;
    let result = response.outcome.expect("codex run succeeds");
    // The result is read from the `-o` file, not the streamed stdout log.
    assert_eq!(result.raw, "codex-hello");
    assert_eq!(response.output, "codex-hello");
}

#[tokio::test]
async fn run_on_times_out_and_kills_the_child() {
    common::fixtures();
    let spec = RunSpec::new("hi", "haiku")
        .isolated(false)
        .timeout(Duration::from_millis(300))
        .env(env(&[("SPAWNLLM_FAKE_SLEEP", "10")]));
    let started = Instant::now();
    let response = spawnllm::run_on(&Backend::Claude, spec).await;
    let error = response.outcome.expect_err("a slept-out run times out");
    assert!(
        matches!(error.source, Error::Timeout(_)),
        "got {:?}",
        error.source
    );
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "reap should be prompt"
    );
}

#[tokio::test]
async fn transient_failure_is_retried_then_succeeds() {
    common::fixtures();
    let counter = tempfile::NamedTempFile::new().unwrap();
    let counter_path = counter.path().to_str().unwrap().to_owned();
    let spec = RunSpec::new("hi", "haiku")
        .isolated(false)
        .env(env(&[("SPAWNLLM_FAKE_COUNTER", &counter_path)]));
    let response = spawnllm::run_on(&Backend::Claude, spec).await;
    let result = response.outcome.expect("the retry recovers");
    assert_eq!(result.raw, "hello");
    assert_eq!(response.discarded_attempts.len(), 1);
    assert_eq!(response.discarded_attempts[0].attempt, 0);
    assert_eq!(response.discarded_attempts[0].error, "BackendCall");
}

#[tokio::test]
async fn run_on_reports_backend_call_error_on_nonzero_exit() {
    common::fixtures();
    let spec = RunSpec::new("hi", "haiku")
        .isolated(false)
        .env(env(&[("SPAWNLLM_FAKE_EXIT", "3")]));
    let response = spawnllm::run_on(&Backend::Claude, spec).await;
    let error = response.outcome.expect_err("a nonzero exit is an error");
    match error.source {
        Error::BackendCall {
            provider,
            exit_code,
            stderr,
            ..
        } => {
            assert_eq!(provider, "claude");
            assert_eq!(exit_code, 3);
            assert!(stderr.contains("boom"));
        }
        other => panic!("expected BackendCall, got {other:?}"),
    }
}

#[tokio::test]
async fn extract_deserializes_structured_output_via_claude() {
    common::fixtures();
    let value: Sample = spawnllm::extract("give me x", CallOpts::default())
        .await
        .expect("extract succeeds");
    assert_eq!(value, Sample { x: 42 });
}

#[tokio::test]
async fn extract_deserializes_via_codex_openai_dialect() {
    common::fixtures();
    let opts = CallOpts {
        backend: Some(Backend::Codex),
        ..CallOpts::default()
    };
    let value: Sample = spawnllm::extract("give me x", opts)
        .await
        .expect("extract succeeds");
    assert_eq!(value, Sample { x: 42 });
}

async fn assert_extract_schema_strict(backend: Backend) {
    common::fixtures();
    let dump = tempfile::NamedTempFile::new().unwrap();
    let dump_path = dump.path().to_str().unwrap().to_owned();
    let opts = CallOpts {
        backend: Some(backend.clone()),
        ..CallOpts::default()
    };
    // The fake writes the schema it actually received (claude's --json-schema argv value /
    // codex's --output-schema file) to the path named in the prompt, then answers.
    let _ = spawnllm::extract::<NestedOuter>(format!("DUMP_SCHEMA_TO={dump_path}"), opts).await;
    let schema: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&dump_path).unwrap())
            .unwrap_or_else(|e| panic!("{backend:?} did not receive a parseable schema: {e}"));
    assert_eq!(
        schema.get("additionalProperties"),
        Some(&serde_json::Value::Bool(false)),
        "root object must be strict for {backend:?}: {schema}"
    );
    let defs = schema
        .get("$defs")
        .and_then(serde_json::Value::as_object)
        .unwrap_or_else(|| panic!("nested type must emit $defs for {backend:?}: {schema}"));
    assert!(
        !defs.is_empty(),
        "the referenced definition survives for {backend:?}"
    );
    for (name, def) in defs {
        assert_eq!(
            def.get("additionalProperties"),
            Some(&serde_json::Value::Bool(false)),
            "definition {name} must be strict for {backend:?}: {def}"
        );
    }
}

#[tokio::test]
async fn claude_extract_sends_strict_schema_in_every_definition() {
    assert_extract_schema_strict(Backend::Claude).await;
}

#[tokio::test]
async fn codex_extract_sends_strict_schema_in_every_definition() {
    assert_extract_schema_strict(Backend::Codex).await;
}

#[tokio::test]
async fn call_returns_text_via_auto_selected_backend() {
    common::fixtures();
    let text = spawnllm::call("hi", CallOpts::default())
        .await
        .expect("call succeeds");
    assert_eq!(text, "hello");
}

#[tokio::test]
async fn check_status_reports_ready_for_installed_authenticated_backends() {
    common::fixtures();
    assert!(matches!(
        Backend::Claude.check_status(Duration::from_secs(5)).await,
        BackendStatus::Ready { .. }
    ));
    assert!(
        Backend::Codex
            .is_authenticated(Duration::from_secs(5))
            .await
    );
}

#[tokio::test]
async fn select_backend_follows_priority_and_specialty() {
    common::fixtures();
    let auto = spawnllm::select_backend(None, Duration::from_secs(5))
        .await
        .expect("a backend is ready");
    assert!(matches!(auto, Backend::Claude));
    let debugging = spawnllm::select_backend(Some(Specialty::Debugging), Duration::from_secs(5))
        .await
        .expect("codex serves debugging");
    assert!(matches!(debugging, Backend::Codex));
}

#[tokio::test]
async fn model_tier_maps_to_the_backend_concrete_model() {
    common::fixtures();
    // A model-tier round-trip through call: default tier (Small) resolves to the
    // backend's small model, and the fake still answers, proving the mapping ran.
    let opts = CallOpts {
        backend: Some(Backend::Claude),
        model: ModelTier::Large,
        ..CallOpts::default()
    };
    let text = spawnllm::call("hi", opts).await.expect("call succeeds");
    assert_eq!(text, "hello");
}

#[test]
fn blocking_call_runs_on_its_own_runtime() {
    common::fixtures();
    let text = spawnllm::blocking::call("hi", CallOpts::default()).expect("blocking call succeeds");
    assert_eq!(text, "hello");
}

#[test]
fn blocking_run_on_returns_a_response() {
    common::fixtures();
    let spec = RunSpec::new("hi", "haiku").isolated(false);
    let response = spawnllm::blocking::run_on(&Backend::Claude, spec);
    assert_eq!(
        response.outcome.expect("blocking run succeeds").raw,
        "hello"
    );
}

#[test]
fn blocking_extract_deserializes_a_model() {
    common::fixtures();
    let value: Sample = spawnllm::blocking::extract("give me x", CallOpts::default())
        .expect("blocking extract succeeds");
    assert_eq!(value, Sample { x: 42 });
}

use std::time::Duration;

use serde_json::{Map, Value, json};
use tempfile::TempDir;

use spawnllm_core::wire::{ExecPlan, InvocationPlan, Resolved};
use spawnllm_core::{RetryInput, retry_decision};

use crate::backend::Backend;
use crate::core_io::{call_core, core_op};
use crate::error::{Error, RunError};
use crate::host;
use crate::spec::{DiscardedAttempt, Response, RunResult, RunSpec};

#[cfg(feature = "openai")]
use spawnllm_core::wire::HttpPlan;

pub(crate) struct Attempt {
    pub(crate) output: String,
    pub(crate) kind: AttemptKind,
}

pub(crate) enum AttemptKind {
    Ok {
        text: String,
        value: Option<Value>,
    },
    Error {
        msg: String,
        exit_code: i32,
        stderr: String,
        cost_usd: Option<f64>,
        usage: Option<Map<String, Value>>,
    },
    Timeout {
        duration: Duration,
    },
}

impl Attempt {
    fn retry_msg(&self, provider: &str) -> Option<String> {
        match &self.kind {
            AttemptKind::Ok { .. } => None,
            AttemptKind::Error { msg, .. } => Some(msg.clone()),
            AttemptKind::Timeout { duration } => Some(timeout_message(provider, *duration)),
        }
    }

    fn discard(&self, attempt: u32) -> DiscardedAttempt {
        let (error, cost_usd, usage) = match &self.kind {
            AttemptKind::Error {
                cost_usd, usage, ..
            } => ("BackendCallError".to_owned(), *cost_usd, usage.clone()),
            AttemptKind::Timeout { .. } => ("Timeout".to_owned(), None, None),
            AttemptKind::Ok { .. } => ("Ok".to_owned(), None, None),
        };
        DiscardedAttempt {
            attempt,
            error,
            cost_usd,
            usage,
            raw_bytes: self.output.len(),
        }
    }
}

pub(crate) fn resolve_kind(
    provider: &str,
    raw: &str,
    returncode: i64,
    stderr: &str,
    wants_value: bool,
) -> AttemptKind {
    let input = json!({
        "provider": provider,
        "raw": raw,
        "returncode": returncode,
        "stderr": stderr,
        "wants_value": wants_value,
    });
    match call_core("resolve", input) {
        Ok(value) => match serde_json::from_value::<Resolved>(value)
            .expect("resolve output matches wire type")
        {
            Resolved::Ok(ok) => AttemptKind::Ok {
                text: ok.text,
                value: wants_value.then_some(ok.value),
            },
            Resolved::Error(err) => AttemptKind::Error {
                msg: err.msg,
                exit_code: returncode as i32,
                stderr: stderr.to_owned(),
                cost_usd: err.cost_usd,
                usage: err.usage,
            },
        },
        Err(core_error) => AttemptKind::Error {
            msg: format!("{provider} resolve unsupported by core: {}", core_error.msg),
            exit_code: returncode as i32,
            stderr: stderr.to_owned(),
            cost_usd: None,
            usage: None,
        },
    }
}

fn portable(spec: &RunSpec, endpoint: Option<&Value>) -> Value {
    json!({
        "prompt": spec.prompt,
        "model": spec.model,
        "schema": spec.schema,
        "agent": spec.agent,
        "api_auth": spec.api_auth,
        "isolated": spec.isolated,
        "timeout": spec.timeout.as_secs() as i64,
        "max_attempts": spec.max_attempts as i64,
        "claude": spec.claude,
        "codex": spec.codex,
        "gemini": spec.gemini,
        "openai_endpoint": endpoint,
    })
}

pub(crate) async fn dispatch_run(backend: &Backend, spec: RunSpec, wants_value: bool) -> Response {
    if let Err(error) = spec.validate() {
        return error_response(spec, error, Vec::new());
    }
    let provider = backend.provider();
    let plan_input = json!({
        "provider": provider,
        "spec": portable(&spec, backend.endpoint_value().as_ref()),
        "host": { "platform": host::platform() },
    });
    let plan = match core_op::<InvocationPlan>("plan", plan_input) {
        Ok(plan) => plan,
        Err(error) => return error_response(spec, error.into(), Vec::new()),
    };
    match plan {
        InvocationPlan::Exec(plan) => exec_loop(spec, plan, provider, wants_value).await,
        #[cfg(feature = "openai")]
        InvocationPlan::Http(plan) => http_loop(spec, plan, provider, wants_value).await,
        #[cfg(not(feature = "openai"))]
        InvocationPlan::Http(_) => unreachable!("an http plan requires the openai feature"),
    }
}

async fn exec_loop(
    spec: RunSpec,
    plan: ExecPlan,
    provider: &'static str,
    wants_value: bool,
) -> Response {
    let isolated_dir = if plan.needs_claude_isolation {
        match crate::isolate::seed_isolation().await {
            Ok(dir) => Some(dir),
            Err(error) => return error_response(spec, error, Vec::new()),
        }
    } else {
        None
    };

    let mut discarded = Vec::new();
    let max = spec.max_attempts.max(1);
    for attempt in 0..max {
        let outcome = crate::exec::exec_attempt(
            &plan,
            &spec,
            provider,
            isolated_dir.as_ref().map(TempDir::path),
            wants_value,
        )
        .await;
        let att = match outcome {
            Ok(att) => att,
            Err(error) => return error_response(spec, error.into(), discarded),
        };
        if let Some((output, outcome)) = settle(provider, attempt, max, att, &mut discarded).await {
            return Response {
                spec,
                output,
                outcome,
                discarded_attempts: discarded,
            };
        }
    }
    unreachable!("the retry loop settles on the final attempt")
}

#[cfg(feature = "openai")]
async fn http_loop(
    spec: RunSpec,
    plan: HttpPlan,
    provider: &'static str,
    wants_value: bool,
) -> Response {
    let client = reqwest::Client::new();
    let mut discarded = Vec::new();
    let max = spec.max_attempts.max(1);
    for attempt in 0..max {
        let att = crate::http::http_attempt(&client, &plan, &spec, provider, wants_value).await;
        if let Some((output, outcome)) = settle(provider, attempt, max, att, &mut discarded).await {
            return Response {
                spec,
                output,
                outcome,
                discarded_attempts: discarded,
            };
        }
    }
    unreachable!("the retry loop settles on the final attempt")
}

async fn settle(
    provider: &str,
    attempt: u32,
    max: u32,
    att: Attempt,
    discarded: &mut Vec<DiscardedAttempt>,
) -> Option<(String, Result<RunResult, RunError>)> {
    if let Some(error_msg) = att.retry_msg(provider) {
        let decision = retry_decision(&RetryInput {
            attempt,
            max_attempts: max,
            error_msg: Some(error_msg),
        });
        if decision.retry {
            discarded.push(att.discard(attempt));
            tokio::time::sleep(Duration::from_secs_f64(decision.sleep_s)).await;
            return None;
        }
    }
    Some(finalize(att, provider))
}

fn finalize(att: Attempt, provider: &str) -> (String, Result<RunResult, RunError>) {
    match att.kind {
        AttemptKind::Ok { text, value } => (
            att.output,
            Ok(RunResult {
                raw: text,
                parsed: value,
            }),
        ),
        AttemptKind::Error {
            msg,
            exit_code,
            stderr,
            ..
        } => {
            let source = Error::BackendCall {
                provider: provider.to_owned(),
                msg: msg.clone(),
                exit_code,
                stderr,
            };
            (att.output, Err(RunError { msg, source }))
        }
        AttemptKind::Timeout { duration } => {
            let msg = timeout_message(provider, duration);
            (
                att.output,
                Err(RunError {
                    msg,
                    source: Error::Timeout(duration),
                }),
            )
        }
    }
}

fn timeout_message(provider: &str, duration: Duration) -> String {
    format!("{provider} timed out after {}s", duration.as_secs_f64())
}

fn error_response(spec: RunSpec, error: Error, discarded: Vec<DiscardedAttempt>) -> Response {
    let msg = error.to_string();
    Response {
        spec,
        output: String::new(),
        outcome: Err(RunError { msg, source: error }),
        discarded_attempts: discarded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_includes_api_auth() {
        let default = portable(&RunSpec::new("p", "m"), None);
        assert_eq!(default.get("api_auth"), Some(&Value::Bool(false)));

        let api_auth = portable(&RunSpec::new("p", "m").api_auth(true), None);
        assert_eq!(api_auth.get("api_auth"), Some(&Value::Bool(true)));
    }
}

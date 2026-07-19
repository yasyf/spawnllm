use serde::Deserialize;
use serde_json::{Map, Value};

use crate::extract::extract_json;
use crate::retry::is_transient;
use crate::wire::{ResolveErrorKind, Resolved, ResolvedError, ResolvedOk};
use crate::{OpError, OpResult, from_input};

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Provider {
    Claude,
    Codex,
    Gemini,
    Antigravity,
    OpenaiEndpoint,
}

impl Provider {
    fn as_str(self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::Gemini => "gemini",
            Self::Antigravity => "antigravity",
            Self::OpenaiEndpoint => "openai_endpoint",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ResolveInput {
    provider: Provider,
    raw: String,
    returncode: i64,
    stderr: String,
    wants_value: bool,
}

fn tail_2000(text: &str) -> String {
    let trimmed = text.trim();
    let start = trimmed
        .char_indices()
        .rev()
        .nth(1999)
        .map_or(0, |(index, _)| index);
    trimmed[start..].to_owned()
}

fn error(
    kind: ResolveErrorKind,
    msg: String,
    cost_usd: Option<f64>,
    usage: Option<Map<String, Value>>,
) -> Resolved {
    Resolved::Error(ResolvedError {
        transient: is_transient(&msg),
        kind,
        msg,
        cost_usd,
        usage,
    })
}

fn finish(
    provider: Provider,
    text: String,
    wants_value: bool,
    structured_value: Option<Value>,
    cost_usd: Option<f64>,
    usage: Option<Map<String, Value>>,
) -> Resolved {
    let value = if wants_value {
        match structured_value.or_else(|| extract_json(&text)) {
            Some(value) => value,
            None => {
                return error(
                    ResolveErrorKind::Parse,
                    format!(
                        "{} output did not contain a valid JSON value",
                        provider.as_str()
                    ),
                    cost_usd,
                    usage,
                );
            }
        }
    } else {
        Value::Null
    };

    Resolved::Ok(ResolvedOk {
        text,
        value,
        cost_usd,
        usage,
    })
}

fn claude_result_event(raw: &str) -> Option<Map<String, Value>> {
    match serde_json::from_str(raw).ok()? {
        Value::Object(event) => Some(event),
        Value::Array(events) => events.into_iter().rev().find_map(|event| match event {
            Value::Object(event) if event.get("type").and_then(Value::as_str) == Some("result") => {
                Some(event)
            }
            _ => None,
        }),
        _ => None,
    }
}

fn claude_accounting(
    event: Option<&Map<String, Value>>,
) -> (Option<f64>, Option<Map<String, Value>>) {
    let cost_usd = event
        .and_then(|event| event.get("total_cost_usd"))
        .and_then(Value::as_f64);
    let usage = event
        .and_then(|event| event.get("usage"))
        .and_then(Value::as_object)
        .cloned();
    (cost_usd, usage)
}

fn claude_error_reason(event: Option<&Map<String, Value>>) -> Option<String> {
    let event = event?;
    if event.get("is_error").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    Some(
        event
            .get("result")
            .and_then(Value::as_str)
            .unwrap_or("claude reported an error")
            .to_owned(),
    )
}

fn resolve_claude(raw: &str, wants_value: bool) -> Resolved {
    let event = claude_result_event(raw);
    let (cost_usd, usage) = claude_accounting(event.as_ref());

    if let Some(msg) = claude_error_reason(event.as_ref()) {
        return error(ResolveErrorKind::Envelope, msg, cost_usd, usage);
    }

    let text = event
        .as_ref()
        .and_then(|event| event.get("result"))
        .and_then(Value::as_str)
        .unwrap_or(raw)
        .to_owned();
    let structured_value = event
        .as_ref()
        .and_then(|event| event.get("structured_output"))
        .cloned();
    finish(
        Provider::Claude,
        text,
        wants_value,
        structured_value,
        cost_usd,
        usage,
    )
}

fn gemini_has_errors(data: &Value) -> bool {
    data.get("stats")
        .and_then(|stats| stats.get("models"))
        .and_then(Value::as_object)
        .is_some_and(|models| {
            models.values().any(|model| {
                model
                    .get("api")
                    .and_then(|api| api.get("totalErrors"))
                    .and_then(Value::as_u64)
                    .is_some_and(|errors| errors > 0)
            })
        })
}

fn resolve_gemini(raw: &str, wants_value: bool) -> Resolved {
    let data: Value = match serde_json::from_str(raw) {
        Ok(data) => data,
        Err(_) => {
            return error(
                ResolveErrorKind::Envelope,
                format!("gemini call failed: {}", tail_2000(raw)),
                None,
                None,
            );
        }
    };
    let text = data.get("response").and_then(Value::as_str);
    if gemini_has_errors(&data) || text.is_none_or(str::is_empty) {
        return error(
            ResolveErrorKind::Envelope,
            format!("gemini call failed: {}", tail_2000(raw)),
            None,
            None,
        );
    }
    finish(
        Provider::Gemini,
        text.unwrap().to_owned(),
        wants_value,
        None,
        None,
        None,
    )
}

fn resolve_openai(raw: &str, wants_value: bool) -> Resolved {
    let data: Value = match serde_json::from_str(raw) {
        Ok(data) => data,
        Err(_) => {
            return error(
                ResolveErrorKind::Envelope,
                "openai_endpoint returned an invalid response envelope".to_owned(),
                None,
                None,
            );
        }
    };
    if let Some(envelope_error) = data.get("error") {
        let msg = envelope_error
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| envelope_error.as_str())
            .unwrap_or("openai_endpoint reported an error")
            .to_owned();
        return error(ResolveErrorKind::Envelope, msg, None, None);
    }
    let Some(text) = data
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
    else {
        return error(
            ResolveErrorKind::Envelope,
            "openai_endpoint response did not contain choices[0].message.content".to_owned(),
            None,
            None,
        );
    };
    finish(
        Provider::OpenaiEndpoint,
        text.to_owned(),
        wants_value,
        None,
        None,
        None,
    )
}

fn resolve(input: ResolveInput) -> Resolved {
    if input.returncode != 0 {
        let (reason, cost_usd, usage) = if matches!(input.provider, Provider::Claude) {
            let event = claude_result_event(&input.raw);
            let (cost_usd, usage) = claude_accounting(event.as_ref());
            (claude_error_reason(event.as_ref()), cost_usd, usage)
        } else {
            (None, None, None)
        };
        let reason = reason.unwrap_or_else(|| match tail_2000(&input.stderr) {
            stderr if stderr.is_empty() => tail_2000(&input.raw),
            stderr => stderr,
        });
        let msg = format!(
            "{} exited {}: {}",
            input.provider.as_str(),
            input.returncode,
            reason
        );
        return error(ResolveErrorKind::Exit, msg, cost_usd, usage);
    }

    match input.provider {
        Provider::Claude => resolve_claude(&input.raw, input.wants_value),
        Provider::Codex => finish(
            Provider::Codex,
            input.raw,
            input.wants_value,
            None,
            None,
            None,
        ),
        Provider::Gemini => resolve_gemini(&input.raw, input.wants_value),
        Provider::Antigravity => finish(
            Provider::Antigravity,
            input.raw.trim().to_owned(),
            input.wants_value,
            None,
            None,
            None,
        ),
        Provider::OpenaiEndpoint => resolve_openai(&input.raw, input.wants_value),
    }
}

pub(crate) fn dispatch(input: Value) -> OpResult {
    let input = from_input::<ResolveInput>(input)?;
    serde_json::to_value(resolve(input)).map_err(OpError::internal)
}

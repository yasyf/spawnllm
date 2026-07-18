use std::sync::LazyLock;

use regex_lite::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{OpError, OpResult, from_input};

static TRANSIENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b529\b|overloaded|rate.?limit|\b5\d\d\b").unwrap());

#[derive(Debug, Clone, Deserialize)]
pub struct RetryInput {
    pub attempt: u32,
    pub max_attempts: u32,
    pub error_msg: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RetryDecision {
    pub retry: bool,
    pub sleep_s: f64,
}

pub(crate) fn is_transient(msg: &str) -> bool {
    TRANSIENT.is_match(msg)
}

pub fn backoff(attempt: u32) -> f64 {
    3u64.checked_pow(attempt)
        .map_or(60.0, |p| p.saturating_mul(5).min(60) as f64)
}

pub fn retry_decision(input: &RetryInput) -> RetryDecision {
    let transient = input.error_msg.as_deref().is_some_and(is_transient);
    let retry = transient && input.attempt.saturating_add(1) < input.max_attempts;
    RetryDecision {
        retry,
        sleep_s: if retry { backoff(input.attempt) } else { 0.0 },
    }
}

pub(crate) fn dispatch(input: Value) -> OpResult {
    let input = from_input::<RetryInput>(input)?;
    serde_json::to_value(retry_decision(&input)).map_err(OpError::internal)
}

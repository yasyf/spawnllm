mod extract;
mod isolate;
mod plan;
mod probe;
mod resolve;
mod retry;
mod route;
mod schema;
pub mod wire;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub use extract::{ExtractInput, extract_json};
pub use retry::{RetryDecision, RetryInput, backoff, retry_decision};
pub use route::{Capabilities, ModelTiers, capabilities};

pub(crate) struct OpError {
    kind: &'static str,
    msg: String,
}

pub(crate) type OpResult = Result<Value, OpError>;

impl OpError {
    pub(crate) fn internal(error: serde_json::Error) -> Self {
        Self {
            kind: "internal",
            msg: error.to_string(),
        }
    }

    fn bad_input(error: &serde_json::Error) -> Self {
        Self {
            kind: "bad_input",
            msg: error.to_string(),
        }
    }

    fn to_json(&self) -> Value {
        json!({ "err": { "kind": self.kind, "msg": self.msg } })
    }
}

pub(crate) fn unimplemented(op: &str) -> OpError {
    OpError {
        kind: "unimplemented",
        msg: op.to_string(),
    }
}

pub(crate) fn from_input<T: DeserializeOwned>(input: Value) -> Result<T, OpError> {
    serde_json::from_value(input).map_err(|error| OpError::bad_input(&error))
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Version {
    pub core_version: &'static str,
    pub source_hash: &'static str,
}

pub fn version() -> Version {
    Version {
        core_version: env!("CARGO_PKG_VERSION"),
        source_hash: option_env!("SPAWNLLM_CORE_SRC_HASH").unwrap_or("dev"),
    }
}

#[derive(Deserialize)]
struct Request {
    op: String,
    #[serde(default)]
    input: Value,
}

fn run(op: &str, input: Value) -> OpResult {
    match op {
        "version" => serde_json::to_value(version()).map_err(OpError::internal),
        "capabilities" => route::dispatch(input),
        "retry_decision" => retry::dispatch(input),
        "extract_json" => extract::dispatch(input),
        "plan" => plan::dispatch(input),
        "resolve" => resolve::dispatch(input),
        "strict_schema" => schema::dispatch(input),
        "auth_probes" => probe::dispatch(input),
        "claude_isolation_sources" | "claude_isolation_seed" => isolate::dispatch(op, input),
        other => Err(OpError {
            kind: "unknown_op",
            msg: other.to_string(),
        }),
    }
}

pub fn dispatch(request_json: &str) -> String {
    let response = match serde_json::from_str::<Request>(request_json) {
        Ok(request) => match run(&request.op, request.input) {
            Ok(value) => json!({ "ok": value }),
            Err(error) => error.to_json(),
        },
        Err(error) => OpError::bad_input(&error).to_json(),
    };
    serde_json::to_string(&response).expect("response value is always serializable")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_reports_crate_version_and_source_hash() {
        let version = version();
        assert_eq!(version.core_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(version.core_version, "0.0.0");
        assert_eq!(
            version.source_hash,
            option_env!("SPAWNLLM_CORE_SRC_HASH").unwrap_or("dev")
        );
    }

    #[test]
    fn dispatch_version_reports_ok_payload() {
        let response: Value =
            serde_json::from_str(&dispatch(r#"{"op":"version","input":{}}"#)).unwrap();
        assert_eq!(response["ok"]["core_version"], "0.0.0");
        assert!(response["ok"]["source_hash"].is_string());
    }

    #[test]
    fn dispatch_malformed_request_is_bad_input_not_panic() {
        let response: Value = serde_json::from_str(&dispatch("not json")).unwrap();
        assert_eq!(response["err"]["kind"], "bad_input");
    }

    #[test]
    fn dispatch_unknown_op_is_flagged() {
        let response: Value =
            serde_json::from_str(&dispatch(r#"{"op":"nope","input":{}}"#)).unwrap();
        assert_eq!(response["err"]["kind"], "unknown_op");
    }

    #[test]
    fn dispatch_op_bad_input_is_error_not_panic() {
        let response: Value = serde_json::from_str(&dispatch(
            r#"{"op":"retry_decision","input":{"attempt":"x"}}"#,
        ))
        .unwrap();
        assert_eq!(response["err"]["kind"], "bad_input");
    }
}

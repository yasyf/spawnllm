use serde::de::DeserializeOwned;
use serde_json::{Value, json};

#[derive(Debug)]
pub(crate) struct CoreError {
    pub(crate) op: String,
    pub(crate) msg: String,
}

impl std::fmt::Display for CoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "core op {} failed: {}", self.op, self.msg)
    }
}

impl From<CoreError> for crate::error::Error {
    fn from(error: CoreError) -> Self {
        Self::Core {
            op: error.op,
            msg: error.msg,
        }
    }
}

pub(crate) fn call_core(op: &str, input: Value) -> Result<Value, CoreError> {
    let request = json!({ "op": op, "input": input }).to_string();
    let response: Value =
        serde_json::from_str(&spawnllm_core::dispatch(&request)).map_err(|error| CoreError {
            op: op.to_owned(),
            msg: format!("invalid response JSON: {error}"),
        })?;
    let mut map = match response {
        Value::Object(map) => map,
        _ => {
            return Err(CoreError {
                op: op.to_owned(),
                msg: "response is not a JSON object".to_owned(),
            });
        }
    };
    if let Some(ok) = map.remove("ok") {
        return Ok(ok);
    }
    match map.remove("err") {
        Some(err) => Err(CoreError {
            op: op.to_owned(),
            msg: err
                .get("msg")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        }),
        None => Err(CoreError {
            op: op.to_owned(),
            msg: "response carries neither ok nor err".to_owned(),
        }),
    }
}

pub(crate) fn core_op<O: DeserializeOwned>(op: &str, input: Value) -> Result<O, CoreError> {
    let value = call_core(op, input)?;
    serde_json::from_value(value).map_err(|error| CoreError {
        op: op.to_owned(),
        msg: format!("output mismatch: {error}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_op_returns_an_error_envelope_instead_of_panicking() {
        let error = core_op::<Value>("unknown-test-op", Value::Null).unwrap_err();

        assert_eq!(error.op, "unknown-test-op");
        assert_eq!(error.msg, "unknown-test-op");
    }
}

use serde::de::DeserializeOwned;
use serde_json::{Value, json};

pub(crate) struct CoreError {
    pub(crate) msg: String,
}

pub(crate) fn call_core(op: &str, input: Value) -> Result<Value, CoreError> {
    let request = json!({ "op": op, "input": input }).to_string();
    let response: Value = serde_json::from_str(&spawnllm_core::dispatch(&request))
        .expect("core dispatch returns valid JSON");
    let mut map = match response {
        Value::Object(map) => map,
        _ => panic!("core response is always a JSON object"),
    };
    if let Some(ok) = map.remove("ok") {
        return Ok(ok);
    }
    match map.remove("err") {
        Some(err) => Err(CoreError {
            msg: err
                .get("msg")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        }),
        None => panic!("core response carries neither ok nor err"),
    }
}

pub(crate) fn core_op<O: DeserializeOwned>(op: &str, input: Value) -> O {
    let value =
        call_core(op, input).unwrap_or_else(|error| panic!("core op {op} failed: {}", error.msg));
    serde_json::from_value(value)
        .unwrap_or_else(|error| panic!("core op {op} output mismatch: {error}"))
}

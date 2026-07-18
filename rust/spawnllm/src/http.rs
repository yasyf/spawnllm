use spawnllm_core::wire::HttpPlan;

use crate::run::{Attempt, AttemptKind, resolve_kind};
use crate::spec::RunSpec;

pub(crate) async fn http_attempt(
    client: &reqwest::Client,
    plan: &HttpPlan,
    spec: &RunSpec,
    provider: &str,
    wants_value: bool,
) -> Attempt {
    let mut request = client
        .post(&plan.url)
        .json(&plan.body)
        .timeout(spec.timeout);
    for (key, value) in &plan.headers {
        request = request.header(key, value);
    }
    match request.send().await {
        Ok(response) => {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            let returncode = if status.is_success() {
                0
            } else {
                i64::from(status.as_u16())
            };
            let stderr = if status.is_success() {
                String::new()
            } else {
                text.clone()
            };
            let kind = resolve_kind(provider, &text, returncode, &stderr, wants_value);
            Attempt { output: text, kind }
        }
        Err(error) if error.is_timeout() => Attempt {
            output: String::new(),
            kind: AttemptKind::Timeout {
                duration: spec.timeout,
            },
        },
        Err(error) => Attempt {
            output: String::new(),
            kind: AttemptKind::Error {
                msg: error.to_string(),
                exit_code: 0,
                stderr: String::new(),
                cost_usd: None,
                usage: None,
            },
        },
    }
}

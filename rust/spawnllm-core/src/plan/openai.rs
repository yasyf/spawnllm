use std::collections::BTreeMap;

use serde_json::{Map, json};

use crate::wire::{HttpPlan, InvocationPlan, RunSpec};

pub(super) fn plan(spec: &RunSpec) -> InvocationPlan {
    let endpoint = spec
        .openai_endpoint
        .as_ref()
        .expect("openai_endpoint provider requires endpoint configuration");
    let mut body = Map::from_iter([
        ("model".to_string(), json!(endpoint.model)),
        (
            "messages".to_string(),
            json!([{"role": "user", "content": spec.prompt}]),
        ),
    ]);
    if let Some(schema) = &spec.schema {
        body.insert(
            "response_format".to_string(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": true,
                    "schema": schema,
                },
            }),
        );
    }

    InvocationPlan::Http(HttpPlan {
        method: "POST".to_string(),
        url: format!(
            "{}/chat/completions",
            endpoint.base_url.trim_end_matches('/')
        ),
        headers: BTreeMap::from([(
            "Authorization".to_string(),
            format!("Bearer {}", endpoint.api_key),
        )]),
        body,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::wire::OpenAiEndpoint;

    fn spec(schema: Option<Value>) -> RunSpec {
        RunSpec {
            prompt: "ping".to_string(),
            model: "qwen3".to_string(),
            agent: false,
            isolated: true,
            timeout: 180,
            max_attempts: 5,
            api_auth: false,
            schema,
            apple: None,
            claude: None,
            codex: None,
            gemini: None,
            openai_endpoint: Some(OpenAiEndpoint {
                api_key: "sk-test".to_string(),
                base_url: "http://local.test/v1".to_string(),
                model: "qwen3".to_string(),
            }),
        }
    }

    #[test]
    fn plain_plan_matches_vector() {
        assert_eq!(
            serde_json::to_value(plan(&spec(None))).unwrap(),
            json!({
                "kind": "http",
                "method": "POST",
                "url": "http://local.test/v1/chat/completions",
                "headers": {"Authorization": "Bearer sk-test"},
                "body": {
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            })
        );
    }

    #[test]
    fn schema_plan_matches_vector() {
        let schema = json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        });

        assert_eq!(
            serde_json::to_value(plan(&spec(Some(schema)))).unwrap(),
            json!({
                "kind": "http",
                "method": "POST",
                "url": "http://local.test/v1/chat/completions",
                "headers": {"Authorization": "Bearer sk-test"},
                "body": {
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "ping"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "strict": true,
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "required": ["answer"],
                            },
                        },
                    },
                },
            })
        );
    }
}

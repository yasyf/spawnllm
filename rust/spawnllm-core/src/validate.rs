use serde::Deserialize;
use serde_json::{Value, json};

use crate::wire::{AppleConfig, AppleSampling, RunSpec};
use crate::{OpError, OpResult, from_input};

#[derive(Debug, Deserialize)]
struct ValidateInput {
    spec: RunSpec,
}

/// Reject a spec whose provider config carries knobs the provider would discard.
pub fn validate_spec(spec: &RunSpec) -> Result<(), &'static str> {
    spec.apple.as_ref().map_or(Ok(()), validate_apple)
}

/// Reject an [`AppleConfig`] whose sampling knobs the Foundation Models framework
/// would silently drop or refuse.
pub fn validate_apple(config: &AppleConfig) -> Result<(), &'static str> {
    if config.sampling != Some(AppleSampling::Random)
        && (config.sampling_top.is_some()
            || config.sampling_probability_threshold.is_some()
            || config.sampling_seed.is_some())
    {
        return Err(
            "AppleConfig sampling_top, sampling_probability_threshold, and sampling_seed require sampling='random'",
        );
    }
    if config.sampling_top.is_some() && config.sampling_probability_threshold.is_some() {
        return Err(
            "AppleConfig accepts either sampling_top or sampling_probability_threshold, not both",
        );
    }
    Ok(())
}

pub(crate) fn dispatch(input: Value) -> OpResult {
    validate_spec(&from_input::<ValidateInput>(input)?.spec)
        .map(|()| json!({}))
        .map_err(OpError::invalid_spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apple(config: AppleConfig) -> Result<(), &'static str> {
        validate_spec(&RunSpec {
            prompt: "hi".to_owned(),
            model: String::new(),
            agent: false,
            isolated: true,
            timeout: 180,
            max_attempts: 5,
            api_auth: false,
            schema: None,
            apple: Some(config),
            claude: None,
            codex: None,
            gemini: None,
            openai_endpoint: None,
        })
    }

    #[test]
    fn a_random_knob_without_random_sampling_is_rejected() {
        for config in [
            AppleConfig {
                sampling_top: Some(20),
                ..AppleConfig::default()
            },
            AppleConfig {
                sampling_probability_threshold: Some(0.9),
                ..AppleConfig::default()
            },
            AppleConfig {
                sampling_seed: Some(7),
                ..AppleConfig::default()
            },
            AppleConfig {
                sampling: Some(AppleSampling::Greedy),
                sampling_seed: Some(7),
                ..AppleConfig::default()
            },
        ] {
            assert_eq!(
                apple(config.clone()),
                Err(
                    "AppleConfig sampling_top, sampling_probability_threshold, and sampling_seed require sampling='random'"
                ),
                "{config:?}"
            );
        }
    }

    #[test]
    fn top_and_probability_threshold_together_are_rejected() {
        assert_eq!(
            apple(AppleConfig {
                sampling: Some(AppleSampling::Random),
                sampling_top: Some(20),
                sampling_probability_threshold: Some(0.9),
                ..AppleConfig::default()
            }),
            Err(
                "AppleConfig accepts either sampling_top or sampling_probability_threshold, not both"
            )
        );
    }

    #[test]
    fn random_sampling_with_one_knob_is_accepted() {
        assert_eq!(
            apple(AppleConfig {
                sampling: Some(AppleSampling::Random),
                sampling_top: Some(20),
                sampling_seed: Some(7),
                ..AppleConfig::default()
            }),
            Ok(())
        );
        assert_eq!(apple(AppleConfig::default()), Ok(()));
    }

    #[test]
    fn dispatch_reports_an_invalid_spec_kind() {
        let response: Value = serde_json::from_str(&crate::dispatch(
            r#"{"op":"validate_spec","input":{"spec":{"prompt":"hi","model":"","agent":false,"isolated":true,"timeout":180,"max_attempts":5,"apple":{"use_case":"general","guardrails":"default","sampling_seed":7}}}}"#,
        ))
        .unwrap();

        assert_eq!(response["err"]["kind"], "invalid_spec");
        assert!(
            response["err"]["msg"]
                .as_str()
                .unwrap()
                .contains("require sampling='random'")
        );
    }

    #[test]
    fn dispatch_accepts_a_spec_with_no_apple_config() {
        let response: Value = serde_json::from_str(&crate::dispatch(
            r#"{"op":"validate_spec","input":{"spec":{"prompt":"hi","model":"haiku","agent":false,"isolated":true,"timeout":180,"max_attempts":5}}}"#,
        ))
        .unwrap();

        assert_eq!(response["ok"], json!({}));
    }
}

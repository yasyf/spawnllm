use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;

use crate::wire::{
    AppleConfig, AppleGuardrails, AppleSampling, AppleUseCase, ExecPlan, InvocationPlan,
    ReadResultFrom, RunSpec,
};

#[derive(Debug, Serialize)]
struct Request<'a> {
    prompt: &'a str,
    instructions: Option<&'a str>,
    use_case: AppleUseCase,
    guardrails: AppleGuardrails,
    options: Option<Options>,
    schema: Option<&'a Value>,
}

#[derive(Debug, Serialize)]
struct Options {
    temperature: Option<f64>,
    maximum_response_tokens: Option<i64>,
    sampling: Option<Sampling>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum Sampling {
    Greedy,
    Random {
        top: Option<i64>,
        probability_threshold: Option<f64>,
        seed: Option<u64>,
    },
}

pub(super) fn plan(spec: &RunSpec) -> InvocationPlan {
    let config = spec.apple.as_ref();
    let request = Request {
        prompt: &spec.prompt,
        instructions: config.and_then(|config| config.instructions.as_deref()),
        use_case: config.map_or(AppleUseCase::General, |config| config.use_case),
        guardrails: config.map_or(AppleGuardrails::Default, |config| config.guardrails),
        options: config.and_then(options),
        schema: spec.schema.as_ref(),
    };

    InvocationPlan::Exec(ExecPlan {
        argv: vec!["spawnllm-apple".into()],
        stdin: serde_json::to_string(&request).expect("the apple request is always serializable"),
        files: Vec::new(),
        stdout_to_file: false,
        read_result_from: ReadResultFrom::Stdout,
        env: BTreeMap::new(),
        env_unset: Vec::new(),
        needs_claude_isolation: false,
    })
}

fn options(config: &AppleConfig) -> Option<Options> {
    match (
        sampling(config),
        config.temperature,
        config.maximum_response_tokens,
    ) {
        (None, None, None) => None,
        (sampling, temperature, maximum_response_tokens) => Some(Options {
            temperature,
            maximum_response_tokens,
            sampling,
        }),
    }
}

fn sampling(config: &AppleConfig) -> Option<Sampling> {
    match config.sampling? {
        AppleSampling::Greedy => Some(Sampling::Greedy),
        AppleSampling::Random => Some(Sampling::Random {
            top: config.sampling_top,
            probability_threshold: config.sampling_probability_threshold,
            seed: config.sampling_seed,
        }),
    }
}

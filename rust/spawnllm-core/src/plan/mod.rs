mod antigravity;
mod claude;
mod codex;
mod gemini;
mod openai;

use serde::Deserialize;
use serde_json::Value;

use crate::wire::{InvocationPlan, RunSpec};
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

#[derive(Debug, Clone, Deserialize)]
struct PlanInput {
    provider: Provider,
    spec: RunSpec,
}

fn plan(input: PlanInput) -> InvocationPlan {
    match input.provider {
        Provider::Claude => claude::plan(&input.spec),
        Provider::Codex => codex::plan(&input.spec),
        Provider::Gemini => gemini::plan(&input.spec),
        Provider::Antigravity => antigravity::plan(&input.spec),
        Provider::OpenaiEndpoint => openai::plan(&input.spec),
    }
}

pub(crate) fn dispatch(input: Value) -> OpResult {
    serde_json::to_value(plan(from_input::<PlanInput>(input)?)).map_err(OpError::internal)
}

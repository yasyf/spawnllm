use std::collections::{BTreeMap, HashMap, HashSet};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::wire::{HttpPlan, InvocationPlan};
use crate::{OpError, OpResult, from_input, unimplemented};

const JEV_URL: &str = "https://api.typesafe.ai/v1/systemone";
const OPENAI_URL: &str = "https://api.openai.com/v1/decisions";
const MAX_OPTIONS: usize = 255;
const MAX_LEVELS: usize = 10;
const BACKOFF_BASE_S: f64 = 0.5;
const BACKOFF_CAP_S: f64 = 5.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Vendor {
    Jev,
    Openai,
}

#[derive(Debug, Clone, Deserialize)]
struct Provider {
    name: Vendor,
    model: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Choice {
    value: String,
    description: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Level {
    label: String,
    description: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Question {
    Binary {
        id: String,
        instructions: String,
        yes: Option<String>,
        no: Option<String>,
    },
    Label {
        id: String,
        instructions: String,
        options: Vec<Choice>,
    },
    Score {
        id: String,
        instructions: String,
        levels: Vec<Level>,
    },
}

impl Question {
    fn id(&self) -> &str {
        match self {
            Self::Binary { id, .. } | Self::Label { id, .. } | Self::Score { id, .. } => id,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct PlanInput {
    provider: Provider,
    api_key: String,
    state: Value,
    questions: Vec<Question>,
}

#[derive(Debug, Clone, Deserialize)]
struct ResolveInput {
    provider: Provider,
    questions: Vec<Question>,
    status: Option<u16>,
    body: String,
    retry_after: Option<String>,
    retry_after_ms: Option<String>,
    attempt: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Answer {
    Binary {
        p_yes: f64,
        confidence: f64,
    },
    Label {
        choice: String,
        probabilities: Vec<f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        probabilities: Vec<f64>,
        confidence: f64,
    },
    Refused,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Outcome {
    Decision {
        model: String,
        input_tokens: u64,
        answers: Map<String, Value>,
    },
    Retry {
        sleep_s: f64,
    },
    Failed {
        status: Option<u16>,
        message: String,
    },
}

#[derive(Deserialize)]
struct Usage {
    input_tokens: u64,
}

#[derive(Deserialize)]
struct Envelope {
    model: String,
    answers: Value,
    usage: Usage,
}

#[derive(Deserialize)]
struct Noul {
    noul: f64,
}

#[derive(Deserialize)]
struct Predicate {
    probability: f64,
}

#[derive(Deserialize)]
struct Distribution<P> {
    probabilities: P,
    confidence: f64,
}

#[derive(Deserialize)]
struct Chosen {
    choice: String,
}

#[derive(Deserialize)]
struct Scored {
    score: f64,
}

#[derive(Deserialize)]
struct Weighted<V> {
    value: V,
    probability: f64,
}

fn validate(state: &Value, questions: &[Question]) -> Result<(), &'static str> {
    if !matches!(state, Value::String(_) | Value::Object(_) | Value::Array(_)) {
        return Err("decide state must be a string, an object, or an array");
    }
    if questions.is_empty() {
        return Err("decide needs at least one question");
    }
    let mut ids = HashSet::new();
    if !questions.iter().all(|question| ids.insert(question.id())) {
        return Err("decide question ids must be unique");
    }
    questions.iter().try_for_each(|question| match question {
        Question::Label { options, .. } if !(2..=MAX_OPTIONS).contains(&options.len()) => {
            Err("a label question takes 2 to 255 options")
        }
        Question::Label { options, .. }
            if options
                .iter()
                .map(|option| &option.value)
                .collect::<HashSet<_>>()
                .len()
                != options.len() =>
        {
            Err("a label question's option values must be unique")
        }
        Question::Score { levels, .. } if !(2..=MAX_LEVELS).contains(&levels.len()) => {
            Err("a score question takes 2 to 10 levels")
        }
        _ => Ok(()),
    })
}

fn jev_question(question: &Question) -> Value {
    match question {
        Question::Binary {
            instructions,
            yes,
            no,
            ..
        } => {
            let criteria: Map<String, Value> = [("true", yes), ("false", no)]
                .into_iter()
                .filter_map(|(key, text)| text.as_ref().map(|text| (key.to_owned(), json!(text))))
                .collect();
            let mut body = json!({"type": "noul", "instructions": instructions});
            if !criteria.is_empty() {
                body["criteria"] = Value::Object(criteria);
            }
            body
        }
        Question::Label {
            instructions,
            options,
            ..
        } => json!({
            "type": "choice",
            "instructions": instructions,
            "criteria": options
                .iter()
                .map(|option| (option.value.clone(), json!(option.description)))
                .collect::<Map<_, _>>(),
        }),
        Question::Score {
            instructions,
            levels,
            ..
        } => json!({
            "type": "score",
            "instructions": instructions,
            "criteria": levels
                .iter()
                .map(|level| match &level.description {
                    Some(description) => format!("{}: {description}", level.label),
                    None => level.label.clone(),
                })
                .collect::<Vec<_>>(),
        }),
    }
}

fn described(key: &str, name: &str, description: Option<&str>) -> Value {
    let mut entry = Map::from_iter([(key.to_owned(), json!(name))]);
    if let Some(description) = description {
        entry.insert("description".to_owned(), json!(description));
    }
    Value::Object(entry)
}

fn predicate_instructions(instructions: &str, yes: Option<&str>, no: Option<&str>) -> String {
    [
        Some(instructions.to_owned()),
        yes.map(|yes| format!("Yes means: {yes}")),
        no.map(|no| format!("No means: {no}")),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join("\n")
}

fn openai_question(question: &Question) -> Value {
    match question {
        Question::Binary {
            id,
            instructions,
            yes,
            no,
        } => json!({
            "type": "predicate",
            "name": id,
            "instructions": predicate_instructions(instructions, yes.as_deref(), no.as_deref()),
        }),
        Question::Label {
            id,
            instructions,
            options,
        } => json!({
            "type": "choice",
            "name": id,
            "instructions": instructions,
            "choices": options
                .iter()
                .map(|option| described("value", &option.value, option.description.as_deref()))
                .collect::<Vec<_>>(),
        }),
        Question::Score {
            id,
            instructions,
            levels,
        } => json!({
            "type": "score",
            "name": id,
            "instructions": instructions,
            "levels": levels
                .iter()
                .map(|level| described("label", &level.label, level.description.as_deref()))
                .collect::<Vec<_>>(),
        }),
    }
}

fn plan(input: &PlanInput) -> InvocationPlan {
    let (url, body) = match input.provider.name {
        Vendor::Jev => (
            JEV_URL,
            Map::from_iter([
                ("state".to_owned(), input.state.clone()),
                ("model".to_owned(), json!(input.provider.model)),
                (
                    "questions".to_owned(),
                    Value::Object(
                        input
                            .questions
                            .iter()
                            .map(|question| (question.id().to_owned(), jev_question(question)))
                            .collect(),
                    ),
                ),
            ]),
        ),
        Vendor::Openai => (
            OPENAI_URL,
            Map::from_iter([
                ("model".to_owned(), json!(input.provider.model)),
                (
                    "input".to_owned(),
                    json!(match &input.state {
                        Value::String(text) => text.clone(),
                        structured => structured.to_string(),
                    }),
                ),
                (
                    "questions".to_owned(),
                    Value::Array(input.questions.iter().map(openai_question).collect()),
                ),
            ]),
        ),
    };
    InvocationPlan::Http(HttpPlan {
        method: "POST".to_owned(),
        url: url.to_owned(),
        headers: BTreeMap::from([(
            "Authorization".to_owned(),
            format!("Bearer {}", input.api_key),
        )]),
        body,
    })
}

fn parse<T: DeserializeOwned>(id: &str, answer: &Value) -> Result<T, String> {
    serde_json::from_value(answer.clone())
        .map_err(|error| format!("answer {id} is malformed: {error}"))
}

fn ordered(
    id: &str,
    probabilities: &HashMap<String, f64>,
    keys: impl Iterator<Item = String>,
) -> Result<Vec<f64>, String> {
    keys.map(|key| {
        probabilities
            .get(&key)
            .copied()
            .ok_or_else(|| format!("answer {id} has no probability for {key:?}"))
    })
    .collect()
}

fn binary(p_yes: f64) -> Answer {
    Answer::Binary {
        p_yes,
        confidence: (2.0 * p_yes - 1.0).abs(),
    }
}

fn option_keys(options: &[Choice]) -> impl Iterator<Item = String> {
    options.iter().map(|option| option.value.clone())
}

fn level_keys(levels: &[Level]) -> impl Iterator<Item = String> {
    (0..levels.len()).map(|index| index.to_string())
}

fn answer_type(answer: &Value) -> Option<&str> {
    answer.get("type").and_then(Value::as_str)
}

fn jev_answer(question: &Question, answer: &Value) -> Result<Answer, String> {
    let id = question.id();
    match (question, answer_type(answer)) {
        (Question::Binary { .. }, Some("noul")) => Ok(binary(parse::<Noul>(id, answer)?.noul)),
        (Question::Label { options, .. }, Some("choice")) => {
            let spread = parse::<Distribution<HashMap<String, f64>>>(id, answer)?;
            Ok(Answer::Label {
                choice: parse::<Chosen>(id, answer)?.choice,
                probabilities: ordered(id, &spread.probabilities, option_keys(options))?,
                confidence: spread.confidence,
            })
        }
        (Question::Score { levels, .. }, Some("score")) => {
            let spread = parse::<Distribution<HashMap<String, f64>>>(id, answer)?;
            Ok(Answer::Score {
                score: parse::<Scored>(id, answer)?.score,
                probabilities: ordered(id, &spread.probabilities, level_keys(levels))?,
                confidence: spread.confidence,
            })
        }
        (_, kind) => Err(format!("answer {id} came back as {kind:?}")),
    }
}

fn openai_answer(question: &Question, answer: &Value) -> Result<Answer, String> {
    let id = question.id();
    if let Some(name) = answer.get("name").and_then(Value::as_str)
        && name != id
    {
        return Err(format!("answer {name} arrived where {id} was asked"));
    }
    match (question, answer_type(answer)) {
        (_, Some("refusal")) => Ok(Answer::Refused),
        (Question::Binary { .. }, Some("predicate")) => {
            Ok(binary(parse::<Predicate>(id, answer)?.probability))
        }
        (Question::Label { options, .. }, Some("choice")) => {
            let spread = parse::<Distribution<Vec<Weighted<String>>>>(id, answer)?;
            let probabilities = spread
                .probabilities
                .into_iter()
                .map(|weighted| (weighted.value, weighted.probability))
                .collect();
            Ok(Answer::Label {
                choice: parse::<Chosen>(id, answer)?.choice,
                probabilities: ordered(id, &probabilities, option_keys(options))?,
                confidence: spread.confidence,
            })
        }
        (Question::Score { levels, .. }, Some("score")) => {
            let spread = parse::<Distribution<Vec<Weighted<usize>>>>(id, answer)?;
            let probabilities = spread
                .probabilities
                .into_iter()
                .map(|weighted| (weighted.value.to_string(), weighted.probability))
                .collect();
            Ok(Answer::Score {
                score: parse::<Scored>(id, answer)?.score,
                probabilities: ordered(id, &probabilities, level_keys(levels))?,
                confidence: spread.confidence,
            })
        }
        (_, kind) => Err(format!("answer {id} came back as {kind:?}")),
    }
}

fn answers(input: &ResolveInput) -> Result<Outcome, String> {
    let envelope: Envelope = serde_json::from_str(&input.body)
        .map_err(|error| format!("the decision response is malformed: {error}"))?;
    let answers: Vec<Answer> = match (input.provider.name, &envelope.answers) {
        (Vendor::Jev, Value::Object(by_id)) => input
            .questions
            .iter()
            .map(|question| {
                by_id
                    .get(question.id())
                    .ok_or_else(|| format!("no answer for {}", question.id()))
                    .and_then(|answer| jev_answer(question, answer))
            })
            .collect::<Result<_, _>>()?,
        (Vendor::Openai, Value::Array(in_order)) if in_order.len() == input.questions.len() => {
            input
                .questions
                .iter()
                .zip(in_order)
                .map(|(question, answer)| openai_answer(question, answer))
                .collect::<Result<_, _>>()?
        }
        _ => return Err("the decision response's answers do not match the questions".to_owned()),
    };
    Ok(Outcome::Decision {
        model: envelope.model,
        input_tokens: envelope.usage.input_tokens,
        answers: input
            .questions
            .iter()
            .zip(answers)
            .map(|(question, answer)| {
                (
                    question.id().to_owned(),
                    serde_json::to_value(answer).expect("answers always serialize"),
                )
            })
            .collect(),
    })
}

fn seconds(header: Option<&str>) -> Option<f64> {
    header
        .and_then(|value| value.trim().parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
}

fn backoff(attempt: u32) -> f64 {
    (BACKOFF_BASE_S * 2f64.powi(attempt.min(16) as i32)).min(BACKOFF_CAP_S)
}

fn resolve(input: &ResolveInput) -> Outcome {
    match input.status {
        Some(200) => answers(input).unwrap_or_else(|message| Outcome::Failed {
            status: Some(200),
            message,
        }),
        None | Some(408 | 429 | 500..=599) => Outcome::Retry {
            sleep_s: seconds(input.retry_after_ms.as_deref())
                .map(|ms| ms / 1000.0)
                .or_else(|| seconds(input.retry_after.as_deref()))
                .unwrap_or_else(|| backoff(input.attempt)),
        },
        Some(status) => Outcome::Failed {
            status: Some(status),
            message: input.body.clone(),
        },
    }
}

pub(crate) fn dispatch(op: &str, input: Value) -> OpResult {
    match op {
        "decide_plan" => {
            let input = from_input::<PlanInput>(input)?;
            validate(&input.state, &input.questions).map_err(OpError::invalid_spec)?;
            serde_json::to_value(plan(&input)).map_err(OpError::internal)
        }
        "decide_resolve" => serde_json::to_value(resolve(&from_input::<ResolveInput>(input)?))
            .map_err(OpError::internal),
        other => Err(unimplemented(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn questions() -> Value {
        json!([
            {"type": "binary", "id": "is_rollback_request", "instructions": "Does the message ask to roll back a release?"},
            {"type": "label", "id": "action", "instructions": "Which release action does the message ask for?", "options": [
                {"value": "none", "description": "No release action"},
                {"value": "release", "description": "Ship a new release"},
                {"value": "rollback", "description": "Roll back a release"},
            ]},
            {"type": "score", "id": "urgency", "instructions": "How urgent is the message?", "levels": [
                {"label": "Not urgent"},
                {"label": "Somewhat urgent"},
                {"label": "Blocking, needs action now"},
            ]},
        ])
    }

    fn plan_for(name: &str, model: &str, state: Value, questions: Value) -> Value {
        dispatch(
            "decide_plan",
            json!({"provider": {"name": name, "model": model}, "api_key": "k", "state": state, "questions": questions}),
        )
        .unwrap_or_else(|error| panic!("{}: {}", error.kind, error.msg))
    }

    fn resolve_for(name: &str, status: Option<u16>, body: &str, attempt: u32) -> Value {
        dispatch(
            "decide_resolve",
            json!({
                "provider": {"name": name, "model": "m"},
                "questions": questions(),
                "status": status,
                "body": body,
                "retry_after": null,
                "retry_after_ms": null,
                "attempt": attempt,
            }),
        )
        .unwrap_or_else(|error| panic!("{}: {}", error.kind, error.msg))
    }

    fn rejection(state: Value, questions: Value) -> String {
        match dispatch(
            "decide_plan",
            json!({"provider": {"name": "jev", "model": "m"}, "api_key": "k", "state": state, "questions": questions}),
        ) {
            Err(error) => {
                assert_eq!(error.kind, "invalid_spec");
                error.msg
            }
            Ok(plan) => panic!("expected a rejection, planned {plan}"),
        }
    }

    #[test]
    fn jev_plan_keys_questions_by_id_in_declared_order() {
        let plan = plan_for("jev", "jev-1.13.0", json!("roll it back"), questions());
        assert_eq!(plan["url"], JEV_URL);
        assert_eq!(plan["headers"], json!({"Authorization": "Bearer k"}));
        assert_eq!(
            plan["body"]["questions"]["action"]["criteria"]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            ["none", "release", "rollback"]
        );
        assert_eq!(
            plan["body"]["questions"]["urgency"]["criteria"],
            json!([
                "Not urgent",
                "Somewhat urgent",
                "Blocking, needs action now"
            ])
        );
        assert!(
            plan["body"]["questions"]["is_rollback_request"]
                .get("criteria")
                .is_none()
        );
    }

    #[test]
    fn jev_binary_sends_only_the_criteria_it_was_given() {
        let plan = plan_for(
            "jev",
            "m",
            json!("x"),
            json!([{"type": "binary", "id": "b", "instructions": "Is it?", "yes": "it is"}]),
        );
        assert_eq!(
            plan["body"]["questions"]["b"],
            json!({"type": "noul", "instructions": "Is it?", "criteria": {"true": "it is"}})
        );
    }

    #[test]
    fn openai_plan_folds_binary_criteria_into_instructions_and_compacts_structured_state() {
        let plan = plan_for(
            "openai",
            "gpt-6-luna",
            json!({"b": 1, "a": [true, null]}),
            json!([{"type": "binary", "id": "b", "instructions": "Is it?", "yes": "it is", "no": "it is not"}]),
        );
        assert_eq!(plan["url"], OPENAI_URL);
        assert_eq!(plan["body"]["input"], r#"{"b":1,"a":[true,null]}"#);
        assert_eq!(
            plan["body"]["questions"],
            json!([{"type": "predicate", "name": "b", "instructions": "Is it?\nYes means: it is\nNo means: it is not"}])
        );
    }

    #[test]
    fn plan_rejects_invalid_specs() {
        assert_eq!(
            rejection(json!(3), questions()),
            "decide state must be a string, an object, or an array"
        );
        assert_eq!(
            rejection(json!("x"), json!([])),
            "decide needs at least one question"
        );
        assert_eq!(
            rejection(
                json!("x"),
                json!([
                    {"type": "binary", "id": "a", "instructions": "?"},
                    {"type": "binary", "id": "a", "instructions": "?"},
                ])
            ),
            "decide question ids must be unique"
        );
        assert_eq!(
            rejection(
                json!("x"),
                json!([{"type": "label", "id": "a", "instructions": "?", "options": [{"value": "only"}]}])
            ),
            "a label question takes 2 to 255 options"
        );
        assert_eq!(
            rejection(
                json!("x"),
                json!([{"type": "label", "id": "a", "instructions": "?", "options": [{"value": "x"}, {"value": "x"}]}])
            ),
            "a label question's option values must be unique"
        );
        let eleven: Vec<Value> = (0..11)
            .map(|level| json!({"label": level.to_string()}))
            .collect();
        assert_eq!(
            rejection(
                json!("x"),
                json!([{"type": "score", "id": "a", "instructions": "?", "levels": eleven}])
            ),
            "a score question takes 2 to 10 levels"
        );
    }

    #[test]
    fn jev_answers_reorder_to_declared_order() {
        let body = json!({
            "model": "jev-1.13.0",
            "answers": {
                "urgency": {"type": "score", "score": 1.5, "confidence": 0.5, "legend": {"0": "a", "1": "b", "2": "c"}, "probabilities": {"2": 0.5, "0": 0.0, "1": 0.5}},
                "action": {"type": "choice", "choice": "rollback", "confidence": 0.7, "probabilities": {"release": 0.1, "rollback": 0.8, "none": 0.1}},
                "is_rollback_request": {"type": "noul", "noul": 0.25},
            },
            "usage": {"input_tokens": 414, "output_tokens": 73},
        });
        let outcome = resolve_for("jev", Some(200), &body.to_string(), 0);
        assert_eq!(
            outcome,
            json!({
                "kind": "decision",
                "model": "jev-1.13.0",
                "input_tokens": 414,
                "answers": {
                    "is_rollback_request": {"type": "binary", "p_yes": 0.25, "confidence": 0.5},
                    "action": {"type": "label", "choice": "rollback", "probabilities": [0.1, 0.1, 0.8], "confidence": 0.7},
                    "urgency": {"type": "score", "score": 1.5, "probabilities": [0.0, 0.5, 0.5], "confidence": 0.5},
                },
            })
        );
    }

    #[test]
    fn openai_refusal_maps_to_refused_while_others_answer() {
        let body = json!({
            "model": "gpt-6-luna",
            "answers": [
                {"type": "refusal", "name": "is_rollback_request"},
                {"type": "choice", "name": "action", "choice": "none", "confidence": 1.0, "probabilities": [
                    {"value": "rollback", "probability": 0.0}, {"value": "none", "probability": 1.0}, {"value": "release", "probability": 0.0},
                ]},
                {"type": "score", "name": "urgency", "score": 0.0, "confidence": 1.0, "probabilities": [
                    {"value": 1, "label": "Somewhat urgent", "probability": 0.0}, {"value": 0, "label": "Not urgent", "probability": 1.0}, {"value": 2, "label": "Blocking, needs action now", "probability": 0.0},
                ]},
            ],
            "usage": {"input_tokens": 419},
        });
        let outcome = resolve_for("openai", Some(200), &body.to_string(), 0);
        assert_eq!(
            outcome["answers"]["is_rollback_request"],
            json!({"type": "refused"})
        );
        assert_eq!(
            outcome["answers"]["action"]["probabilities"],
            json!([1.0, 0.0, 0.0])
        );
        assert_eq!(
            outcome["answers"]["urgency"]["probabilities"],
            json!([1.0, 0.0, 0.0])
        );
    }

    #[test]
    fn mismatched_success_bodies_fail_instead_of_guessing() {
        let misnamed = json!({
            "model": "gpt-6-luna",
            "answers": [{"type": "refusal", "name": "urgency"}, {"type": "refusal"}, {"type": "refusal"}],
            "usage": {"input_tokens": 1},
        });
        assert_eq!(
            resolve_for("openai", Some(200), &misnamed.to_string(), 0),
            json!({"kind": "failed", "status": 200, "message": "answer urgency arrived where is_rollback_request was asked"})
        );
        let short = json!({"model": "gpt-6-luna", "answers": [], "usage": {"input_tokens": 1}});
        assert_eq!(
            resolve_for("openai", Some(200), &short.to_string(), 0)["message"],
            "the decision response's answers do not match the questions"
        );
        let missing = json!({"model": "jev-1.13.0", "answers": {}, "usage": {"input_tokens": 1}});
        assert_eq!(
            resolve_for("jev", Some(200), &missing.to_string(), 0)["message"],
            "no answer for is_rollback_request"
        );
    }

    #[test]
    fn transient_statuses_back_off_to_the_cap_and_others_fail() {
        assert_eq!(
            resolve_for("jev", Some(529), "overloaded", 0),
            json!({"kind": "retry", "sleep_s": 0.5})
        );
        assert_eq!(resolve_for("jev", Some(503), "", 2)["sleep_s"], json!(2.0));
        assert_eq!(resolve_for("jev", None, "", 9)["sleep_s"], json!(5.0));
        assert_eq!(resolve_for("jev", Some(408), "", 1)["sleep_s"], json!(1.0));
        assert_eq!(
            resolve_for("openai", Some(401), "bad key", 0),
            json!({"kind": "failed", "status": 401, "message": "bad key"})
        );
    }

    #[test]
    fn retry_after_headers_override_the_backoff() {
        let outcome = |retry_after: Value, retry_after_ms: Value| {
            dispatch(
                "decide_resolve",
                json!({
                    "provider": {"name": "openai", "model": "m"},
                    "questions": questions(),
                    "status": 429,
                    "body": "",
                    "retry_after": retry_after,
                    "retry_after_ms": retry_after_ms,
                    "attempt": 0,
                }),
            )
            .unwrap_or_else(|error| panic!("{}", error.msg))
        };
        assert_eq!(outcome(json!("3"), json!("250"))["sleep_s"], json!(0.25));
        assert_eq!(outcome(json!("3"), Value::Null)["sleep_s"], json!(3.0));
        assert_eq!(
            outcome(json!("Wed, 21 Oct 2026 07:28:00 GMT"), Value::Null)["sleep_s"],
            json!(0.5)
        );
    }
}

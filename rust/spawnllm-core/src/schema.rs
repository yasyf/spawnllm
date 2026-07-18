use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{OpResult, from_input};

const SUPPORTED_STRING_FORMATS: &[&str] = &[
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Dialect {
    Anthropic,
    Openai,
}

#[derive(Debug, Deserialize)]
struct SchemaInput {
    dialect: Dialect,
    schema: Value,
}

pub(crate) fn dispatch(input: Value) -> OpResult {
    let SchemaInput { dialect, schema } = from_input(input)?;
    let transformed = match dialect {
        Dialect::Anthropic => transform_anthropic(schema),
        Dialect::Openai => {
            let root = schema.clone();
            ensure_strict(schema, &root)
        }
    };
    Ok(serde_json::json!({ "schema": transformed }))
}

fn transform_anthropic(schema: Value) -> Value {
    let mut src = match schema {
        Value::Object(src) => src,
        other => return other,
    };
    let mut out = Map::new();

    if let Some(Value::Object(defs)) = src.remove("$defs") {
        out.insert(
            "$defs".into(),
            Value::Object(
                defs.into_iter()
                    .map(|(name, def)| (name, transform_anthropic(def)))
                    .collect(),
            ),
        );
    }

    if let Some(reference) = src.remove("$ref") {
        out.insert("$ref".into(), reference);
        return Value::Object(out);
    }

    let type_value = src.remove("type");
    let type_kind = type_value
        .as_ref()
        .and_then(Value::as_str)
        .map(str::to_string);
    let any_of = src.remove("anyOf");
    let one_of = src.remove("oneOf");
    let all_of = src.remove("allOf");

    if let Some(Value::Array(variants)) = any_of {
        out.insert("anyOf".into(), transform_variants(variants));
    } else if let Some(Value::Array(variants)) = one_of {
        out.insert("anyOf".into(), transform_variants(variants));
    } else if let Some(Value::Array(variants)) = all_of {
        out.insert("allOf".into(), transform_variants(variants));
    } else if let Some(type_value) = type_value {
        out.insert("type".into(), type_value);
    }

    if let Some(Value::Array(enum_values)) = src.remove("enum") {
        out.insert("enum".into(), Value::Array(enum_values));
    }

    if let Some(description) = src.remove("description")
        && !description.is_null()
    {
        out.insert("description".into(), description);
    }

    if let Some(title) = src.remove("title")
        && !title.is_null()
    {
        out.insert("title".into(), title);
    }

    match type_kind.as_deref() {
        Some("object") => {
            let properties = match src.remove("properties") {
                Some(Value::Object(properties)) => properties,
                _ => Map::new(),
            };
            out.insert(
                "properties".into(),
                Value::Object(
                    properties
                        .into_iter()
                        .map(|(key, prop)| (key, transform_anthropic(prop)))
                        .collect(),
                ),
            );
            src.remove("additionalProperties");
            out.insert("additionalProperties".into(), Value::Bool(false));
            if let Some(required) = src.remove("required")
                && !required.is_null()
            {
                out.insert("required".into(), required);
            }
        }
        Some("string") => {
            if let Some(format) = src.remove("format") {
                match format.as_str() {
                    Some(name) if SUPPORTED_STRING_FORMATS.contains(&name) => {
                        out.insert("format".into(), format);
                    }
                    Some(name) if !name.is_empty() => {
                        src.insert("format".into(), format);
                    }
                    _ => {}
                }
            }
        }
        Some("array") => {
            if let Some(items) = src.remove("items")
                && !items.is_null()
            {
                out.insert("items".into(), transform_anthropic(items));
            }
            if let Some(min_items) = src.remove("minItems") {
                if matches!(min_items.as_i64(), Some(0 | 1)) {
                    out.insert("minItems".into(), min_items);
                } else {
                    src.insert("minItems".into(), min_items);
                }
            }
        }
        _ => {}
    }

    if !src.is_empty() {
        let body = src
            .iter()
            .map(|(key, value)| format!("{key}: {}", py_str(value)))
            .collect::<Vec<_>>()
            .join(", ");
        let description = match out.get("description").and_then(Value::as_str) {
            Some(existing) => format!("{existing}\n\n{{{body}}}"),
            None => format!("{{{body}}}"),
        };
        out.insert("description".into(), Value::String(description));
    }

    Value::Object(out)
}

fn transform_variants(variants: Vec<Value>) -> Value {
    Value::Array(variants.into_iter().map(transform_anthropic).collect())
}

fn ensure_strict(schema: Value, root: &Value) -> Value {
    let mut obj = match schema {
        Value::Object(obj) => obj,
        other => return other,
    };

    if let Some(Value::Object(defs)) = obj.get_mut("$defs") {
        for def in defs.values_mut() {
            *def = ensure_strict(std::mem::take(def), root);
        }
    }

    if let Some(Value::Object(definitions)) = obj.get_mut("definitions") {
        for definition in definitions.values_mut() {
            *definition = ensure_strict(std::mem::take(definition), root);
        }
    }

    if obj.get("type").and_then(Value::as_str) == Some("object")
        && !obj.contains_key("additionalProperties")
    {
        obj.insert("additionalProperties".into(), Value::Bool(false));
    }

    if obj.get("properties").is_some_and(Value::is_object) {
        let prop_keys: Vec<String> = obj
            .get("properties")
            .and_then(Value::as_object)
            .unwrap()
            .keys()
            .cloned()
            .collect();
        let required = ordered_required(&prop_keys, obj.get("required"));
        if let Some(Value::Object(properties)) = obj.get_mut("properties") {
            for prop in properties.values_mut() {
                *prop = ensure_strict(std::mem::take(prop), root);
            }
        }
        obj.insert("required".into(), Value::Array(required));
    }

    if obj.get("items").is_some_and(Value::is_object) {
        let items = obj.remove("items").unwrap();
        obj.insert("items".into(), ensure_strict(items, root));
    }

    if obj.get("anyOf").is_some_and(Value::is_array)
        && let Some(Value::Array(variants)) = obj.remove("anyOf")
    {
        obj.insert(
            "anyOf".into(),
            Value::Array(
                variants
                    .into_iter()
                    .map(|variant| ensure_strict(variant, root))
                    .collect(),
            ),
        );
    }

    if obj.get("allOf").is_some_and(Value::is_array)
        && let Some(Value::Array(mut entries)) = obj.remove("allOf")
    {
        if entries.len() == 1 {
            if let Value::Object(merged) = ensure_strict(entries.remove(0), root) {
                for (key, value) in merged {
                    obj.insert(key, value);
                }
            }
        } else {
            obj.insert(
                "allOf".into(),
                Value::Array(
                    entries
                        .into_iter()
                        .map(|entry| ensure_strict(entry, root))
                        .collect(),
                ),
            );
        }
    }

    if obj.get("default").is_some_and(Value::is_null) {
        obj.remove("default");
    }

    let inline_ref = obj
        .get("$ref")
        .and_then(Value::as_str)
        .filter(|reference| !reference.is_empty())
        .map(str::to_string);
    if let Some(reference) = inline_ref.filter(|_| obj.len() > 1) {
        let Value::Object(mut merged) = resolve_ref(root, &reference) else {
            panic!("$ref {reference} did not resolve to an object");
        };
        for (key, value) in obj {
            merged.insert(key, value);
        }
        merged.remove("$ref");
        return ensure_strict(Value::Object(merged), root);
    }

    Value::Object(obj)
}

fn ordered_required(prop_keys: &[String], existing: Option<&Value>) -> Vec<Value> {
    let existing: Vec<&str> = existing
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    existing
        .iter()
        .copied()
        .filter(|key| prop_keys.iter().any(|prop| prop == key))
        .chain(
            prop_keys
                .iter()
                .map(String::as_str)
                .filter(|key| !existing.contains(key)),
        )
        .map(|key| Value::String(key.to_string()))
        .collect()
}

fn resolve_ref(root: &Value, reference: &str) -> Value {
    reference
        .strip_prefix("#/")
        .expect("$ref must start with '#/'")
        .split('/')
        .fold(root, |current, key| &current[key])
        .clone()
}

fn py_str(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => text.clone(),
        Value::Array(items) => {
            format!(
                "[{}]",
                items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
            )
        }
        Value::Object(entries) => format!(
            "{{{}}}",
            entries
                .iter()
                .map(|(key, value)| format!("{}: {}", py_repr_str(key), py_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn py_repr(value: &Value) -> String {
    match value {
        Value::String(text) => py_repr_str(text),
        other => py_str(other),
    }
}

fn py_repr_str(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ if character == quote => {
                out.push('\\');
                out.push(character);
            }
            _ => out.push(character),
        }
    }
    out.push(quote);
    out
}

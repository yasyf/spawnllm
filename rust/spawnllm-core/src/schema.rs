use std::collections::{BTreeMap, BTreeSet};

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

const APPLE_SCALARS: &[&str] = &["string", "integer", "number", "boolean"];
const DEFS_PREFIX: &str = "#/$defs/";
const DEFINITIONS_PREFIX: &str = "#/definitions/";

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Dialect {
    Anthropic,
    Openai,
    Apple,
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
        Dialect::Apple => transform_apple(schema),
    };
    Ok(serde_json::json!({ "schema": transformed }))
}

fn transform_anthropic(schema: Value) -> Value {
    let mut src = match schema {
        Value::Object(src) => src,
        other => return other,
    };
    if src
        .get("$ref")
        .and_then(Value::as_str)
        .is_some_and(|reference| !reference.starts_with("#/"))
    {
        return Value::Object(src);
    }
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
        let Some(Value::Object(mut merged)) = resolve_ref(root, &reference) else {
            return Value::Object(obj);
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

fn resolve_ref(root: &Value, reference: &str) -> Option<Value> {
    reference
        .strip_prefix("#/")?
        .split('/')
        .try_fold(root, |current, key| current.get(key))
        .cloned()
}

fn transform_apple(schema: Value) -> Value {
    let mut out = match rewrite_definitions(schema) {
        Value::Object(out) => out,
        other => return other,
    };

    if let Some(Value::Object(defs)) = out.remove("$defs") {
        let titled: Map<String, Value> = defs
            .into_iter()
            .filter_map(|(name, def)| match def {
                Value::Object(mut def) => {
                    def.insert("title".into(), Value::String(name.clone()));
                    Some((name, Value::Object(def)))
                }
                _ => None,
            })
            .collect();
        let inline = expand_inlinable(&titled);
        out.insert("$defs".into(), Value::Object(titled));

        if !inline.is_empty() {
            out = substitute_entries(out, &inline);
            if let Some(Value::Object(defs)) = out.get_mut("$defs") {
                defs.retain(|name, _| !inline.contains_key(name));
                if defs.is_empty() {
                    out.remove("$defs");
                }
            }
        }
    }

    if let Some(name) = out
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|reference| reference.strip_prefix(DEFS_PREFIX))
        .map(str::to_string)
    {
        let mut merged = match out.get("$defs").and_then(|defs| defs.get(&name)) {
            Some(Value::Object(target)) => target.clone(),
            _ => Map::new(),
        };
        out.remove("$ref");
        merged.extend(out);
        out = merged;
    }

    let mut taken = BTreeSet::new();
    if let Some(defs) = out.get("$defs") {
        collect_titles(defs, &mut taken);
    }
    walk(Value::Object(out), &mut taken, "Schema")
}

fn rewrite_definitions(schema: Value) -> Value {
    if schema.get("definitions").is_none() {
        return schema;
    }
    let mut out = match rewrite_defs_prefix(schema) {
        Value::Object(out) => out,
        other => return other,
    };
    let mut defs = match out.remove("definitions") {
        Some(Value::Object(defs)) => defs,
        _ => Map::new(),
    };
    if let Some(Value::Object(existing)) = out.remove("$defs") {
        defs.extend(existing);
    }
    out.insert("$defs".into(), Value::Object(defs));
    Value::Object(out)
}

fn rewrite_defs_prefix(value: Value) -> Value {
    match value {
        Value::String(text) => Value::String(text.replace(DEFINITIONS_PREFIX, DEFS_PREFIX)),
        Value::Array(items) => Value::Array(items.into_iter().map(rewrite_defs_prefix).collect()),
        Value::Object(entries) => Value::Object(
            entries
                .into_iter()
                .map(|(name, value)| {
                    (
                        name.replace(DEFINITIONS_PREFIX, DEFS_PREFIX),
                        rewrite_defs_prefix(value),
                    )
                })
                .collect(),
        ),
        other => other,
    }
}

fn is_inlinable(def: &Value) -> bool {
    def.get("type").and_then(Value::as_str) != Some("object")
        && def.get("enum").is_none()
        && def.get("properties").is_none()
}

fn ref_names(node: &Value, names: &mut BTreeSet<String>) {
    match node {
        Value::Object(entries) => {
            if let Some(name) = entries
                .get("$ref")
                .and_then(Value::as_str)
                .and_then(|reference| reference.strip_prefix(DEFS_PREFIX))
            {
                names.insert(name.to_owned());
            }
            entries.values().for_each(|value| ref_names(value, names));
        }
        Value::Array(items) => items.iter().for_each(|item| ref_names(item, names)),
        _ => {}
    }
}

fn expand_inlinable(titled: &Map<String, Value>) -> BTreeMap<String, Map<String, Value>> {
    let candidates: BTreeMap<String, Map<String, Value>> = titled
        .iter()
        .filter(|(_, def)| is_inlinable(def))
        .filter_map(|(name, def)| match def {
            Value::Object(def) => {
                let mut def = def.clone();
                def.remove("title");
                Some((name.clone(), def))
            }
            _ => None,
        })
        .collect();

    let mut dependents: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut unresolved: BTreeMap<String, usize> = BTreeMap::new();
    for name in candidates.keys() {
        let mut targets = BTreeSet::new();
        ref_names(&titled[name.as_str()], &mut targets);
        targets.retain(|target| candidates.contains_key(target));
        unresolved.insert(name.clone(), targets.len());
        for target in targets {
            dependents.entry(target).or_default().push(name.clone());
        }
    }

    let mut ready: Vec<String> = unresolved
        .iter()
        .filter(|(_, pending)| **pending == 0)
        .map(|(name, _)| name.clone())
        .collect();
    let mut expanded: BTreeMap<String, Map<String, Value>> = BTreeMap::new();
    while let Some(name) = ready.pop() {
        let def = substitute_entries(candidates[&name].clone(), &expanded);
        expanded.insert(name.clone(), def);
        for dependent in dependents.get(&name).into_iter().flatten() {
            let pending = unresolved.get_mut(dependent).unwrap();
            *pending -= 1;
            if *pending == 0 {
                ready.push(dependent.clone());
            }
        }
    }
    expanded
}

fn substitute_refs(node: Value, inline: &BTreeMap<String, Map<String, Value>>) -> Value {
    match node {
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| substitute_refs(item, inline))
                .collect(),
        ),
        Value::Object(entries) => Value::Object(substitute_entries(entries, inline)),
        other => other,
    }
}

fn substitute_entries(
    entries: Map<String, Value>,
    inline: &BTreeMap<String, Map<String, Value>>,
) -> Map<String, Value> {
    let entries: Map<String, Value> = entries
        .into_iter()
        .map(|(key, value)| (key, substitute_refs(value, inline)))
        .collect();
    let Some(mut merged) = entries
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|reference| reference.strip_prefix(DEFS_PREFIX))
        .and_then(|name| inline.get(name).cloned())
    else {
        return entries;
    };
    merged.extend(entries.into_iter().filter(|(key, _)| key != "$ref"));
    merged.remove("title");
    merged
}

fn collect_titles(node: &Value, taken: &mut BTreeSet<String>) {
    match node {
        Value::Object(entries) => {
            if let Some(title) = entries.get("title").and_then(Value::as_str) {
                taken.insert(title.to_owned());
            }
            entries
                .values()
                .for_each(|value| collect_titles(value, taken));
        }
        Value::Array(items) => items.iter().for_each(|item| collect_titles(item, taken)),
        _ => {}
    }
}

fn unique_title(hint: &str, fallback: &str, taken: &mut BTreeSet<String>) -> String {
    let titled = py_title(hint);
    let base = if titled.is_empty() { fallback } else { &titled };
    let candidate = (1u32..)
        .map(|suffix| match suffix {
            1 => base.to_owned(),
            _ => format!("{base}{suffix}"),
        })
        .find(|candidate| !taken.contains(candidate))
        .unwrap();
    taken.insert(candidate.clone());
    candidate
}

fn py_title(text: &str) -> String {
    text.chars()
        .scan(false, |previous_alpha, character| {
            let cased = match previous_alpha {
                true => character.to_lowercase().collect::<String>(),
                false => character.to_uppercase().collect::<String>(),
            };
            *previous_alpha = character.is_alphabetic();
            Some(cased)
        })
        .collect()
}

fn normalize_type(mut node: Map<String, Value>) -> Map<String, Value> {
    let Some(Value::Array(types)) = node.get("type").cloned() else {
        return node;
    };
    let mut rest: Vec<Value> = types
        .into_iter()
        .filter(|kind| kind.as_str() != Some("null"))
        .collect();
    if rest.len() == 1 {
        node.insert("type".into(), rest.remove(0));
        return node;
    }
    node.remove("type");
    node.insert(
        "anyOf".into(),
        Value::Array(
            rest.into_iter()
                .map(|kind| serde_json::json!({ "type": kind }))
                .collect(),
        ),
    );
    node
}

fn walk(node: Value, taken: &mut BTreeSet<String>, hint: &str) -> Value {
    let node = match node {
        Value::Array(items) => {
            return Value::Array(
                items
                    .into_iter()
                    .map(|item| walk(item, taken, hint))
                    .collect(),
            );
        }
        Value::Object(node) => node,
        other => return other,
    };
    let mut out = normalize_type(node);

    for key in ["properties", "$defs"] {
        if let Some(Value::Object(entries)) = out.get_mut(key) {
            *entries = std::mem::take(entries)
                .into_iter()
                .map(|(name, value)| {
                    let walked = walk(value, taken, &name);
                    (name, walked)
                })
                .collect();
        }
    }
    for key in ["items", "additionalItems", "contains"] {
        if let Some(value) = out.get_mut(key).filter(|value| value.is_object()) {
            *value = walk(std::mem::take(value), taken, hint);
        }
    }
    for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
        if let Some(Value::Array(variants)) = out.get_mut(key) {
            *variants = std::mem::take(variants)
                .into_iter()
                .map(|variant| walk(variant, taken, hint))
                .collect();
        }
    }

    let properties = out.get("properties").and_then(Value::as_object);
    if out.get("type").and_then(Value::as_str) == Some("object") || properties.is_some() {
        let prop_keys: Vec<String> = properties
            .map(|properties| properties.keys().cloned().collect())
            .unwrap_or_default();
        let required: Vec<Value> = out
            .get("required")
            .and_then(Value::as_array)
            .map(|required| {
                required
                    .iter()
                    .filter(|key| {
                        key.as_str()
                            .is_some_and(|key| prop_keys.iter().any(|prop| prop == key))
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        let title = match out.get("title") {
            Some(Value::String(title)) => title.clone(),
            _ => unique_title(hint, "Schema", taken),
        };
        if !out.get("properties").is_some_and(Value::is_object) {
            out.insert("properties".into(), Value::Object(Map::new()));
        }
        if !out.contains_key("additionalProperties") {
            out.insert("additionalProperties".into(), Value::Bool(false));
        }
        out.insert("type".into(), Value::String("object".into()));
        out.insert("title".into(), Value::String(title));
        out.insert("required".into(), Value::Array(required));
        out.insert(
            "x-order".into(),
            Value::Array(prop_keys.into_iter().map(Value::String).collect()),
        );
    } else if out.get("anyOf").is_some_and(Value::is_array)
        && !out.get("title").is_some_and(Value::is_string)
    {
        let title = unique_title(hint, "Value", taken);
        out.insert("title".into(), Value::String(title));
    } else if out
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| APPLE_SCALARS.contains(&kind))
        && !out.contains_key("enum")
    {
        out.remove("title");
    }

    Value::Object(out)
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn external_refs_with_siblings_are_unchanged_in_every_dialect() {
        let schema = json!({
            "$ref": "https://example.com/x",
            "description": "external schema",
        });

        for dialect in ["anthropic", "openai", "apple"] {
            let output = dispatch(json!({ "dialect": dialect, "schema": schema.clone() }))
                .unwrap_or_else(|_| panic!("{dialect} transform succeeds"));
            assert_eq!(output["schema"], schema, "dialect: {dialect}");
        }
    }

    fn apple(schema: Value) -> Value {
        dispatch(json!({ "dialect": "apple", "schema": schema }))
            .unwrap_or_else(|_| panic!("apple transform succeeds"))["schema"]
            .take()
    }

    #[test]
    fn apple_object_gains_every_mandatory_key() {
        assert_eq!(
            apple(json!({
                "type": "object",
                "title": "Flat",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            })),
            json!({
                "type": "object",
                "title": "Flat",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
                "additionalProperties": false,
                "x-order": ["name", "age"],
            })
        );
    }

    #[test]
    fn apple_x_order_lists_exactly_the_emitted_property_keys() {
        let output = apple(json!({
            "type": "object",
            "title": "Ordered",
            "properties": {"zeta": {"type": "integer"}, "alpha": {"type": "integer"}},
        }));
        let keys: Vec<&String> = output["properties"]
            .as_object()
            .expect("properties survives")
            .keys()
            .collect();

        assert_eq!(output["x-order"], json!(keys));
        assert_eq!(output["required"], json!([]));
    }

    #[test]
    fn apple_preserves_partial_required() {
        let output = apple(json!({
            "type": "object",
            "title": "Partial",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}, "c": {"type": "integer"}},
            "required": ["c", "a", "missing"],
        }));

        assert_eq!(output["required"], json!(["c", "a"]));
        assert_eq!(output["x-order"], json!(["a", "b", "c"]));
    }

    #[test]
    fn apple_keeps_bare_refs_and_titles_defs_by_key() {
        assert_eq!(
            apple(json!({
                "$defs": {"Inner": {
                    "type": "object",
                    "title": "Renamed",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                }},
                "type": "object",
                "title": "Root",
                "properties": {"inner": {"$ref": "#/$defs/Inner"}},
                "required": ["inner"],
            })),
            json!({
                "$defs": {"Inner": {
                    "type": "object",
                    "title": "Inner",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                    "additionalProperties": false,
                    "x-order": ["a"],
                }},
                "type": "object",
                "title": "Root",
                "properties": {"inner": {"$ref": "#/$defs/Inner"}},
                "required": ["inner"],
                "additionalProperties": false,
                "x-order": ["inner"],
            })
        );
    }

    #[test]
    fn apple_strips_string_titles_without_an_enum() {
        let output = apple(json!({
            "type": "object",
            "title": "Titles",
            "properties": {
                "plain": {"type": "string", "title": "Plain"},
                "choice": {"type": "string", "title": "Choice", "enum": ["a", "b"]},
                "count": {"type": "integer", "title": "Count"},
                "tags": {"type": "array", "title": "Tags", "items": {"type": "string"}},
            },
            "required": ["plain", "choice", "count", "tags"],
        }));

        assert_eq!(output["properties"]["plain"], json!({"type": "string"}));
        assert_eq!(
            output["properties"]["choice"],
            json!({"type": "string", "title": "Choice", "enum": ["a", "b"]})
        );
        assert_eq!(output["properties"]["count"], json!({"type": "integer"}));
        assert_eq!(
            output["properties"]["tags"],
            json!({"type": "array", "title": "Tags", "items": {"type": "string"}})
        );
    }

    #[test]
    fn apple_titles_untitled_any_of_nodes() {
        let output = apple(json!({
            "type": "object",
            "title": "Choice",
            "properties": {
                "zip_code": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "kept": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Kept"},
            },
            "required": ["zip_code", "kept"],
        }));

        assert_eq!(output["properties"]["zip_code"]["title"], json!("Zip_Code"));
        assert_eq!(output["properties"]["kept"]["title"], json!("Kept"));
    }

    #[test]
    fn apple_deduplicates_generated_titles() {
        let output = apple(json!({
            "$defs": {"Value": {"type": "object", "properties": {}, "required": []}},
            "type": "object",
            "title": "Root",
            "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
            "required": ["value"],
        }));

        assert_eq!(output["$defs"]["Value"]["title"], json!("Value"));
        assert_eq!(output["properties"]["value"]["title"], json!("Value2"));
    }

    #[test]
    fn apple_preserves_generation_constraints() {
        let output = apple(json!({
            "type": "object",
            "title": "Coded",
            "properties": {
                "code": {"type": "string", "pattern": "^[a-z]+$", "minLength": 3},
                "counts": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 9}, "minItems": 1, "maxItems": 4},
                "grade": {"type": "string", "enum": ["a", "b"]},
            },
            "required": ["code"],
        }));

        assert_eq!(
            output["properties"]["code"],
            json!({"type": "string", "pattern": "^[a-z]+$", "minLength": 3})
        );
        assert_eq!(
            output["properties"]["counts"],
            json!({"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 9}, "minItems": 1, "maxItems": 4})
        );
        assert_eq!(output["properties"]["grade"]["enum"], json!(["a", "b"]));
    }

    #[test]
    fn apple_collapses_nullable_type_lists() {
        let output = apple(json!({
            "type": "object",
            "title": "Types",
            "properties": {
                "nullable": {"type": ["string", "null"]},
                "multi": {"type": ["string", "integer"]},
            },
            "required": ["nullable"],
        }));

        assert_eq!(output["properties"]["nullable"], json!({"type": "string"}));
        assert_eq!(
            output["properties"]["multi"],
            json!({"anyOf": [{"type": "string"}, {"type": "integer"}], "title": "Multi"})
        );
    }

    #[test]
    fn apple_rewrites_definitions_refs() {
        let output = apple(json!({
            "definitions": {"Inner": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
            }},
            "type": "object",
            "title": "Root",
            "properties": {"inner": {"$ref": "#/definitions/Inner"}},
            "required": ["inner"],
        }));

        assert!(output.get("definitions").is_none());
        assert_eq!(
            output["properties"]["inner"],
            json!({"$ref": "#/$defs/Inner"})
        );
        assert_eq!(output["$defs"]["Inner"]["title"], json!("Inner"));
    }

    #[test]
    fn apple_inlines_unnameable_string_defs() {
        let output = apple(json!({
            "$defs": {"Alias": {"type": "string", "minLength": 2}},
            "type": "object",
            "title": "Root",
            "properties": {"a": {"$ref": "#/$defs/Alias"}},
            "required": ["a"],
        }));

        assert!(output.get("$defs").is_none());
        assert_eq!(
            output["properties"]["a"],
            json!({"type": "string", "minLength": 2})
        );
    }

    #[test]
    fn apple_inlines_a_bare_ref_root() {
        let output = apple(json!({
            "$defs": {"Root": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            }},
            "$ref": "#/$defs/Root",
        }));

        assert!(output.get("$ref").is_none());
        assert_eq!(output["type"], json!("object"));
        assert_eq!(output["title"], json!("Root"));
        assert_eq!(output["x-order"], json!(["x"]));
    }

    #[test]
    fn apple_terminates_on_recursive_schemas() {
        let recursive = apple(json!({
            "$defs": {"Node": {
                "type": "object",
                "properties": {"next": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}], "title": "Next"}},
                "required": [],
            }},
            "$ref": "#/$defs/Node",
        }));
        assert_eq!(
            recursive["$defs"]["Node"]["properties"]["next"]["anyOf"][0],
            json!({"$ref": "#/$defs/Node"})
        );

        let mutual = apple(json!({
            "$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}},
            "type": "object",
            "title": "Cyclic",
            "properties": {"a": {"$ref": "#/$defs/A"}},
            "required": ["a"],
        }));
        assert_eq!(mutual["properties"]["a"], json!({"$ref": "#/$defs/A"}));
    }

    fn dangling_refs(schema: &Value) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        ref_names(schema, &mut names);
        let defs = schema.get("$defs").and_then(Value::as_object);
        names.retain(|name| !defs.is_some_and(|defs| defs.contains_key(name)));
        names
    }

    #[test]
    fn apple_inlines_a_long_flat_ref_chain() {
        const LINKS: usize = 4000;
        let defs: Map<String, Value> = (0..LINKS)
            .map(|index| {
                let def = match index + 1 == LINKS {
                    true => json!({"type": "string"}),
                    false => json!({"$ref": format!("{DEFS_PREFIX}D{}", index + 1)}),
                };
                (format!("D{index}"), def)
            })
            .collect();
        let output = apple(json!({
            "$defs": defs,
            "type": "object",
            "title": "Root",
            "properties": {"a": {"$ref": "#/$defs/D0"}},
            "required": ["a"],
        }));

        assert!(output.get("$defs").is_none());
        assert_eq!(output["properties"]["a"], json!({"type": "string"}));
        assert_eq!(apple(output.clone()), output);
    }

    #[test]
    fn apple_keeps_cyclic_inlinable_defs_resolvable() {
        let mutual = apple(json!({
            "$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}},
            "type": "object",
            "title": "Root",
            "properties": {"a": {"$ref": "#/$defs/A"}},
            "required": ["a"],
        }));

        assert_eq!(dangling_refs(&mutual), BTreeSet::new());
        assert_eq!(
            mutual["$defs"],
            json!({"A": {"$ref": "#/$defs/B", "title": "A"}, "B": {"$ref": "#/$defs/A", "title": "B"}})
        );
        assert_eq!(apple(mutual.clone()), mutual);

        let looped = apple(json!({
            "$defs": {"A": {"$ref": "#/$defs/A"}},
            "type": "object",
            "title": "Root",
            "properties": {"a": {"$ref": "#/$defs/A"}},
            "required": ["a"],
        }));

        assert_eq!(dangling_refs(&looped), BTreeSet::new());
        assert_eq!(apple(looped.clone()), looped);
    }

    #[test]
    fn apple_round_trips_a_self_referential_object_def() {
        let output = apple(json!({
            "$defs": {"Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}, "label": {"type": "string"}},
                "required": ["label"],
            }},
            "type": "object",
            "title": "Root",
            "properties": {"root": {"$ref": "#/$defs/Node"}},
            "required": ["root"],
        }));

        assert_eq!(dangling_refs(&output), BTreeSet::new());
        assert_eq!(
            output["$defs"]["Node"]["properties"]["child"],
            json!({"$ref": "#/$defs/Node"})
        );
        assert_eq!(
            output["$defs"]["Node"]["x-order"],
            json!(["child", "label"])
        );
        assert_eq!(apple(output.clone()), output);
    }

    #[test]
    fn apple_transform_is_idempotent() {
        let schema = json!({
            "$defs": {
                "Color": {"type": "string", "enum": ["red", "blue"], "title": "Color"},
                "Alias": {"type": "string", "maxLength": 8},
                "Inner": {
                    "type": "object",
                    "properties": {"a": {"type": "integer", "title": "A"}, "b": {"type": "string", "title": "B"}},
                    "required": ["a"],
                },
            },
            "type": "object",
            "title": "NestedRefs",
            "properties": {
                "color": {"$ref": "#/$defs/Color"},
                "inner": {"$ref": "#/$defs/Inner"},
                "alias": {"$ref": "#/$defs/Alias"},
                "name": {"type": "string", "title": "Name", "pattern": "^x"},
                "opt": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": null},
                "rows": {"type": "array", "title": "Rows", "items": {"$ref": "#/$defs/Inner"}},
            },
            "required": ["name", "inner", "rows", "color"],
        });

        let once = apple(schema);
        assert_eq!(apple(once.clone()), once);
    }
}

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Number, Value, json};
use spawnllm_core::dispatch;

fn vectors_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/vectors")
}

fn sorted_children(dir: &Path, keep: impl Fn(&Path) -> bool) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| keep(path))
        .collect();
    paths.sort();
    paths
}

fn number_eq(left: &Number, right: &Number) -> bool {
    let left = left.to_string();
    let right = right.to_string();
    if left == right {
        return true;
    }
    let float_shaped = |value: &str| value.contains(['.', 'e', 'E']);
    if !float_shaped(&left) && !float_shaped(&right) {
        return false;
    }
    matches!(
        (left.parse::<f64>(), right.parse::<f64>()),
        (Ok(a), Ok(b)) if a == b && a.is_finite()
    )
}

fn value_eq(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Number(a), Value::Number(b)) => number_eq(a, b),
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| value_eq(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, x)| b.get(key).is_some_and(|y| value_eq(x, y)))
        }
        _ => left == right,
    }
}

#[test]
fn conformance_vectors() {
    let mut passed = 0usize;
    let mut replayed = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for op_dir in sorted_children(&vectors_dir(), |path| path.is_dir()) {
        let op = op_dir.file_name().unwrap().to_string_lossy().into_owned();
        for path in sorted_children(&op_dir, |path| {
            path.extension().is_some_and(|ext| ext == "json")
        }) {
            replayed += 1;
            let vector: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
            let label = format!("{op}/{}", vector["name"].as_str().unwrap());
            let request = json!({ "op": vector["op"], "input": vector["input"] });
            let response: Value = serde_json::from_str(&dispatch(&request.to_string())).unwrap();

            match (response.get("ok"), response.get("err")) {
                (Some(ok), None) if value_eq(ok, &vector["expected"]) => passed += 1,
                (Some(ok), None) => failures.push(format!(
                    "MISMATCH {label}\n  expected: {}\n  actual:   {}",
                    vector["expected"], ok
                )),
                (None, Some(err)) if err["kind"] == "unimplemented" => {
                    failures.push(format!("UNIMPLEMENTED {label}"))
                }
                (None, Some(err)) => failures.push(format!("ERROR {label}: {err}")),
                _ => failures.push(format!("MALFORMED {label}: {response}")),
            }
        }
    }

    println!(
        "conformance: {passed} passed, {} failed ({replayed} replayed)",
        failures.len()
    );

    assert!(replayed > 0, "no conformance vectors replayed");
    assert!(
        failures.is_empty(),
        "conformance failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn pure_integer_numbers_do_not_compare_through_f64() {
    let left = serde_json::from_str::<Number>("9007199254740992").unwrap();
    let right = serde_json::from_str::<Number>("9007199254740993").unwrap();

    assert!(!number_eq(&left, &right));
}

#[test]
fn float_shaped_numbers_can_compare_through_f64() {
    let integer = serde_json::from_str::<Number>("1").unwrap();
    let float = serde_json::from_str::<Number>("1.0").unwrap();

    assert!(number_eq(&integer, &float));
}

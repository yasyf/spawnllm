use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::{env, fs};

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
    left.to_string() == right.to_string()
        || matches!((left.as_f64(), right.as_f64()), (Some(a), Some(b)) if a == b && a.is_finite())
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

fn ops_filter() -> Option<BTreeSet<String>> {
    env::var("SPAWNLLM_CONFORMANCE_OPS").ok().map(|raw| {
        raw.split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(String::from)
            .collect()
    })
}

#[test]
fn conformance_vectors() {
    let strict = env::var("SPAWNLLM_CONFORMANCE_STRICT").is_ok_and(|value| value == "1");
    let filter = ops_filter();

    let mut passed = 0usize;
    let mut skipped: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for op_dir in sorted_children(&vectors_dir(), |path| path.is_dir()) {
        let op = op_dir.file_name().unwrap().to_string_lossy().into_owned();
        if filter.as_ref().is_some_and(|ops| !ops.contains(&op)) {
            continue;
        }
        for path in sorted_children(&op_dir, |path| {
            path.extension().is_some_and(|ext| ext == "json")
        }) {
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
                (None, Some(err)) if err["kind"] == "unimplemented" && !strict => {
                    skipped.push(label)
                }
                (None, Some(err)) if err["kind"] == "unimplemented" => failures.push(format!(
                    "UNIMPLEMENTED {label} (SPAWNLLM_CONFORMANCE_STRICT=1)"
                )),
                (None, Some(err)) => failures.push(format!("ERROR {label}: {err}")),
                _ => failures.push(format!("MALFORMED {label}: {response}")),
            }
        }
    }

    let skipped_ops: BTreeSet<&str> = skipped
        .iter()
        .filter_map(|label| label.split('/').next())
        .collect();
    println!(
        "conformance: {passed} passed, {} skipped {skipped_ops:?}, {} failed",
        skipped.len(),
        failures.len()
    );

    assert!(
        failures.is_empty(),
        "conformance failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

mod cases;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use cases::Case;

pub enum Mode {
    Diff,
    Write,
}

fn conformance_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance")
}

pub fn render(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap() + "\n"
}

fn expected(case: &Case) -> Value {
    let request = json!({"op": case.op, "input": case.input});
    let response: Value =
        serde_json::from_str(&spawnllm_core::dispatch(&request.to_string())).unwrap();
    match response.get("ok") {
        Some(ok) => ok.clone(),
        None => panic!("dispatch error for {}/{}: {response}", case.op, case.name),
    }
}

pub fn artifacts() -> BTreeMap<PathBuf, String> {
    let conf = conformance_dir();
    let mut map = BTreeMap::new();
    for case in cases::all_cases() {
        let vector = json!({
            "name": case.name,
            "op": case.op,
            "input": case.input,
            "expected": expected(&case),
        });
        map.insert(
            conf.join("vectors")
                .join(case.op)
                .join(format!("{}.json", case.name)),
            render(&vector),
        );
    }
    for (name, body) in cases::contract_schemas() {
        map.insert(
            conf.join("schema").join(format!("{name}.schema.json")),
            body.to_owned(),
        );
    }
    map
}

fn collect_json(dir: &Path, out: &mut BTreeSet<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_json(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "json") {
            out.insert(path);
        }
    }
}

pub fn committed_files() -> BTreeSet<PathBuf> {
    let conf = conformance_dir();
    let mut files = BTreeSet::new();
    collect_json(&conf.join("vectors"), &mut files);
    collect_json(&conf.join("schema"), &mut files);
    files
}

fn rel(path: &Path) -> String {
    path.strip_prefix(conformance_dir())
        .unwrap_or(path)
        .display()
        .to_string()
}

pub fn run(mode: Mode) -> bool {
    let artifacts = artifacts();
    let artifact_paths: BTreeSet<PathBuf> = artifacts.keys().cloned().collect();
    let orphans: Vec<PathBuf> = committed_files()
        .difference(&artifact_paths)
        .cloned()
        .collect();

    match mode {
        Mode::Diff => {
            let (mut matched, mut changed, mut missing) = (0usize, 0usize, 0usize);
            for (path, content) in &artifacts {
                match fs::read_to_string(path) {
                    Ok(on_disk) if &on_disk == content => matched += 1,
                    Ok(_) => {
                        changed += 1;
                        println!("CHANGED {}", rel(path));
                    }
                    Err(_) => {
                        missing += 1;
                        println!("MISSING {}", rel(path));
                    }
                }
            }
            for orphan in &orphans {
                println!("ORPHAN  {}", rel(orphan));
            }
            println!(
                "conformance-gen: {} artifacts, {matched} matched, {changed} changed, {missing} missing, {} orphan",
                artifacts.len(),
                orphans.len(),
            );
            changed == 0 && missing == 0 && orphans.is_empty()
        }
        Mode::Write => {
            let (mut created, mut updated, mut unchanged) = (0usize, 0usize, 0usize);
            for (path, content) in &artifacts {
                match fs::read_to_string(path).ok() {
                    Some(ref on_disk) if on_disk == content => unchanged += 1,
                    Some(_) => {
                        write_file(path, content);
                        updated += 1;
                    }
                    None => {
                        write_file(path, content);
                        created += 1;
                    }
                }
            }
            for orphan in &orphans {
                fs::remove_file(orphan).unwrap();
                println!("DELETE  {}", rel(orphan));
            }
            println!(
                "conformance-gen: wrote {} artifacts ({created} created, {updated} updated, {unchanged} unchanged), deleted {} orphans",
                artifacts.len(),
                orphans.len(),
            );
            true
        }
    }
}

fn write_file(path: &Path, content: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

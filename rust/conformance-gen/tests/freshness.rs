use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use conformance_gen::{artifacts, committed_files};

#[test]
fn committed_tree_matches_fresh_regeneration() {
    let artifacts = artifacts();

    for (path, content) in &artifacts {
        let on_disk = fs::read_to_string(path)
            .unwrap_or_else(|_| panic!("missing committed artifact {}", path.display()));
        assert_eq!(
            &on_disk,
            content,
            "stale committed artifact {}",
            path.display()
        );
    }

    let generated: BTreeSet<PathBuf> = artifacts.keys().cloned().collect();
    let committed = committed_files();

    let orphans: Vec<&PathBuf> = committed.difference(&generated).collect();
    assert!(
        orphans.is_empty(),
        "orphan committed artifacts: {orphans:?}"
    );

    let uncommitted: Vec<&PathBuf> = generated.difference(&committed).collect();
    assert!(
        uncommitted.is_empty(),
        "generated artifacts missing from tree: {uncommitted:?}"
    );
}

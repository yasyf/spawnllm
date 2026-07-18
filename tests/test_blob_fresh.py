"""Guard that the packaged wasm blob was built from the committed rust source.

`scripts/build_wasm.sh --hash-only` re-derives the metadata-free content hash of
`rust/spawnllm-core`, `rust/spawnllm-wasm`, and `rust/Cargo.lock`; the core stamps
that same hash into `version().source_hash` at build time. A blob rebuilt from
different source, or committed rust drifting past a stale blob, diverges here.
Skipped for sdist consumers, who ship only the prebuilt blob without the rust tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spawnllm import _core

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (REPO_ROOT / "rust").is_dir(),
    reason="rust/ source tree absent (sdist consumers ship only the prebuilt blob)",
)
def test_blob_matches_committed_rust_source() -> None:
    result = subprocess.run(
        ["bash", "scripts/build_wasm.sh", "--hash-only"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == _core.version()["source_hash"]

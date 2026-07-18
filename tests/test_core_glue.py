from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from spawnllm._core import CoreError, dispatch, dispatch_raw, version

VECTORS = Path(__file__).parent.parent / "conformance" / "vectors"


def load_vector(op: str, name: str) -> dict:
    return json.loads((VECTORS / op / f"{name}.json").read_text())


def test_version_round_trips_core_version_and_source_hash() -> None:
    result = version()
    assert result["core_version"] == "0.0.0"
    assert len(result["source_hash"]) == 64
    assert all(c in "0123456789abcdef" for c in result["source_hash"])


def test_unknown_op_raises_core_error_with_unknown_op_kind() -> None:
    with pytest.raises(CoreError) as exc:
        dispatch("definitely-not-an-op")
    assert exc.value.kind == "unknown_op"
    assert exc.value.msg == "definitely-not-an-op"


def test_plan_op_matches_committed_vector() -> None:
    vector = load_vector("plan", "claude-default")
    assert dispatch("plan", vector["input"]) == vector["expected"]


def test_dispatch_raw_preserves_float_bytes_parse_free() -> None:
    vector = load_vector("retry_decision", "transient-503-attempt-2")
    raw = dispatch_raw(json.dumps({"op": "retry_decision", "input": vector["input"]}))
    assert f'"sleep_s":{vector["expected"]["sleep_s"]}' in raw
    assert '"sleep_s":45.0' in raw


def test_dispatch_is_thread_safe_under_concurrent_load() -> None:
    expected = version()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [f.result() for f in [pool.submit(lambda: dispatch("version")) for _ in range(8 * 50)]]
    assert len(results) == 400
    assert all(result == expected for result in results)

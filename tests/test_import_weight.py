from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

ROOT = Path(__file__).resolve().parent.parent


def fresh_python(program: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", dedent(program)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    ("module", "symbol"),
    [("spawnllm", "TModel"), ("spawnllm.decide", "BinaryAnswer")],
    ids=["model-metadata", "decision-metadata"],
)
def test_metadata_import_does_not_load_wasmtime(module: str, symbol: str) -> None:
    observed = fresh_python(
        f"""
        import importlib
        import json
        import sys
        from pathlib import Path

        loaded = []

        def record(event, args):
            if event == "ctypes.dlopen":
                loaded.append(str(args[0]))

        sys.addaudithook(record)
        metadata = getattr(importlib.import_module({module!r}), {symbol!r})
        from spawnllm import _core

        print(json.dumps({{
            "runtime_built": _core.RUNTIME is not None,
            "native_libraries": [name for name in loaded if "wasmtime" in Path(name).name.lower()],
            "metadata_available": metadata is not None,
        }}))
        """
    )
    assert observed["runtime_built"] is False
    assert observed["metadata_available"] is True
    assert observed["native_libraries"] == []


def test_first_dispatch_builds_the_real_runtime_once_and_warm_dispatch_reuses_it() -> None:
    observed = fresh_python(
        """
        import json
        from spawnllm import _core

        assert _core.RUNTIME is None
        builds = []
        original = _core.build

        def record():
            runtime = original()
            builds.append(runtime)
            return runtime

        _core.build = record
        first = _core.version()
        runtime = _core.RUNTIME
        second = _core.version()
        print(json.dumps({
            "first": first,
            "second": second,
            "builds": len(builds),
            "same_runtime": runtime is _core.RUNTIME,
        }))
        """
    )
    assert observed["builds"] == 1
    assert observed["same_runtime"] is True
    assert observed["first"] == observed["second"]
    assert len(observed["first"]["source_hash"]) == 64

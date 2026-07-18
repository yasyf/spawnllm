"""Replay every committed golden vector through the wasm core.

The Rust core owns the drift-prone logic; `conformance/vectors/<op>/*.json` pins
what it must return for a fixed input. Each vector is replayed by constructing the
same `{op, input}` request the Go and Rust harnesses build, dispatching it through
`_core.dispatch_raw`, and comparing the `ok` payload against `expected` with a
faithful Python port of the Go harness's number-tolerant `valueEqual`/`numberEqual`
(`go/internal/core/conformance_test.go`): distinct integer lexemes never collapse,
but a decimal or exponent lexeme compares by float value, so `45.0` equals `45`
while `45.1` does not. Every op replays — there is no skip tier.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from spawnllm import _core

REPO_ROOT = Path(__file__).resolve().parents[1]
VECTORS_DIR = REPO_ROOT / "conformance" / "vectors"
VECTORS = sorted(VECTORS_DIR.rglob("*.json"))
MIN_VECTORS = 100
FLOAT_MARKERS = frozenset(".eE")


@dataclass(frozen=True, slots=True)
class Number:
    lexeme: str


def load(text: str) -> object:
    return json.loads(text, parse_int=Number, parse_float=Number)


def number_equal(a: Number, b: Number) -> bool:
    if a.lexeme == b.lexeme:
        return True
    if not (FLOAT_MARKERS & set(a.lexeme)) and not (FLOAT_MARKERS & set(b.lexeme)):
        return False
    return (af := float(a.lexeme)) == float(b.lexeme) and not math.isinf(af) and not math.isnan(af)


def value_equal(a: object, b: object) -> bool:
    match a:
        case Number():
            return isinstance(b, Number) and number_equal(a, b)
        case list():
            return (
                isinstance(b, list) and len(a) == len(b) and all(value_equal(x, y) for x, y in zip(a, b, strict=True))
            )
        case dict():
            return isinstance(b, dict) and a.keys() == b.keys() and all(value_equal(v, b[k]) for k, v in a.items())
        case _:
            return type(a) is type(b) and a == b


def test_vector_corpus_present() -> None:
    assert len(VECTORS) >= MIN_VECTORS, f"expected >= {MIN_VECTORS} vectors, found {len(VECTORS)} under {VECTORS_DIR}"


@pytest.mark.parametrize("path", VECTORS, ids=lambda p: str(p.relative_to(VECTORS_DIR).with_suffix("")))
def test_vector_replays_through_core(path: Path) -> None:
    text = path.read_text()
    vector = json.loads(text)
    envelope = load(_core.dispatch_raw(json.dumps({"op": vector["op"], "input": vector["input"]})))
    assert isinstance(envelope, dict)
    assert envelope.get("err") is None, f"{vector['name']}: core returned error envelope {envelope.get('err')}"
    expected = load(text)["expected"]
    assert value_equal(envelope["ok"], expected), (
        f"{vector['name']}: core output diverged from expected\n got: {envelope['ok']}\nwant: {expected}"
    )

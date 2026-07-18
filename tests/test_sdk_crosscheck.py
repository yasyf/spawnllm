"""Drift canary: the wasm core's strict-schema transform must equal the live SDKs.

The core reimplements each provider's strict-schema transform so the runtime needs
no anthropic/openai dependency. This test pins that reimplementation to the real
SDK output: for the nine `STRICT_SCHEMA_CASES` models (recovered from the deleted
`tests/conformance/cases.py` generator via `git show 'HEAD:tests/conformance/cases.py'`),
`_core.dispatch("strict_schema", ...)` must deep-equal
`anthropic.lib._parse._transform.transform_schema` (anthropic) and
`openai.lib._pydantic.to_strict_json_schema` (openai). An SDK bump that changes
either transform breaks here until the core is regenerated.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import pytest
from anthropic.lib._parse._transform import transform_schema
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, Field

from spawnllm import _core

if TYPE_CHECKING:
    from collections.abc import Callable


class Color(StrEnum):
    red = "red"
    blue = "blue"


class Flat(BaseModel):
    x: int


class OptionalField(BaseModel):
    a: int
    b: str | None = None


class Inner(BaseModel):
    a: int
    b: str = "default"


class NestedRefs(BaseModel):
    name: str
    inner: Inner
    tags: list[str]
    color: Color
    opt: int | None = None


class Arrays(BaseModel):
    items: list[int]
    matrix: list[list[str]]


class Enums(BaseModel):
    color: Color


class Defaults(BaseModel):
    a: int = 1
    b: str = "hi"
    c: bool = False


class ListOfModels(BaseModel):
    rows: list[Inner]


class Union(BaseModel):
    val: int | str


class Constrained(BaseModel):
    n: int = Field(ge=1, le=10)
    label: str = Field(min_length=2)


STRICT_SCHEMA_CASES: tuple[tuple[str, type[BaseModel]], ...] = (
    ("flat", Flat),
    ("optional-field", OptionalField),
    ("nested-refs", NestedRefs),
    ("arrays", Arrays),
    ("enums", Enums),
    ("defaults", Defaults),
    ("list-of-models", ListOfModels),
    ("union", Union),
    ("constrained", Constrained),
)

DIALECTS: tuple[tuple[str, Callable[[type[BaseModel]], dict[str, object]]], ...] = (
    ("anthropic", transform_schema),
    ("openai", to_strict_json_schema),
)


@pytest.mark.parametrize("dialect,transform", DIALECTS, ids=[name for name, _ in DIALECTS])
@pytest.mark.parametrize("case_name,model", STRICT_SCHEMA_CASES, ids=[name for name, _ in STRICT_SCHEMA_CASES])
def test_core_strict_schema_matches_live_sdk(
    case_name: str,
    model: type[BaseModel],
    dialect: str,
    transform: Callable[[type[BaseModel]], dict[str, object]],
) -> None:
    core = _core.dispatch("strict_schema", {"dialect": dialect, "schema": model.model_json_schema()})["schema"]
    assert core == transform(model), f"{case_name}-{dialect}: core transform diverged from the live SDK"

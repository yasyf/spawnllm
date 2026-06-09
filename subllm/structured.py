"""Structured-output helpers: JSON-schema build, schema-path resolution, response parsing."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, cast

from subllm.backends.codex import CodexCliBackend

if TYPE_CHECKING:
    from pydantic import BaseModel

    from subllm.backends.base import LlmBackend

__all__ = [
    "extract_structured",
    "parse_result_envelope",
    "parse_structured_output",
    "resolve_schema_path",
    "schema_for",
]


def schema_for(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema() | {"additionalProperties": False})


def resolve_schema_path(backend: LlmBackend, schema: str | None) -> str | None:
    if not schema:
        return None
    if isinstance(backend, CodexCliBackend):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.write(fd, schema.encode())
        os.close(fd)
        return path
    return schema


def extract_structured(events: list[dict[str, Any]], model: type[BaseModel]) -> BaseModel | None:
    """Return the validated ``structured_output`` from a stream-json event list, if present."""
    for e in events:
        if e.get("type") == "result" and "structured_output" in e:
            return model.model_validate(e["structured_output"])
    return None


def parse_structured_output(raw: str, response_model: type[BaseModel] | None) -> str | BaseModel:
    if not response_model:
        return raw
    data = json.loads(raw)
    if isinstance(data, list) and data:
        return extract_structured(
            cast(list[dict[str, Any]], data), response_model
        ) or response_model.model_validate_json(raw)
    return response_model.model_validate_json(raw)


def parse_result_envelope(stdout: bytes, *, argv: list[str], stderr: bytes) -> str:
    data = json.loads(stdout)
    if data["is_error"]:
        raise subprocess.CalledProcessError(0, argv, output=stdout, stderr=stderr)
    return data["result"]

"""Structured-output helpers: schema-path resolution and response parsing."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, cast, overload

from spawnllm.backends.codex import CodexCliBackend

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.backends.base import LlmBackend

__all__ = [
    "extract_json_block",
    "extract_structured",
    "parse_result_envelope",
    "parse_structured_output",
    "resolve_schema_path",
]

JSON_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def extract_json_block(text: str) -> str:
    """Extract a JSON object/array from model text, tolerating ```json fences or surrounding prose."""
    if match := JSON_FENCE.search(text):
        return match.group(1).strip()
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=0)
    return text[start : max(text.rfind("}"), text.rfind("]")) + 1]


def resolve_schema_path(backend: LlmBackend, schema: str | None) -> str | None:
    """Resolve a JSON schema into the argument form the backend's CLI expects.

    Args:
        backend: The backend the schema is destined for.
        schema: The JSON schema string, or `None`.

    Returns:
        A temp-file path for `CodexCliBackend`, the schema unchanged for other backends, or `None` without a schema.
    """
    if not schema:
        return None
    if isinstance(backend, CodexCliBackend):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.write(fd, schema.encode())
        os.close(fd)
        return path
    return schema


def extract_structured[M: BaseModel](events: list[dict[str, Any]], model: type[M]) -> M | None:
    """Return the validated `structured_output` from a stream-json event list, if present."""
    for e in events:
        if e.get("type") == "result" and "structured_output" in e:
            return model.model_validate(e["structured_output"])
    return None


@overload
def parse_structured_output(raw: str, response_model: None) -> str: ...
@overload
def parse_structured_output[M: BaseModel](raw: str, response_model: type[M]) -> M: ...
def parse_structured_output[M: BaseModel](raw: str, response_model: type[M] | None) -> str | M:
    """Parse raw CLI stdout into text or a validated model.

    Args:
        raw: Raw stdout from the backend CLI.
        response_model: The Pydantic model to validate against, or `None` for text.

    Returns:
        `raw` for text calls; otherwise `structured_output` from the stream-json events, else `raw` validated as JSON.
    """
    if not response_model:
        return raw
    data = json.loads(raw)
    events = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
    return extract_structured(
        cast(list[dict[str, Any]], events), response_model
    ) or response_model.model_validate_json(raw)


def parse_result_envelope(stdout: bytes, *, argv: list[str], stderr: bytes) -> str:
    """Parse a `{is_error, result}` JSON envelope into its result text.

    Args:
        stdout: Raw stdout bytes holding the envelope.
        argv: The argv that produced the output, for error reporting.
        stderr: Raw stderr bytes, attached to the error.

    Returns:
        The `result` payload.

    Raises:
        subprocess.CalledProcessError: When the envelope marks the run as an error.
    """
    data = json.loads(stdout)
    if data["is_error"]:
        raise subprocess.CalledProcessError(0, argv, output=stdout, stderr=stderr)
    return data["result"]

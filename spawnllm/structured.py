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
    from spawnllm.proc import RunResult

__all__ = [
    "backoff",
    "extract_json_block",
    "extract_structured",
    "is_transient",
    "parse_result_envelope",
    "parse_structured_output",
    "resolve_schema_path",
]

JSON_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
TRANSIENT = re.compile(r"\b529\b|overloaded|rate.?limit|\b5\d\d\b", re.I)


def first_json_value(source: str) -> str | None:
    """Return the first complete JSON object/array in `source`, or `None` when there is none."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(source):
        if char in "{[":
            try:
                end = decoder.raw_decode(source, index)[1]
            except (json.JSONDecodeError, RecursionError):
                continue
            return source[index:end]
    return None


def extract_json_block(text: str) -> str:
    """Extract the first complete JSON value from model text, tolerating ```json fences or surrounding prose."""
    fenced = match.group(1) if (match := JSON_FENCE.search(text)) else None
    for source in (fenced, text):
        if source is not None and (value := first_json_value(source)) is not None:
            return value
    raise ValueError(f"no JSON value found in model output: {text!r}")


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


def envelope_is_error(stdout: str) -> bool:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return False
    match data:
        case dict():
            return bool(data.get("is_error"))
        case list():
            return any(e.get("type") == "result" and e.get("is_error") for e in data)
        case _:
            return False


def is_transient(rr: RunResult) -> bool:
    """Report whether a run failed with a retryable transient error.

    A run is transient iff the `TRANSIENT` pattern (529, overloaded, rate-limit,
    or any `5xx`) matches its stdout or stderr **and** the run actually failed —
    either a nonzero exit code or a `{is_error: true}` stdout envelope (a dict
    or any `type=="result"` element of a JSON array).

    Args:
        rr: The captured outcome of a single backend run.

    Returns:
        `True` when the run should be retried.
    """
    return bool(TRANSIENT.search(rr.stdout) or TRANSIENT.search(rr.stderr)) and (
        rr.returncode != 0 or envelope_is_error(rr.stdout)
    )


def backoff(attempt: int) -> float:
    """Return the seconds to sleep before retry `attempt`, capped at 60.

    Args:
        attempt: The zero-based retry attempt number.

    Returns:
        `min(5 * 3**attempt, 60)` seconds.
    """
    return min(5 * 3**attempt, 60)

"""Structured-output helpers: schema-path resolution, JSON extraction, transient detection."""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import TYPE_CHECKING, cast

from spawnllm.backends.codex import CodexCliBackend

if TYPE_CHECKING:
    from spawnllm.backends.base import LlmBackend
    from spawnllm.response import Response

__all__ = [
    "backoff",
    "extract_json_block",
    "is_transient",
    "resolve_schema_path",
    "structured_value",
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


def structured_value(raw: str) -> object:
    """Return the JSON value to validate from a stream-json envelope.

    Parses `raw` as JSON; when a `type=="result"` event carries a
    `structured_output` field (claude/mlx stream-json), returns that field,
    otherwise returns the parsed value itself.

    Args:
        raw: Raw stdout holding a JSON value or a list of stream-json events.

    Returns:
        The `structured_output` payload when present, else the parsed JSON.
    """
    data = json.loads(raw)
    events = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
    for e in cast(list[dict[str, object]], events):
        if e.get("type") == "result" and "structured_output" in e:
            return e["structured_output"]
    return data


def is_transient(resp: Response) -> bool:
    """Report whether a response failed with a retryable transient error.

    A response is transient iff it carries an `error` whose text matches the
    `TRANSIENT` pattern (529, overloaded, rate-limit, or any `5xx`).

    Args:
        resp: The backend's resolved response.

    Returns:
        `True` when the response should be retried.
    """
    return resp.error is not None and bool(TRANSIENT.search(resp.error))


def backoff(attempt: int) -> float:
    """Return the seconds to sleep before retry `attempt`, capped at 60.

    Args:
        attempt: The zero-based retry attempt number.

    Returns:
        `min(5 * 3**attempt, 60)` seconds.
    """
    return min(5 * 3**attempt, 60)

"""Structured-output helpers still needed natively by MLX and the JSON-extraction wrapper."""

from __future__ import annotations

import json
from typing import cast

from spawnllm import _core

__all__ = ["extract_json_block", "structured_value"]


def extract_json_block(text: str) -> str:
    """Extract the first complete JSON value from model text, tolerating ```json fences or surrounding prose.

    Args:
        text: The model output to scan for a JSON object or array.

    Returns:
        The extracted JSON value re-serialized as a string.

    Raises:
        ValueError: When the core finds no JSON value in `text`.
    """
    if (value := _core.dispatch("extract_json", {"text": text})["value"]) is None:
        raise ValueError(f"no JSON value found in model output: {text!r}")
    return json.dumps(value)


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
    for e in cast("list[dict[str, object]]", events):
        if e.get("type") == "result" and "structured_output" in e:
            return e["structured_output"]
    return data

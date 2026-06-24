"""The structured object every backend hands back: spec, raw output, result, and error."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.spec import RunSpec


@dataclass(frozen=True, slots=True)
class Error:
    """A failed run: a human-readable message plus the underlying exception.

    `ex` preserves the real exception so a caller can re-raise it unchanged —
    a `BackendCallError` for a nonzero exit or error envelope, a normalized
    timeout error, or a `pydantic.ValidationError` for a non-conforming model.

    Example:
        >>> Error(msg="codex exited 127: codex: not found", ex=RuntimeError("..."))
    """

    msg: str
    ex: Exception


@dataclass(frozen=True, slots=True)
class Result:
    """A successful run: the extracted final text and the optional validated model.

    `raw` is the extracted final text; `parsed` is the validated model, set only
    when the `RunSpec` carried a `response_model`.

    Example:
        >>> Result(raw="hello", parsed=None)
    """

    raw: str
    parsed: BaseModel | None = None


@dataclass(frozen=True, slots=True)
class Output:
    """The full unparsed transport stream, present on success and failure alike.

    `raw` is the complete output the provider wrote — the `claude
    --output-format json` event stream, the `codex -o` file, or plain stdout —
    before any extraction.

    Example:
        >>> Output(raw='[{"type": "system"}, {"type": "result", "result": "hi"}]')
    """

    raw: str


@dataclass(frozen=True, slots=True)
class Response:
    """A backend's fully-resolved outcome: the spec, the raw output, and exactly one of result/error.

    A backend runs the process, reads its output wherever the provider writes it,
    detects failure, and validates — then hands back one `Response`. `spec` and
    `output` are always present (the raw bytes live in `output.raw` even on
    failure); exactly one of `result`/`error` is set. Every failure — a nonzero
    exit, an error envelope, a timeout, or a validation error — routes through
    `error`, never a raise from `run`.

    Example:
        >>> Response(spec=spec, output=Output(raw="hi"), result=Result(raw="hi"))
    """

    spec: RunSpec
    output: Output
    result: Result | None = None
    error: Error | None = None

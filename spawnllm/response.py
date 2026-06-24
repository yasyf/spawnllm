"""The single object every backend hands back: error, text, and validated model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class Response:
    """A backend's fully-resolved outcome: the provider error, the text, and the model.

    A backend runs the process, reads its output wherever the provider writes it,
    detects failure, and validates — then hands back one `Response`. No
    subprocess plumbing escapes: `error` carries the provider failure (a nonzero
    exit with stderr, or an error envelope) and is `None` on success; `result`
    is the final text output; `parsed` is the validated model, set only when the
    `RunSpec` carried a `response_model`.

    Example:
        >>> Response(error=None, result="hello", parsed=None)
    """

    error: str | None
    result: str | None
    parsed: BaseModel | None = None

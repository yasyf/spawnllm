"""Spec-driven run entries with error-aware transient retry."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from spawnllm.backends.registry import select_backend
from spawnllm.response import DiscardedAttempt
from spawnllm.structured import backoff, is_transient

if TYPE_CHECKING:
    from spawnllm.backends.base import LlmBackend
    from spawnllm.response import Response
    from spawnllm.spec import RunSpec

__all__ = ["run", "run_sync"]


def _discarded(backend: LlmBackend, resp: Response, attempt: int) -> DiscardedAttempt:
    """Summarize a transient `resp` the retry loop is about to discard on attempt `attempt`.

    `accounting` is best-effort by contract, so a parse failure on a malformed
    envelope degrades to no cost/usage rather than aborting the retry loop.
    """
    try:
        cost_usd, usage = backend.accounting(resp.output.raw)
    except Exception:
        cost_usd, usage = None, None
    return DiscardedAttempt(
        attempt=attempt,
        error=type(resp.error.ex).__name__,
        cost_usd=cost_usd,
        usage=usage,
        raw_bytes=len(resp.output.raw.encode()),
    )


async def run(spec: RunSpec, *, backend: LlmBackend | None = None) -> Response:
    """Execute a `RunSpec` asynchronously, retrying transient failures with backoff.

    Each attempt runs through `backend.aexecute`; a transient `Response.error`
    (a 529, overloaded, rate-limit, or `5xx`) triggers a capped exponential
    backoff and another attempt, up to `spec.max_attempts`. The final `Response`
    — success or last failure — is returned without raising; every operational
    failure (nonzero exit, error envelope, timeout, validation) lives in
    `resp.error`.

    Args:
        spec: The configured run to execute.
        backend: The backend to run on; defaults to `select_backend()`.

    Returns:
        The resolved `Response` of the last attempt, carrying every retried-away
        attempt in `discarded_attempts`.
    """
    backend = backend or select_backend()
    discarded: list[DiscardedAttempt] = []
    for attempt in range(spec.max_attempts):
        resp = await backend.aexecute(spec)
        if not is_transient(resp) or attempt + 1 == spec.max_attempts:
            break
        discarded.append(_discarded(backend, resp, attempt))
        await asyncio.sleep(backoff(attempt))
    return replace(resp, discarded_attempts=tuple(discarded)) if discarded else resp


def run_sync(spec: RunSpec, *, backend: LlmBackend | None = None) -> Response:
    """Execute a `RunSpec` synchronously, retrying transient failures with backoff.

    The synchronous companion to `run`: each attempt runs through
    `backend.execute`, transient outcomes sleep and retry up to
    `spec.max_attempts`, and the last `Response` is returned without raising.

    Args:
        spec: The configured run to execute.
        backend: The backend to run on; defaults to `select_backend()`.

    Returns:
        The resolved `Response` of the last attempt, carrying every retried-away
        attempt in `discarded_attempts`.
    """
    backend = backend or select_backend()
    discarded: list[DiscardedAttempt] = []
    for attempt in range(spec.max_attempts):
        resp = backend.execute(spec)
        if not is_transient(resp) or attempt + 1 == spec.max_attempts:
            break
        discarded.append(_discarded(backend, resp, attempt))
        time.sleep(backoff(attempt))
    return replace(resp, discarded_attempts=tuple(discarded)) if discarded else resp

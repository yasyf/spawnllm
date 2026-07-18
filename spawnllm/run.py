"""Spec-driven run entries with core-decided transient retry."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from spawnllm import _core
from spawnllm.backends.registry import select_backend
from spawnllm.response import DiscardedAttempt

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


def _retry_decision(resp: Response, attempt: int, max_attempts: int) -> dict[str, Any]:
    return _core.dispatch(
        "retry_decision", {"attempt": attempt, "max_attempts": max_attempts, "error_msg": resp.error.msg}
    )


async def run(spec: RunSpec, *, backend: LlmBackend | None = None) -> Response:
    """Execute a `RunSpec` asynchronously, retrying transient failures with backoff.

    Each attempt runs through `backend.aexecute`; the core's `retry_decision` op
    classifies a `Response.error` (a 529, overloaded, rate-limit, or `5xx`) and
    returns the backoff to sleep before another attempt, up to `spec.max_attempts`.
    The final `Response` — success or last failure — is returned without raising;
    every operational failure (nonzero exit, error envelope, timeout, validation)
    lives in `resp.error`.

    Args:
        spec: The configured run to execute.
        backend: The backend to run on; defaults to `select_backend()`.

    Returns:
        The resolved `Response` of the last attempt, carrying every retried-away
        attempt in `discarded_attempts`.
    """
    backend = backend or select_backend()
    discarded: list[DiscardedAttempt] = []
    attempt = 0
    while True:
        resp = await backend.aexecute(spec)
        if resp.error is None or not (decision := _retry_decision(resp, attempt, spec.max_attempts))["retry"]:
            break
        discarded.append(_discarded(backend, resp, attempt))
        await asyncio.sleep(decision["sleep_s"])
        attempt += 1
    return replace(resp, discarded_attempts=tuple(discarded)) if discarded else resp


def run_sync(spec: RunSpec, *, backend: LlmBackend | None = None) -> Response:
    """Execute a `RunSpec` synchronously, retrying transient failures with backoff.

    The synchronous companion to `run`: each attempt runs through
    `backend.execute`, the core decides retry and backoff, and the last
    `Response` is returned without raising.

    Args:
        spec: The configured run to execute.
        backend: The backend to run on; defaults to `select_backend()`.

    Returns:
        The resolved `Response` of the last attempt, carrying every retried-away
        attempt in `discarded_attempts`.
    """
    backend = backend or select_backend()
    discarded: list[DiscardedAttempt] = []
    attempt = 0
    while True:
        resp = backend.execute(spec)
        if resp.error is None or not (decision := _retry_decision(resp, attempt, spec.max_attempts))["retry"]:
            break
        discarded.append(_discarded(backend, resp, attempt))
        time.sleep(decision["sleep_s"])
        attempt += 1
    return replace(resp, discarded_attempts=tuple(discarded)) if discarded else resp

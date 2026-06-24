"""Spec-driven run entries with error-aware transient retry."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from spawnllm.backends.registry import select_backend
from spawnllm.structured import backoff, is_transient

if TYPE_CHECKING:
    from spawnllm.backends.base import LlmBackend
    from spawnllm.response import Response
    from spawnllm.spec import RunSpec

__all__ = ["run", "run_sync"]


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
        The resolved `Response` of the last attempt.
    """
    backend = backend or select_backend()
    for attempt in range(spec.max_attempts):
        resp = await backend.aexecute(spec)
        if not is_transient(resp):
            break
        await asyncio.sleep(backoff(attempt))
    return resp


def run_sync(spec: RunSpec, *, backend: LlmBackend | None = None) -> Response:
    """Execute a `RunSpec` synchronously, retrying transient failures with backoff.

    The synchronous companion to `run`: each attempt runs through
    `backend.execute`, transient outcomes sleep and retry up to
    `spec.max_attempts`, and the last `Response` is returned without raising.

    Args:
        spec: The configured run to execute.
        backend: The backend to run on; defaults to `select_backend()`.

    Returns:
        The resolved `Response` of the last attempt.
    """
    backend = backend or select_backend()
    for attempt in range(spec.max_attempts):
        resp = backend.execute(spec)
        if not is_transient(resp):
            break
        time.sleep(backoff(attempt))
    return resp

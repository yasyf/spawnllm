"""Spec-driven run entries with envelope-aware transient retry."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from spawnllm.backends.registry import select_backend
from spawnllm.structured import backoff, is_transient

if TYPE_CHECKING:
    from spawnllm.backends.base import LlmBackend
    from spawnllm.proc import RunResult
    from spawnllm.spec import RunSpec

__all__ = ["run", "run_sync"]


async def run(spec: RunSpec, *, backend: LlmBackend | None = None) -> RunResult:
    """Execute a `RunSpec` asynchronously, retrying transient failures with backoff.

    Each attempt runs through `backend.aexecute`; a transient outcome (a 529,
    overloaded, rate-limit, or `5xx` envelope) triggers a capped exponential
    backoff and another attempt, up to `spec.max_attempts`. The final outcome —
    success or last transient failure — is returned without raising.

    Args:
        spec: The configured run to execute.
        backend: The backend to run on; defaults to `select_backend()`.

    Returns:
        The captured outcome of the last attempt.
    """
    backend = backend or select_backend()
    for attempt in range(spec.max_attempts):
        rr = await backend.aexecute(spec)
        if not is_transient(rr):
            return rr
        await asyncio.sleep(backoff(attempt))
    return rr


def run_sync(spec: RunSpec, *, backend: LlmBackend | None = None) -> RunResult:
    """Execute a `RunSpec` synchronously, retrying transient failures with backoff.

    The synchronous companion to `run`: each attempt runs through
    `backend.execute`, transient outcomes sleep and retry up to
    `spec.max_attempts`, and the last outcome is returned without raising.

    Args:
        spec: The configured run to execute.
        backend: The backend to run on; defaults to `select_backend()`.

    Returns:
        The captured outcome of the last attempt.
    """
    backend = backend or select_backend()
    for attempt in range(spec.max_attempts):
        rr = backend.execute(spec)
        if not is_transient(rr):
            return rr
        time.sleep(backoff(attempt))
    return rr

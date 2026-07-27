"""Prompt-ergonomic text LLM call over a backend, with the abstract model-tier map."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spawnllm.backends.registry import select_backend
from spawnllm.run import run, run_sync
from spawnllm.spec import RunSpec

if TYPE_CHECKING:
    from spawnllm.backends.base import LlmBackend
    from spawnllm.types import TModel, TSpecialty

__all__ = ["call", "call_sync"]

DEFAULT_MODEL: TModel = "small"
"""Abstract tier `call`/`call_sync` resolve a backend against when the caller names none."""


async def call(
    prompt: str,
    *,
    backend: LlmBackend | None = None,
    specialty: TSpecialty | None = None,
    model: TModel | str = DEFAULT_MODEL,
    agent: bool = False,
    cwd: str | None = None,
    api_auth: bool = False,
    timeout: int = 180,
) -> str:
    """Run one LLM call asynchronously and return its text response.

    Resolves a backend, maps the abstract model tier to the provider's literal
    model id, and executes through `run` (transient retry included). The backend
    fully resolves the outcome; a provider error raises `BackendCallError`.

    Args:
        prompt: The user prompt, delivered to the backend over stdin.
        backend: The `LlmBackend` to invoke; when `None`, auto-selects the first
            ready backend via the priority chain, optionally scoped by `specialty`.
        specialty: Specialty used to scope auto-selection when `backend` is
            `None`; ignored when `backend` is given.
        model: Abstract model tier (`small`/`medium`/`large`), or a concrete
            provider model id passed through unchanged.
        agent: Whether the call may use tools / agent capabilities.
        cwd: Working directory for the backend process; `None` inherits the caller's.
        api_auth: Whether to inherit provider API-key environment variables.
        timeout: Seconds to wait before the backend process is killed.

    Returns:
        The text response.

    Raises:
        BackendCallError: When the backend returns a provider error.
    """
    backend = backend or select_backend(specialty=specialty, model=model)
    resp = await run(
        RunSpec(
            prompt=prompt,
            model=backend.resolve_model(model),
            agent=agent,
            cwd=cwd,
            api_auth=api_auth,
            timeout=timeout,
        ),
        backend=backend,
    )
    if resp.error is not None:
        raise resp.error.ex
    return resp.result.raw


def call_sync(
    prompt: str,
    *,
    backend: LlmBackend | None = None,
    specialty: TSpecialty | None = None,
    model: TModel | str = DEFAULT_MODEL,
    agent: bool = False,
    cwd: str | None = None,
    api_auth: bool = False,
    timeout: int = 180,
) -> str:
    """Run one LLM call synchronously and return its text response.

    The synchronous companion to `call`: resolves a backend, maps the abstract
    model tier, executes through `run_sync` (transient retry included), and
    returns the text. A provider error raises `BackendCallError`.

    Args:
        prompt: The user prompt, delivered to the backend over stdin.
        backend: The `LlmBackend` to invoke; when `None`, auto-selects the first
            ready backend via the priority chain, optionally scoped by `specialty`.
        specialty: Specialty used to scope auto-selection when `backend` is
            `None`; ignored when `backend` is given.
        model: Abstract model tier (`small`/`medium`/`large`), or a concrete
            provider model id passed through unchanged.
        agent: Whether the call may use tools / agent capabilities.
        cwd: Working directory for the backend process; `None` inherits the caller's.
        api_auth: Whether to inherit provider API-key environment variables.
        timeout: Seconds to wait before the backend process is killed.

    Returns:
        The text response.

    Raises:
        BackendCallError: When the backend returns a provider error.
    """
    backend = backend or select_backend(specialty=specialty, model=model)
    resp = run_sync(
        RunSpec(
            prompt=prompt,
            model=backend.resolve_model(model),
            agent=agent,
            cwd=cwd,
            api_auth=api_auth,
            timeout=timeout,
        ),
        backend=backend,
    )
    if resp.error is not None:
        raise resp.error.ex
    return resp.result.raw

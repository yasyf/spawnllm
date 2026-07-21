"""Prompt-ergonomic structured LLM call: prompt plus model in, validated model out."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from spawnllm.backends.registry import select_backend
from spawnllm.run import run, run_sync
from spawnllm.spec import RunSpec

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.backends.base import LlmBackend
    from spawnllm.types import TModel, TSpecialty

__all__ = ["extract", "extract_sync"]


async def extract[T: BaseModel](
    prompt: str,
    response_model: type[T],
    *,
    backend: LlmBackend | None = None,
    specialty: TSpecialty | None = None,
    model: TModel = "small",
    agent: bool = False,
    cwd: str | None = None,
    api_auth: bool = False,
    timeout: int = 180,
) -> T:
    """Run one LLM call asynchronously and return a validated `response_model`.

    Resolves a backend, maps the abstract model tier, and executes through `run`.
    The backend runs, reads, and validates; a provider error raises
    `BackendCallError`, and a `pydantic.ValidationError` from a non-conforming
    model propagates out of this call.

    Args:
        prompt: The user prompt, delivered to the backend over stdin.
        response_model: The Pydantic model the structured output is validated against.
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
        The validated `response_model` instance.

    Raises:
        BackendCallError: When the backend returns a provider error.
        pydantic.ValidationError: When the model's output fails validation.
    """
    backend = backend or select_backend(specialty=specialty)
    spec = RunSpec(
        prompt=prompt,
        model=backend.resolve_model(model),
        response_model=response_model,
        agent=agent,
        cwd=cwd,
        api_auth=api_auth,
        timeout=timeout,
    )
    resp = await run(spec, backend=backend)
    if resp.error is not None:
        raise resp.error.ex
    return cast(T, resp.result.parsed)


def extract_sync[T: BaseModel](
    prompt: str,
    response_model: type[T],
    *,
    backend: LlmBackend | None = None,
    specialty: TSpecialty | None = None,
    model: TModel = "small",
    agent: bool = False,
    cwd: str | None = None,
    api_auth: bool = False,
    timeout: int = 180,
) -> T:
    """Run one LLM call synchronously and return a validated `response_model`.

    The synchronous companion to `extract`: resolves a backend, maps the model
    tier, executes through `run_sync`, and returns the validated model. A
    provider error raises `BackendCallError`; a `pydantic.ValidationError`
    propagates.

    Args:
        prompt: The user prompt, delivered to the backend over stdin.
        response_model: The Pydantic model the structured output is validated against.
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
        The validated `response_model` instance.

    Raises:
        BackendCallError: When the backend returns a provider error.
        pydantic.ValidationError: When the model's output fails validation.
    """
    backend = backend or select_backend(specialty=specialty)
    spec = RunSpec(
        prompt=prompt,
        model=backend.resolve_model(model),
        response_model=response_model,
        agent=agent,
        cwd=cwd,
        api_auth=api_auth,
        timeout=timeout,
    )
    resp = run_sync(spec, backend=backend)
    if resp.error is not None:
        raise resp.error.ex
    return cast(T, resp.result.parsed)

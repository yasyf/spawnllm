"""Prompt-ergonomic LLM call over a backend, with the abstract model-tier map."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spawnllm.backends.registry import select_backend
from spawnllm.run import run, run_sync
from spawnllm.spec import RunSpec

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.backends.base import LlmBackend
    from spawnllm.types import TModel, TSpecialty

__all__ = ["call", "call_sync"]


async def call(
    prompt: str,
    *,
    backend: LlmBackend | None = None,
    specialty: TSpecialty | None = None,
    model: TModel = "small",
    agent: bool = False,
    response_model: type[BaseModel] | None = None,
    cwd: str | None = None,
    timeout: int = 180,
) -> str | BaseModel:
    """Run one LLM call asynchronously and parse its response.

    The ergonomic companion to `run`: it resolves a backend, maps the abstract
    model tier to the provider's literal model id, serializes `response_model`
    into a JSON schema, executes through `run` (transient retry included), and
    parses the raw stdout into text or a validated model.

    Args:
        prompt: The user prompt, delivered to the backend over stdin.
        backend: The `LlmBackend` to invoke; when `None`, auto-selects the first
            ready backend via the priority chain, optionally scoped by `specialty`.
        specialty: Specialty used to scope auto-selection when `backend` is
            `None`; ignored when `backend` is given.
        model: Abstract model tier (`small`/`medium`/`large`).
        agent: Whether the call may use tools / agent capabilities.
        response_model: Pydantic model for structured output, or `None` for text.
        cwd: Working directory for the backend process; `None` inherits the caller's.
        timeout: Seconds to wait before the backend process is killed.

    Returns:
        The raw text response, or a validated `response_model` instance.
    """
    backend = backend or select_backend(specialty=specialty)
    schema = backend.schema_for(response_model) if response_model is not None else None
    spec = RunSpec(prompt=prompt, model=backend.models[model], schema=schema, agent=agent, cwd=cwd, timeout=timeout)
    rr = await run(spec, backend=backend)
    return backend.parse_response(rr.stdout, response_model)


def call_sync(
    prompt: str,
    *,
    backend: LlmBackend | None = None,
    specialty: TSpecialty | None = None,
    model: TModel = "small",
    agent: bool = False,
    response_model: type[BaseModel] | None = None,
    cwd: str | None = None,
    timeout: int = 180,
) -> str | BaseModel:
    """Run one LLM call synchronously and parse its response.

    The synchronous companion to `call`: it resolves a backend, maps the abstract
    model tier to the provider's literal model id, serializes `response_model`
    into a JSON schema, executes through `run_sync` (transient retry included),
    and parses the raw stdout into text or a validated model.

    Args:
        prompt: The user prompt, delivered to the backend over stdin.
        backend: The `LlmBackend` to invoke; when `None`, auto-selects the first
            ready backend via the priority chain, optionally scoped by `specialty`.
        specialty: Specialty used to scope auto-selection when `backend` is
            `None`; ignored when `backend` is given.
        model: Abstract model tier (`small`/`medium`/`large`).
        agent: Whether the call may use tools / agent capabilities.
        response_model: Pydantic model for structured output, or `None` for text.
        cwd: Working directory for the backend process; `None` inherits the caller's.
        timeout: Seconds to wait before the backend process is killed.

    Returns:
        The raw text response, or a validated `response_model` instance.
    """
    backend = backend or select_backend(specialty=specialty)
    schema = backend.schema_for(response_model) if response_model is not None else None
    spec = RunSpec(prompt=prompt, model=backend.models[model], schema=schema, agent=agent, cwd=cwd, timeout=timeout)
    rr = run_sync(spec, backend=backend)
    return backend.parse_response(rr.stdout, response_model)

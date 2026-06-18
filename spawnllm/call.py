"""One-shot synchronous LLM call over a CLI backend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from spawnllm.backends.registry import select_backend
from spawnllm.proc import run_cli
from spawnllm.structured import resolve_schema_path

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.backends.base import LlmBackend
    from spawnllm.types import TModel, TSpecialty


def call(
    prompt: str,
    *,
    backend: LlmBackend | None = None,
    specialty: TSpecialty | None = None,
    model: TModel = "small",
    agent: bool = False,
    response_model: type[BaseModel] | None = None,
) -> str | BaseModel:
    """Run one CLI-backed LLM call and parse its response.

    Args:
        prompt: The user prompt, delivered to the backend over stdin.
        backend: The `LlmBackend` to invoke; when `None`, auto-selects the first
            ready backend via the priority chain, optionally scoped by `specialty`.
        specialty: Specialty used to scope auto-selection when `backend` is
            `None`; ignored when `backend` is given.
        model: Abstract model tier (`small`/`medium`/`large`).
        agent: Whether the call may use tools / agent capabilities.
        response_model: Pydantic model for structured output, or `None` for text.

    Returns:
        The raw text response, or a validated `response_model` instance.
    """
    backend = backend or select_backend(specialty=specialty)
    schema = backend.schema_for(response_model) if response_model is not None else None
    schema_path = resolve_schema_path(backend, schema)
    argv, stdin = backend.invocation(prompt, model=backend.models[model], schema_path=schema_path, agent=agent)
    raw = run_cli(argv, input=stdin, env=os.environ | backend.env(), timeout=180)
    return backend.parse_response(raw, response_model)

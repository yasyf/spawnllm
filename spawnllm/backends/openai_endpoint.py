"""HTTP backend for any OpenAI-compatible `/chat/completions` endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import httpx

from spawnllm.backends.base import BackendReady, LlmBackend

if TYPE_CHECKING:
    from spawnllm.backends.base import BackendStatus
    from spawnllm.response import Response
    from spawnllm.spec import RunSpec
    from spawnllm.types import ProviderName, TModel


class OpenAiEndpointBackend(LlmBackend):
    """`LlmBackend` that POSTs to an OpenAI-compatible `/chat/completions` endpoint.

    Unlike the CLI backends this drives a raw `httpx` request rather than a
    subprocess, and it is never auto-selected; the consumer constructs it
    explicitly with a `base_url` and a literal `model`. The core plans the HTTP
    request (url, headers, body — a strict `json_schema` `response_format` when a
    `response_model` is set) and resolves the response; every abstract tier maps to
    the one pinned `model`, and `RunSpec.model` is ignored at request time.

    Args:
        base_url: Root URL of the server; `/chat/completions` is appended.
        model: The literal model id sent in every request body.
        api_key: Bearer token for the `Authorization` header; defaults to
            `"local"` for self-hosted servers that ignore it.
        transport: Async transport injected into the `httpx.AsyncClient` used by
            `aexecute` — e.g. a record/replay caching transport; `None` uses
            httpx's default transport. The synchronous `execute` path always uses
            the default transport.

    Example:
        >>> backend = OpenAiEndpointBackend("http://localhost:8000/v1", "qwen3")
        >>> backend.execute(RunSpec(prompt="ping", model="qwen3"))
    """

    provider: ClassVar[ProviderName] = "openai_endpoint"
    schema_dialect: ClassVar[str | None] = "openai"

    def __init__(
        self, base_url: str, model: str, *, api_key: str = "local", transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.transport = transport
        self.models: dict[TModel, str] = {"small": model, "medium": model, "large": model}

    def openai_section(self) -> dict[str, Any]:
        """Return the `openai_endpoint` wire section the core turns into the HTTP request."""
        return {"api_key": self.api_key, "base_url": self.base_url, "model": self.model}

    def resolve(self, resp: httpx.Response, spec: RunSpec) -> Response:
        return self.to_response(
            resp.text,
            returncode=0 if resp.is_success else resp.status_code,
            stderr="" if resp.is_success else resp.text,
            spec=spec,
        )

    async def aexecute(self, spec: RunSpec) -> Response:
        plan = self.core_plan(spec)
        async with httpx.AsyncClient(timeout=spec.timeout, transport=self.transport) as client:
            resp = await client.post(plan["url"], headers=plan["headers"], json=plan["body"])
        return self.resolve(resp, spec)

    def execute(self, spec: RunSpec) -> Response:
        plan = self.core_plan(spec)
        with httpx.Client(timeout=spec.timeout) as client:
            resp = client.post(plan["url"], headers=plan["headers"], json=plan["body"])
        return self.resolve(resp, spec)

    def env(self, _spec: RunSpec) -> dict[str, str]:
        """Return no extra environment variables; the endpoint is reached over HTTP with nothing to isolate."""
        return {}

    def is_authenticated(self, *, timeout: int) -> bool:
        """Report `True`; credentials travel inline as the `Authorization` bearer token on every request."""
        return True

    def check_status(self, *, timeout: int = 10) -> BackendStatus:
        """Report `BackendReady`; the endpoint is reached per-request and carries its own auth."""
        return BackendReady(binary="openai_endpoint")

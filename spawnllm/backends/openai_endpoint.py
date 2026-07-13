"""HTTP backend for any OpenAI-compatible `/chat/completions` endpoint."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

import httpx

from spawnllm.backends.base import BackendReady, LlmBackend
from spawnllm.structured import extract_json_block

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.backends.base import BackendStatus
    from spawnllm.response import Response
    from spawnllm.spec import RunSpec
    from spawnllm.types import ProviderName, TModel


class OpenAiEndpointBackend(LlmBackend):
    """`LlmBackend` that POSTs to an OpenAI-compatible `/chat/completions` endpoint.

    Unlike the CLI backends this drives a raw `httpx` request rather than a
    subprocess, and it is never auto-selected; the consumer constructs it
    explicitly with a `base_url` and a literal `model`. Every abstract tier maps
    to that one pinned `model`, and `RunSpec.model` is ignored at request time —
    the endpoint always serves the constructor's `model`. Structured output rides
    on `response_format` with a strict `json_schema`; text output reads
    `choices[0].message.content`.

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

    def __init__(
        self, base_url: str, model: str, *, api_key: str = "local", transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.transport = transport
        self.models: dict[TModel, str] = {"small": model, "medium": model, "large": model}

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def payload(self, spec: RunSpec) -> dict[str, object]:
        base: dict[str, object] = {"model": self.model, "messages": [{"role": "user", "content": spec.prompt}]}
        if (schema := self.schema_arg(spec)) is None:
            return base
        json_schema = {"name": "response", "strict": True, "schema": json.loads(schema)}
        return base | {"response_format": {"type": "json_schema", "json_schema": json_schema}}

    def resolve(self, resp: httpx.Response, spec: RunSpec) -> Response:
        return self.to_response(
            resp.text,
            returncode=0 if resp.is_success else resp.status_code,
            stderr="" if resp.is_success else resp.text,
            spec=spec,
        )

    async def aexecute(self, spec: RunSpec) -> Response:
        async with httpx.AsyncClient(timeout=spec.timeout, transport=self.transport) as client:
            resp = await client.post(self.url, headers=self.headers(), json=self.payload(spec))
        return self.resolve(resp, spec)

    def execute(self, spec: RunSpec) -> Response:
        with httpx.Client(timeout=spec.timeout) as client:
            resp = client.post(self.url, headers=self.headers(), json=self.payload(spec))
        return self.resolve(resp, spec)

    def schema_for(self, model: type[BaseModel]) -> str:
        """Serialize a Pydantic model into an OpenAI strict JSON schema for `response_format`.

        Uses the OpenAI SDK's `to_strict_json_schema`, which recursively sets
        `additionalProperties: false` and forces every property into `required`
        across `$defs`, `anyOf`, and array items — the form a `json_schema`
        `response_format` requires.

        Args:
            model: The Pydantic model describing the structured output.

        Returns:
            A strict JSON-schema string embedded in the request body.
        """
        from openai.lib._pydantic import to_strict_json_schema

        return json.dumps(to_strict_json_schema(model))

    def result_text(self, raw: str) -> str:
        """Return `choices[0].message.content` from the chat-completion response body."""
        return json.loads(raw)["choices"][0]["message"]["content"]

    def result_value(self, raw: str) -> object:
        """Return the JSON value parsed from the message content, tolerating fences or surrounding prose."""
        return json.loads(extract_json_block(self.result_text(raw)))

    def envelope_error(self, raw: str) -> str | None:
        """Return the message from an OpenAI `{"error": {...}}` body carried on a 2xx response, else `None`."""
        match json.loads(raw):
            case {"error": {"message": str(msg)}}:
                return msg
            case {"error": str(msg)}:
                return msg
            case _:
                return None

    def env(self, _spec: RunSpec) -> dict[str, str]:
        """Return no extra environment variables; the endpoint is reached over HTTP with nothing to isolate."""
        return {}

    def is_authenticated(self, *, timeout: int) -> bool:
        """Report `True`; credentials travel inline as the `Authorization` bearer token on every request."""
        return True

    def check_status(self, *, timeout: int = 10) -> BackendStatus:
        """Report `BackendReady`; the endpoint is reached per-request and carries its own auth."""
        return BackendReady(binary="openai_endpoint")

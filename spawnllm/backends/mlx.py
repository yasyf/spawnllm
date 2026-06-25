"""In-process MLX backend adapting `MlxEngine` to the `LlmBackend` contract."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import BackendReady, LlmBackend
from spawnllm.structured import structured_value

if TYPE_CHECKING:
    from spawnllm.backends.base import BackendStatus
    from spawnllm.mlx import MlxEngine
    from spawnllm.response import Response
    from spawnllm.spec import RunSpec
    from spawnllm.types import ProviderName, TModel


class MlxBackend(LlmBackend):
    """In-process backend that runs a prompt through a local `MlxEngine`.

    Unlike the CLI backends this is never auto-selected; the consumer constructs
    it explicitly with a loaded `MlxEngine`. `RunSpec.model` is ignored — the
    engine is already bound to a fused model — and every provider config and CLI
    flag is irrelevant, so `models` is the empty identity mapping.

    Example:
        >>> backend = MlxBackend(engine=engine, max_tokens=512)
        >>> backend.execute(RunSpec(prompt="ping", model="local"))
    """

    models: ClassVar[dict[TModel, str]] = {}
    provider: ClassVar[ProviderName] = "mlx"

    def __init__(self, engine: MlxEngine, *, max_tokens: int = 512) -> None:
        self.engine = engine
        self.max_tokens = max_tokens

    async def aexecute(self, spec: RunSpec) -> Response:
        await self.engine.ensure_loaded()
        texts = await self.engine.generate(
            [[{"role": "user", "content": spec.prompt}]],
            lambda _: None,
            max_tokens=self.max_tokens,
        )
        return self.to_response(texts[0], returncode=0, stderr="", spec=spec)

    def execute(self, spec: RunSpec) -> Response:
        return asyncio.run(self.aexecute(spec))

    def result_value(self, raw: str) -> object:
        """Return the `structured_output` from a stream-json result event, else `raw` parsed as JSON."""
        return structured_value(raw)

    def env(self, _spec: RunSpec) -> dict[str, str]:
        """Return no extra environment variables; MLX runs in-process with nothing to isolate."""
        return {}

    def is_authenticated(self, *, timeout: int) -> bool:
        """Report `True`; the engine is local and needs no credentials."""
        return True

    def check_status(self, *, timeout: int = 10) -> BackendStatus:
        """Report `BackendReady`; the engine is local and always available once loaded."""
        return BackendReady(binary="mlx")

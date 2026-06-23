"""In-process MLX backend adapting `MlxEngine` to the `LlmBackend` contract."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import BackendReady, LlmBackend
from spawnllm.proc import RunResult
from spawnllm.structured import parse_structured_output

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.backends.base import BackendStatus
    from spawnllm.mlx import MlxEngine
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

    async def aexecute(self, spec: RunSpec) -> RunResult:
        await self.engine.ensure_loaded()
        texts = await self.engine.generate(
            [[{"role": "user", "content": spec.prompt}]],
            lambda _: None,
            max_tokens=self.max_tokens,
        )
        return RunResult(texts[0], "", 0)

    def execute(self, spec: RunSpec) -> RunResult:
        return asyncio.run(self.aexecute(spec))

    def parse_response(self, raw: str, response_model: type[BaseModel] | None) -> str | BaseModel:
        return parse_structured_output(raw, response_model)

    def env(self) -> dict[str, str]:
        """Return no extra environment variables; MLX runs in-process."""
        return {}

    def is_authenticated(self, *, timeout: int) -> bool:
        """Report `True`; the engine is local and needs no credentials."""
        return True

    def check_status(self, *, timeout: int = 10) -> BackendStatus:
        """Report `BackendReady`; the engine is local and always available once loaded."""
        return BackendReady(binary="mlx")

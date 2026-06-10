"""LlmBackend for the OpenAI ``codex`` CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import LlmBackend

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.types import TModel


class CodexCliBackend(LlmBackend):
    """:class:`LlmBackend` for the OpenAI ``codex`` CLI."""

    models: ClassVar[dict[TModel, str]] = {
        "small": "gpt-5.3-codex-spark",
        "medium": "gpt-5.4-mini",
        "large": "gpt-5.5",
    }

    def build_command(self, model: str, schema_path: str | None, agent: bool) -> list[str]:
        return [
            "codex",
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--model",
            model,
            *([] if agent else ["-c", "features.codex_hooks=false", "-c", "features.mcp_servers=false"]),
            *(["--output-schema", schema_path] if schema_path else []),
        ]

    def parse_response(self, raw: str, response_model: type[BaseModel] | None) -> str | BaseModel:
        return raw if not response_model else response_model.model_validate_json(raw)

    def env(self) -> dict[str, str]:
        return {}

"""LlmBackend for the OpenAI `codex` CLI."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import LlmBackend

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.types import TModel


class CodexCliBackend(LlmBackend):
    """`LlmBackend` for the OpenAI `codex` CLI.

    Invokes `codex exec` with an ephemeral session and a read-only sandbox.

    Attributes:
        models: Mapping from abstract model size to an OpenAI model name.
    """

    models: ClassVar[dict[TModel, str]] = {
        "small": "gpt-5.3-codex-spark",
        "medium": "gpt-5.4-mini",
        "large": "gpt-5.5",
    }
    binary: ClassVar[str] = "codex"
    install_hint: ClassVar[str] = "npm install -g @openai/codex"

    def build_command(self, model: str, schema_path: str | None, agent: bool) -> list[str]:
        """Build the `codex exec` argv for one stdin-prompted invocation.

        Every invocation runs an ephemeral session in a read-only sandbox.
        Non-agent invocations disable Codex hooks and MCP servers. A schema
        path adds `--output-schema`.

        Args:
            model: OpenAI model name, e.g. `gpt-5.5`.
            schema_path: Path to a JSON schema file passed to
                `--output-schema`, or `None`.
            agent: Whether the invocation may use tools / agent capabilities.

        Returns:
            The argv list to execute.
        """
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
        """Parse `codex` stdout into text or a validated model.

        Args:
            raw: Raw stdout from the `codex` CLI.
            response_model: Model to validate against, or `None` for raw text.

        Returns:
            `raw` when `response_model` is `None`; otherwise `raw` validated as JSON against `response_model`.
        """
        return raw if not response_model else response_model.model_validate_json(raw)

    def env(self) -> dict[str, str]:
        """Return no extra environment variables; the `codex` CLI runs with the inherited environment."""
        return {}

    def is_authenticated(self, *, timeout: int) -> bool:
        """Report whether `codex login status` exits cleanly, i.e. the CLI is logged in.

        Args:
            timeout: Seconds to wait for `codex login status`.

        Returns:
            `True` when `codex login status` exits 0.
        """
        return (
            subprocess.run(
                ["codex", "login", "status"], capture_output=True, text=True, timeout=timeout, check=False
            ).returncode
            == 0
        )

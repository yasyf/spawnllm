"""Spec-driven run configuration shared across every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.types import ProviderName


@dataclass(frozen=True, slots=True)
class ClaudeConfig:
    """Claude CLI flag passthrough applied only by the Claude backend.

    Fields map one-to-one onto `claude` flags: the agent/system-prompt knobs
    (`permission_mode`, `mcp_config`, `strict_mcp`, `append_system_prompt`,
    `system_prompt`, `settings`, `disallowed_tools`) and orthogonal extras
    (`max_turns`, `max_budget_usd`, `tools`, `disable_slash_commands`,
    `output_format`, `verbose`).

    Example:
        >>> ClaudeConfig(permission_mode="bypassPermissions", strict_mcp=True)
    """

    permission_mode: str | None = None
    mcp_config: str | None = None
    strict_mcp: bool = False
    append_system_prompt: str | None = None
    system_prompt: str | None = None
    settings: str | None = None
    disallowed_tools: tuple[str, ...] = ()
    max_turns: int | None = None
    max_budget_usd: float | None = None
    tools: str | None = None
    disable_slash_commands: bool = False
    output_format: str | None = None
    verbose: bool = False


@dataclass(frozen=True, slots=True)
class CodexConfig:
    """Codex CLI knobs applied only by the Codex backend.

    Example:
        >>> CodexConfig(sandbox="read-only", enable_mcp=True)
    """

    sandbox: str | None = None
    enable_hooks: bool = False
    enable_mcp: bool = False


@dataclass(frozen=True, slots=True)
class GeminiConfig:
    """Gemini CLI knobs applied only by the Gemini and Antigravity backends.

    Example:
        >>> GeminiConfig(approval_mode="auto", extensions=("search",))
    """

    approval_mode: str | None = None
    extensions: tuple[str, ...] | None = None


type ProviderConfig = ClaudeConfig | CodexConfig | GeminiConfig


@dataclass(frozen=True, slots=True)
class RunSpec:
    """A single configured run, translated per backend at execution time.

    Common fields are interpreted by every backend; `provider_configs` carries
    optional per-provider flag passthrough that only the matching backend reads.
    `model` is a literal provider model id (`opus`, `sonnet`, …) passed straight
    through with no tier mapping. `isolated` (default `True`) runs the backend
    against a fresh, host-free config home so a spawned CLI ignores ambient
    settings, MCP servers, and hooks. Structured output comes from either a
    `response_model` (validated to a model) or a raw `schema` (a JSON-Schema dict
    or pre-serialized string, passed to the provider verbatim, with nothing to
    validate); setting both raises `ValueError`.

    Example:
        >>> RunSpec(prompt="ping", model="opus")
    """

    prompt: str
    model: str
    response_model: type[BaseModel] | None = None
    schema: dict[str, object] | str | None = None
    agent: bool = False
    isolated: bool = True
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout: int = 180
    max_attempts: int = 5
    provider_configs: dict[ProviderName, ProviderConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.response_model is not None and self.schema is not None:
            raise ValueError("RunSpec accepts either response_model or schema, not both")

    def config_for[T: ProviderConfig](self, kind: type[T]) -> T | None:
        """Return the first provider config that is an instance of `kind`, or None."""
        return next((c for c in self.provider_configs.values() if isinstance(c, kind)), None)

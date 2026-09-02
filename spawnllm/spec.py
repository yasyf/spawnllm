"""Spec-driven run configuration shared across every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.types import ProviderName, TSettingSource


@dataclass(frozen=True, slots=True)
class AppleConfig:
    """Apple Foundation Models knobs applied only by the Apple backend.

    `use_case` and `guardrails` select the `SystemLanguageModelUseCase` and
    `SystemLanguageModelGuardrails` the `spawnllm-apple` sidecar builds its
    session with. The sampling knobs are flat rather than a nested mode because
    Apple exposes `SamplingMode` as a factory (`greedy()` / `random()`) that no
    serializable value can carry: `sampling` picks the factory and `sampling_top`,
    `sampling_probability_threshold`, and `sampling_seed` are the arguments
    `random` takes. `None` everywhere leaves the framework default.

    Example:
        >>> AppleConfig(use_case="content_tagging", sampling="random", sampling_top=20)

    Raises:
        ValueError: When a `random`-only argument is set without `sampling="random"`,
            a combination the framework would silently discard; when `sampling_top`
            and `sampling_probability_threshold` are set together, a pair the
            framework rejects; or when `sampling_seed` is negative, which the
            framework's `UInt64` cannot carry.
    """

    use_case: Literal["general", "content_tagging"] = "general"
    guardrails: Literal["default", "permissive_content_transformations"] = "default"
    instructions: str | None = None
    temperature: float | None = None
    maximum_response_tokens: int | None = None
    sampling: Literal["greedy", "random"] | None = None
    sampling_top: int | None = None
    sampling_probability_threshold: float | None = None
    sampling_seed: int | None = None

    def __post_init__(self) -> None:
        if self.sampling != "random" and any(
            knob is not None for knob in (self.sampling_top, self.sampling_probability_threshold, self.sampling_seed)
        ):
            raise ValueError(
                "AppleConfig sampling_top, sampling_probability_threshold, and sampling_seed require sampling='random'"
            )
        if self.sampling_top is not None and self.sampling_probability_threshold is not None:
            raise ValueError("AppleConfig accepts either sampling_top or sampling_probability_threshold, not both")
        if self.sampling_seed is not None and self.sampling_seed < 0:
            raise ValueError("AppleConfig sampling_seed is unsigned; the framework takes a UInt64")


@dataclass(frozen=True, slots=True)
class ClaudeConfig:
    """Claude CLI flag passthrough applied only by the Claude backend.

    Fields map one-to-one onto `claude` flags: the agent/system-prompt knobs
    (`permission_mode`, `mcp_config`, `strict_mcp`, `append_system_prompt`,
    `system_prompt`, `settings`, `disallowed_tools`) and orthogonal extras
    (`max_turns`, `max_budget_usd`, `tools`, `disable_slash_commands`,
    `output_format`, `verbose`). `tools` selects the built-in toolset: `None`
    keeps the CLI default, `()` disables every built-in tool, and names
    restrict the session to those tools. `setting_sources` selects
    `--setting-sources`: `None` drops the host's user settings (`("project",)`,
    or nothing when `isolated`), so a run never inherits the caller's hooks.

    Example:
        >>> ClaudeConfig(permission_mode="bypassPermissions", strict_mcp=True)
        >>> ClaudeConfig(tools=())  # bare session: no built-in tools
        >>> ClaudeConfig(setting_sources=("user", "project", "local"))  # CLI default
    """

    permission_mode: str | None = None
    mcp_config: str | None = None
    strict_mcp: bool = False
    setting_sources: tuple[TSettingSource, ...] | None = None
    append_system_prompt: str | None = None
    system_prompt: str | None = None
    settings: str | None = None
    disallowed_tools: tuple[str, ...] = ()
    max_turns: int | None = None
    max_budget_usd: float | None = None
    tools: tuple[str, ...] | None = None
    disable_slash_commands: bool = False
    output_format: str | None = None
    verbose: bool = False


@dataclass(frozen=True, slots=True)
class CodexConfig:
    """Codex CLI knobs applied only by the Codex backend.

    `service_tier` (default `"fast"`) is emitted as `-c service_tier=<value>` on
    every invocation, isolated or not: isolated runs pass `--ignore-user-config`,
    which drops a `service_tier` pin in the user's `~/.codex/config.toml`, and the
    standard tier turns long prompts into multi-minute runs. Set it to `None` to
    drop the flag; an isolated run still passes `--ignore-user-config`, so a
    user-level tier pin applies only with `isolated=False`. `developer_instructions` injects
    the system-prompt layer via `-c developer_instructions=<value>`, serialized as
    a TOML string so any text — multi-line, or TOML-ambiguous words like `true` —
    arrives as a string.

    Example:
        >>> CodexConfig(sandbox="read-only", enable_mcp=True)
    """

    sandbox: str | None = None
    enable_hooks: bool = False
    enable_mcp: bool = False
    service_tier: str | None = "fast"
    developer_instructions: str | None = None


@dataclass(frozen=True, slots=True)
class GeminiConfig:
    """Gemini CLI knobs applied only by the Gemini and Antigravity backends.

    Example:
        >>> GeminiConfig(approval_mode="auto", extensions=("search",))
    """

    approval_mode: str | None = None
    extensions: tuple[str, ...] | None = None


type ProviderConfig = AppleConfig | ClaudeConfig | CodexConfig | GeminiConfig


@dataclass(frozen=True, slots=True)
class RunSpec:
    """A single configured run, translated per backend at execution time.

    Common fields are interpreted by every backend; `provider_configs` carries
    optional per-provider flag passthrough that only the matching backend reads.
    `model` is a literal provider model id (`opus`, `sonnet`, …) passed straight
    through with no tier mapping. `isolated` (default `True`) runs the backend
    against a fresh, host-free config home so a spawned CLI ignores ambient
    settings, MCP servers, and hooks. `api_auth` (default `False`) strips the
    provider's API-key environment variables from the child process so the CLI
    bills the logged-in subscription; `True` inherits the environment untouched.
    Structured output comes from either a `response_model` (validated to a model)
    or a raw `schema` (a JSON-Schema dict or pre-serialized string, passed to the
    provider verbatim, with nothing to validate); setting both raises `ValueError`.

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
    api_auth: bool = False
    timeout: int = 180
    max_attempts: int = 5
    provider_configs: dict[ProviderName, ProviderConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.response_model is not None and self.schema is not None:
            raise ValueError("RunSpec accepts either response_model or schema, not both")

    def config_for[T: ProviderConfig](self, kind: type[T]) -> T | None:
        """Return the first provider config that is an instance of `kind`, or None."""
        return next((c for c in self.provider_configs.values() if isinstance(c, kind)), None)

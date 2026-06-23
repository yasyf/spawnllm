"""CliBackend for the Anthropic `claude` CLI, plus install/auth status checks."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import CliBackend
from spawnllm.spec import ClaudeConfig
from spawnllm.structured import parse_structured_output

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.spec import RunSpec
    from spawnllm.types import ProviderName, TModel

CLAUDE_MODELS: dict[TModel, str] = {"small": "haiku", "medium": "sonnet", "large": "opus"}


class ClaudeCliBackend(CliBackend):
    """`CliBackend` for the Anthropic `claude` CLI.

    `build_command` translates a `RunSpec` into a `claude -p` argv with the prompt
    delivered over stdin. The permission and system-prompt flags resolve through
    three mutually exclusive branches: explicit `ClaudeConfig` agent fields, an
    agent run, or a locked-down default. Orthogonal `ClaudeConfig` extras and the
    output format are appended after.

    Attributes:
        models: Mapping from abstract model size to a Claude model alias
            (`haiku`/`sonnet`/`opus`).

    Example:
        >>> from spawnllm.spec import RunSpec
        >>> ClaudeCliBackend().build_command(RunSpec(prompt="hi", model="haiku"))[:5]
        ['claude', '-p', '--no-session-persistence', '--model', 'haiku']
    """

    models: ClassVar[dict[TModel, str]] = CLAUDE_MODELS
    provider: ClassVar[ProviderName] = "claude"
    binary: ClassVar[str] = "claude"
    install_hint: ClassVar[str] = "curl -fsSL https://claude.ai/install.sh | bash"

    def build_command(self, spec: RunSpec) -> list[str]:
        """Build the `claude -p` argv for one stdin-prompted invocation.

        Args:
            spec: The configured run to translate into argv.

        Returns:
            The argv list to execute; the prompt is delivered over stdin.
        """
        cfg = spec.config_for(ClaudeConfig) or ClaudeConfig()
        explicit = (
            cfg.permission_mode is not None
            or cfg.mcp_config is not None
            or cfg.append_system_prompt is not None
            or cfg.system_prompt is not None
            or cfg.settings is not None
            or bool(cfg.disallowed_tools)
            or cfg.strict_mcp
        )
        return [
            "claude",
            "-p",
            "--no-session-persistence",
            "--model",
            spec.model,
            *(
                [
                    *(["--permission-mode", cfg.permission_mode] if cfg.permission_mode is not None else []),
                    *(["--mcp-config", cfg.mcp_config] if cfg.mcp_config is not None else []),
                    *(["--strict-mcp-config"] if cfg.strict_mcp else []),
                    *(["--disallowedTools", *cfg.disallowed_tools] if cfg.disallowed_tools else []),
                    *(
                        ["--append-system-prompt", cfg.append_system_prompt]
                        if cfg.append_system_prompt is not None
                        else []
                    ),
                    *(["--settings", cfg.settings] if cfg.settings is not None else []),
                    *(["--max-budget-usd", str(cfg.max_budget_usd)] if cfg.max_budget_usd is not None else []),
                ]
                if explicit
                else ["--permission-mode", "auto", "--max-budget-usd", "1"]
                if spec.agent
                else ["--system-prompt", "", "--setting-sources", "", "--strict-mcp-config"]
            ),
            *(["--system-prompt", cfg.system_prompt] if cfg.system_prompt is not None else []),
            *(["--max-turns", str(cfg.max_turns)] if cfg.max_turns is not None else []),
            *(["--tools", cfg.tools] if cfg.tools is not None else []),
            *(["--disable-slash-commands"] if cfg.disable_slash_commands else []),
            *(
                ["--json-schema", spec.schema, "--output-format", "json"]
                if spec.schema
                else ["--output-format", cfg.output_format]
                if cfg.output_format
                else []
            ),
            *(["--verbose"] if cfg.verbose else []),
        ]

    def schema_for(self, model: type[BaseModel]) -> str:
        """Serialize a Pydantic model into Anthropic's structured-output JSON schema.

        Uses the Anthropic SDK's `transform_schema`, which recursively sets
        `additionalProperties: false` while preserving Pydantic's `required`,
        producing the standard JSON Schema the `claude --json-schema` flag expects.

        Args:
            model: The Pydantic model describing the structured output.

        Returns:
            A JSON-schema string passed inline to `--json-schema`.
        """
        from anthropic.lib._parse._transform import transform_schema

        return json.dumps(transform_schema(model))

    def parse_response(self, raw: str, response_model: type[BaseModel] | None) -> str | BaseModel:
        """Parse `claude` stdout into text or a validated model.

        Args:
            raw: Raw stdout from the `claude` CLI.
            response_model: Model to validate against, or `None` for raw text.

        Returns:
            `raw` for text calls; otherwise the validated `structured_output` from the result event, else `raw` as JSON.
        """
        return parse_structured_output(raw, response_model)

    def env(self) -> dict[str, str]:
        """Return no extra environment variables; the `claude` CLI runs with the inherited environment."""
        # CLAUDE_CODE_SIMPLE=1 breaks claude.ai keychain auth ("Not logged in")
        # on current CLIs; --setting-sources ""/--strict-mcp-config already trim startup.
        return {}

    def is_authenticated(self, *, timeout: int) -> bool:
        """Report whether `claude auth status` exits cleanly, i.e. a claude.ai login is stored.

        Args:
            timeout: Seconds to wait for `claude auth status`.

        Returns:
            `True` when the OAuth-aware probe reports a stored claude.ai login.
        """
        return (
            subprocess.run(
                ["claude", "auth", "status"], capture_output=True, text=True, timeout=timeout, check=False
            ).returncode
            == 0
        )

"""CliBackend for the Anthropic `claude` CLI, plus install/auth status checks."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import CliBackend
from spawnllm.spec import ClaudeConfig
from spawnllm.structured import structured_value

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.spec import RunSpec
    from spawnllm.types import ProviderName, TModel

CLAUDE_MODELS: dict[TModel, str] = {"small": "haiku", "medium": "sonnet", "large": "opus"}


def result_event(raw: str) -> dict[str, object] | None:
    """Return the `claude` result envelope: the dict itself, or the `type=="result"` stream-json event, else `None`."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    match data:
        case {"is_error": _} | {"result": _}:
            return data
        case list():
            return next((e for e in data if isinstance(e, dict) and e.get("type") == "result"), None)
        case _:
            return None


def keychain_credentials(home: Path) -> str | None:
    """Return the claude.ai OAuth credentials for `home` from the macOS Keychain, or `None` off darwin / on a miss."""
    if sys.platform != "darwin":
        return None
    service = f"Claude Code-credentials-{hashlib.sha256(str(home).encode()).hexdigest()[:8]}"
    proc = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


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

    _isolated_config_dir: str | None = None

    def build_command(self, spec: RunSpec) -> list[str]:
        """Build the `claude -p` argv for one stdin-prompted invocation.

        Args:
            spec: The configured run to translate into argv.

        Returns:
            The argv list to execute; the prompt is delivered over stdin.
        """
        cfg = spec.config_for(ClaudeConfig) or ClaudeConfig()
        schema = self.schema_arg(spec)
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
            *(["--setting-sources", ""] if spec.isolated else []),
            *(["--strict-mcp-config"] if spec.isolated or cfg.strict_mcp else []),
            *(
                [
                    *(["--permission-mode", cfg.permission_mode] if cfg.permission_mode is not None else []),
                    *(["--mcp-config", cfg.mcp_config] if cfg.mcp_config is not None else []),
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
                else ["--system-prompt", ""]
            ),
            *(["--system-prompt", cfg.system_prompt] if cfg.system_prompt is not None else []),
            *(["--max-turns", str(cfg.max_turns)] if cfg.max_turns is not None else []),
            *(["--tools", cfg.tools] if cfg.tools is not None else []),
            *(["--disable-slash-commands"] if cfg.disable_slash_commands else []),
            *(
                ["--json-schema", schema, "--output-format", "json"]
                if schema
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

    def result_text(self, raw: str) -> str:
        """Return the `result` text from the `claude` envelope, falling back to `raw` for plain text."""
        if (event := result_event(raw)) is not None and isinstance(text := event.get("result"), str):
            return text
        return raw

    def result_value(self, raw: str) -> object:
        """Return the `structured_output` from the `claude` stream-json result event, else `raw` parsed as JSON."""
        return structured_value(raw)

    def envelope_error(self, raw: str) -> str | None:
        """Return the error message when the `claude` result event marks the run as an error, else `None`."""
        if (event := result_event(raw)) is not None and event.get("is_error"):
            return event["result"] if isinstance(event.get("result"), str) else "claude reported an error"
        return None

    def env(self, spec: RunSpec) -> dict[str, str]:
        """Point an isolated run at a fresh, host-free `CLAUDE_CONFIG_DIR`; otherwise add nothing.

        Defense in depth behind the argv flags: a config home seeded with nothing
        but the account pointer and OAuth token means plugin and
        `~/.claude.json`-driven loading finds no host settings, plugins, or hooks
        even if a flag is ever dropped.

        Args:
            spec: The configured run; `spec.isolated` gates the override.

        Returns:
            `{"CLAUDE_CONFIG_DIR": <isolated dir>}` for an isolated run, else `{}`.
        """
        if not spec.isolated:
            return {}
        return {"CLAUDE_CONFIG_DIR": self._isolated_dir()}

    def _isolated_dir(self) -> str:
        """Return the process-lifetime isolated config home, creating and seeding it once.

        The home is a fresh temp dir seeded with only the two auth-bearing files,
        sourced from the caller's effective config home (`$CLAUDE_CONFIG_DIR` when
        set, else `~/.claude`): the active-account pointer
        (`$CLAUDE_CONFIG_DIR/.claude.json`, or `~/.claude.json` under the default
        home — the CLI's own lookup rule) minus its `mcpServers` block, so no host
        MCP servers leak even absent `--strict-mcp-config`, and the claude.ai
        OAuth token — the home's `.credentials.json` when present, else on darwin
        the `Claude Code-credentials-<sha256(home)[:8]>` Keychain item, written
        out with mode 0600. When neither exists nothing is seeded and the run
        fails with the CLI's own not-logged-in error. Host `settings.json`,
        plugins, and hooks are never copied. The dir is cached on the backend and
        removed at interpreter exit.
        """
        if self._isolated_config_dir is not None:
            return self._isolated_config_dir
        config_dir = Path(tempfile.mkdtemp(prefix="spawnllm-claude-config-"))
        home = Path(env_home) if (env_home := os.environ.get("CLAUDE_CONFIG_DIR")) else Path.home() / ".claude"
        account_path = home / ".claude.json" if env_home else Path.home() / ".claude.json"
        if account_path.exists():
            account = json.loads(account_path.read_text())
            account.pop("mcpServers", None)
            (config_dir / ".claude.json").write_text(json.dumps(account))
        if (credentials_path := home / ".credentials.json").exists():
            shutil.copyfile(credentials_path, config_dir / ".credentials.json")
        elif (token := keychain_credentials(home)) is not None:
            (seeded := config_dir / ".credentials.json").write_text(token)
            seeded.chmod(0o600)
        atexit.register(shutil.rmtree, config_dir, ignore_errors=True)
        self._isolated_config_dir = str(config_dir)
        return self._isolated_config_dir

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

"""LlmBackend for the Anthropic `claude` CLI, plus install/auth status checks."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import LlmBackend
from spawnllm.structured import parse_result_envelope, parse_structured_output

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.types import TModel

CLAUDE_MODELS: dict[TModel, str] = {"small": "haiku", "medium": "sonnet", "large": "opus"}


@dataclass(frozen=True)
class ClaudeReady:
    """The `claude` CLI is installed and authenticated."""


@dataclass(frozen=True)
class ClaudeNotInstalled:
    """The `claude` CLI is not on PATH.

    Attributes:
        brew_available: Whether the `brew` executable is on PATH to install it with.
    """

    brew_available: bool


@dataclass(frozen=True)
class ClaudeNotAuthenticated:
    """The `claude` CLI is installed but not authenticated."""


ClaudeStatus = ClaudeReady | ClaudeNotInstalled | ClaudeNotAuthenticated
"""Result of `check_status`: `ClaudeReady`, `ClaudeNotInstalled`, or `ClaudeNotAuthenticated`."""


def check_status(timeout: int = 10) -> ClaudeStatus:
    """Check whether the `claude` CLI is installed and authenticated.

    Looks for `claude` on PATH, then runs `claude auth status`.

    Args:
        timeout: Seconds to wait for `claude auth status` before
            `subprocess.TimeoutExpired` is raised.

    Returns:
        `ClaudeReady` when authenticated, `ClaudeNotInstalled` when not on PATH, else `ClaudeNotAuthenticated`.
    """
    if not shutil.which("claude"):
        return ClaudeNotInstalled(brew_available=bool(shutil.which("brew")))
    result = subprocess.run(["claude", "auth", "status"], capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode == 0:
        return ClaudeReady()
    return ClaudeNotAuthenticated()


@dataclass(frozen=True)
class ClaudeCliBackend(LlmBackend):
    """`LlmBackend` for the Anthropic `claude` CLI.

    The default (no-arg) construction delivers the prompt over stdin with abstract
    model tiers and structured-output parsing. The `cc_sentiment` preset
    configures inline `-p` prompting with `{is_error, result}` envelope parsing.

    Attributes:
        models: Mapping from abstract model size to a Claude model alias
            (`haiku`/`sonnet`/`opus`).
        inline_system_prompt: System prompt that `build_argv` passes via
            `--system-prompt`.
        verbose: Whether `build_argv` appends `--verbose`.

    Example:
        >>> ClaudeCliBackend().build_command("haiku", None, agent=False)[:5]
        ['claude', '-p', '--no-session-persistence', '--model', 'haiku']
    """

    models: ClassVar[dict[TModel, str]] = CLAUDE_MODELS

    inline_system_prompt: str = ""
    verbose: bool = False

    @classmethod
    def cc_sentiment(cls, *, system_prompt: str, verbose: bool = False) -> ClaudeCliBackend:
        """Build a backend preset for the sentiment/pushback scoring path.

        Args:
            system_prompt: System prompt that `build_argv` passes via
                `--system-prompt`.
            verbose: Whether `build_argv` appends `--verbose`.

        Returns:
            A `ClaudeCliBackend` for inline `-p` prompting; parse its stdout with `parse_result_envelope`.
        """
        return cls(inline_system_prompt=system_prompt, verbose=verbose)

    def build_command(self, model: str, schema_path: str | None, agent: bool) -> list[str]:
        """Build the `claude -p` argv for one stdin-prompted invocation.

        Every invocation runs without session persistence. Agent invocations
        add `--permission-mode auto` and a $1 `--max-budget-usd` cap;
        non-agent invocations empty the system prompt, disable setting
        sources, and load no MCP servers. A schema adds `--json-schema` with
        `--output-format json`.

        Args:
            model: Claude model name or alias, e.g. `haiku`.
            schema_path: Inline JSON schema passed to `--json-schema`, or `None`.
            agent: Whether the invocation may use tools / agent capabilities.

        Returns:
            The argv list to execute.
        """
        return [
            "claude",
            "-p",
            "--no-session-persistence",
            "--model",
            model,
            *(
                ["--permission-mode", "auto", "--max-budget-usd", "1"]
                if agent
                else ["--system-prompt", "", "--setting-sources", "", "--strict-mcp-config"]
            ),
            *(["--json-schema", schema_path, "--output-format", "json"] if schema_path else []),
        ]

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

    def build_argv(self, content: str, *, model: str) -> list[str]:
        """Build the inline `-p` argv for the sentiment/pushback scoring path.

        The prompt travels inline as the `-p` argument instead of over stdin.
        The invocation uses `inline_system_prompt` as the system prompt, JSON
        output, a single turn, no tools, and no slash commands; `verbose`
        appends `--verbose`.

        Args:
            content: Prompt text passed inline via `-p`.
            model: Claude model name or alias, e.g. `haiku`.

        Returns:
            The argv list to execute; parse its stdout with `parse_result_envelope`.
        """
        argv = [
            "claude",
            "-p",
            content,
            "--model",
            model,
            "--system-prompt",
            self.inline_system_prompt,
            "--output-format",
            "json",
            "--max-turns",
            "1",
            "--tools",
            "",
            "--disable-slash-commands",
        ]
        if self.verbose:
            argv.append("--verbose")
        return argv

    @staticmethod
    def parse_result_envelope(stdout: bytes, *, argv: list[str], stderr: bytes) -> str:
        """Parse the `{is_error, result}` JSON envelope from `claude -p --output-format json`.

        Args:
            stdout: Raw stdout bytes holding the JSON envelope.
            argv: The argv that produced the output, recorded on the raised error.
            stderr: Raw stderr bytes, recorded on the raised error.

        Returns:
            The envelope's `result` string.

        Raises:
            subprocess.CalledProcessError: If the envelope's `is_error` flag is set.
        """
        return parse_result_envelope(stdout, argv=argv, stderr=stderr)

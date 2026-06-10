"""LlmBackend for the Anthropic ``claude`` CLI, plus install/auth status checks."""

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
    """The ``claude`` CLI is installed and authenticated."""


@dataclass(frozen=True)
class ClaudeNotInstalled:
    """The ``claude`` CLI is not on PATH.

    Attributes:
        brew_available: Whether Homebrew is available to install it.
    """

    brew_available: bool


@dataclass(frozen=True)
class ClaudeNotAuthenticated:
    """The ``claude`` CLI is installed but not authenticated."""


ClaudeStatus = ClaudeReady | ClaudeNotInstalled | ClaudeNotAuthenticated


def check_status(timeout: int = 10) -> ClaudeStatus:
    """Return the install/auth status of the ``claude`` CLI."""
    if not shutil.which("claude"):
        return ClaudeNotInstalled(brew_available=bool(shutil.which("brew")))
    result = subprocess.run(["claude", "auth", "status"], capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode == 0:
        return ClaudeReady()
    return ClaudeNotAuthenticated()


@dataclass(frozen=True)
class ClaudeCliBackend(LlmBackend):
    """:class:`LlmBackend` for the Anthropic ``claude`` CLI.

    The default (no-arg) construction delivers the prompt over stdin with abstract
    model tiers and structured-output parsing. The :meth:`cc_sentiment` preset
    configures inline ``-p`` prompting with ``{is_error, result}`` envelope parsing.

    Example:
        >>> ClaudeCliBackend().build_command("haiku", None, agent=False)[0]
        'claude'
    """

    models: ClassVar[dict[TModel, str]] = CLAUDE_MODELS

    inline_system_prompt: str = ""
    verbose: bool = False

    @classmethod
    def cc_sentiment(cls, *, system_prompt: str, verbose: bool = False) -> ClaudeCliBackend:
        """Return a backend configured for inline ``-p`` prompting + envelope parsing."""
        return cls(inline_system_prompt=system_prompt, verbose=verbose)

    def build_command(self, model: str, schema_path: str | None, agent: bool) -> list[str]:
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
        return parse_structured_output(raw, response_model)

    def env(self) -> dict[str, str]:
        # CLAUDE_CODE_SIMPLE=1 breaks claude.ai keychain auth ("Not logged in")
        # on current CLIs; --setting-sources ""/--strict-mcp-config already trim startup.
        return {}

    def build_argv(self, content: str, *, model: str) -> list[str]:
        """Build the inline ``-p`` argv for the sentiment/pushback scoring path."""
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
        """Parse the ``{is_error, result}`` JSON envelope; raise ``CalledProcessError`` on error."""
        return parse_result_envelope(stdout, argv=argv, stderr=stderr)

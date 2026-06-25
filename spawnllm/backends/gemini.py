"""LlmBackends for the Gemini CLI family (`gemini` and the `agy`/Antigravity successor)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import CliBackend, Invocation
from spawnllm.spec import GeminiConfig
from spawnllm.structured import extract_json_block

if TYPE_CHECKING:
    from spawnllm.spec import RunSpec
    from spawnllm.types import ProviderName, TModel

SCHEMA_PROMPT = (
    "Respond with ONLY a single JSON object that conforms to this JSON Schema. "
    "No prose, no explanation, no markdown code fences.\nJSON Schema:"
)


class GeminiFamilyBackend(CliBackend, ABC):
    """Shared logic for the Gemini-family CLIs (`gemini`, `agy`)."""

    api_key_envs: ClassVar[tuple[str, ...]]

    def env(self, _spec: RunSpec) -> dict[str, str]:
        """Return no extra environment variables.

        Gemini-family CLIs read settings and OAuth from the same config home (`~/.gemini`,
        `~/.gemini/antigravity-cli`), with no isolation flag and no way to relocate config
        without stranding auth, so isolated runs read the real config — the no-flag exception.
        """
        return {}

    def is_authenticated(self, *, timeout: int) -> bool:
        """Report whether cached OAuth credentials exist or an API-key env var is set.

        Args:
            timeout: Unused; the cached-credential and env-var probes are instant.

        Returns:
            `True` when cached credentials are present or any `api_key_envs` var is set.
        """
        return self.has_cached_credentials() or any(os.environ.get(k) for k in self.api_key_envs)

    def invocation(self, spec: RunSpec) -> Invocation:
        """Build the argv and inline prompt for a single invocation.

        The prompt travels inline via `-p`; structured output appends the JSON
        schema and an instruction to emit only conforming JSON. An empty stdin
        forces the CLI into non-interactive mode, and the result is read from stdout.

        Args:
            spec: The configured run to translate into an invocation.

        Returns:
            An `Invocation` with an empty stdin that forces non-interactive output.
        """
        schema = self.schema_arg(spec)
        text = spec.prompt if schema is None else f"{spec.prompt}\n\n{SCHEMA_PROMPT}\n{schema}"
        return Invocation(self.build_command(spec) + ["-p", text], "")

    def result_text(self, raw: str) -> str:
        """Return the model's text output, extracted from this CLI's stdout envelope."""
        return self.extract_text(raw)

    def result_value(self, raw: str) -> object:
        """Return the JSON block parsed from the model's text output."""
        return json.loads(extract_json_block(self.result_text(raw)))

    @abstractmethod
    def extract_text(self, raw: str) -> str: ...

    @abstractmethod
    def has_cached_credentials(self) -> bool: ...


class GeminiCliBackend(GeminiFamilyBackend):
    """`LlmBackend` for Google's `gemini` CLI.

    Invokes `gemini --model … -o json` with the prompt delivered inline via
    `-p`. Authentication prefers cached OAuth credentials and falls back to a
    `GEMINI_API_KEY`/`GOOGLE_API_KEY` environment key.

    Attributes:
        models: Mapping from abstract model size to a Gemini model name.
        api_key_envs: Environment variables consulted for an API key when no
            OAuth credentials are cached.

    Example:
        >>> GeminiCliBackend().build_command(RunSpec(prompt="hi", model="gemini-2.5-flash"))[:5]
        ['gemini', '--model', 'gemini-2.5-flash', '-o', 'json']
    """

    models: ClassVar[dict[TModel, str]] = {
        "small": "gemini-2.5-flash-lite",
        "medium": "gemini-2.5-flash",
        "large": "gemini-3-pro-preview",
    }
    provider: ClassVar[ProviderName] = "gemini"
    binary: ClassVar[str] = "gemini"
    install_hint: ClassVar[str] = "npm install -g @google/gemini-cli"
    api_key_envs: ClassVar[tuple[str, ...]] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def build_command(self, spec: RunSpec) -> list[str]:
        """Build the `gemini` argv for one inline-prompted invocation.

        Every invocation requests JSON output via `-o json`. Agent invocations
        auto-approve tool use with `--approval-mode yolo`; non-agent invocations
        keep the default approval mode and disable tool extensions with
        `-e none`. A `GeminiConfig` overrides the derived approval mode and
        replaces the extension set.

        Args:
            spec: The configured run to translate into argv.

        Returns:
            The argv list to execute.
        """
        config = spec.config_for(GeminiConfig)
        return [
            "gemini",
            "--model",
            spec.model,
            "-o",
            "json",
            "--approval-mode",
            (config and config.approval_mode) or ("yolo" if spec.agent else "default"),
            *self.extension_args(spec, config),
        ]

    def extension_args(self, spec: RunSpec, config: GeminiConfig | None) -> list[str]:
        match config and config.extensions:
            case None if spec.agent:
                return []
            case None:
                return ["-e", "none"]
            case extensions:
                return [arg for e in extensions for arg in ("-e", e)]

    def extract_text(self, raw: str) -> str:
        return json.loads(raw)["response"]

    def envelope_error(self, raw: str) -> str | None:
        """Return the raw failure payload when `gemini` reports `totalErrors` or an empty response, else `None`.

        The whole payload tail is folded into the message so any transient marker the CLI emits
        (529/overloaded/rate-limit) lands in `Response.error` and `is_transient` can fire a retry.
        """
        data = json.loads(raw)
        models = data.get("stats", {}).get("models", {})
        if sum(m["api"]["totalErrors"] for m in models.values()) > 0 or not data.get("response"):
            return f"gemini call failed: {raw.strip()[-2000:]}"
        return None

    def has_cached_credentials(self) -> bool:
        return (Path.home() / ".gemini" / "oauth_creds.json").exists()


class AntigravityCliBackend(GeminiFamilyBackend):
    """`LlmBackend` for the Antigravity `agy` CLI, a Gemini-family successor.

    Invokes `agy --model … -p` and reads its plain-text stdout. Authentication
    prefers an Antigravity login stored in the macOS keychain and falls back to
    a `GEMINI_API_KEY`/`ANTIGRAVITY_API_KEY` environment key.

    Attributes:
        models: Mapping from abstract model size to an Antigravity model name.
        api_key_envs: Environment variables consulted for an API key when no
            cached login is present.

    Example:
        >>> AntigravityCliBackend().build_command(RunSpec(prompt="hi", model="gemini-3.5"))
        ['agy', '--model', 'gemini-3.5', '--print-timeout', '120s']
    """

    models: ClassVar[dict[TModel, str]] = {
        "small": "gemini-3.5-flash",
        "medium": "gemini-3.5",
        "large": "gemini-3.5-pro",
    }
    provider: ClassVar[ProviderName] = "antigravity"
    binary: ClassVar[str] = "agy"
    install_hint: ClassVar[str] = "curl -fsSL https://antigravity.google/cli/install.sh | bash"
    api_key_envs: ClassVar[tuple[str, ...]] = ("GEMINI_API_KEY", "ANTIGRAVITY_API_KEY")

    def build_command(self, spec: RunSpec) -> list[str]:
        """Build the `agy` argv for one inline-prompted invocation.

        Agent invocations auto-approve tool use with
        `--dangerously-skip-permissions`; `agy` has no flag to disable tools for
        non-agent calls. Every invocation caps non-interactive runs with
        `--print-timeout 120s`.

        Args:
            spec: The configured run to translate into argv.

        Returns:
            The argv list to execute.
        """
        return [
            "agy",
            "--model",
            spec.model,
            *(["--dangerously-skip-permissions"] if spec.agent else []),
            "--print-timeout",
            "120s",
        ]

    def extract_text(self, raw: str) -> str:
        return raw.strip()

    def has_cached_credentials(self) -> bool:
        if sys.platform != "darwin" or not shutil.which("security"):
            return False
        return (
            subprocess.run(
                ["security", "find-generic-password", "-s", "gemini", "-a", "antigravity"],
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )

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

from spawnllm.backends.base import Invocation, LlmBackend
from spawnllm.structured import extract_json_block

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.types import TModel

SCHEMA_PROMPT = (
    "Respond with ONLY a single JSON object that conforms to this JSON Schema. "
    "No prose, no explanation, no markdown code fences.\nJSON Schema:"
)


class GeminiFamilyBackend(LlmBackend, ABC):
    """Shared logic for the Gemini-family CLIs (`gemini`, `agy`)."""

    api_key_envs: ClassVar[tuple[str, ...]]

    def env(self) -> dict[str, str]:
        """Return no extra environment variables; Gemini-family CLIs authenticate via OAuth, never an injected API key."""
        return {}

    def is_authenticated(self, *, timeout: int) -> bool:
        """Report whether cached OAuth credentials exist or an API-key env var is set.

        Args:
            timeout: Unused; the cached-credential and env-var probes are instant.

        Returns:
            `True` when cached credentials are present or any `api_key_envs` var is set.
        """
        return self.has_cached_credentials() or any(os.environ.get(k) for k in self.api_key_envs)

    def prompt_args(self, text: str) -> list[str]:
        return ["-p", text]

    def invocation(self, prompt: str, *, model: str, schema_path: str | None, agent: bool) -> Invocation:
        """Build the argv and inline prompt for a single invocation.

        The prompt travels inline via `-p`; structured output appends the JSON
        schema and an instruction to emit only conforming JSON. An empty stdin
        forces the CLI into non-interactive mode, and the result is read from stdout.

        Args:
            prompt: The prompt text to deliver inline.
            model: Provider-specific model name.
            schema_path: Inline JSON schema appended to the prompt, or `None`.
            agent: Whether the invocation may use tools / agent capabilities.

        Returns:
            An `Invocation` with an empty stdin that forces non-interactive output.
        """
        text = prompt if schema_path is None else f"{prompt}\n\n{SCHEMA_PROMPT}\n{schema_path}"
        return Invocation(self.build_command(model, None, agent) + self.prompt_args(text), "")

    def parse_response(self, raw: str, response_model: type[BaseModel] | None) -> str | BaseModel:
        """Parse Gemini-family stdout into text or a validated model.

        Args:
            raw: Raw stdout from the CLI.
            response_model: Model to validate against, or `None` for raw text.

        Returns:
            The extracted text when `response_model` is `None`; otherwise the JSON block validated against it.
        """
        text = self.extract_text(raw)
        if response_model is None:
            return text
        return response_model.model_validate_json(extract_json_block(text))

    @abstractmethod
    def build_command(self, model: str, schema_path: str | None, agent: bool) -> list[str]: ...

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
        >>> GeminiCliBackend().build_command("gemini-2.5-flash", None, agent=False)[:5]
        ['gemini', '--model', 'gemini-2.5-flash', '-o', 'json']
    """

    models: ClassVar[dict[TModel, str]] = {
        "small": "gemini-2.5-flash-lite",
        "medium": "gemini-2.5-flash",
        "large": "gemini-3-pro-preview",
    }
    binary: ClassVar[str] = "gemini"
    install_hint: ClassVar[str] = "npm install -g @google/gemini-cli"
    api_key_envs: ClassVar[tuple[str, ...]] = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def build_command(self, model: str, schema_path: str | None, agent: bool) -> list[str]:
        """Build the `gemini` argv for one inline-prompted invocation.

        Every invocation requests JSON output via `-o json`. Agent invocations
        auto-approve tool use with `--approval-mode yolo`; non-agent invocations
        keep the default approval mode and disable tool extensions with
        `-e none`. The schema travels inline within the prompt, so `schema_path`
        is unused here.

        Args:
            model: Gemini model name, e.g. `gemini-2.5-flash`.
            schema_path: Unused; the schema is appended to the prompt instead.
            agent: Whether the invocation may use tools / agent capabilities.

        Returns:
            The argv list to execute.
        """
        return [
            "gemini",
            "--model",
            model,
            "-o",
            "json",
            "--approval-mode",
            "yolo" if agent else "default",
            *([] if agent else ["-e", "none"]),
        ]

    def extract_text(self, raw: str) -> str:
        data = json.loads(raw)
        if (
            sum(m["api"]["totalErrors"] for m in data.get("stats", {}).get("models", {}).values()) > 0
            or not data.get("response")
        ):
            raise RuntimeError(f"gemini call failed: {data.get('stats', {}).get('models')}")
        return data["response"]

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
        >>> AntigravityCliBackend().build_command("gemini-3.5", None, agent=False)
        ['agy', '--model', 'gemini-3.5', '--print-timeout', '120s']
    """

    models: ClassVar[dict[TModel, str]] = {
        "small": "gemini-3.5-flash",
        "medium": "gemini-3.5",
        "large": "gemini-3.5-pro",
    }
    binary: ClassVar[str] = "agy"
    install_hint: ClassVar[str] = "curl -fsSL https://antigravity.google/cli/install.sh | bash"
    api_key_envs: ClassVar[tuple[str, ...]] = ("GEMINI_API_KEY", "ANTIGRAVITY_API_KEY")

    def build_command(self, model: str, schema_path: str | None, agent: bool) -> list[str]:
        """Build the `agy` argv for one inline-prompted invocation.

        Agent invocations auto-approve tool use with
        `--dangerously-skip-permissions`; `agy` has no flag to disable tools for
        non-agent calls. Every invocation caps non-interactive runs with
        `--print-timeout 120s`.

        Args:
            model: Antigravity model name, e.g. `gemini-3.5`.
            schema_path: Unused; the schema is appended to the prompt instead.
            agent: Whether the invocation may use tools / agent capabilities.

        Returns:
            The argv list to execute.
        """
        return [
            "agy",
            "--model",
            model,
            *(["--dangerously-skip-permissions"] if agent else []),
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

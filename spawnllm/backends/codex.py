"""CliBackend for the OpenAI `codex` CLI."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import CliBackend, Invocation
from spawnllm.spec import CodexConfig

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.spec import RunSpec
    from spawnllm.types import ProviderName, TModel


class CodexCliBackend(CliBackend):
    """`CliBackend` for the OpenAI `codex` CLI.

    `build_command` translates a `RunSpec` into a `codex exec` argv that runs an
    ephemeral session in a read-only sandbox; `invocation` resolves the schema to
    a temp file and captures the final message to an `-o` file. An optional
    `CodexConfig` overrides the sandbox and re-enables Codex hooks or MCP servers.

    Attributes:
        models: Mapping from abstract model size to an OpenAI model name.

    Example:
        >>> from spawnllm.spec import RunSpec
        >>> CodexCliBackend().build_command(RunSpec(prompt="hi", model="gpt-5.5"))[:4]
        ['codex', 'exec', '--ephemeral', '--sandbox']
    """

    models: ClassVar[dict[TModel, str]] = {
        "small": "gpt-5.4-mini:low",
        "medium": "gpt-5.4-mini:medium",
        "large": "gpt-5.5:medium",
    }
    provider: ClassVar[ProviderName] = "codex"
    binary: ClassVar[str] = "codex"
    install_hint: ClassVar[str] = "npm install -g @openai/codex"

    def build_command(self, spec: RunSpec) -> list[str]:
        """Build the `codex exec` argv for one stdin-prompted invocation.

        Derives the schema from `spec.response_model`, writes it to a temp file
        via `resolve_schema_path`, and adds `--output-schema` when present;
        `invocation` reuses that path and cleans it up after the run.

        Args:
            spec: The configured run to translate into argv.

        Returns:
            The argv list to execute; the prompt is delivered over stdin.
        """
        from spawnllm.structured import resolve_schema_path

        return self.command_for(spec, resolve_schema_path(self, self.schema_arg(spec)))

    def command_for(self, spec: RunSpec, schema_path: str | None) -> list[str]:
        cfg = spec.config_for(CodexConfig) or CodexConfig()
        model, _, effort = spec.model.partition(":")
        return [
            "codex",
            "exec",
            "--ephemeral",
            "--sandbox",
            cfg.sandbox or "read-only",
            "--model",
            model,
            *(["-c", f"model_reasoning_effort={effort}"] if effort else []),
            *(["--ignore-user-config"] if spec.isolated else []),
            *(
                []
                if spec.agent
                else [
                    *([] if cfg.enable_hooks else ["-c", "features.hooks=false"]),
                    *([] if cfg.enable_mcp else ["-c", "features.mcp_servers=false"]),
                ]
            ),
            *(["--output-schema", schema_path] if schema_path else []),
        ]

    def invocation(self, spec: RunSpec) -> Invocation:
        """Build the `codex exec` invocation, capturing the final message to a file.

        `codex exec` streams an interactive log to stdout, so the result is read
        from the `-o` file instead. The schema is resolved to a temp file once
        here; that file and the result file are removed after the run.

        Args:
            spec: The configured run to translate into an invocation.

        Returns:
            An `Invocation` whose result is read from the `-o` file.
        """
        from spawnllm.structured import resolve_schema_path

        schema_path = resolve_schema_path(self, self.schema_arg(spec))
        fd, result_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        return Invocation(
            [*self.command_for(spec, schema_path), "-o", result_path],
            spec.prompt,
            result_path=result_path,
            cleanup_paths=(result_path, *((schema_path,) if schema_path else ())),
        )

    def schema_for(self, model: type[BaseModel]) -> str:
        """Serialize a Pydantic model into an OpenAI strict JSON schema.

        Uses the OpenAI SDK's `to_strict_json_schema`, which recursively sets
        `additionalProperties: false` and forces every property into `required`
        across `$defs`, `anyOf`, and array items — the form the Responses API
        requires behind `codex exec --output-schema`.

        Args:
            model: The Pydantic model describing the structured output.

        Returns:
            A strict JSON-schema string written to the `--output-schema` file.
        """
        from openai.lib._pydantic import to_strict_json_schema

        return json.dumps(to_strict_json_schema(model))

    def env(self, _spec: RunSpec) -> dict[str, str]:
        """Return no extra environment variables; `--ignore-user-config` isolates config while `CODEX_HOME` keeps auth.

        `codex` keeps `auth.json` in `CODEX_HOME`, so relocating it would strand a
        single-use OAuth refresh token; the `--ignore-user-config` flag isolates
        `config.toml` without touching auth.
        """
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

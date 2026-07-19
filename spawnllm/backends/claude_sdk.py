"""In-process host for the Claude Agent SDK's bundled Claude Code CLI."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from spawnllm import _core
from spawnllm.backends.base import (
    BackendCallError,
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    LlmBackend,
)
from spawnllm.backends.claude import CLAUDE_MODELS
from spawnllm.response import Error, Output, Response, Result
from spawnllm.spec import ClaudeConfig

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk.types import SystemPromptPreset

    from spawnllm.backends.base import BackendStatus
    from spawnllm.spec import RunSpec
    from spawnllm.types import ProviderName, TModel


def sdk_cli_path() -> str | None:
    """Return the Claude Agent SDK's bundled CLI path, falling back to `PATH`."""
    import claude_agent_sdk

    bundled_name = "claude.exe" if sys.platform == "win32" else "claude"
    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / bundled_name
    return str(bundled) if bundled.is_file() else shutil.which("claude")


class ClaudeSdkBackend(LlmBackend):
    """Claude backend hosted through the optional `claude-agent-sdk` package.

    The SDK bundles Claude Code, so `pip install 'spawnllm[sdk]'` is the only
    installation step. Authentication comes from the ambient subscription OAuth
    session: the bundled CLI resolves `/login` credentials from the platform
    keychain or uses `CLAUDE_CODE_OAUTH_TOKEN`.
    Here `isolated=True` makes settings and MCP configuration hermetic through
    `setting_sources=[]` and `strict_mcp_config`, but unlike `ClaudeCliBackend`
    it does not seed a fresh `CLAUDE_CONFIG_DIR`, so credentials come from the
    ambient config home or `CLAUDE_CODE_OAUTH_TOKEN`.

    `api_auth=False` blanks Claude's API-key environment variables because the
    SDK can only overlay its subprocess environment, not truly unset inherited
    keys. An explicit `RunSpec.env` value still wins. `ClaudeConfig.mcp_config`
    passes through the SDK's path-capable `mcp_servers` field. Its `output_format`
    and `verbose` fields are ignored because the SDK transport always uses
    stream-JSON with verbose output internally.

    Attributes:
        models: Mapping from abstract model size to a Claude model alias.
        provider: Python-only backend identifier used for selection and config.
        schema_dialect: Anthropic strict-schema dialect applied by the core.
    """

    models: ClassVar[dict[TModel, str]] = CLAUDE_MODELS
    provider: ClassVar[ProviderName] = "claude-sdk"
    resolve_provider: ClassVar[ProviderName] = "claude"
    binary: ClassVar[str] = "claude-sdk"
    install_hint: ClassVar[str] = "uv pip install 'spawnllm[sdk]'"
    schema_dialect: ClassVar[str | None] = "anthropic"

    def build_options(self, spec: RunSpec) -> ClaudeAgentOptions:
        """Translate a `RunSpec` into the SDK options matching the Claude CLI plan.

        Args:
            spec: The configured run to translate.

        Returns:
            Options for the SDK's one-shot `query` generator.
        """
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk.types import PermissionMode

        cfg = spec.config_for(ClaudeConfig) or ClaudeConfig()
        api_key_vars = _core.dispatch("capabilities")["api_key_vars"]["claude"]
        schema = self.wire_schema(spec)
        options = ClaudeAgentOptions(
            model=spec.model,
            cwd=spec.cwd,
            env=({} if spec.api_auth else {key: "" for key in api_key_vars}) | (spec.env or {}),
            setting_sources=[] if spec.isolated else ["user", "project", "local"],
            strict_mcp_config=spec.isolated or cfg.strict_mcp,
            max_turns=cfg.max_turns,
            tools=list(cfg.tools) if cfg.tools is not None else None,
            extra_args={"no-session-persistence": None}
            | ({"disable-slash-commands": None} if cfg.disable_slash_commands else {}),
            output_format={"type": "json_schema", "schema": schema} if schema is not None else None,
        )
        explicit = (
            cfg.permission_mode is not None
            or cfg.mcp_config is not None
            or cfg.append_system_prompt is not None
            or cfg.system_prompt is not None
            or cfg.settings is not None
            or bool(cfg.disallowed_tools)
            or cfg.strict_mcp
        )
        if explicit:
            system_prompt: str | SystemPromptPreset
            if cfg.system_prompt is not None:
                system_prompt = cfg.system_prompt
            elif cfg.append_system_prompt is not None:
                system_prompt = {
                    "type": "preset",
                    "preset": "claude_code",
                    "append": cfg.append_system_prompt,
                }
            else:
                system_prompt = {"type": "preset", "preset": "claude_code"}
            return dataclasses.replace(
                options,
                permission_mode=cast(PermissionMode | None, cfg.permission_mode),
                mcp_servers=cfg.mcp_config if cfg.mcp_config is not None else {},
                disallowed_tools=list(cfg.disallowed_tools),
                settings=cfg.settings,
                max_budget_usd=cfg.max_budget_usd,
                system_prompt=system_prompt,
                extra_args=options.extra_args
                | (
                    {"append-system-prompt": cfg.append_system_prompt}
                    if cfg.system_prompt is not None and cfg.append_system_prompt is not None
                    else {}
                ),
            )
        if spec.agent:
            agent_prompt: SystemPromptPreset = {"type": "preset", "preset": "claude_code"}
            return dataclasses.replace(
                options,
                permission_mode="auto",
                max_budget_usd=1.0,
                system_prompt=agent_prompt,
            )
        return options

    async def aexecute(self, spec: RunSpec) -> Response:
        """Execute one SDK query asynchronously and resolve its Claude result event.

        Args:
            spec: The configured run to execute.

        Returns:
            The resolved `Response`, including normalized SDK and timeout errors.
        """
        from claude_agent_sdk import ClaudeSDKError, ResultMessage, query

        try:
            async with contextlib.aclosing(query(prompt=spec.prompt, options=self.build_options(spec))) as stream:
                async with asyncio.timeout(spec.timeout):
                    messages = [message async for message in stream]
        except TimeoutError:
            msg = f"{self.provider} timed out after {spec.timeout}s"
            return Response(spec=spec, output=Output(""), error=Error(msg, TimeoutError(msg)))
        except ClaudeSDKError as e:
            return self.to_response("", returncode=1, stderr=str(e), spec=spec)
        result = next(message for message in reversed(messages) if isinstance(message, ResultMessage))
        envelope = {
            key: value
            for key, value in {
                "type": "result",
                "subtype": result.subtype,
                "is_error": result.is_error,
                "result": result.result,
                "structured_output": result.structured_output,
                "total_cost_usd": result.total_cost_usd,
                "usage": result.usage,
                "num_turns": result.num_turns,
                "session_id": result.session_id,
            }.items()
            if value is not None
        }
        return self.to_response(json.dumps(envelope), returncode=0, stderr="", spec=spec)

    def execute(self, spec: RunSpec) -> Response:
        """Execute one SDK query synchronously via `asyncio.run`.

        Args:
            spec: The configured run to execute.

        Returns:
            The resolved `Response`.

        Raises:
            RuntimeError: If called from a thread already running an event loop.
        """
        return asyncio.run(self.aexecute(spec))

    def to_response(self, raw: str, *, returncode: int, stderr: str, spec: RunSpec) -> Response:
        """Resolve the SDK's reconstructed event through the core's Claude path.

        Args:
            raw: The reconstructed Claude result event.
            returncode: The synthesized transport exit code.
            stderr: The SDK error text, when present.
            spec: The configured run carrying any response model.

        Returns:
            The resolved `Response`.
        """
        import pydantic

        output = Output(raw)
        resolved = _core.dispatch(
            "resolve",
            {
                "provider": self.resolve_provider,
                "raw": raw,
                "returncode": returncode,
                "stderr": stderr,
                "wants_value": spec.response_model is not None,
            },
        )
        if resolved["status"] != "ok":
            return Response(
                spec=spec,
                output=output,
                error=Error(resolved["msg"], BackendCallError(resolved["msg"])),
            )
        if spec.response_model is None:
            return Response(spec=spec, output=output, result=Result(raw=resolved["text"]))
        try:
            parsed = spec.response_model.model_validate(resolved["value"])
        except pydantic.ValidationError as e:
            return Response(spec=spec, output=output, error=Error(str(e), e))
        return Response(spec=spec, output=output, result=Result(raw=resolved["text"], parsed=parsed))

    def accounting(self, raw: str) -> tuple[float | None, dict[str, object] | None]:
        """Return cost and usage from a reconstructed event via Claude resolution.

        Args:
            raw: The reconstructed Claude result event.

        Returns:
            The event's `(cost_usd, usage)` pair.
        """
        resolved = _core.dispatch(
            "resolve",
            {
                "provider": self.resolve_provider,
                "raw": raw,
                "returncode": 0,
                "stderr": "",
                "wants_value": False,
            },
        )
        return resolved["cost_usd"], resolved["usage"]

    def env(self, _spec: RunSpec) -> dict[str, str]:
        """Return no host-side environment; the SDK owns its subprocess environment."""
        return {}

    def is_authenticated(self, *, timeout: int) -> bool:
        """Report whether the SDK's Claude Code executable has an active login.

        Args:
            timeout: Seconds to wait for `claude auth status`.

        Returns:
            `True` when the authentication probe exits successfully.
        """
        if (cli := sdk_cli_path()) is None:
            return False
        return (
            subprocess.run(
                [cli, "auth", "status"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            ).returncode
            == 0
        )

    def check_status(self, *, timeout: int = 10) -> BackendStatus:
        """Check whether the optional SDK is installed and its bundled CLI authenticated.

        Args:
            timeout: Seconds to wait for the authentication probe.

        Returns:
            `BackendReady` when authenticated, `BackendNotInstalled` without the
            SDK extra, else `BackendNotAuthenticated`.
        """
        if importlib.util.find_spec("claude_agent_sdk") is None:
            return BackendNotInstalled(binary=self.binary, install_hint=self.install_hint)
        if self.is_authenticated(timeout=timeout):
            return BackendReady(binary=self.binary)
        return BackendNotAuthenticated(binary=self.binary)

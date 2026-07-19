from __future__ import annotations

import asyncio
import dataclasses
import subprocess
from pathlib import Path

import claude_agent_sdk
import pytest
from claude_agent_sdk import ClaudeSDKError, ResultMessage
from pydantic import BaseModel

import spawnllm.backends.claude_sdk as claude_sdk_module
from spawnllm.backends.base import BackendNotAuthenticated, BackendNotInstalled, BackendReady
from spawnllm.backends.claude_sdk import ClaudeSdkBackend, sdk_cli_path
from spawnllm.spec import ClaudeConfig, RunSpec


class StructuredResult(BaseModel):
    answer: int


def result_message(**changes: object) -> ResultMessage:
    message = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=80,
        is_error=False,
        num_turns=1,
        session_id="session-1",
        total_cost_usd=0.125,
        usage={"input_tokens": 3, "output_tokens": 4},
        result="done",
    )
    return dataclasses.replace(message, **changes)


def spec(*, config: ClaudeConfig | None = None, **changes: object) -> RunSpec:
    base = RunSpec(
        prompt="hello",
        model="sonnet",
        provider_configs={"claude": config} if config is not None else {},
    )
    return dataclasses.replace(base, **changes)


class TestBuildOptions:
    def test_lockdown_default(self) -> None:
        options = ClaudeSdkBackend().build_options(spec())

        assert options.model == "sonnet"
        assert options.system_prompt is None
        assert options.permission_mode is None
        assert options.max_budget_usd is None
        assert options.setting_sources == []
        assert options.strict_mcp_config is True
        assert options.max_turns is None
        assert options.tools is None
        assert options.extra_args == {"no-session-persistence": None}
        assert options.output_format is None

    def test_isolated_false_loads_every_setting_source(self) -> None:
        options = ClaudeSdkBackend().build_options(spec(isolated=False))

        assert options.setting_sources == ["user", "project", "local"]
        assert options.strict_mcp_config is False

    def test_agent_branch_uses_auto_permissions_and_claude_code_prompt(self) -> None:
        options = ClaudeSdkBackend().build_options(spec(agent=True))

        assert options.permission_mode == "auto"
        assert options.max_budget_usd == 1.0
        assert options.system_prompt == {"type": "preset", "preset": "claude_code"}

    def test_full_explicit_config(self) -> None:
        config = ClaudeConfig(
            permission_mode="bypassPermissions",
            mcp_config="/tmp/mcp.json",
            strict_mcp=True,
            append_system_prompt="additional system prompt",
            system_prompt="custom system prompt",
            settings="/tmp/settings.json",
            disallowed_tools=("Bash", "Edit"),
            max_turns=4,
            max_budget_usd=2.5,
        )
        options = ClaudeSdkBackend().build_options(spec(config=config, isolated=False))

        assert options.permission_mode == "bypassPermissions"
        assert options.mcp_servers == "/tmp/mcp.json"
        assert options.strict_mcp_config is True
        assert options.disallowed_tools == ["Bash", "Edit"]
        assert options.settings == "/tmp/settings.json"
        assert options.max_turns == 4
        assert options.max_budget_usd == 2.5
        assert options.system_prompt == "custom system prompt"
        assert options.extra_args == {
            "no-session-persistence": None,
            "append-system-prompt": "additional system prompt",
        }

    def test_append_system_prompt_alone_selects_explicit_branch(self) -> None:
        options = ClaudeSdkBackend().build_options(spec(config=ClaudeConfig(append_system_prompt="remember this")))

        assert options.permission_mode is None
        assert options.system_prompt == {
            "type": "preset",
            "preset": "claude_code",
            "append": "remember this",
        }

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [
            pytest.param(None, None, id="default"),
            pytest.param((), [], id="disabled"),
            pytest.param(("Read", "Bash"), ["Read", "Bash"], id="restricted"),
        ],
    )
    def test_tools_tri_state(
        self,
        configured: tuple[str, ...] | None,
        expected: list[str] | None,
    ) -> None:
        assert ClaudeSdkBackend().build_options(spec(config=ClaudeConfig(tools=configured))).tools == expected

    def test_disable_slash_commands_is_an_extra_flag(self) -> None:
        options = ClaudeSdkBackend().build_options(spec(config=ClaudeConfig(disable_slash_commands=True)))

        assert options.extra_args == {
            "no-session-persistence": None,
            "disable-slash-commands": None,
        }

    def test_schema_becomes_sdk_json_schema_output_format(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}

        options = ClaudeSdkBackend().build_options(spec(schema=schema))

        assert options.output_format == {"type": "json_schema", "schema": schema}

    def test_default_blanks_claude_api_key_variables(self) -> None:
        options = ClaudeSdkBackend().build_options(spec())

        assert options.env == {
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
        }

    def test_api_auth_leaves_api_key_environment_unmodified(self) -> None:
        assert ClaudeSdkBackend().build_options(spec(api_auth=True)).env == {}

    def test_explicit_environment_wins_over_blanking(self) -> None:
        options = ClaudeSdkBackend().build_options(spec(env={"ANTHROPIC_API_KEY": "explicit", "OTHER": "value"}))

        assert options.env == {
            "ANTHROPIC_API_KEY": "explicit",
            "ANTHROPIC_AUTH_TOKEN": "",
            "OTHER": "value",
        }


async def test_aexecute_resolves_structured_result_and_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = result_message(result="structured result", structured_output={"answer": 7})

    async def fake_query(**_: object):
        yield message

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    backend = ClaudeSdkBackend()
    response = await backend.aexecute(spec(response_model=StructuredResult))

    assert response.error is None
    assert response.result.raw == "structured result"
    assert response.result.parsed == StructuredResult(answer=7)
    assert backend.accounting(response.output.raw) == (
        0.125,
        {"input_tokens": 3, "output_tokens": 4},
    )


async def test_aexecute_parses_result_json_without_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = result_message(result='{"answer": 7}', structured_output=None)

    async def fake_query(**_: object):
        yield message

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    response = await ClaudeSdkBackend().aexecute(spec(response_model=StructuredResult))

    assert response.error is None
    assert response.result.parsed == StructuredResult(answer=7)


async def test_aexecute_result_error_becomes_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_query(**_: object):
        yield result_message(subtype="error_during_execution", is_error=True, result="permission denied")

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    response = await ClaudeSdkBackend().aexecute(spec())

    assert response.result is None
    assert response.error is not None
    assert "permission denied" in response.error.msg


async def test_aexecute_sdk_error_becomes_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_query(**_: object):
        if False:
            yield result_message()
        raise ClaudeSDKError("sdk connection failed")

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    response = await ClaudeSdkBackend().aexecute(spec())

    assert response.result is None
    assert response.error is not None
    assert "sdk connection failed" in response.error.msg


async def test_aexecute_timeout_closes_query_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    cleaned_up: list[bool] = []

    async def fake_query(**_: object):
        try:
            await asyncio.sleep(60)
            yield result_message()
        finally:
            await asyncio.sleep(0)
            cleaned_up.append(True)

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    response = await ClaudeSdkBackend().aexecute(spec(timeout=0.01))

    assert response.result is None
    assert response.error is not None
    assert response.error.msg == "claude-sdk timed out after 0.01s"
    assert isinstance(response.error.ex, TimeoutError)
    assert cleaned_up == [True]


def test_execute_bridges_to_async_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_query(**_: object):
        yield result_message(result="sync result")

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    response = ClaudeSdkBackend().execute(spec())

    assert response.error is None
    assert response.result.raw == "sync result"


class TestStatus:
    @pytest.mark.parametrize(
        ("platform", "bundled_name"),
        [
            pytest.param("win32", "claude.exe", id="windows"),
            pytest.param("darwin", "claude", id="non_windows"),
        ],
    )
    def test_sdk_cli_path_uses_platform_bundled_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        platform: str,
        bundled_name: str,
    ) -> None:
        monkeypatch.setattr(claude_sdk_module.sys, "platform", platform)
        monkeypatch.setattr(claude_sdk_module.Path, "is_file", lambda _: True)

        assert Path(sdk_cli_path()).name == bundled_name

    def test_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(claude_sdk_module.importlib.util, "find_spec", lambda name: None)

        assert ClaudeSdkBackend().check_status() == BackendNotInstalled(
            binary="claude-sdk",
            install_hint="uv pip install 'spawnllm[sdk]'",
        )

    @pytest.mark.parametrize(
        ("authenticated", "expected"),
        [
            pytest.param(True, BackendReady(binary="claude-sdk"), id="ready"),
            pytest.param(False, BackendNotAuthenticated(binary="claude-sdk"), id="not_authenticated"),
        ],
    )
    def test_installed_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
        authenticated: bool,
        expected: BackendReady | BackendNotAuthenticated,
    ) -> None:
        monkeypatch.setattr(claude_sdk_module.importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(ClaudeSdkBackend, "is_authenticated", lambda self, *, timeout: authenticated)

        assert ClaudeSdkBackend().check_status() == expected

    @pytest.mark.parametrize("returncode", [pytest.param(0, id="ready"), pytest.param(1, id="not_authenticated")])
    def test_authentication_probe_uses_bundled_cli(
        self,
        monkeypatch: pytest.MonkeyPatch,
        returncode: int,
    ) -> None:
        bundled_cli = sdk_cli_path()
        assert bundled_cli is not None
        assert Path(bundled_cli).is_file()
        captured: dict[str, object] = {}

        def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            captured["argv"] = argv
            captured.update(kwargs)
            return subprocess.CompletedProcess(argv, returncode, stdout="", stderr="")

        monkeypatch.setattr(claude_sdk_module.subprocess, "run", fake_run)

        assert ClaudeSdkBackend().is_authenticated(timeout=7) is (returncode == 0)
        assert captured["argv"] == [bundled_cli, "auth", "status"]
        assert captured["timeout"] == 7

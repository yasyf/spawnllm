from __future__ import annotations

import subprocess

import pytest
from pydantic import BaseModel

from subllm import ClaudeCliBackend, CodexCliBackend, LlmBackends


class TestClaudeArgv:
    def test_non_agent_no_schema(self) -> None:
        assert ClaudeCliBackend().build_command("haiku", None, agent=False) == [
            "claude",
            "-p",
            "--no-session-persistence",
            "--model",
            "haiku",
            "--system-prompt",
            "",
            "--setting-sources",
            "",
            "--strict-mcp-config",
        ]

    def test_agent_with_schema(self) -> None:
        assert ClaudeCliBackend().build_command("opus", '{"a":1}', agent=True) == [
            "claude",
            "-p",
            "--no-session-persistence",
            "--model",
            "opus",
            "--permission-mode",
            "auto",
            "--max-budget-usd",
            "1",
            "--json-schema",
            '{"a":1}',
            "--output-format",
            "json",
        ]

    def test_models_and_env(self) -> None:
        backend = ClaudeCliBackend()
        assert backend.models == {"small": "haiku", "medium": "sonnet", "large": "opus"}
        assert backend.env() == {"CLAUDE_CODE_SIMPLE": "1"}

    def test_parse_response_passthrough_without_model(self) -> None:
        assert ClaudeCliBackend().parse_response("raw text", None) == "raw text"


class TestClaudeInlinePreset:
    def test_inline_argv_no_verbose(self) -> None:
        backend = ClaudeCliBackend.cc_sentiment(system_prompt="SP")
        assert backend.build_argv("hi", model="claude-haiku-4-5") == [
            "claude",
            "-p",
            "hi",
            "--model",
            "claude-haiku-4-5",
            "--system-prompt",
            "SP",
            "--output-format",
            "json",
            "--max-turns",
            "1",
            "--tools",
            "",
            "--disable-slash-commands",
        ]

    def test_inline_argv_verbose_appends_flag(self) -> None:
        backend = ClaudeCliBackend.cc_sentiment(system_prompt="SP", verbose=True)
        assert backend.build_argv("hi", model="m")[-1] == "--verbose"

    def test_parse_result_envelope_returns_result(self) -> None:
        out = ClaudeCliBackend.parse_result_envelope(b'{"is_error": false, "result": "4"}', argv=["claude"], stderr=b"")
        assert out == "4"

    def test_parse_result_envelope_raises_on_is_error(self) -> None:
        raw = b'{"is_error": true, "result": "rate limit"}'
        with pytest.raises(subprocess.CalledProcessError) as exc:
            ClaudeCliBackend.parse_result_envelope(raw, argv=["claude"], stderr=b"e")
        assert exc.value.returncode == 0
        assert exc.value.output == raw
        assert exc.value.stderr == b"e"


class TestCodexArgv:
    def test_non_agent_no_schema(self) -> None:
        assert CodexCliBackend().build_command("gpt-5.5", None, agent=False) == [
            "codex",
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--model",
            "gpt-5.5",
            "-c",
            "features.codex_hooks=false",
            "-c",
            "features.mcp_servers=false",
        ]

    def test_agent_with_schema(self) -> None:
        assert CodexCliBackend().build_command("gpt-5.4-mini", "/tmp/s.json", agent=True) == [
            "codex",
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--model",
            "gpt-5.4-mini",
            "--output-schema",
            "/tmp/s.json",
        ]

    def test_models(self) -> None:
        assert CodexCliBackend().models == {
            "small": "gpt-5.3-codex-spark",
            "medium": "gpt-5.4-mini",
            "large": "gpt-5.5",
        }

    def test_parse_response_validates_model(self) -> None:
        class Verdict(BaseModel):
            block: bool
            reason: str

        result = CodexCliBackend().parse_response('{"block": true, "reason": "bad"}', Verdict)
        assert isinstance(result, Verdict)
        assert result.block is True
        assert result.reason == "bad"


class TestRegistry:
    @pytest.mark.parametrize(
        "specialty, backend_cls",
        [("general", ClaudeCliBackend), ("debugging", CodexCliBackend), ("review", CodexCliBackend)],
        ids=["general-claude", "debugging-codex", "review-codex"],
    )
    def test_for_specialty(self, specialty: str, backend_cls: type) -> None:
        assert isinstance(LlmBackends.for_specialty(specialty), backend_cls)

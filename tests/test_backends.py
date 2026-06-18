from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from spawnllm import AntigravityCliBackend, ClaudeCliBackend, CodexCliBackend, GeminiCliBackend, LlmBackends


class M(BaseModel):
    x: int


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
        assert backend.env() == {}  # CLAUDE_CODE_SIMPLE breaks claude.ai keychain auth

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

    def test_invocation_adds_output_file_with_schema(self) -> None:
        inv = CodexCliBackend().invocation("hi", model="gpt-5.5", schema_path="/tmp/s.json", agent=False)
        assert inv.argv[-2:] == ["-o", inv.result_path]
        assert "--output-schema" in inv.argv
        assert inv.stdin == "hi"
        assert inv.cleanup_paths == (inv.result_path, "/tmp/s.json")
        assert Path(inv.result_path).exists()
        Path(inv.result_path).unlink()

    def test_invocation_no_schema_cleans_only_output_file(self) -> None:
        inv = CodexCliBackend().invocation("hi", model="gpt-5.5", schema_path=None, agent=False)
        assert "--output-schema" not in inv.argv
        assert inv.cleanup_paths == (inv.result_path,)
        Path(inv.result_path).unlink()


class TestRegistry:
    @pytest.mark.parametrize(
        "specialty, backend_cls",
        [("general", ClaudeCliBackend), ("debugging", CodexCliBackend), ("review", CodexCliBackend)],
        ids=["general-claude", "debugging-codex", "review-codex"],
    )
    def test_for_specialty(self, specialty: str, backend_cls: type) -> None:
        assert isinstance(LlmBackends.for_specialty(specialty), backend_cls)


class TestGeminiBackend:
    @pytest.mark.parametrize(
        "agent, expected",
        [
            (
                False,
                ["gemini", "--model", "gemini-2.5-flash", "-o", "json", "--approval-mode", "default", "-e", "none"],
            ),
            (True, ["gemini", "--model", "gemini-2.5-flash", "-o", "json", "--approval-mode", "yolo"]),
        ],
        ids=["non-agent-default-disables-extensions", "agent-yolo-keeps-extensions"],
    )
    def test_build_command(self, agent: bool, expected: list[str]) -> None:
        assert GeminiCliBackend().build_command("gemini-2.5-flash", None, agent=agent) == expected

    def test_invocation_inline_prompt_empty_stdin(self) -> None:
        backend = GeminiCliBackend()
        inv = backend.invocation("hi", model="gemini-2.5-flash", schema_path=None, agent=False)
        assert inv.argv == backend.build_command("gemini-2.5-flash", None, agent=False) + ["-p", "hi"]
        assert inv.stdin == ""
        assert inv.result_path is None

    def test_invocation_injects_schema_into_prompt(self) -> None:
        backend = GeminiCliBackend()
        inv = backend.invocation("hi", model="gemini-2.5-flash", schema_path='{"type":"object"}', agent=False)
        assert inv.argv[-2] == "-p"
        assert "hi" in inv.argv[-1]
        assert '{"type":"object"}' in inv.argv[-1]
        assert inv.stdin == ""

    def test_parse_response_extracts_envelope_text(self) -> None:
        raw = json.dumps({"response": "hello", "stats": {"models": {"gemini-2.5-flash": {"api": {"totalErrors": 0}}}}})
        assert GeminiCliBackend().parse_response(raw, None) == "hello"

    def test_parse_response_raises_on_error_envelope(self) -> None:
        raw = json.dumps({"response": "", "stats": {"models": {"gemini-2.5-flash": {"api": {"totalErrors": 1}}}}})
        with pytest.raises(RuntimeError):
            GeminiCliBackend().parse_response(raw, None)

    def test_parse_response_validates_structured_from_envelope(self) -> None:
        raw = json.dumps({"response": '```json\n{"x": 1}\n```', "stats": {"models": {"g": {"api": {"totalErrors": 0}}}}})
        assert GeminiCliBackend().parse_response(raw, M) == M(x=1)


class TestAntigravityBackend:
    def test_build_command_agent_skips_permissions_with_timeout(self) -> None:
        argv = AntigravityCliBackend().build_command("gemini-3.5", None, agent=True)
        assert "--dangerously-skip-permissions" in argv
        assert "--print-timeout" in argv
        assert "--output-format" not in argv
        assert "-o" not in argv

    def test_build_command_non_agent_omits_skip_permissions(self) -> None:
        assert "--dangerously-skip-permissions" not in AntigravityCliBackend().build_command(
            "gemini-3.5", None, agent=False
        )

    def test_extract_text_strips_whitespace(self) -> None:
        assert AntigravityCliBackend().extract_text("  ok  \n") == "ok"

    def test_parse_response_passthrough(self) -> None:
        assert AntigravityCliBackend().parse_response("ok", None) == "ok"

    def test_parse_response_validates_structured(self) -> None:
        assert AntigravityCliBackend().parse_response('```json\n{"x": 2}\n```', M) == M(x=2)

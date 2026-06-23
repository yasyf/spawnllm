from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from spawnllm import (
    AntigravityCliBackend,
    ClaudeCliBackend,
    ClaudeConfig,
    CodexCliBackend,
    CodexConfig,
    GeminiCliBackend,
    GeminiConfig,
    LlmBackends,
    RunSpec,
)


class M(BaseModel):
    x: int


class TestClaudeArgv:
    def test_lockdown_non_agent(self) -> None:
        assert ClaudeCliBackend().build_command(RunSpec(prompt="hi", model="haiku")) == [
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
        assert ClaudeCliBackend().build_command(
            RunSpec(prompt="hi", model="opus", schema='{"a":1}', agent=True)
        ) == [
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

    def test_full_config_agent_passthrough(self) -> None:
        spec = RunSpec(
            prompt="ping",
            model="opus",
            provider_configs={
                "claude": ClaudeConfig(
                    permission_mode="bypassPermissions",
                    mcp_config="{mcp}",
                    strict_mcp=True,
                    disallowed_tools=("Bash", "Write"),
                    append_system_prompt="extra",
                    settings="{settings}",
                )
            },
        )
        assert ClaudeCliBackend().build_command(spec) == [
            "claude",
            "-p",
            "--no-session-persistence",
            "--model",
            "opus",
            "--permission-mode",
            "bypassPermissions",
            "--mcp-config",
            "{mcp}",
            "--strict-mcp-config",
            "--disallowedTools",
            "Bash",
            "Write",
            "--append-system-prompt",
            "extra",
            "--settings",
            "{settings}",
        ]

    def test_folded_sentiment_shape(self) -> None:
        spec = RunSpec(
            prompt="hi",
            model="haiku",
            provider_configs={
                "claude": ClaudeConfig(
                    system_prompt="SP",
                    max_turns=1,
                    tools="",
                    disable_slash_commands=True,
                    output_format="json",
                )
            },
        )
        assert ClaudeCliBackend().build_command(spec) == [
            "claude",
            "-p",
            "--no-session-persistence",
            "--model",
            "haiku",
            "--system-prompt",
            "SP",
            "--max-turns",
            "1",
            "--tools",
            "",
            "--disable-slash-commands",
            "--output-format",
            "json",
        ]

    def test_folded_sentiment_verbose_appends_flag(self) -> None:
        spec = RunSpec(
            prompt="hi",
            model="haiku",
            provider_configs={"claude": ClaudeConfig(system_prompt="SP", verbose=True)},
        )
        assert ClaudeCliBackend().build_command(spec)[-1] == "--verbose"

    def test_models_and_env(self) -> None:
        backend = ClaudeCliBackend()
        assert backend.models == {"small": "haiku", "medium": "sonnet", "large": "opus"}
        assert backend.env() == {}  # CLAUDE_CODE_SIMPLE breaks claude.ai keychain auth

    def test_invocation_delivers_prompt_over_stdin(self) -> None:
        inv = ClaudeCliBackend().invocation(RunSpec(prompt="hi", model="haiku"))
        assert inv.stdin == "hi"
        assert inv.result_path is None

    def test_parse_response_passthrough_without_model(self) -> None:
        assert ClaudeCliBackend().parse_response("raw text", None) == "raw text"


class TestCodexArgv:
    def test_non_agent_no_schema(self) -> None:
        assert CodexCliBackend().build_command(RunSpec(prompt="hi", model="gpt-5.5")) == [
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

    def test_agent_omits_feature_toggles(self) -> None:
        assert CodexCliBackend().build_command(RunSpec(prompt="hi", model="gpt-5.4-mini", agent=True)) == [
            "codex",
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--model",
            "gpt-5.4-mini",
        ]

    def test_config_overrides_sandbox_and_reenables_features(self) -> None:
        spec = RunSpec(
            prompt="hi",
            model="gpt-5.5",
            provider_configs={"codex": CodexConfig(sandbox="workspace-write", enable_hooks=True, enable_mcp=True)},
        )
        assert CodexCliBackend().build_command(spec) == [
            "codex",
            "exec",
            "--ephemeral",
            "--sandbox",
            "workspace-write",
            "--model",
            "gpt-5.5",
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
        inv = CodexCliBackend().invocation(RunSpec(prompt="hi", model="gpt-5.5", schema='{"type":"object"}'))
        assert inv.argv[-2:] == ["-o", inv.result_path]
        assert "--output-schema" in inv.argv
        assert inv.stdin == "hi"
        schema_path = inv.argv[inv.argv.index("--output-schema") + 1]
        assert inv.cleanup_paths == (inv.result_path, schema_path)
        assert Path(inv.result_path).exists()
        for path in inv.cleanup_paths:
            Path(path).unlink()

    def test_invocation_no_schema_cleans_only_output_file(self) -> None:
        inv = CodexCliBackend().invocation(RunSpec(prompt="hi", model="gpt-5.5"))
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
        assert GeminiCliBackend().build_command(RunSpec(prompt="hi", model="gemini-2.5-flash", agent=agent)) == expected

    def test_build_command_config_overrides_approval_and_extensions(self) -> None:
        spec = RunSpec(
            prompt="hi",
            model="gemini-2.5-flash",
            provider_configs={"gemini": GeminiConfig(approval_mode="auto", extensions=("search", "fs"))},
        )
        assert GeminiCliBackend().build_command(spec) == [
            "gemini",
            "--model",
            "gemini-2.5-flash",
            "-o",
            "json",
            "--approval-mode",
            "auto",
            "-e",
            "search",
            "-e",
            "fs",
        ]

    def test_invocation_inline_prompt_empty_stdin(self) -> None:
        backend = GeminiCliBackend()
        spec = RunSpec(prompt="hi", model="gemini-2.5-flash")
        inv = backend.invocation(spec)
        assert inv.argv == backend.build_command(spec) + ["-p", "hi"]
        assert inv.stdin == ""
        assert inv.result_path is None

    def test_invocation_injects_schema_into_prompt(self) -> None:
        inv = GeminiCliBackend().invocation(
            RunSpec(prompt="hi", model="gemini-2.5-flash", schema='{"type":"object"}')
        )
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
        stats = {"models": {"g": {"api": {"totalErrors": 0}}}}
        raw = json.dumps({"response": '```json\n{"x": 1}\n```', "stats": stats})
        assert GeminiCliBackend().parse_response(raw, M) == M(x=1)


class TestAntigravityBackend:
    def test_build_command_agent_skips_permissions_with_timeout(self) -> None:
        argv = AntigravityCliBackend().build_command(RunSpec(prompt="hi", model="gemini-3.5", agent=True))
        assert argv == ["agy", "--model", "gemini-3.5", "--dangerously-skip-permissions", "--print-timeout", "120s"]

    def test_build_command_non_agent_omits_skip_permissions(self) -> None:
        argv = AntigravityCliBackend().build_command(RunSpec(prompt="hi", model="gemini-3.5"))
        assert argv == ["agy", "--model", "gemini-3.5", "--print-timeout", "120s"]
        assert "--dangerously-skip-permissions" not in argv

    def test_extract_text_strips_whitespace(self) -> None:
        assert AntigravityCliBackend().extract_text("  ok  \n") == "ok"

    def test_parse_response_passthrough(self) -> None:
        assert AntigravityCliBackend().parse_response("ok", None) == "ok"

    def test_parse_response_validates_structured(self) -> None:
        assert AntigravityCliBackend().parse_response('```json\n{"x": 2}\n```', M) == M(x=2)

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

from spawnllm import (
    AntigravityCliBackend,
    ClaudeCliBackend,
    ClaudeConfig,
    CodexCliBackend,
    CodexConfig,
    Error,
    GeminiCliBackend,
    GeminiConfig,
    LlmBackends,
    Output,
    Response,
    RunSpec,
)
from spawnllm.structured import is_transient


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
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--system-prompt",
            "",
        ]

    def test_isolated_false_drops_isolation_flags(self) -> None:
        argv = ClaudeCliBackend().build_command(RunSpec(prompt="hi", model="haiku", isolated=False))
        assert "--setting-sources" not in argv
        assert "--strict-mcp-config" not in argv
        assert argv == [
            "claude",
            "-p",
            "--no-session-persistence",
            "--model",
            "haiku",
            "--system-prompt",
            "",
        ]

    def test_agent_with_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ClaudeCliBackend, "schema_for", lambda self, model: '{"a":1}')
        assert ClaudeCliBackend().build_command(RunSpec(prompt="hi", model="opus", response_model=M, agent=True)) == [
            "claude",
            "-p",
            "--no-session-persistence",
            "--model",
            "opus",
            "--setting-sources",
            "",
            "--strict-mcp-config",
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
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--permission-mode",
            "bypassPermissions",
            "--mcp-config",
            "{mcp}",
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
            "--setting-sources",
            "",
            "--strict-mcp-config",
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

    def test_models(self) -> None:
        assert ClaudeCliBackend().models == {"small": "haiku", "medium": "sonnet", "large": "opus"}

    def test_env_isolates_config_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        (tmp_path / ".claude.json").write_text(
            json.dumps({"oauthAccount": {"accountUuid": "a"}, "mcpServers": {"semble": {}}})
        )
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude" / ".credentials.json").write_text('{"claudeAiOauth": {"accessToken": "tok"}}')
        backend = ClaudeCliBackend()
        env = backend.env(RunSpec(prompt="hi", model="haiku"))
        config_dir = Path(env["CLAUDE_CONFIG_DIR"])
        assert config_dir.is_dir()
        # The dir is created once and cached on the instance, not regenerated per call.
        assert backend.env(RunSpec(prompt="hi", model="haiku"))["CLAUDE_CONFIG_DIR"] == str(config_dir)
        # The account pointer is seeded sans host mcpServers; settings/plugins/hooks never leak.
        assert json.loads((config_dir / ".claude.json").read_text()) == {"oauthAccount": {"accountUuid": "a"}}
        # The OAuth token is seeded so the relocated home stays logged in.
        assert json.loads((config_dir / ".credentials.json").read_text()) == {"claudeAiOauth": {"accessToken": "tok"}}

    def test_env_isolated_seeds_from_claude_config_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # A non-default config home (e.g. cc-pool accounts) holds its own pointer + token; ~ is never read.
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        (account_home := tmp_path / "acct").mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(account_home))
        (account_home / ".claude.json").write_text(
            json.dumps({"oauthAccount": {"accountUuid": "b"}, "mcpServers": {"semble": {}}})
        )
        (account_home / ".credentials.json").write_text('{"claudeAiOauth": {"accessToken": "acct-tok"}}')
        config_dir = Path(ClaudeCliBackend().env(RunSpec(prompt="hi", model="haiku"))["CLAUDE_CONFIG_DIR"])
        assert json.loads((config_dir / ".claude.json").read_text()) == {"oauthAccount": {"accountUuid": "b"}}
        assert json.loads((config_dir / ".credentials.json").read_text()) == {
            "claudeAiOauth": {"accessToken": "acct-tok"}
        }

    def test_env_isolated_falls_back_to_keychain(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (account_home := tmp_path / "acct").mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(account_home))
        (account_home / ".claude.json").write_text(json.dumps({"oauthAccount": {"accountUuid": "c"}}))
        monkeypatch.setattr(sys, "platform", "darwin")
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(argv)
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout='{"claudeAiOauth": {"accessToken": "kc-tok"}}\n'
            )

        monkeypatch.setattr("spawnllm.backends.claude.subprocess.run", fake_run)
        config_dir = Path(ClaudeCliBackend().env(RunSpec(prompt="hi", model="haiku"))["CLAUDE_CONFIG_DIR"])
        # The service name hashes the effective home path, matching the CLI's Keychain item.
        digest = hashlib.sha256(str(account_home).encode()).hexdigest()[:8]
        assert calls == [["security", "find-generic-password", "-s", f"Claude Code-credentials-{digest}", "-w"]]
        credentials = config_dir / ".credentials.json"
        assert json.loads(credentials.read_text()) == {"claudeAiOauth": {"accessToken": "kc-tok"}}
        assert credentials.stat().st_mode & 0o777 == 0o600

    def test_env_isolated_keychain_miss_seeds_no_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (account_home := tmp_path / "acct").mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(account_home))
        (account_home / ".claude.json").write_text(json.dumps({"oauthAccount": {"accountUuid": "d"}}))
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(
            "spawnllm.backends.claude.subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=44, stdout="", stderr="not found"),
        )
        config_dir = Path(ClaudeCliBackend().env(RunSpec(prompt="hi", model="haiku"))["CLAUDE_CONFIG_DIR"])
        # A Keychain miss seeds nothing; the CLI itself then fails loudly as not logged in.
        assert not (config_dir / ".credentials.json").exists()

    def test_env_non_isolated_adds_nothing(self) -> None:
        assert ClaudeCliBackend().env(RunSpec(prompt="hi", model="haiku", isolated=False)) == {}

    def test_invocation_delivers_prompt_over_stdin(self) -> None:
        inv = ClaudeCliBackend().invocation(RunSpec(prompt="hi", model="haiku"))
        assert inv.stdin == "hi"
        assert inv.result_path is None

    def test_result_text_passthrough_for_plain_text(self) -> None:
        assert ClaudeCliBackend().result_text("raw text") == "raw text"

    def test_result_text_reads_envelope_result(self) -> None:
        raw = json.dumps({"type": "result", "is_error": False, "result": "hello"})
        assert ClaudeCliBackend().result_text(raw) == "hello"

    def test_envelope_error_surfaces_message(self) -> None:
        raw = json.dumps({"type": "result", "is_error": True, "result": "Overloaded"})
        assert ClaudeCliBackend().envelope_error(raw) == "Overloaded"

    def test_envelope_error_none_on_success(self) -> None:
        raw = json.dumps({"type": "result", "is_error": False, "result": "ok"})
        assert ClaudeCliBackend().envelope_error(raw) is None

    def test_result_value_extracts_structured_output(self) -> None:
        raw = json.dumps([{"type": "result", "structured_output": {"x": 7}}])
        assert ClaudeCliBackend().result_value(raw) == {"x": 7}


class TestCodexArgv:
    def test_non_agent_no_schema(self) -> None:
        assert CodexCliBackend().build_command(RunSpec(prompt="hi", model="gpt-5.5")) == [
            "codex",
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--model",
            "gpt-5.5",
            "--ignore-user-config",
            "-c",
            "features.hooks=false",
            "-c",
            "features.mcp_servers=false",
        ]

    def test_isolated_false_drops_ignore_user_config(self) -> None:
        argv = CodexCliBackend().build_command(RunSpec(prompt="hi", model="gpt-5.5", isolated=False))
        assert "--ignore-user-config" not in argv
        assert argv == [
            "codex",
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--model",
            "gpt-5.5",
            "-c",
            "features.hooks=false",
            "-c",
            "features.mcp_servers=false",
        ]

    def test_agent_keeps_isolation_omits_feature_toggles(self) -> None:
        assert CodexCliBackend().build_command(RunSpec(prompt="hi", model="gpt-5.4-mini", agent=True)) == [
            "codex",
            "exec",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
            "--model",
            "gpt-5.4-mini",
            "--ignore-user-config",
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
            "--skip-git-repo-check",
            "--model",
            "gpt-5.5",
            "--ignore-user-config",
        ]

    def test_models(self) -> None:
        assert CodexCliBackend().models == {
            "small": "gpt-5.4-mini:low",
            "medium": "gpt-5.4-mini:medium",
            "large": "gpt-5.5:medium",
        }

    def test_reasoning_effort_split_from_model(self) -> None:
        argv = CodexCliBackend().build_command(RunSpec(prompt="hi", model="gpt-5.4-mini:medium"))
        assert argv[argv.index("--model") + 1] == "gpt-5.4-mini"
        assert "model_reasoning_effort=medium" in argv

    def test_bare_model_has_no_effort_flag(self) -> None:
        argv = CodexCliBackend().build_command(RunSpec(prompt="hi", model="gpt-5.5"))
        assert not any(a.startswith("model_reasoning_effort") for a in argv)

    def test_result_value_parses_raw_json(self) -> None:
        assert CodexCliBackend().result_value('{"block": true, "reason": "bad"}') == {"block": True, "reason": "bad"}

    def test_invocation_adds_output_file_with_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(CodexCliBackend, "schema_for", lambda self, model: '{"type":"object"}')
        inv = CodexCliBackend().invocation(RunSpec(prompt="hi", model="gpt-5.5", response_model=M))
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

    def test_select_backend_never_auto_picks_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from spawnllm.backends import registry
        from spawnllm.backends.base import BackendNotInstalled, BackendReady, BackendUnavailable

        def absent(self: object, *, timeout: int = 10) -> BackendNotInstalled:
            return BackendNotInstalled(binary=self.binary, install_hint="x")

        for cls in (ClaudeCliBackend, CodexCliBackend, AntigravityCliBackend):
            monkeypatch.setattr(cls, "check_status", absent)
        monkeypatch.setattr(GeminiCliBackend, "check_status", lambda self, *, timeout=10: BackendReady(binary="gemini"))
        # Gemini is the only "ready" backend, yet auto-selection refuses it.
        with pytest.raises(BackendUnavailable):
            registry.select_backend()


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

    def test_invocation_injects_schema_into_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(GeminiCliBackend, "schema_for", lambda self, model: '{"type":"object"}')
        inv = GeminiCliBackend().invocation(RunSpec(prompt="hi", model="gemini-2.5-flash", response_model=M))
        assert inv.argv[-2] == "-p"
        assert "hi" in inv.argv[-1]
        assert '{"type":"object"}' in inv.argv[-1]
        assert inv.stdin == ""

    def test_result_text_extracts_envelope_text(self) -> None:
        raw = json.dumps({"response": "hello", "stats": {"models": {"gemini-2.5-flash": {"api": {"totalErrors": 0}}}}})
        assert GeminiCliBackend().result_text(raw) == "hello"

    def test_envelope_error_returns_message_on_total_errors(self) -> None:
        raw = json.dumps({"response": "", "stats": {"models": {"gemini-2.5-flash": {"api": {"totalErrors": 1}}}}})
        assert "gemini call failed" in GeminiCliBackend().envelope_error(raw)

    def test_envelope_error_none_on_success(self) -> None:
        raw = json.dumps({"response": "hi", "stats": {"models": {"g": {"api": {"totalErrors": 0}}}}})
        assert GeminiCliBackend().envelope_error(raw) is None

    def test_envelope_error_surfaces_transient_marker_for_retry(self) -> None:
        raw = json.dumps(
            {"response": "", "error": "503 model overloaded", "stats": {"models": {"g": {"api": {"totalErrors": 1}}}}}
        )
        msg = GeminiCliBackend().envelope_error(raw)
        spec = RunSpec(prompt="hi", model="gemini-2.5-flash")
        assert msg is not None and is_transient(
            Response(spec=spec, output=Output(raw), error=Error(msg, RuntimeError(msg)))
        )

    def test_result_value_extracts_json_block_from_envelope(self) -> None:
        stats = {"models": {"g": {"api": {"totalErrors": 0}}}}
        raw = json.dumps({"response": '```json\n{"x": 1}\n```', "stats": stats})
        assert M.model_validate(GeminiCliBackend().result_value(raw)) == M(x=1)

    def test_env_never_isolates(self) -> None:
        # Gemini reads settings + OAuth from one config home with no isolation flag, so it can't be isolated.
        assert GeminiCliBackend().env(RunSpec(prompt="hi", model="gemini-2.5-flash")) == {}


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

    def test_result_text_strips_whitespace(self) -> None:
        assert AntigravityCliBackend().result_text("  ok  \n") == "ok"

    def test_result_value_validates_structured(self) -> None:
        assert M.model_validate(AntigravityCliBackend().result_value('```json\n{"x": 2}\n```')) == M(x=2)

    def test_env_never_isolates(self) -> None:
        # agy has no config-home override and entangles auth/onboarding with its config dir, so it never relocates.
        assert AntigravityCliBackend().env(RunSpec(prompt="hi", model="gemini-3.5")) == {}


class TestSchemaOrModel:
    def test_raw_schema_dict_dumped_into_claude_argv_verbatim(self) -> None:
        spec = RunSpec(prompt="hi", model="haiku", schema={"type": "object", "x": 1})
        argv = ClaudeCliBackend().build_command(spec)
        i = argv.index("--json-schema")
        assert json.loads(argv[i + 1]) == {"type": "object", "x": 1}
        assert argv[i + 2 : i + 4] == ["--output-format", "json"]

    def test_raw_schema_string_passes_through_verbatim(self) -> None:
        spec = RunSpec(prompt="hi", model="haiku", schema='{"raw":true}')
        argv = ClaudeCliBackend().build_command(spec)
        assert argv[argv.index("--json-schema") + 1] == '{"raw":true}'

    def test_raw_schema_goes_into_codex_output_schema_file(self) -> None:
        spec = RunSpec(prompt="hi", model="gpt-5.5", schema={"type": "object"})
        inv = CodexCliBackend().invocation(spec)
        schema_path = inv.argv[inv.argv.index("--output-schema") + 1]
        assert json.loads(Path(schema_path).read_text()) == {"type": "object"}
        for path in inv.cleanup_paths:
            Path(path).unlink()

    def test_raw_schema_injected_into_gemini_prompt(self) -> None:
        inv = GeminiCliBackend().invocation(RunSpec(prompt="hi", model="gemini-2.5-flash", schema={"type": "object"}))
        assert '{"type": "object"}' in inv.argv[-1]

    def test_schema_and_response_model_together_raise(self) -> None:
        with pytest.raises(ValueError, match="either response_model or schema"):
            RunSpec(prompt="hi", model="haiku", schema={"type": "object"}, response_model=M)

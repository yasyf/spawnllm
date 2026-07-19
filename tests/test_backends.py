from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from spawnllm import (
    AntigravityCliBackend,
    BackendReady,
    ClaudeCliBackend,
    ClaudeConfig,
    ClaudeSdkBackend,
    CodexCliBackend,
    CodexConfig,
    GeminiCliBackend,
    GeminiConfig,
    LlmBackends,
    OpenAiEndpointBackend,
    RunSpec,
    _core,
    call_sync,
    extract_sync,
)
from spawnllm.backends import base
from spawnllm.proc import RunResult

ENDPOINT = "http://local.test/v1"


class M(BaseModel):
    x: int


class TestWireSpec:
    """The only guard on the RunSpec -> portable wire mapping the core plan/resolve ops consume."""

    def test_defaults_serialize_with_null_schema_and_absent_configs(self) -> None:
        assert ClaudeCliBackend().wire_spec(RunSpec(prompt="hi", model="haiku")) == {
            "prompt": "hi",
            "model": "haiku",
            "schema": None,
            "agent": False,
            "isolated": True,
            "api_auth": False,
            "timeout": 180,
            "max_attempts": 5,
            "claude": None,
            "codex": None,
            "gemini": None,
            "openai_endpoint": None,
        }

    def test_every_non_default_field_reaches_the_wire(self) -> None:
        spec = RunSpec(
            prompt="do the thing",
            model="opus",
            schema={"type": "object", "properties": {}, "additionalProperties": False},
            agent=True,
            isolated=False,
            api_auth=True,
            timeout=42,
            max_attempts=2,
            provider_configs={"claude": ClaudeConfig(tools=("Bash", "Read"))},
        )
        assert ClaudeCliBackend().wire_spec(spec) == {
            "prompt": "do the thing",
            "model": "opus",
            "schema": {"type": "object", "properties": {}, "additionalProperties": False},
            "agent": True,
            "isolated": False,
            "api_auth": True,
            "timeout": 42,
            "max_attempts": 2,
            "claude": dataclasses.asdict(ClaudeConfig(tools=("Bash", "Read"))),
            "codex": None,
            "gemini": None,
            "openai_endpoint": None,
        }

    def test_tools_none_versus_empty_tuple_is_preserved(self) -> None:
        # Assert the JSON the core actually consumes: None -> null, () -> [].
        def wire(cfg: ClaudeConfig) -> dict:
            spec = RunSpec(prompt="hi", model="haiku", provider_configs={"claude": cfg})
            return json.loads(json.dumps(ClaudeCliBackend().wire_spec(spec)))

        assert wire(ClaudeConfig())["claude"]["tools"] is None
        assert wire(ClaudeConfig(tools=()))["claude"]["tools"] == []

    def test_config_section_is_dataclass_asdict(self) -> None:
        spec = RunSpec(
            prompt="hi",
            model="gpt-5.5",
            provider_configs={"codex": CodexConfig(sandbox="workspace-write", service_tier="standard")},
        )
        assert CodexCliBackend().wire_spec(spec)["codex"] == {
            "sandbox": "workspace-write",
            "enable_hooks": False,
            "enable_mcp": False,
            "service_tier": "standard",
            "developer_instructions": None,
        }

    def test_gemini_config_distinguishes_nil_from_empty_extensions(self) -> None:
        spec = RunSpec(
            prompt="hi",
            model="gemini-2.5-flash",
            provider_configs={"gemini": GeminiConfig(approval_mode="auto", extensions=())},
        )
        gemini = json.loads(json.dumps(GeminiCliBackend().wire_spec(spec)))["gemini"]
        assert gemini == {"approval_mode": "auto", "extensions": []}

    def test_response_model_becomes_dialect_strict_schema(self) -> None:
        schema = CodexCliBackend().wire_spec(RunSpec(prompt="hi", model="gpt-5.5", response_model=M))["schema"]
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["x"]

    def test_raw_dict_schema_passes_through_as_object(self) -> None:
        spec = RunSpec(prompt="hi", model="haiku", schema={"type": "object", "x": 1})
        assert ClaudeCliBackend().wire_spec(spec)["schema"] == {"type": "object", "x": 1}

    def test_raw_string_schema_parsed_to_object(self) -> None:
        spec = RunSpec(prompt="hi", model="haiku", schema='{"raw": true}')
        assert ClaudeCliBackend().wire_spec(spec)["schema"] == {"raw": True}

    def test_openai_endpoint_section_set_only_by_owning_backend(self) -> None:
        assert ClaudeCliBackend().wire_spec(RunSpec(prompt="hi", model="haiku"))["openai_endpoint"] is None
        endpoint = OpenAiEndpointBackend(ENDPOINT, "q", api_key="sk").wire_spec(RunSpec(prompt="hi", model="q"))
        assert endpoint["openai_endpoint"] == {"api_key": "sk", "base_url": ENDPOINT, "model": "q"}

    def test_schema_and_response_model_together_raise(self) -> None:
        with pytest.raises(ValueError, match="either response_model or schema"):
            RunSpec(prompt="hi", model="haiku", schema={"type": "object"}, response_model=M)


class TestInvocationMaterialization:
    """The core plans argv/files/env; the host materializes temp files and wires the result source."""

    def test_claude_mints_stdout_file_and_registers_it_for_cleanup(self) -> None:
        inv = ClaudeCliBackend().invocation(RunSpec(prompt="hi", model="haiku", isolated=False))
        assert inv.stdin == "hi"
        assert inv.result_path is None
        assert inv.stdout_path is not None and Path(inv.stdout_path).exists()
        assert inv.cleanup_paths == (inv.stdout_path,)
        for path in inv.cleanup_paths:
            Path(path).unlink()

    def test_codex_mints_schema_file_with_content_and_substitutes_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(CodexCliBackend, "schema_for", lambda self, model: '{"type": "object"}')
        inv = CodexCliBackend().invocation(RunSpec(prompt="hi", model="gpt-5.5", response_model=M))
        assert "${file:" not in " ".join(inv.argv)
        assert inv.argv[-2:] == ["-o", inv.result_path]
        schema_path = inv.argv[inv.argv.index("--output-schema") + 1]
        assert json.loads(Path(schema_path).read_text()) == {"type": "object"}
        assert Path(inv.result_path).exists()
        assert set(inv.cleanup_paths) == {schema_path, inv.result_path}
        for path in inv.cleanup_paths:
            Path(path).unlink()

    def test_codex_no_schema_cleans_only_the_result_file(self) -> None:
        inv = CodexCliBackend().invocation(RunSpec(prompt="hi", model="gpt-5.5"))
        assert "--output-schema" not in inv.argv
        assert inv.result_path is not None
        assert inv.cleanup_paths == (inv.result_path,)
        Path(inv.result_path).unlink()

    def test_raw_dict_schema_written_into_codex_output_schema_file(self) -> None:
        inv = CodexCliBackend().invocation(RunSpec(prompt="hi", model="gpt-5.5", schema={"type": "object"}))
        schema_path = inv.argv[inv.argv.index("--output-schema") + 1]
        assert json.loads(Path(schema_path).read_text()) == {"type": "object"}
        for path in inv.cleanup_paths:
            Path(path).unlink()

    def test_gemini_delivers_prompt_inline_with_empty_stdin_and_no_files(self) -> None:
        inv = GeminiCliBackend().invocation(RunSpec(prompt="hi", model="gemini-2.5-flash"))
        assert inv.argv[-2:] == ["-p", "hi"]
        assert inv.stdin == ""
        assert inv.result_path is None
        assert inv.stdout_path is None
        assert inv.cleanup_paths == ()

    def test_gemini_injects_schema_into_the_inline_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(GeminiCliBackend, "schema_for", lambda self, model: '{"type": "object"}')
        inv = GeminiCliBackend().invocation(RunSpec(prompt="hi", model="gemini-2.5-flash", response_model=M))
        assert inv.argv[-2] == "-p"
        assert "hi" in inv.argv[-1]
        assert json.loads(inv.argv[-1].splitlines()[-1]) == {"type": "object"}


class TestCliEnvironment:
    def test_execute_strips_planned_keys_with_two_plan_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "inherited-key")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "inherited-token")
        monkeypatch.setenv("SPAWNLLM_KEEP", "kept")
        backend = ClaudeCliBackend()
        original_core_plan = backend.core_plan
        plan_calls = 0
        captured: dict[str, object] = {}

        def counted_core_plan(spec: RunSpec) -> dict[str, object]:
            nonlocal plan_calls
            plan_calls += 1
            return original_core_plan(spec)

        def fake_capture_cli(argv: list[str], **kwargs: object) -> RunResult:
            captured.update(kwargs)
            return RunResult(json.dumps({"type": "result", "is_error": False, "result": "ok"}), "", 0)

        monkeypatch.setattr(backend, "core_plan", counted_core_plan)
        monkeypatch.setattr(base, "capture_cli", fake_capture_cli)
        backend.execute(RunSpec(prompt="hi", model="haiku", isolated=False))

        env = captured["env"]
        assert isinstance(env, dict)
        assert "ANTHROPIC_API_KEY" not in env
        assert "ANTHROPIC_AUTH_TOKEN" not in env
        assert env["SPAWNLLM_KEEP"] == "kept"
        assert plan_calls == 2

    async def test_aexecute_api_auth_preserves_parent_env_with_two_plan_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "inherited-key")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "inherited-token")
        backend = ClaudeCliBackend()
        original_core_plan = backend.core_plan
        plan_calls = 0
        captured: dict[str, object] = {}

        def counted_core_plan(spec: RunSpec) -> dict[str, object]:
            nonlocal plan_calls
            plan_calls += 1
            return original_core_plan(spec)

        async def fake_acapture_cli(argv: list[str], **kwargs: object) -> RunResult:
            captured.update(kwargs)
            return RunResult(json.dumps({"type": "result", "is_error": False, "result": "ok"}), "", 0)

        monkeypatch.setattr(backend, "core_plan", counted_core_plan)
        monkeypatch.setattr(base, "acapture_cli", fake_acapture_cli)
        await backend.aexecute(RunSpec(prompt="hi", model="haiku", isolated=False, api_auth=True))

        env = captured["env"]
        assert isinstance(env, dict)
        assert env["ANTHROPIC_API_KEY"] == "inherited-key"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "inherited-token"
        assert plan_calls == 2

    def test_execute_explicit_env_restores_stripped_key_with_two_plan_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "inherited-key")
        backend = ClaudeCliBackend()
        original_core_plan = backend.core_plan
        plan_calls = 0
        captured: dict[str, object] = {}

        def counted_core_plan(spec: RunSpec) -> dict[str, object]:
            nonlocal plan_calls
            plan_calls += 1
            return original_core_plan(spec)

        def fake_capture_cli(argv: list[str], **kwargs: object) -> RunResult:
            captured.update(kwargs)
            return RunResult(json.dumps({"type": "result", "is_error": False, "result": "ok"}), "", 0)

        monkeypatch.setattr(backend, "core_plan", counted_core_plan)
        monkeypatch.setattr(base, "capture_cli", fake_capture_cli)
        backend.execute(
            RunSpec(
                prompt="hi",
                model="haiku",
                isolated=False,
                env={"ANTHROPIC_API_KEY": "explicit-key"},
            )
        )

        env = captured["env"]
        assert isinstance(env, dict)
        assert env["ANTHROPIC_API_KEY"] == "explicit-key"
        assert plan_calls == 2


class TestClaudeIsolation:
    def test_env_isolates_and_seeds_config_dir_from_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
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
        # The token is substituted in place of the plan's ${isolated_config_dir} placeholder.
        assert "${isolated_config_dir}" not in env["CLAUDE_CONFIG_DIR"]
        # The account pointer is seeded sans host mcpServers; the OAuth token comes along.
        assert json.loads((config_dir / ".claude.json").read_text()) == {"oauthAccount": {"accountUuid": "a"}}
        assert json.loads((config_dir / ".credentials.json").read_text()) == {"claudeAiOauth": {"accessToken": "tok"}}

    def test_env_seeds_from_claude_config_dir_over_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        (account_home := tmp_path / "acct").mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(account_home))
        (account_home / ".claude.json").write_text(json.dumps({"oauthAccount": {"accountUuid": "b"}}))
        (account_home / ".credentials.json").write_text('{"claudeAiOauth": {"accessToken": "acct-tok"}}')
        config_dir = Path(ClaudeCliBackend().env(RunSpec(prompt="hi", model="haiku"))["CLAUDE_CONFIG_DIR"])
        assert json.loads((config_dir / ".claude.json").read_text()) == {"oauthAccount": {"accountUuid": "b"}}
        assert json.loads((config_dir / ".credentials.json").read_text()) == {
            "claudeAiOauth": {"accessToken": "acct-tok"}
        }

    def test_env_falls_back_to_keychain_for_credentials(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (account_home := tmp_path / "acct").mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(account_home))
        (account_home / ".claude.json").write_text(json.dumps({"oauthAccount": {"accountUuid": "c"}}))
        monkeypatch.setattr("spawnllm.backends.claude.sys.platform", "darwin")
        calls: list[list[str]] = []

        def fake_run(argv: list[str], **kwargs: object) -> object:
            calls.append(argv)
            return type("P", (), {"returncode": 0, "stdout": '{"claudeAiOauth": {"accessToken": "kc-tok"}}\n'})()

        monkeypatch.setattr("spawnllm.backends.claude.subprocess.run", fake_run)
        config_dir = Path(ClaudeCliBackend().env(RunSpec(prompt="hi", model="haiku"))["CLAUDE_CONFIG_DIR"])
        # The service name hashes the effective home path, matching the CLI's Keychain item.
        digest = hashlib.sha256(str(account_home).encode()).hexdigest()[:8]
        assert calls == [["security", "find-generic-password", "-s", f"Claude Code-credentials-{digest}", "-w"]]
        credentials = config_dir / ".credentials.json"
        assert json.loads(credentials.read_text()) == {"claudeAiOauth": {"accessToken": "kc-tok"}}
        assert credentials.stat().st_mode & 0o777 == 0o600

    def test_env_keychain_miss_seeds_no_credentials(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (account_home := tmp_path / "acct").mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(account_home))
        (account_home / ".claude.json").write_text(json.dumps({"oauthAccount": {"accountUuid": "d"}}))
        monkeypatch.setattr("spawnllm.backends.claude.sys.platform", "darwin")
        monkeypatch.setattr(
            "spawnllm.backends.claude.subprocess.run",
            lambda *a, **k: type("P", (), {"returncode": 44, "stdout": ""})(),
        )
        config_dir = Path(ClaudeCliBackend().env(RunSpec(prompt="hi", model="haiku"))["CLAUDE_CONFIG_DIR"])
        assert not (config_dir / ".credentials.json").exists()

    def test_env_non_isolated_adds_nothing(self) -> None:
        assert ClaudeCliBackend().env(RunSpec(prompt="hi", model="haiku", isolated=False)) == {}


class TestModels:
    def test_claude(self) -> None:
        assert ClaudeCliBackend().models == {"small": "haiku", "medium": "sonnet", "large": "opus"}

    def test_codex(self) -> None:
        assert CodexCliBackend().models == {
            "small": "gpt-5.4-mini:low",
            "medium": "gpt-5.4-mini:medium",
            "large": "gpt-5.5:medium",
        }

    def test_openai_endpoint_pins_every_tier(self) -> None:
        assert OpenAiEndpointBackend(ENDPOINT, "m").models == {"small": "m", "medium": "m", "large": "m"}


class TestRegistry:
    @pytest.mark.parametrize(
        "specialty, backend_cls",
        [("general", ClaudeSdkBackend), ("debugging", CodexCliBackend), ("review", CodexCliBackend)],
        ids=["general-claude-sdk", "debugging-codex", "review-codex"],
    )
    def test_for_specialty(self, specialty: str, backend_cls: type) -> None:
        assert isinstance(LlmBackends.for_specialty(specialty), backend_cls)

    def test_select_backend_never_auto_picks_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from spawnllm.backends import registry
        from spawnllm.backends.base import BackendNotInstalled, BackendReady, BackendUnavailable

        def absent(self: object, *, timeout: int = 10) -> BackendNotInstalled:
            return BackendNotInstalled(binary=self.binary, install_hint="x")

        for cls in (ClaudeSdkBackend, ClaudeCliBackend, CodexCliBackend, AntigravityCliBackend):
            monkeypatch.setattr(cls, "check_status", absent)
        monkeypatch.setattr(GeminiCliBackend, "check_status", lambda self, *, timeout=10: BackendReady(binary="gemini"))
        with pytest.raises(BackendUnavailable):
            registry.select_backend()

    def test_native_tables_match_the_capabilities_op(self) -> None:
        from spawnllm.backends.registry import BACKENDS_BY_NAME, PRIORITY, LlmBackends

        caps = _core.dispatch("capabilities")
        core_providers = set(caps["providers"])
        native_backends = {name: backend for name, backend in BACKENDS_BY_NAME.items() if name in core_providers}
        specialty_aliases = {"claude-sdk": "claude"}
        assert set(BACKENDS_BY_NAME) - core_providers == {"claude-sdk"}
        assert caps["providers"] == list(native_backends)
        assert caps["priority"] == [backend.provider for backend in PRIORITY if backend.provider in core_providers]
        assert caps["specialties"] == {
            specialty: specialty_aliases.get(backend.provider, backend.provider)
            for specialty, backend in LlmBackends.LLM_BACKENDS.items()
        }
        assert caps["models"] == {name: dict(backend.models) for name, backend in native_backends.items()}
        assert caps["binaries"] == {name: backend.binary for name, backend in native_backends.items()}
        assert caps["install_hints"] == {name: backend.install_hint for name, backend in native_backends.items()}
        assert caps["auto_select_excludes"] == ["gemini"]
        assert caps["api_key_vars"]["claude"] == ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"]


def completion(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def mock_transport(monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    client, aclient = httpx.Client, httpx.AsyncClient
    override = {"transport": httpx.MockTransport(handler)}
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(**(kw | override)))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: aclient(**(kw | override)))


class TestOpenAiEndpointBackend:
    def test_provider_is_openai_endpoint(self) -> None:
        assert OpenAiEndpointBackend(ENDPOINT, "q").provider == "openai_endpoint"

    def test_env_adds_nothing(self) -> None:
        assert OpenAiEndpointBackend(ENDPOINT, "q").env(RunSpec(prompt="p", model="q")) == {}

    def test_is_authenticated_and_check_status_ready(self) -> None:
        backend = OpenAiEndpointBackend(ENDPOINT, "q")
        assert backend.is_authenticated(timeout=1) is True
        assert backend.check_status() == BackendReady(binary="openai_endpoint")

    def test_execute_posts_to_chat_completions_and_reads_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=completion("pong"))

        mock_transport(monkeypatch, handler)
        # A trailing slash on base_url is stripped before the path is appended.
        resp = OpenAiEndpointBackend(ENDPOINT + "/", "qwen3").execute(RunSpec(prompt="ping", model="ignored"))
        assert seen["url"] == "http://local.test/v1/chat/completions"
        assert seen["auth"] == "Bearer local"
        assert seen["body"] == {"model": "qwen3", "messages": [{"role": "user", "content": "ping"}]}
        assert resp.error is None
        assert resp.result.raw == "pong"
        assert resp.result.parsed is None

    async def test_aexecute_posts_and_reads_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_transport(monkeypatch, lambda _request: httpx.Response(200, json=completion("pong")))
        resp = await OpenAiEndpointBackend(ENDPOINT, "qwen3").aexecute(RunSpec(prompt="ping", model="q"))
        assert resp.result.raw == "pong"

    async def test_aexecute_routes_through_injected_transport(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(200, json=completion("pong"))

        backend = OpenAiEndpointBackend(ENDPOINT, "qwen3", transport=httpx.MockTransport(handler))
        resp = await backend.aexecute(RunSpec(prompt="ping", model="q"))
        assert seen["url"] == "http://local.test/v1/chat/completions"
        assert resp.result.raw == "pong"

    def test_execute_structured_embeds_strict_schema_and_validates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            json_schema = body["response_format"]["json_schema"]
            assert body["response_format"]["type"] == "json_schema"
            assert json_schema["strict"] is True
            assert json_schema["schema"]["additionalProperties"] is False
            return httpx.Response(200, json=completion(json.dumps({"x": 7})))

        mock_transport(monkeypatch, handler)
        resp = OpenAiEndpointBackend(ENDPOINT, "q").execute(RunSpec(prompt="hi", model="q", response_model=M))
        assert resp.error is None
        assert resp.result.parsed == M(x=7)

    def test_text_payload_has_no_response_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=completion("hi"))

        mock_transport(monkeypatch, handler)
        OpenAiEndpointBackend(ENDPOINT, "q").execute(RunSpec(prompt="hi", model="q"))
        assert "response_format" not in seen["body"]

    def test_http_error_routes_through_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_transport(monkeypatch, lambda _request: httpx.Response(503, text="overloaded"))
        resp = OpenAiEndpointBackend(ENDPOINT, "q").execute(RunSpec(prompt="p", model="q"))
        assert resp.result is None
        assert "503" in resp.error.msg

    def test_error_body_on_200_routes_through_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_transport(monkeypatch, lambda _request: httpx.Response(200, json={"error": {"message": "no such model"}}))
        resp = OpenAiEndpointBackend(ENDPOINT, "q").execute(RunSpec(prompt="p", model="q"))
        assert resp.result is None
        assert resp.error.msg == "no such model"

    def test_call_sync_drives_text_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["model"] == "qwen3"
            return httpx.Response(200, json=completion("hello"))

        mock_transport(monkeypatch, handler)
        assert call_sync("hi", backend=OpenAiEndpointBackend(ENDPOINT, "qwen3")) == "hello"

    def test_extract_sync_drives_structured_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_transport(monkeypatch, lambda _request: httpx.Response(200, json=completion(json.dumps({"x": 9}))))
        assert extract_sync("hi", M, backend=OpenAiEndpointBackend(ENDPOINT, "qwen3")) == M(x=9)

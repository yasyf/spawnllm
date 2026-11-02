from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from spawnllm import ClaudeCliBackend, CodexCliBackend, RunSpec, _core
from spawnllm.backends import base
from spawnllm.backends.base import ClaudeIsolation
from spawnllm.proc import RunResult


@pytest.mark.parametrize("backend_type", [ClaudeCliBackend, CodexCliBackend])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("isolated", [False, True], ids=["host", "isolated"])
async def test_one_expensive_plan_per_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend_type: type[base.CliBackend],
    asynchronous: bool,
    isolated: bool,
) -> None:
    class Answer(BaseModel):
        schema_calls: ClassVar[int] = 0
        x: int

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            cls.schema_calls += 1
            return super().model_json_schema(*args, **kwargs)

    if expected_source := os.environ.get("SPAWNLLM_EXPECTED_SOURCE"):
        assert Path(base.__file__).resolve().is_relative_to(Path(expected_source).resolve())
    backend = backend_type()
    spec = RunSpec(
        prompt="private-plan-fixture",
        model="sonnet" if backend_type is ClaudeCliBackend else "gpt-5.5",
        response_model=Answer,
        isolated=isolated,
        api_auth=True,
        env={"SPAWNLLM_KEEP": "explicit"},
        cwd=str(tmp_path),
        timeout=42,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-parent")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "synthetic-auth")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "synthetic-inherited")
    monkeypatch.setenv("SPAWNLLM_KEEP", "parent")
    original_dispatch = _core.dispatch
    plan_calls = 0
    isolation_calls = 0
    captures: list[dict[str, Any]] = []
    minted: list[str] = []
    original_mkstemp = base.tempfile.mkstemp

    def dispatch(op: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        nonlocal plan_calls
        if op == "plan":
            plan_calls += 1
        return original_dispatch(op, payload)

    def isolation(api_auth: bool) -> ClaudeIsolation:
        nonlocal isolation_calls
        assert api_auth is True
        isolation_calls += 1
        return ClaudeIsolation(str(tmp_path / "config"), {"CLAUDE_CODE_OAUTH_TOKEN": f"synthetic-{isolation_calls}"})

    def mkstemp(*args: Any, **kwargs: Any) -> tuple[int, str]:
        fd, path = original_mkstemp(*args, **kwargs)
        minted.append(path)
        return fd, path

    def capture(argv: list[str], **kwargs: Any) -> RunResult:
        paths = minted.copy()
        normalized = []
        for value in argv:
            for index, path in enumerate(paths):
                value = value.replace(path, f"<file:{index}>")
            normalized.append(value.replace(str(tmp_path), "<fixture>"))
        content = [Path(path).read_text() for path in paths]
        if backend_type is ClaudeCliBackend and isolated:
            assert kwargs["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == f"synthetic-{2 * len(captures) + 1}"
        captures.append(
            {
                "argv": normalized,
                "stdin": kwargs["input"],
                "env": {
                    key: (
                        "<fresh-first-token>"
                        if key == "CLAUDE_CODE_OAUTH_TOKEN"
                        else value.replace(str(tmp_path), "<fixture>")
                    )
                    for key, value in kwargs["env"].items()
                    if key in {"ANTHROPIC_API_KEY", "SPAWNLLM_KEEP", "CLAUDE_CONFIG_DIR", "CLAUDE_CODE_OAUTH_TOKEN"}
                },
                "cwd": "<fixture>" if kwargs["cwd"] == str(tmp_path) else kwargs["cwd"],
                "timeout": kwargs["timeout"],
                "files": content,
                "stdout_file": kwargs["stdout_path"] in paths,
            }
        )
        if backend_type is CodexCliBackend:
            Path(argv[argv.index("-o") + 1]).write_text('{"x":7}')
            return RunResult("", "", 0)
        return RunResult(json.dumps({"type": "result", "is_error": False, "result": '{"x":7}'}), "", 0)

    async def acapture(argv: list[str], **kwargs: Any) -> RunResult:
        return capture(argv, **kwargs)

    monkeypatch.setattr(_core, "dispatch", dispatch)
    monkeypatch.setattr(backend, "claude_isolation", isolation)
    monkeypatch.setattr(base.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(base, "capture_cli", capture)
    monkeypatch.setattr(base, "acapture_cli", acapture)
    results: list[dict[str, Any]] = []
    for _ in range(32):
        minted.clear()
        response = await backend.aexecute(spec) if asynchronous else backend.execute(spec)
        assert response.error is None
        assert response.result.parsed == Answer(x=7)
        assert all(not Path(path).exists() for path in minted)
        results.append({"raw": response.output.raw, "parsed": response.result.parsed.model_dump()})
    expected_isolation = 64 if backend_type is ClaudeCliBackend and isolated else 0
    assert isolation_calls == expected_isolation
    assert len(captures) == len(results) == 32
    assert all(row == captures[0] for row in captures)
    observation = {
        "backend": backend.provider,
        "asynchronous": asynchronous,
        "isolated": isolated,
        "captures": captures,
        "results": results,
        "isolation_calls": isolation_calls,
    }
    if destination := os.environ.get("SPAWNLLM_PLAN_OBSERVATIONS"):
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{backend.provider}-{asynchronous}-{isolated}.json").write_text(
            json.dumps(observation, sort_keys=True) + "\n"
        )
    print(
        f"PLAN_COUNTS backend={backend.provider} async={asynchronous} isolated={isolated} "
        f"plans={plan_calls} schemas={Answer.schema_calls} attempts=32"
    )
    assert plan_calls == 32, f"duplicate expensive planning: actual={plan_calls} expected=32"
    assert Answer.schema_calls == 32, f"duplicate schema generation: actual={Answer.schema_calls} expected=32"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_failed_schema_does_not_survive_the_attempt(monkeypatch: pytest.MonkeyPatch, asynchronous: bool) -> None:
    class Answer(BaseModel):
        fail: ClassVar[bool] = True
        schema_calls: ClassVar[int] = 0
        x: int

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            cls.schema_calls += 1
            if cls.fail:
                raise ValueError("private-schema-failure")
            return super().model_json_schema(*args, **kwargs)

    backend = ClaudeCliBackend()
    spec = RunSpec(prompt="private", model="sonnet", response_model=Answer, isolated=False)
    captures = 0

    def capture(*args: Any, **kwargs: Any) -> RunResult:
        nonlocal captures
        captures += 1
        return RunResult(json.dumps({"type": "result", "is_error": False, "result": '{"x":7}'}), "", 0)

    async def acapture(*args: Any, **kwargs: Any) -> RunResult:
        return capture(*args, **kwargs)

    monkeypatch.setattr(base, "capture_cli", capture)
    monkeypatch.setattr(base, "acapture_cli", acapture)
    with pytest.raises(ValueError, match="^private-schema-failure$"):
        if asynchronous:
            await backend.aexecute(spec)
        else:
            backend.execute(spec)
    assert captures == 0
    assert Answer.schema_calls == 1
    Answer.fail = False
    response = await backend.aexecute(spec) if asynchronous else backend.execute(spec)
    assert response.error is None
    assert response.result.parsed == Answer(x=7)
    assert captures == 1
    assert Answer.schema_calls == 2


async def test_concurrent_calls_resolve_their_own_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    backend = ClaudeCliBackend()
    original_dispatch = _core.dispatch
    plan_prompts: list[str] = []
    captured: list[str] = []

    def dispatch(op: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if op == "plan":
            plan_prompts.append(payload["spec"]["prompt"])
        return original_dispatch(op, payload)

    async def capture(argv: list[str], **kwargs: Any) -> RunResult:
        await asyncio.sleep(0)
        captured.append(kwargs["input"])
        return RunResult(json.dumps({"type": "result", "is_error": False, "result": kwargs["input"]}), "", 0)

    monkeypatch.setattr(_core, "dispatch", dispatch)
    monkeypatch.setattr(base, "acapture_cli", capture)
    specs = [RunSpec(prompt=f"private-{index}", model="sonnet", isolated=False) for index in range(32)]
    responses = await asyncio.gather(*(backend.aexecute(spec) for spec in specs))
    assert plan_prompts == [spec.prompt for spec in specs]
    assert captured == [spec.prompt for spec in specs]
    assert all(response.error is None for response in responses)
    assert [response.result.raw for response in responses] == captured


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_public_invocation_env_mutation_is_not_an_execution_overlay(
    monkeypatch: pytest.MonkeyPatch, asynchronous: bool
) -> None:
    class InvocationEnvBackend(CodexCliBackend):
        def invocation(self, spec: RunSpec) -> base.Invocation:
            inv = super().invocation(spec)
            inv.env["SPAWNLLM_INVOCATION_ONLY"] = "private"
            return inv

    backend = InvocationEnvBackend()
    spec = RunSpec(prompt="private", model="gpt-5.5", isolated=False)
    monkeypatch.delenv("SPAWNLLM_INVOCATION_ONLY", raising=False)
    captured = 0

    def capture(argv: list[str], **kwargs: Any) -> RunResult:
        nonlocal captured
        captured += 1
        assert "SPAWNLLM_INVOCATION_ONLY" not in kwargs["env"]
        Path(argv[argv.index("-o") + 1]).write_text("private-result")
        return RunResult("", "", 0)

    async def acapture(argv: list[str], **kwargs: Any) -> RunResult:
        return capture(argv, **kwargs)

    monkeypatch.setattr(base, "capture_cli", capture)
    monkeypatch.setattr(base, "acapture_cli", acapture)
    response = await backend.aexecute(spec) if asynchronous else backend.execute(spec)
    assert captured == 1
    assert response.error is None
    assert response.result.raw == "private-result"

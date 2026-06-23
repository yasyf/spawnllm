from __future__ import annotations

import importlib
import subprocess
from typing import TYPE_CHECKING

import pytest

from spawnllm import RunResult, RunSpec, run, run_sync
from spawnllm.backends.base import LlmBackend
from spawnllm.proc import arun_cli, map_concurrent, run_cli

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.backends.base import BackendStatus

RUN_MODULE = importlib.import_module("spawnllm.run")

NONZERO_OVERLOADED = RunResult(stdout="", stderr="API Error: 529 Overloaded", returncode=1)
ENVELOPE_OVERLOADED = RunResult(stdout='{"is_error": true, "result": "Overloaded"}', stderr="", returncode=0)
SUCCESS = RunResult(stdout='{"is_error": false, "result": "ok"}', stderr="", returncode=0)


class ScriptedBackend(LlmBackend):
    models = {}
    provider = "claude"

    def __init__(self, results: list[RunResult]) -> None:
        self.results = results
        self.attempts = 0

    def _next(self) -> RunResult:
        result = self.results[self.attempts]
        self.attempts += 1
        return result

    async def aexecute(self, spec: RunSpec) -> RunResult:
        return self._next()

    def execute(self, spec: RunSpec) -> RunResult:
        return self._next()

    def parse_response(self, raw: str, response_model: type[BaseModel] | None) -> str | BaseModel:
        return raw

    def env(self) -> dict[str, str]:
        return {}

    def is_authenticated(self, *, timeout: int) -> bool:
        return True

    def check_status(self, *, timeout: int = 10) -> BackendStatus:
        raise NotImplementedError


class TestRunCli:
    def test_success_returns_stdout(self) -> None:
        assert "hello" in run_cli(["echo", "hello"])

    def test_raises_with_str_attrs_and_notes(self) -> None:
        with pytest.raises(subprocess.CalledProcessError) as exc:
            run_cli(["sh", "-c", "echo out; echo err >&2; exit 3"])
        err = exc.value
        assert err.returncode == 3
        assert isinstance(err.stderr, str) and "err" in err.stderr
        assert isinstance(err.output, str) and "out" in err.output
        notes = getattr(err, "__notes__", [])
        assert any(n.startswith("argv:") for n in notes)
        assert any(n.startswith("exit_code:") for n in notes)
        assert any("err" in n for n in notes if n.startswith("stderr:"))
        assert any("out" in n for n in notes if n.startswith("stdout:"))

    def test_cwd_passthrough(self, tmp_path) -> None:
        assert run_cli(["pwd"], cwd=str(tmp_path)).strip() == str(tmp_path)


class TestArunCli:
    async def test_returns_stdout_bytes(self) -> None:
        assert await arun_cli(["echo", "hi"]) == b"hi\n"

    async def test_raises_bytes_on_failure(self) -> None:
        argv = ["sh", "-c", "echo out; echo err >&2; exit 2"]
        with pytest.raises(subprocess.CalledProcessError) as exc:
            await arun_cli(argv)
        err = exc.value
        assert err.returncode == 2
        assert err.cmd == argv
        assert err.output == b"out\n"
        assert err.stderr == b"err\n"

    async def test_stdin_passthrough(self) -> None:
        assert await arun_cli(["cat"], input="piped") == b"piped"

    async def test_stderr_tee_receives_lines(self) -> None:
        seen: list[bytes] = []
        await arun_cli(["sh", "-c", "echo a >&2; echo b >&2"], stderr_tee=seen.append)
        assert b"".join(seen) == b"a\nb\n"


class TestMapConcurrent:
    async def test_preserves_order_and_counts_completions(self) -> None:
        async def double(x: int) -> int:
            return x * 2

        done: list[int] = []
        result = await map_concurrent([1, 2, 3, 4], double, limit=2, on_done=done.append)
        assert result == [2, 4, 6, 8]
        assert done == [1, 1, 1, 1]


SPEC = RunSpec(prompt="hi", model="haiku", max_attempts=3)


class TestRetry:
    @pytest.mark.parametrize(
        "transient",
        [NONZERO_OVERLOADED, ENVELOPE_OVERLOADED],
        ids=["nonzero-exit-stderr", "zero-exit-error-envelope"],
    )
    async def test_async_retries_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, transient: RunResult
    ) -> None:
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr(RUN_MODULE.asyncio, "sleep", fake_sleep)
        backend = ScriptedBackend([transient, SUCCESS])
        assert await run(SPEC, backend=backend) is SUCCESS
        assert backend.attempts == 2
        assert slept == [5.0]

    @pytest.mark.parametrize(
        "transient",
        [NONZERO_OVERLOADED, ENVELOPE_OVERLOADED],
        ids=["nonzero-exit-stderr", "zero-exit-error-envelope"],
    )
    def test_sync_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch, transient: RunResult) -> None:
        slept: list[float] = []
        monkeypatch.setattr(RUN_MODULE.time, "sleep", slept.append)
        backend = ScriptedBackend([transient, SUCCESS])
        assert run_sync(SPEC, backend=backend) is SUCCESS
        assert backend.attempts == 2
        assert slept == [5.0]

    async def test_async_all_fail_returns_last_without_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr(RUN_MODULE.asyncio, "sleep", fake_sleep)
        last = RunResult(stdout="", stderr="529 Overloaded again", returncode=1)
        backend = ScriptedBackend([NONZERO_OVERLOADED, ENVELOPE_OVERLOADED, last])
        assert await run(SPEC, backend=backend) is last
        assert backend.attempts == 3

    def test_sync_all_fail_returns_last_without_raising(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(RUN_MODULE.time, "sleep", lambda _: None)
        last = RunResult(stdout="", stderr="529 Overloaded again", returncode=1)
        backend = ScriptedBackend([NONZERO_OVERLOADED, ENVELOPE_OVERLOADED, last])
        assert run_sync(SPEC, backend=backend) is last
        assert backend.attempts == 3


class TestMlxBackend:
    async def test_aexecute_returns_generated_text(self) -> None:
        from spawnllm.backends.mlx import MlxBackend

        class FakeEngine:
            def __init__(self) -> None:
                self.calls: list[object] = []

            async def ensure_loaded(self) -> None:
                self.calls.append("loaded")

            async def generate(self, message_lists, on_progress, *, max_tokens):  # noqa: ANN001
                self.calls.append((message_lists, max_tokens))
                return ["pong"]

        engine = FakeEngine()
        result = await MlxBackend(engine, max_tokens=64).aexecute(RunSpec(prompt="ping", model="local"))
        assert result == RunResult("pong", "", 0)
        assert engine.calls == ["loaded", ([[{"role": "user", "content": "ping"}]], 64)]

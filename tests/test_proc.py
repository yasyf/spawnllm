from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import pytest

from spawnllm import ClaudeCliBackend, DiscardedAttempt, Error, Output, Response, Result, RunSpec, run, run_sync
from spawnllm.backends.base import LlmBackend
from spawnllm.proc import acapture_cli, arun_cli, capture_cli, map_concurrent, run_cli

if TYPE_CHECKING:
    from spawnllm.backends.base import BackendStatus

RUN_MODULE = importlib.import_module("spawnllm.run")
PROC_MODULE = importlib.import_module("spawnllm.proc")

SPEC = RunSpec(prompt="hi", model="haiku", max_attempts=3)
TRANSIENT = Response(
    spec=SPEC, output=Output("529"), error=Error("codex exited 1: API Error: 529 Overloaded", RuntimeError("529"))
)
TRANSIENT_2 = Response(
    spec=SPEC, output=Output("rl"), error=Error("claude reported an error: rate limit", RuntimeError("rl"))
)
SUCCESS = Response(spec=SPEC, output=Output("ok"), result=Result(raw="ok"))
CLAUDE_COST_ENVELOPE = json.dumps(
    {
        "type": "result",
        "is_error": True,
        "result": "rate limit exceeded",
        "total_cost_usd": 0.02,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
)
TRANSIENT_WITH_COST = Response(
    spec=SPEC,
    output=Output(CLAUDE_COST_ENVELOPE),
    error=Error("claude reported an error: rate limit exceeded", RuntimeError("rl")),
)


def patch_plan_argv(monkeypatch: pytest.MonkeyPatch, backend: ClaudeCliBackend, argv: list[str]) -> None:
    core_plan = backend.core_plan
    monkeypatch.setattr(backend, "core_plan", lambda spec: core_plan(spec) | {"argv": argv})


class ScriptedBackend(LlmBackend):
    models = {}
    provider = "claude"

    def __init__(self, results: list[Response]) -> None:
        self.results = results
        self.attempts = 0

    def _next(self) -> Response:
        result = self.results[self.attempts]
        self.attempts += 1
        return result

    async def aexecute(self, spec: RunSpec) -> Response:
        return self._next()

    def execute(self, spec: RunSpec) -> Response:
        return self._next()

    def env(self, spec: RunSpec) -> dict[str, str]:
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

    async def test_stderr_newline_free_blob_past_64k(self) -> None:
        # A newline-free stderr blob past the 64 KiB readline limit is captured whole,
        # not dropped to a LimitOverrunError mid-drain.
        blob = "e" * 200_000
        seen: list[bytes] = []
        argv = [sys.executable, "-c", f"import sys; sys.stderr.write({blob!r}); sys.exit(4)"]
        with pytest.raises(subprocess.CalledProcessError) as exc:
            await arun_cli(argv, stderr_tee=seen.append)
        assert exc.value.returncode == 4
        assert exc.value.stderr == blob.encode()
        assert b"".join(seen) == blob.encode()


class TestMapConcurrent:
    async def test_preserves_order_and_counts_completions(self) -> None:
        async def double(x: int) -> int:
            return x * 2

        done: list[int] = []
        result = await map_concurrent([1, 2, 3, 4], double, limit=2, on_done=done.append)
        assert result == [2, 4, 6, 8]
        assert done == [1, 1, 1, 1]


class TestFileBackedStdout:
    LARGE = 100_000  # a single write past the 64 KiB pipe-drain boundary

    async def test_acapture_captures_full_large_single_write(self, tmp_path) -> None:
        # The child tags its output by fd-1 kind, so a regression to PIPE capture fails loudly.
        payload = "x" * self.LARGE
        script = (
            "import os, stat, sys; "
            f"sys.stdout.write(('REG:' if stat.S_ISREG(os.fstat(1).st_mode) else 'PIPE:') + {payload!r}); "
            "sys.stdout.flush()"
        )
        rr = await acapture_cli([sys.executable, "-c", script], stdout_path=str(tmp_path / "out"))
        assert rr.returncode == 0
        assert rr.stdout == "REG:" + payload

    def test_capture_captures_full_large_single_write(self, tmp_path) -> None:
        payload = "y" * self.LARGE
        script = (
            "import os, stat, sys; "
            f"sys.stdout.write(('REG:' if stat.S_ISREG(os.fstat(1).st_mode) else 'PIPE:') + {payload!r}); "
            "sys.stdout.flush()"
        )
        rr = capture_cli([sys.executable, "-c", script], stdout_path=str(tmp_path / "out"))
        assert rr.returncode == 0
        assert rr.stdout == "REG:" + payload

    async def test_acapture_still_captures_stderr_with_file_backed_stdout(self, tmp_path) -> None:
        script = "import sys; sys.stdout.write('out'); sys.stderr.write('err'); sys.exit(3)"
        rr = await acapture_cli([sys.executable, "-c", script], stdout_path=str(tmp_path / "out"))
        assert rr.stdout == "out"
        assert rr.stderr == "err"
        assert rr.returncode == 3

    async def test_claude_aexecute_file_backs_large_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = "z" * self.LARGE
        backend = ClaudeCliBackend()
        script = f"import sys; sys.stdout.write({payload!r}); sys.stdout.flush()"
        patch_plan_argv(monkeypatch, backend, [sys.executable, "-c", script])
        resp = await backend.aexecute(RunSpec(prompt="hi", model="haiku", isolated=False))
        assert resp.error is None
        assert resp.output.raw == payload

    def test_claude_execute_file_backs_large_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = "w" * self.LARGE
        backend = ClaudeCliBackend()
        script = f"import sys; sys.stdout.write({payload!r}); sys.stdout.flush()"
        patch_plan_argv(monkeypatch, backend, [sys.executable, "-c", script])
        resp = backend.execute(RunSpec(prompt="hi", model="haiku", isolated=False))
        assert resp.error is None
        assert resp.output.raw == payload


class TestFileBackedStdin:
    # The child tags its stdin by fd-0 kind, so a regression to a pipe (whose write the
    # parent must schedule after the spawn) fails loudly on the REG: assertion.
    KIND_ECHO = (
        "import os, stat, sys; "
        "sys.stdout.write(('REG:' if stat.S_ISREG(os.fstat(0).st_mode) else 'PIPE:') + sys.stdin.read())"
    )

    async def test_acapture_feeds_stdin_as_regular_file(self) -> None:
        rr = await acapture_cli([sys.executable, "-c", self.KIND_ECHO], input="ping")
        assert rr.returncode == 0
        assert rr.stdout == "REG:ping"

    async def test_arun_feeds_stdin_as_regular_file(self) -> None:
        assert await arun_cli([sys.executable, "-c", self.KIND_ECHO], input="ping") == b"REG:ping"

    async def test_acapture_large_stdin_survives_pipe_boundary(self) -> None:
        payload = "s" * 100_000  # a single feed past the 64 KiB pipe-drain boundary
        rr = await acapture_cli([sys.executable, "-c", self.KIND_ECHO], input=payload)
        assert rr.stdout == "REG:" + payload


class TestStdinSurvivesSaturatedLoop:
    """The child aborts if stdin has no data within `deadline`s of exec, as `claude --print` does.

    A file-backed stdin is complete at exec, so the child clears its deadline even when a
    saturated event loop cannot run the parent's post-spawn steps for `BLOCK` seconds.
    """

    DEADLINE = 0.5
    BLOCK = 1.0
    FAKE_CLI = (
        "import os, select, stat, sys\n"
        "deadline = float(sys.argv[1])\n"
        "kind = 'REG:' if stat.S_ISREG(os.fstat(0).st_mode) else 'PIPE:'\n"
        "if not select.select([sys.stdin], [], [], deadline)[0]:\n"
        "    sys.stderr.write('no stdin data received')\n"
        "    sys.exit(1)\n"
        "sys.stdout.write(kind + sys.stdin.read())\n"
    )

    def saturate_after_spawn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_create = PROC_MODULE.asyncio.create_subprocess_exec

        async def saturated_create(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
            proc = await real_create(*args, **kwargs)
            time.sleep(self.BLOCK)  # the loop is jammed past the child's stdin deadline before the parent resumes
            return proc

        monkeypatch.setattr(PROC_MODULE.asyncio, "create_subprocess_exec", saturated_create)

    async def test_acapture_stdin_survives_loop_block_after_spawn(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        cli = tmp_path / "deadline_cli.py"
        cli.write_text(self.FAKE_CLI)
        self.saturate_after_spawn(monkeypatch)
        rr = await acapture_cli([sys.executable, str(cli), str(self.DEADLINE)], input="payload")
        assert rr.returncode == 0
        assert rr.stdout == "REG:payload"

    async def test_arun_stdin_survives_loop_block_after_spawn(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        cli = tmp_path / "deadline_cli.py"
        cli.write_text(self.FAKE_CLI)
        self.saturate_after_spawn(monkeypatch)
        assert await arun_cli([sys.executable, str(cli), str(self.DEADLINE)], input="payload") == b"REG:payload"


class TestCliBackendEnvironment:
    async def test_aexecute_uses_overridden_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class CustomEnvBackend(ClaudeCliBackend):
            def env(self, spec: RunSpec) -> dict[str, str]:
                return {"CUSTOM_KEY": "custom-value"}

        backend = CustomEnvBackend()
        script = "import os; print(os.environ.get('CUSTOM_KEY'))"
        patch_plan_argv(monkeypatch, backend, [sys.executable, "-c", script])
        resp = await backend.aexecute(RunSpec(prompt="hi", model="haiku", isolated=False))
        assert resp.error is None
        assert resp.output.raw == "custom-value\n"

    def test_execute_uses_overridden_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class CustomEnvBackend(ClaudeCliBackend):
            def env(self, spec: RunSpec) -> dict[str, str]:
                return {"CUSTOM_KEY": "custom-value"}

        backend = CustomEnvBackend()
        script = "import os; print(os.environ.get('CUSTOM_KEY'))"
        patch_plan_argv(monkeypatch, backend, [sys.executable, "-c", script])
        resp = backend.execute(RunSpec(prompt="hi", model="haiku", isolated=False))
        assert resp.error is None
        assert resp.output.raw == "custom-value\n"


class TestTimeoutTermination:
    SLEEPER = "import os, time; open({path!r}, 'w').write(str(os.getpid())); time.sleep(30)"

    async def test_acapture_timeout_terminates_and_reaps_child(self, tmp_path) -> None:
        pid_file = tmp_path / "pid"
        script = self.SLEEPER.format(path=str(pid_file))
        with pytest.raises(TimeoutError):
            await acapture_cli([sys.executable, "-c", script], timeout=1, stdout_path=str(tmp_path / "out"))
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    def test_capture_sync_timeout_terminates_and_reaps_child(self, tmp_path) -> None:
        # subprocess.run already kills+reaps on timeout; this locks that in for the file-backed path.
        pid_file = tmp_path / "pid"
        script = self.SLEEPER.format(path=str(pid_file))
        with pytest.raises(subprocess.TimeoutExpired):
            capture_cli([sys.executable, "-c", script], timeout=1, stdout_path=str(tmp_path / "out"))
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_claude_aexecute_timeout_kills_child_and_cleans_tempfile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        pid_file = tmp_path / "pid"
        backend = ClaudeCliBackend()
        script = self.SLEEPER.format(path=str(pid_file))
        patch_plan_argv(monkeypatch, backend, [sys.executable, "-c", script])
        captured: dict[str, str] = {}
        real_invocation = backend.invocation

        def spy(spec):
            inv = real_invocation(spec)
            captured["stdout_path"] = inv.stdout_path
            return inv

        monkeypatch.setattr(backend, "invocation", spy)
        resp = await backend.aexecute(RunSpec(prompt="hi", model="haiku", isolated=False, timeout=1))
        assert isinstance(resp.error.ex, TimeoutError)
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not os.path.exists(captured["stdout_path"])


class ScriptedClaudeBackend(ScriptedBackend):
    """A scripted backend that parses discarded-attempt cost the way `claude` does."""

    def accounting(self, raw: str) -> tuple[float | None, dict[str, object] | None]:
        return ClaudeCliBackend().accounting(raw)


class TestRetry:
    @pytest.mark.parametrize(
        "transient",
        [TRANSIENT, TRANSIENT_2],
        ids=["exit-529-error", "rate-limit-error"],
    )
    async def test_async_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch, transient: Response) -> None:
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr(RUN_MODULE.asyncio, "sleep", fake_sleep)
        backend = ScriptedBackend([transient, SUCCESS])
        resp = await run(SPEC, backend=backend)
        assert resp.result is SUCCESS.result
        assert resp.output is SUCCESS.output
        assert resp.error is None
        assert backend.attempts == 2
        assert slept == [5.0]
        assert [(d.attempt, d.error, d.raw_bytes) for d in resp.discarded_attempts] == [
            (0, "RuntimeError", len(transient.output.raw.encode()))
        ]

    @pytest.mark.parametrize(
        "transient",
        [TRANSIENT, TRANSIENT_2],
        ids=["exit-529-error", "rate-limit-error"],
    )
    def test_sync_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch, transient: Response) -> None:
        slept: list[float] = []
        monkeypatch.setattr(RUN_MODULE.time, "sleep", slept.append)
        backend = ScriptedBackend([transient, SUCCESS])
        resp = run_sync(SPEC, backend=backend)
        assert resp.result is SUCCESS.result
        assert resp.output is SUCCESS.output
        assert resp.error is None
        assert backend.attempts == 2
        assert slept == [5.0]
        assert [d.attempt for d in resp.discarded_attempts] == [0]

    async def test_async_discarded_attempt_carries_cost_and_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr(RUN_MODULE.asyncio, "sleep", fake_sleep)
        backend = ScriptedClaudeBackend([TRANSIENT_WITH_COST, SUCCESS])
        resp = await run(SPEC, backend=backend)
        assert resp.result is SUCCESS.result
        assert resp.discarded_attempts == (
            DiscardedAttempt(
                attempt=0,
                error="RuntimeError",
                cost_usd=0.02,
                usage={"input_tokens": 10, "output_tokens": 2},
                raw_bytes=len(CLAUDE_COST_ENVELOPE.encode()),
            ),
        )

    async def test_async_accounting_failure_degrades_not_aborts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr(RUN_MODULE.asyncio, "sleep", fake_sleep)

        class BoomBackend(ScriptedBackend):
            def accounting(self, raw: str) -> tuple[float | None, dict[str, object] | None]:
                raise ValueError("malformed envelope")

        backend = BoomBackend([TRANSIENT, SUCCESS])
        resp = await run(SPEC, backend=backend)
        assert resp.result is SUCCESS.result
        assert len(resp.discarded_attempts) == 1
        d = resp.discarded_attempts[0]
        assert d.cost_usd is None
        assert d.usage is None
        assert d.raw_bytes == len(TRANSIENT.output.raw.encode())

    async def test_async_discarded_raw_bytes_counts_utf8_not_code_points(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr(RUN_MODULE.asyncio, "sleep", fake_sleep)
        multibyte = Response(
            spec=SPEC,
            output=Output("café"),  # 4 code points, 5 UTF-8 bytes
            error=Error("claude reported an error: rate limit", RuntimeError("rl")),
        )
        backend = ScriptedBackend([multibyte, SUCCESS])
        resp = await run(SPEC, backend=backend)
        assert resp.discarded_attempts[0].raw_bytes == 5

    async def test_async_all_fail_returns_last_with_earlier_attempts_discarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_sleep(seconds: float) -> None:
            return None

        monkeypatch.setattr(RUN_MODULE.asyncio, "sleep", fake_sleep)
        last = Response(
            spec=SPEC, output=Output("529"), error=Error("codex exited 1: 529 Overloaded again", RuntimeError("529"))
        )
        backend = ScriptedBackend([TRANSIENT, TRANSIENT_2, last])
        resp = await run(SPEC, backend=backend)
        assert resp.error is last.error
        assert resp.output is last.output
        assert resp.result is None
        assert backend.attempts == 3
        assert [d.attempt for d in resp.discarded_attempts] == [0, 1]

    def test_sync_all_fail_returns_last_with_earlier_attempts_discarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(RUN_MODULE.time, "sleep", lambda _: None)
        last = Response(
            spec=SPEC, output=Output("529"), error=Error("codex exited 1: 529 Overloaded again", RuntimeError("529"))
        )
        backend = ScriptedBackend([TRANSIENT, TRANSIENT_2, last])
        resp = run_sync(SPEC, backend=backend)
        assert resp.error is last.error
        assert backend.attempts == 3
        assert [d.attempt for d in resp.discarded_attempts] == [0, 1]


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
        spec = RunSpec(prompt="ping", model="local")
        result = await MlxBackend(engine, max_tokens=64).aexecute(spec)
        assert result == Response(spec=spec, output=Output("pong"), result=Result(raw="pong"))
        assert engine.calls == ["loaded", ([[{"role": "user", "content": "ping"}]], 64)]

from __future__ import annotations

import subprocess

import pytest

from spawnllm.proc import arun_cli, map_concurrent, run_cli


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

"""Subprocess transport for CLI-backed LLM calls (sync ``run_cli`` + async ``arun_cli``)."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Awaitable, Callable, Sequence

__all__ = ["arun_cli", "collect_process", "map_concurrent", "run_cli"]


def run_cli(
    argv: list[str],
    *,
    input: str | None = None,
    timeout: int = 30,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> str:
    result = subprocess.run(
        argv,
        input=input,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=cwd,
    )
    if result.returncode != 0:
        err = subprocess.CalledProcessError(result.returncode, argv, output=result.stdout, stderr=result.stderr)
        err.add_note(f"argv: {argv}")
        err.add_note(f"exit_code: {result.returncode}")
        err.add_note(f"stderr: {result.stderr[-4096:]}")
        err.add_note(f"stdout: {result.stdout[-4096:]}")
        raise err
    return result.stdout


async def collect_process(
    proc: asyncio.subprocess.Process,
    *,
    stderr_tee: Callable[[bytes], None] | None = None,
) -> tuple[bytes, bytes, int]:
    assert proc.stderr is not None, "create_subprocess_exec was called with stderr=PIPE"
    assert proc.stdout is not None, "create_subprocess_exec was called with stdout=PIPE"
    stderr_buf = bytearray()
    async with asyncio.TaskGroup() as tg:
        tg.create_task(_tee_stderr(proc.stderr, stderr_buf, stderr_tee))
        stdout_task = tg.create_task(proc.stdout.read())
        rc_task = tg.create_task(proc.wait())
    return stdout_task.result(), bytes(stderr_buf), rc_task.result()


async def _tee_stderr(
    stream: asyncio.StreamReader,
    buf: bytearray,
    stderr_tee: Callable[[bytes], None] | None,
) -> None:
    async for raw in stream:
        buf.extend(raw)
        if stderr_tee is not None:
            stderr_tee(raw)


async def arun_cli(
    argv: list[str],
    *,
    input: str | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    stderr_tee: Callable[[bytes], None] | None = None,
) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if input is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=cwd,
    )
    if input is not None:
        assert proc.stdin is not None, "create_subprocess_exec was called with stdin=PIPE"
        proc.stdin.write(input.encode())
        await proc.stdin.drain()
        proc.stdin.close()
    stdout, stderr, rc = await collect_process(proc, stderr_tee=stderr_tee)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, argv, output=stdout, stderr=stderr)
    return stdout


async def map_concurrent[T, R](
    items: Sequence[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    limit: int,
    on_done: Callable[[int], None] | None = None,
) -> list[R]:
    sem = asyncio.Semaphore(limit)

    async def one(item: T) -> R:
        async with sem:
            result = await fn(item)
        if on_done is not None:
            on_done(1)
        return result

    return list(await asyncio.gather(*(one(item) for item in items)))

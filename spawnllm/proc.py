"""Subprocess transport for CLI-backed LLM calls (sync `run_cli` + async `arun_cli`)."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

__all__ = ["RunResult", "acapture_cli", "arun_cli", "capture_cli", "collect_process", "map_concurrent", "run_cli"]


@dataclass(frozen=True, slots=True)
class RunResult:
    """The raw outcome of a CLI invocation.

    Attributes:
        stdout: The decoded stdout.
        stderr: The decoded stderr.
        returncode: The process exit code.
    """

    stdout: str
    stderr: str
    returncode: int


def run_cli(
    argv: list[str],
    *,
    input: str | None = None,
    timeout: int = 30,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> str:
    """Run a CLI command to completion and return its stdout.

    Args:
        argv: The command and its arguments.
        input: Text delivered to the process over stdin.
        timeout: Seconds to wait before the process is killed.
        env: Environment for the process; `None` inherits the current environment.
        cwd: Working directory for the process.

    Returns:
        The decoded stdout.

    Raises:
        subprocess.CalledProcessError: On a nonzero exit code, with the argv,
            exit code, and stdout/stderr tails attached as notes.
        subprocess.TimeoutExpired: When the process outlives `timeout`.
    """
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


def capture_cli(
    argv: list[str],
    *,
    input: str | None = None,
    timeout: int = 180,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> RunResult:
    """Run a CLI command to completion and capture its full outcome.

    Unlike `run_cli`, a nonzero exit does not raise; the stdout, stderr, and exit
    code come back intact so callers can inspect failures and 0-exit error envelopes.

    Args:
        argv: The command and its arguments.
        input: Text delivered to the process over stdin.
        timeout: Seconds to wait before the process is killed.
        env: Environment for the process; `None` inherits the current environment.
        cwd: Working directory for the process.

    Returns:
        The captured stdout, stderr, and exit code.

    Raises:
        subprocess.TimeoutExpired: When the process outlives `timeout`.
    """
    result = subprocess.run(
        argv,
        input=input,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=cwd,
    )
    return RunResult(result.stdout, result.stderr, result.returncode)


async def collect_process(
    proc: asyncio.subprocess.Process,
    *,
    stderr_tee: Callable[[bytes], None] | None = None,
) -> tuple[bytes, bytes, int]:
    """Drain a subprocess's stdout and stderr concurrently and wait for it to exit.

    Args:
        proc: A process created with both stdout and stderr piped.
        stderr_tee: Callback invoked with each stderr line as it arrives.

    Returns:
        A `(stdout, stderr, returncode)` tuple.
    """
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
    """Run a CLI command asynchronously and return its stdout.

    Args:
        argv: The command and its arguments.
        input: Text delivered to the process over stdin.
        env: Environment for the process; `None` inherits the current environment.
        cwd: Working directory for the process.
        stderr_tee: Callback invoked with each stderr line as it arrives.

    Returns:
        The raw stdout bytes.

    Raises:
        subprocess.CalledProcessError: On a nonzero exit code.
    """
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


async def acapture_cli(
    argv: list[str],
    *,
    input: str | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: int | None = None,
) -> RunResult:
    """Run a CLI command asynchronously and capture its full outcome.

    Unlike `arun_cli`, a nonzero exit does not raise; the stdout, stderr, and exit
    code come back intact so callers can inspect failures and 0-exit error envelopes.

    Args:
        argv: The command and its arguments.
        input: Text delivered to the process over stdin.
        env: Environment for the process; `None` inherits the current environment.
        cwd: Working directory for the process.
        timeout: Seconds to wait before the wait is abandoned; `None` waits forever.

    Returns:
        The captured stdout, stderr, and exit code.

    Raises:
        TimeoutError: When the process outlives `timeout`.
    """
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
    collect = collect_process(proc)
    stdout, stderr, rc = await (asyncio.wait_for(collect, timeout) if timeout is not None else collect)
    return RunResult(stdout.decode(), stderr.decode(), rc)


async def map_concurrent[T, R](
    items: Sequence[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    limit: int,
    on_done: Callable[[int], None] | None = None,
) -> list[R]:
    """Map an async function over items with bounded concurrency.

    Args:
        items: The inputs to process.
        fn: Async function applied to each item.
        limit: Maximum number of in-flight calls.
        on_done: Progress callback invoked with `1` as each item completes.

    Returns:
        The results, in input order.
    """
    sem = asyncio.Semaphore(limit)

    async def one(item: T) -> R:
        async with sem:
            result = await fn(item)
        if on_done is not None:
            on_done(1)
        return result

    return list(await asyncio.gather(*(one(item) for item in items)))

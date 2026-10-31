from __future__ import annotations

import importlib.util
import importlib
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Callable

import pytest

from spawnllm.proc import run_cli


class Abandoned(BaseException):
    pass


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_inflight_checkpoint_reaps_only_its_owned_child(tmp_path: Path) -> None:
    legacy = os.environ.get("SPAWNLLM_PROC_BEFORE")
    if legacy:
        spec = importlib.util.spec_from_file_location("spawnllm_proc_before", legacy)
        before = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = before
        spec.loader.exec_module(before)
        invoke = before.run_cli
    else:
        invoke = run_cli
    database, pid_path = tmp_path / "blocked.sqlite", tmp_path / "child.pid"
    cancelled = threading.Event()
    errors: list[BaseException] = []
    code = "import os,pathlib,sqlite3,sys; pathlib.Path(sys.argv[2]).write_text(str(os.getpid())); sqlite3.connect(sys.argv[1],timeout=10).execute('BEGIN IMMEDIATE')"
    argv = [sys.executable, "-c", code, str(database), str(pid_path)]

    def checkpoint() -> None:
        if cancelled.is_set():
            raise Abandoned

    def call() -> None:
        try:
            invoke(argv, timeout=10, **({} if legacy else {"checkpoint": checkpoint}))
        except BaseException as error:
            errors.append(error)

    writer = sqlite3.connect(database)
    writer.execute("BEGIN IMMEDIATE")
    worker = threading.Thread(target=call)
    with subprocess.Popen([sys.executable, "-c", "import sys;sys.stdin.read(1)"], stdin=subprocess.PIPE, text=True) as unrelated:
        try:
            worker.start()
            deadline = time.monotonic() + 5
            while not pid_path.exists():
                assert time.monotonic() < deadline, "SQLite-blocked owned child did not start"
                time.sleep(0.01)
            pid = int(pid_path.read_text())
            cancelled.set()
            worker.join(2)
            assert not worker.is_alive(), f"owned child survived request cancellation: pid={pid}"
            assert len(errors) == 1 and isinstance(errors[0], Abandoned)
            assert not alive(pid)
            with pytest.raises(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)
            assert unrelated.poll() is None
        finally:
            writer.rollback()
            writer.close()
            worker.join(12)
            unrelated.communicate(input="x", timeout=5)
        assert not worker.is_alive()


def test_checkpoint_before_spawn_preserves_exception() -> None:
    error = Abandoned()

    def checkpoint() -> None:
        raise error

    with pytest.raises(Abandoned) as raised:
        run_cli(["/does-not-exist"], checkpoint=checkpoint)
    assert raised.value is error


@pytest.mark.parametrize("checkpoint", [None, lambda: None], ids=["default", "checkpoint"])
def test_checkpoint_transport_keeps_input_environment_cwd_and_error_notes(tmp_path: Path, checkpoint: Callable[[], None] | None) -> None:
    argv = [sys.executable, "-c", "import os,sys; print(os.getcwd());print(os.environ['CHECKPOINT_VALUE']);print(sys.stdin.read());print('error-tail',file=sys.stderr);sys.exit(3)"]
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_cli(argv, input="payload", cwd=str(tmp_path), env=os.environ | {"CHECKPOINT_VALUE": "value"}, checkpoint=checkpoint)
    error = raised.value
    assert error.returncode == 3
    assert error.output == f"{tmp_path}\nvalue\npayload\n"
    assert error.stderr == "error-tail\n"
    assert error.__notes__ == [f"argv: {argv}", "exit_code: 3", "stderr: error-tail\n", f"stdout: {tmp_path}\nvalue\npayload\n"]


def test_checkpoint_timeout_preserves_partial_bytes_and_reaps(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    code = "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); print('before',flush=True); print('error',file=sys.stderr,flush=True); time.sleep(10)"
    argv = [sys.executable, "-c", code, str(pid_path)]
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        run_cli(argv, timeout=1, checkpoint=lambda: None)
    assert raised.value.timeout == 1
    assert raised.value.output == b"before\n"
    assert raised.value.stderr == b"error\n"
    pid = int(pid_path.read_text())
    assert not alive(pid)
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)


def test_checkpoint_transport_drains_large_text_input_once() -> None:
    payload = "ü" * 100_000
    argv = [sys.executable, "-c", "import sys,time;time.sleep(0.1);sys.stdout.write(sys.stdin.read())"]
    assert run_cli(argv, input=payload, checkpoint=lambda: None) == payload


def test_checkpoint_rejects_unsupported_platform_before_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = importlib.import_module("spawnllm.proc")
    monkeypatch.setattr(proc, "os", SimpleNamespace(name="nt"))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "default", "")

    def spawn(*args, **kwargs):
        pytest.fail("unsupported checkpoint spawned a process")

    monkeypatch.setattr(proc.subprocess, "run", run)
    monkeypatch.setattr(proc.subprocess, "Popen", spawn)
    assert proc.run_cli(["default"]) == "default"
    with pytest.raises(NotImplementedError, match="require POSIX"):
        proc.run_cli(["unsupported"], checkpoint=lambda: None)
    assert calls == [["default"]]


def test_posix_cancel_reaps_child_without_waiting_for_descendant_pipe_eof(tmp_path: Path) -> None:
    database, pid_path = tmp_path / "blocked.sqlite", tmp_path / "child.pid"
    descendant_path, release = tmp_path / "descendant.pid", tmp_path / "release"
    cancelled = threading.Event()
    errors: list[BaseException] = []
    descendant = "import pathlib,sys,time\nrelease=pathlib.Path(sys.argv[1])\ndeadline=time.monotonic()+10\nwhile not release.exists() and time.monotonic()<deadline:\n time.sleep(0.01)\nrelease.with_suffix('.done').write_text('done')\n"
    code = "import os,pathlib,sqlite3,subprocess,sys; child=subprocess.Popen([sys.executable,'-c',sys.argv[5],sys.argv[4]]);pathlib.Path(sys.argv[3]).write_text(str(child.pid));pathlib.Path(sys.argv[2]).write_text(str(os.getpid()));sqlite3.connect(sys.argv[1],timeout=10).execute('BEGIN IMMEDIATE')"
    argv = [sys.executable, "-c", code, str(database), str(pid_path), str(descendant_path), str(release), descendant]

    def checkpoint() -> None:
        if cancelled.is_set():
            raise Abandoned

    def call() -> None:
        try:
            run_cli(argv, timeout=10, checkpoint=checkpoint)
        except BaseException as error:
            errors.append(error)

    writer = sqlite3.connect(database)
    writer.execute("BEGIN IMMEDIATE")
    worker = threading.Thread(target=call)
    try:
        worker.start()
        deadline = time.monotonic() + 5
        while not pid_path.exists() or not descendant_path.exists():
            assert time.monotonic() < deadline, "owned child and inherited-pipe descendant did not start"
            time.sleep(0.01)
        pid, descendant_pid = int(pid_path.read_text()), int(descendant_path.read_text())
        cancelled.set()
        worker.join(2)
        assert not worker.is_alive(), "direct-child cleanup waited for descendant pipe EOF"
        assert len(errors) == 1 and isinstance(errors[0], Abandoned)
        assert not alive(pid)
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        assert alive(descendant_pid)
    finally:
        release.write_text("release")
        writer.rollback()
        writer.close()
        worker.join(12)
        if descendant_path.exists():
            deadline = time.monotonic() + 5
            while not release.with_suffix(".done").exists():
                assert time.monotonic() < deadline, "owned descendant did not reach cooperative exit"
                time.sleep(0.01)
    assert not worker.is_alive()

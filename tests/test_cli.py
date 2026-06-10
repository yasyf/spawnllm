from __future__ import annotations

import pytest
from click.testing import CliRunner

from spawnllm.cli import main


def test_help_exits_cleanly() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert result.output.startswith("Usage: main")


def test_backends_lists_available() -> None:
    result = CliRunner().invoke(main, ["backends"])
    assert result.exit_code == 0
    assert result.output.splitlines() == ["claude", "codex", "mlx"]


def test_call_dispatches_to_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_call(prompt: str, *, backend, model, agent):
        captured.update(prompt=prompt, backend=type(backend).__name__, model=model, agent=agent)
        return "RESULT"

    monkeypatch.setattr("spawnllm.cli.call_backend", fake_call)
    result = CliRunner().invoke(main, ["call", "--backend", "claude", "hello"])
    assert result.exit_code == 0
    assert result.output == "RESULT\n"
    assert captured == {"prompt": "hello", "backend": "ClaudeCliBackend", "model": "small", "agent": False}

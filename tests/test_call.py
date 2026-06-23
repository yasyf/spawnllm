from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from spawnllm import ClaudeCliBackend, CodexCliBackend, RunResult, call_sync
from spawnllm.backends import base

if TYPE_CHECKING:
    import pytest


class M(BaseModel):
    x: int


def test_codex_text_call_reads_final_message_not_log(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_capture_cli(argv: list[str], **_: object) -> RunResult:
        captured["result"] = argv[argv.index("-o") + 1]
        Path(captured["result"]).write_text("the final answer")
        return RunResult("noisy codex log line 1\nlog line 2\n", "", 0)

    monkeypatch.setattr(base, "capture_cli", fake_capture_cli)

    assert call_sync("hi", backend=CodexCliBackend(), response_model=None) == "the final answer"
    assert not Path(captured["result"]).exists()


def test_codex_structured_call_reads_file_and_cleans_schema_and_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CodexCliBackend, "schema_for", lambda self, model: '{"type": "object"}')
    captured: dict[str, str] = {}

    def fake_capture_cli(argv: list[str], **_: object) -> RunResult:
        captured["schema"] = argv[argv.index("--output-schema") + 1]
        captured["result"] = argv[argv.index("-o") + 1]
        Path(captured["result"]).write_text('{"x": 7}')
        return RunResult("noisy codex log that must be ignored", "", 0)

    monkeypatch.setattr(base, "capture_cli", fake_capture_cli)

    assert call_sync("hi", backend=CodexCliBackend(), response_model=M) == M(x=7)
    assert not Path(captured["schema"]).exists()
    assert not Path(captured["result"]).exists()


def test_call_sync_text_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "capture_cli", lambda argv, **_: RunResult("plain answer", "", 0))
    assert call_sync("hi", backend=ClaudeCliBackend(), response_model=None) == "plain answer"


def test_call_sync_threads_cwd_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_capture_cli(argv: list[str], **kwargs: object) -> RunResult:
        captured.update(kwargs)
        return RunResult("ok", "", 0)

    monkeypatch.setattr(base, "capture_cli", fake_capture_cli)

    assert call_sync("hi", backend=ClaudeCliBackend(), cwd="/tmp/work", timeout=42) == "ok"
    assert captured["cwd"] == "/tmp/work"
    assert captured["timeout"] == 42


def test_call_sync_maps_tier_to_literal_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_capture_cli(argv: list[str], **_: object) -> RunResult:
        captured["argv"] = argv
        return RunResult("ok", "", 0)

    monkeypatch.setattr(base, "capture_cli", fake_capture_cli)

    assert call_sync("hi", backend=ClaudeCliBackend(), model="large") == "ok"
    assert captured["argv"][captured["argv"].index("--model") + 1] == "opus"

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from spawnllm import BackendCallError, ClaudeCliBackend, CodexCliBackend, call_sync, extract_sync
from spawnllm.backends import base
from spawnllm.proc import RunResult


class M(BaseModel):
    x: int


def test_codex_text_call_reads_final_message_not_log(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_capture_cli(argv: list[str], **_: object) -> RunResult:
        captured["result"] = argv[argv.index("-o") + 1]
        Path(captured["result"]).write_text("the final answer")
        return RunResult("noisy codex log line 1\nlog line 2\n", "", 0)

    monkeypatch.setattr(base, "capture_cli", fake_capture_cli)

    assert call_sync("hi", backend=CodexCliBackend()) == "the final answer"
    assert not Path(captured["result"]).exists()


def test_codex_structured_extract_reads_file_and_cleans_schema_and_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CodexCliBackend, "schema_for", lambda self, model: '{"type": "object"}')
    captured: dict[str, str] = {}

    def fake_capture_cli(argv: list[str], **_: object) -> RunResult:
        captured["schema"] = argv[argv.index("--output-schema") + 1]
        captured["result"] = argv[argv.index("-o") + 1]
        Path(captured["result"]).write_text('{"x": 7}')
        return RunResult("noisy codex log that must be ignored", "", 0)

    monkeypatch.setattr(base, "capture_cli", fake_capture_cli)

    assert extract_sync("hi", M, backend=CodexCliBackend()) == M(x=7)
    assert not Path(captured["schema"]).exists()
    assert not Path(captured["result"]).exists()


def test_call_sync_text_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "capture_cli", lambda argv, **_: RunResult("plain answer", "", 0))
    assert call_sync("hi", backend=ClaudeCliBackend()) == "plain answer"


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


def test_failed_codex_surfaces_stderr_not_eof_on_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "capture_cli", lambda argv, **_: RunResult("", "codex: not found", 127))

    with pytest.raises(BackendCallError) as exc:
        call_sync("hi", backend=CodexCliBackend())

    assert "codex: not found" in str(exc.value)
    assert "127" in str(exc.value)
    assert "EOF while parsing" not in str(exc.value)


def test_failed_codex_surfaces_stderr_not_eof_on_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CodexCliBackend, "schema_for", lambda self, model: '{"type": "object"}')

    def fake_capture_cli(argv: list[str], **_: object) -> RunResult:
        # The -o file was created empty by mkstemp and codex died before writing.
        return RunResult("", "codex: not found", 127)

    monkeypatch.setattr(base, "capture_cli", fake_capture_cli)

    with pytest.raises(BackendCallError) as exc:
        extract_sync("hi", M, backend=CodexCliBackend())

    assert "codex: not found" in str(exc.value)
    assert "EOF while parsing" not in str(exc.value)


def test_malformed_json_raises_validation_error_out_of_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CodexCliBackend, "schema_for", lambda self, model: '{"type": "object"}')

    def fake_capture_cli(argv: list[str], **_: object) -> RunResult:
        Path(argv[argv.index("-o") + 1]).write_text('{"x": "not an int"}')
        return RunResult("log", "", 0)

    monkeypatch.setattr(base, "capture_cli", fake_capture_cli)

    with pytest.raises(ValidationError):
        extract_sync("hi", M, backend=CodexCliBackend())

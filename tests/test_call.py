from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from spawnllm import CodexCliBackend, call

if TYPE_CHECKING:
    import pytest

# `spawnllm.call` the attribute resolves to the function, so reach the module explicitly to patch its `run_cli`.
CALL_MODULE = importlib.import_module("spawnllm.call")


class M(BaseModel):
    x: int


def test_codex_text_call_reads_final_message_not_log(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_run_cli(argv: list[str], **_: object) -> str:
        captured["result"] = argv[argv.index("-o") + 1]
        Path(captured["result"]).write_text("the final answer")
        return "noisy codex log line 1\nlog line 2\n"

    monkeypatch.setattr(CALL_MODULE, "run_cli", fake_run_cli)

    assert call("hi", backend=CodexCliBackend(), response_model=None) == "the final answer"
    assert not Path(captured["result"]).exists()


def test_codex_structured_call_reads_file_and_cleans_schema_and_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CodexCliBackend, "schema_for", lambda self, model: '{"type": "object"}')
    captured: dict[str, str] = {}

    def fake_run_cli(argv: list[str], **_: object) -> str:
        captured["schema"] = argv[argv.index("--output-schema") + 1]
        captured["result"] = argv[argv.index("-o") + 1]
        Path(captured["result"]).write_text('{"x": 7}')
        return "noisy codex log that must be ignored"

    monkeypatch.setattr(CALL_MODULE, "run_cli", fake_run_cli)

    assert call("hi", backend=CodexCliBackend(), response_model=M) == M(x=7)
    assert not Path(captured["schema"]).exists()
    assert not Path(captured["result"]).exists()

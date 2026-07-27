from __future__ import annotations

import re

import pytest
from click.testing import CliRunner

from spawnllm import (
    AntigravityCliBackend,
    AppleBackend,
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    ClaudeCliBackend,
    ClaudeSdkBackend,
    CodexCliBackend,
    GeminiCliBackend,
    LlmBackend,
)
from spawnllm.backends.base import BackendStatus
from spawnllm.cli import main


def _patch_statuses(monkeypatch: pytest.MonkeyPatch, statuses: dict[type[LlmBackend], BackendStatus]) -> None:
    for cls, status in statuses.items():
        monkeypatch.setattr(cls, "check_status", lambda self, *, timeout=10, _s=status: _s)


def test_help_exits_cleanly() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert result.output.startswith("Usage: main")


def test_backends_lists_available() -> None:
    result = CliRunner().invoke(main, ["backends"])
    assert result.exit_code == 0
    assert result.output.splitlines() == ["claude-sdk", "claude", "codex", "antigravity", "gemini", "apple", "mlx"]


@pytest.mark.parametrize(
    ("name", "backend_cls"),
    [
        pytest.param("claude-sdk", ClaudeSdkBackend, id="claude-sdk"),
        pytest.param("claude", ClaudeCliBackend, id="claude"),
        pytest.param("gemini", GeminiCliBackend, id="gemini"),
        pytest.param("antigravity", AntigravityCliBackend, id="antigravity"),
    ],
)
def test_call_dispatches_to_backend(monkeypatch: pytest.MonkeyPatch, name: str, backend_cls: type[LlmBackend]) -> None:
    captured: dict[str, object] = {}

    def fake_call(prompt: str, *, backend, model, agent, api_auth):
        captured.update(prompt=prompt, backend=backend, model=model, agent=agent, api_auth=api_auth)
        return "RESULT"

    monkeypatch.setattr("spawnllm.cli.call_sync", fake_call)
    result = CliRunner().invoke(main, ["call", "--backend", name, "--api-auth", "hello"])
    assert result.exit_code == 0
    assert result.output == "RESULT\n"
    assert isinstance(captured["backend"], backend_cls)
    assert captured["prompt"] == "hello"
    assert captured["model"] == "small"
    assert captured["agent"] is False
    assert captured["api_auth"] is True


def test_status_reports_per_backend_and_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_statuses(
        monkeypatch,
        {
            ClaudeSdkBackend: BackendReady("claude-sdk"),
            ClaudeCliBackend: BackendReady("claude"),
            CodexCliBackend: BackendNotInstalled(binary="codex", install_hint="npm install -g @openai/codex"),
            AntigravityCliBackend: BackendNotAuthenticated("agy"),
            GeminiCliBackend: BackendNotAuthenticated("gemini"),
            AppleBackend: BackendReady("apple"),
        },
    )
    result = CliRunner().invoke(main, ["status"])
    assert result.exit_code == 0
    *lines, core_line = result.output.splitlines()
    assert lines == [
        "claude-sdk: ready",
        "claude: ready",
        "codex: not installed — install with: npm install -g @openai/codex",
        "agy: not authenticated",
        "gemini: not authenticated",
        "apple: ready",
        "selected: claude-sdk",
    ]
    assert re.fullmatch(r"core: \d+\.\d+\.\d+@[0-9a-f]{12}", core_line)


def test_status_selects_apple_when_it_is_the_only_ready_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_statuses(
        monkeypatch,
        {
            ClaudeSdkBackend: BackendNotAuthenticated("claude-sdk"),
            ClaudeCliBackend: BackendNotAuthenticated("claude"),
            CodexCliBackend: BackendNotAuthenticated("codex"),
            AntigravityCliBackend: BackendNotAuthenticated("agy"),
            GeminiCliBackend: BackendNotAuthenticated("gemini"),
            AppleBackend: BackendReady("apple"),
        },
    )
    result = CliRunner().invoke(main, ["status"])
    assert result.exit_code == 0
    *lines, _ = result.output.splitlines()
    assert lines == [
        "claude-sdk: not authenticated",
        "claude: not authenticated",
        "codex: not authenticated",
        "agy: not authenticated",
        "gemini: not authenticated",
        "apple: ready",
        "selected: apple",
    ]


def test_status_reports_none_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_statuses(
        monkeypatch,
        {
            ClaudeSdkBackend: BackendNotAuthenticated("claude-sdk"),
            ClaudeCliBackend: BackendNotAuthenticated("claude"),
            CodexCliBackend: BackendNotAuthenticated("codex"),
            AntigravityCliBackend: BackendNotAuthenticated("agy"),
            GeminiCliBackend: BackendNotAuthenticated("gemini"),
            AppleBackend: BackendNotAuthenticated("apple"),
        },
    )
    result = CliRunner().invoke(main, ["status"])
    assert result.exit_code == 0
    *lines, core_line = result.output.splitlines()
    assert lines == [
        "claude-sdk: not authenticated",
        "claude: not authenticated",
        "codex: not authenticated",
        "agy: not authenticated",
        "gemini: not authenticated",
        "apple: not authenticated",
        "selected: none available",
    ]
    assert re.fullmatch(r"core: \d+\.\d+\.\d+@[0-9a-f]{12}", core_line)

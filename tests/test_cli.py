from __future__ import annotations

import json
import re

import pytest
from click.testing import CliRunner

from spawnllm import (
    JEV,
    AntigravityCliBackend,
    AppleBackend,
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    Binary,
    ClaudeCliBackend,
    ClaudeSdkBackend,
    CodexCliBackend,
    DecideKeyMissing,
    Decision,
    GeminiCliBackend,
    Label,
    LabelAnswer,
    LlmBackend,
    Refused,
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
            AppleBackend: BackendReady("spawnllm-apple"),
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
        "spawnllm-apple: ready",
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
            AppleBackend: BackendReady("spawnllm-apple"),
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
        "spawnllm-apple: ready",
        "selected: spawnllm-apple",
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
            AppleBackend: BackendNotAuthenticated("spawnllm-apple"),
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
        "spawnllm-apple: not authenticated",
        "selected: none available",
    ]
    assert re.fullmatch(r"core: \d+\.\d+\.\d+@[0-9a-f]{12}", core_line)


def test_decide_asks_the_json_questions_about_its_state_and_prints_each_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: dict[str, object] = {}

    def fake_decide(state, questions, *, provider, timeout):
        asked.update(state=state, questions=questions, provider=provider, timeout=timeout)
        answers = {"decides": LabelAnswer("a1", {"none": 0.1, "a1": 0.9}, 0.8), "urgent": Refused()}
        return Decision(answers, "jev-1.13.0", 42, 120.0)

    monkeypatch.setattr("spawnllm.cli.decide_sync", fake_decide)
    request = {
        "state": {"question": "Where do a run's files go?"},
        "questions": {
            "decides": {
                "label": {"instructions": "Which answer decides it?", "options": {"none": "None does", "a1": None}}
            },
            "urgent": {"binary": {"instructions": "Is it urgent?"}},
        },
    }
    result = CliRunner().invoke(main, ["decide", "--timeout", "20"], input=json.dumps(request))
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "model": "jev-1.13.0",
        "answers": {
            "decides": {"type": "label", "choice": "a1", "probabilities": {"none": 0.1, "a1": 0.9}, "confidence": 0.8},
            "urgent": {"type": "refused"},
        },
    }
    assert asked == {
        "state": {"question": "Where do a run's files go?"},
        "questions": {
            "decides": Label("Which answer decides it?", {"none": "None does", "a1": None}),
            "urgent": Binary("Is it urgent?"),
        },
        "provider": JEV,
        "timeout": 20.0,
    }
    assert list(asked["questions"]["decides"].options) == ["none", "a1"]


def test_decide_exits_with_why_no_decision_came(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: object, **kwargs: object) -> Decision:
        raise DecideKeyMissing("no jev API key: set TYPESAFE_API_KEY or run `spawnllm key set jev`")

    monkeypatch.setattr("spawnllm.cli.decide_sync", missing)
    asked = {"state": "x", "questions": {"q": {"binary": {"instructions": "?"}}}}
    unknown = {"state": "x", "questions": {"q": {"poll": {"instructions": "?"}}}}
    results = [CliRunner().invoke(main, ["decide"], input=json.dumps(request)) for request in (asked, unknown)]
    assert [result.exit_code for result in results] == [1, 1]
    assert "no jev API key" in results[0].output
    assert "unknown question type 'poll': use binary, label, score" in results[1].output

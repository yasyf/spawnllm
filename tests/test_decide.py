from __future__ import annotations

import importlib
import json
import subprocess
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from click.testing import CliRunner

from spawnllm import (
    JEV,
    OPENAI,
    Binary,
    BinaryAnswer,
    DecideError,
    DecideKeyMissing,
    Label,
    LabelAnswer,
    Refused,
    Score,
    ScoreAnswer,
    decide,
    decide_sync,
)
from spawnllm.cli import main

if TYPE_CHECKING:
    from collections.abc import Callable

module = importlib.import_module("spawnllm.decide")

STATE = "Help! Our nightly release has been failing for 3 days and customers are blocked. Please roll it back now."
QUESTIONS = {
    "is_rollback_request": Binary("Does the message ask to roll back a release?"),
    "action": Label(
        "Which release action does the message ask for?",
        {"none": "No release action", "release": "Ship a new release", "rollback": "Roll back a release"},
    ),
    "urgency": Score(
        "How urgent is the message?", {"Not urgent": None, "Somewhat urgent": None, "Blocking, needs action now": None}
    ),
}
JEV_RECORDED = {
    "model": "jev-1.13.0",
    "answers": {
        "is_rollback_request": {"type": "noul", "noul": 0.99},
        "action": {
            "type": "choice",
            "choice": "rollback",
            "confidence": 1.0,
            "probabilities": {"release": 0.0, "rollback": 1.0, "none": 0.0},
        },
        "urgency": {
            "type": "score",
            "score": 2.0,
            "confidence": 1.0,
            "legend": {"0": "Not urgent", "1": "Somewhat urgent", "2": "Blocking, needs action now"},
            "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0},
        },
    },
    "usage": {"input_tokens": 414, "output_tokens": 73},
}
OPENAI_REFUSAL = {
    "model": "gpt-6-luna",
    "answers": [
        {"type": "refusal", "name": "is_rollback_request"},
        {
            "type": "choice",
            "name": "action",
            "choice": "none",
            "confidence": 0.7,
            "probabilities": [
                {"value": "rollback", "probability": 0.1},
                {"value": "none", "probability": 0.8},
                {"value": "release", "probability": 0.1},
            ],
        },
        {
            "type": "score",
            "name": "urgency",
            "score": 0.9,
            "confidence": 0.2,
            "probabilities": [
                {"value": 2, "label": "Blocking, needs action now", "probability": 0.2},
                {"value": 0, "label": "Not urgent", "probability": 0.3},
                {"value": 1, "label": "Somewhat urgent", "probability": 0.5},
            ],
        },
    ],
    "usage": {"input_tokens": 419},
}

type Reply = httpx.Response | httpx.TransportError


def serve(monkeypatch: pytest.MonkeyPatch, *replies: Reply) -> list[httpx.Request]:
    seen: list[httpx.Request] = []
    pending = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        match pending.pop(0):
            case httpx.TransportError() as error:
                raise error
            case httpx.Response() as response:
                return response

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(module, "sync_client", lambda: httpx.Client(transport=transport))
    monkeypatch.setattr(module, "async_client", lambda: httpx.AsyncClient(transport=transport))
    return seen


def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(module.time, "sleep", slept.append)
    return slept


def test_jev_round_trip_sends_the_planned_request_and_orders_answers_as_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = serve(monkeypatch, httpx.Response(200, json=JEV_RECORDED))

    decision = decide_sync(STATE, QUESTIONS, api_key="test-key")

    assert str(seen[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert seen[0].headers["authorization"] == "Bearer test-key"
    body = json.loads(seen[0].content)
    assert body["model"] == "jev-1.13.0"
    assert list(body["questions"]["action"]["criteria"]) == ["none", "release", "rollback"]
    assert decision.model == "jev-1.13.0"
    assert decision.input_tokens == 414
    assert list(decision.answers) == ["is_rollback_request", "action", "urgency"]
    assert decision.answers["is_rollback_request"] == BinaryAnswer(p_yes=0.99, confidence=0.98)
    assert decision.answers["action"] == LabelAnswer(
        choice="rollback", probabilities={"none": 0.0, "release": 0.0, "rollback": 1.0}, confidence=1.0
    )
    match decision.answers["action"]:
        case LabelAnswer(probabilities=probabilities):
            assert list(probabilities) == ["none", "release", "rollback"]
    assert decision.answers["urgency"] == ScoreAnswer(
        score=2.0,
        probabilities={"Not urgent": 0.0, "Somewhat urgent": 0.0, "Blocking, needs action now": 1.0},
        confidence=1.0,
    )
    assert decision.latency_ms > 0


def test_openai_refusal_comes_back_beside_the_other_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = serve(monkeypatch, httpx.Response(200, json=OPENAI_REFUSAL))

    decision = decide_sync({"message": STATE}, QUESTIONS, provider=OPENAI, api_key="test-key")

    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.openai.com/v1/decisions"
    assert body["input"] == json.dumps({"message": STATE}, separators=(",", ":"))
    assert [question["name"] for question in body["questions"]] == ["is_rollback_request", "action", "urgency"]
    assert decision.answers["is_rollback_request"] == Refused()
    assert decision.answers["action"] == LabelAnswer(
        choice="none", probabilities={"none": 0.8, "release": 0.1, "rollback": 0.1}, confidence=0.7
    )
    assert decision.answers["urgency"] == ScoreAnswer(
        score=0.9,
        probabilities={"Not urgent": 0.3, "Somewhat urgent": 0.5, "Blocking, needs action now": 0.2},
        confidence=0.2,
    )


@pytest.mark.parametrize(
    ("first", "slept"),
    [
        pytest.param(httpx.Response(529, text="overloaded"), [0.5], id="529-backs-off"),
        pytest.param(httpx.Response(429, headers={"retry-after-ms": "250"}), [0.25], id="429-honors-retry-after-ms"),
        pytest.param(httpx.ConnectError("reset"), [0.5], id="lost-connection-retries"),
    ],
)
def test_transient_failures_retry_inside_the_deadline(
    monkeypatch: pytest.MonkeyPatch, first: Reply, slept: list[float]
) -> None:
    serve(monkeypatch, first, httpx.Response(200, json=JEV_RECORDED))
    recorded = sleeps(monkeypatch)

    decision = decide_sync(STATE, QUESTIONS, api_key="test-key")

    assert recorded == slept
    assert decision.answers["action"] == LabelAnswer(
        choice="rollback", probabilities={"none": 0.0, "release": 0.0, "rollback": 1.0}, confidence=1.0
    )


def test_a_retry_that_would_outlast_the_deadline_raises_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(monkeypatch, httpx.Response(429, headers={"retry-after": "30"}))
    recorded = sleeps(monkeypatch)

    with pytest.raises(TimeoutError, match="the last try got HTTP 429"):
        decide_sync(STATE, QUESTIONS, api_key="test-key", timeout=2.0)
    assert recorded == []


def test_a_rejected_request_raises_decide_error_without_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = serve(monkeypatch, httpx.Response(401, text="Incorrect API key provided"))

    with pytest.raises(DecideError) as raised:
        decide_sync(STATE, QUESTIONS, provider=OPENAI, api_key="bad")

    assert (raised.value.status, raised.value.message) == (401, "Incorrect API key provided")
    assert len(seen) == 1


def test_an_invalid_question_set_fails_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = serve(monkeypatch)

    with pytest.raises(ValueError, match="a label question takes 2 to 255 options"):
        decide_sync(STATE, {"only": Label("Which?", {"one": None})}, api_key="test-key")
    assert seen == []


async def test_async_decide_shares_the_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    serve(monkeypatch, httpx.Response(503), httpx.Response(200, json=JEV_RECORDED))
    slept: list[float] = []

    async def record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(module.asyncio, "sleep", record)

    decision = await decide(STATE, QUESTIONS, api_key="test-key")

    assert slept == [0.5]
    assert decision.answers["is_rollback_request"] == BinaryAnswer(p_yes=0.99, confidence=0.98)


def security(monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str = "") -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    real: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run

    def fake(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[0] != "security":
            return real(argv, **kwargs)
        calls.append({"argv": argv} | kwargs)
        return subprocess.CompletedProcess(argv, returncode, stdout, "")

    monkeypatch.setattr(module.subprocess, "run", fake)
    return calls


def test_the_environment_variable_outranks_the_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")
    calls = security(monkeypatch, 0, "from-keychain\n")

    assert module.api_key(JEV) == "from-env"
    assert calls == []


def test_the_keychain_answers_when_the_variable_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    calls = security(monkeypatch, 0, "from-keychain\n")

    assert module.api_key(OPENAI) == "from-keychain"
    assert calls[0]["argv"] == [
        "security",
        "find-generic-password",
        "-s",
        "spawnllm-openai-api-key",
        "-a",
        "openai",
        "-w",
    ]


def test_no_key_anywhere_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    security(monkeypatch, 44)

    with pytest.raises(DecideKeyMissing, match="spawnllm key set jev"):
        module.api_key(JEV)


def test_key_set_writes_the_keychain_over_stdin_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = security(monkeypatch, 0)

    result = CliRunner().invoke(main, ["key", "set", "jev"], input="ts_live-AbC.123\n")

    assert result.exit_code == 0, result.output
    assert calls[0]["argv"] == ["security", "-i"]
    assert calls[0]["input"] == "add-generic-password -U -s spawnllm-jev-api-key -a jev -w ts_live-AbC.123\n"
    assert all("ts_live" not in arg for arg in calls[0]["argv"])


def test_key_set_refuses_a_key_that_could_break_the_security_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = security(monkeypatch, 0)

    result = CliRunner().invoke(main, ["key", "set", "openai"], input='sk-x" -A\n')

    assert isinstance(result.exception, ValueError)
    assert calls == []

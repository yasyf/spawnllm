"""Typed yes/no, label, and score decisions over TypeSafe Jev and OpenAI Decisions.

The Rust core maps one question set onto either provider's wire format and resolves the
answer, retry, or failure each response means; this module only moves bytes over one
keep-alive HTTP client per process and keeps every attempt inside the caller's deadline.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from spawnllm import _core

__all__ = [
    "JEV",
    "OPENAI",
    "Binary",
    "BinaryAnswer",
    "DecideError",
    "DecideKeyMissing",
    "Decision",
    "Label",
    "LabelAnswer",
    "Provider",
    "Refused",
    "Score",
    "ScoreAnswer",
    "TDecideProvider",
    "api_key",
    "decide",
    "decide_sync",
    "store_key",
]

TDecideProvider = Literal["jev", "openai"]
"""A decision provider: TypeSafe's Jev (`jev`) or OpenAI Decisions (`openai`)."""

KEY_ENV: dict[TDecideProvider, str] = {"jev": "TYPESAFE_API_KEY", "openai": "OPENAI_API_KEY"}
KEY_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
DEFAULT_TIMEOUT = 10.0

CLIENT_LOCK = threading.Lock()
CLIENT: httpx.Client | None = None


@dataclass(frozen=True, slots=True)
class Provider:
    """A decision provider and the model it runs.

    Attributes:
        name: `jev` posts to TypeSafe's System One endpoint, `openai` to OpenAI's `/v1/decisions`.
        model: The literal model id sent with every request; pin a versioned id once a
            threshold is tuned against it.

    Example:
        >>> Provider("jev", "jev-1.13.0")
    """

    name: TDecideProvider
    model: str


JEV = Provider("jev", "jev-1.13.0")
"""TypeSafe Jev pinned to `jev-1.13.0`, the model `jev-latest` resolves to today."""

OPENAI = Provider("openai", "gpt-6-luna")
"""OpenAI Decisions on `gpt-6-luna`, its only model."""


@dataclass(frozen=True, slots=True)
class Binary:
    """A yes/no question; its answer is the probability of yes.

    Attributes:
        instructions: The question, about one observable fact.
        yes: What a yes means, when the question alone leaves it open.
        no: What a no means, when the question alone leaves it open.

    Example:
        >>> Binary("Does the message ask to roll back a release?")
    """

    instructions: str
    yes: str | None = None
    no: str | None = None


@dataclass(frozen=True, slots=True)
class Label:
    """Pick one option from 2 to 255; put the fail-safe option first, since Jev leans toward it.

    Attributes:
        instructions: What to decide.
        options: Each option value mapped to its description, or `None`, in declared order.

    Example:
        >>> Label("Which release action does the message ask for?", {"none": None, "rollback": "Undo a release"})
    """

    instructions: str
    options: Mapping[str, str | None]


@dataclass(frozen=True, slots=True)
class Score:
    """Rate along 2 to 10 ordered levels, lowest first; the score is the probability-weighted level index.

    Attributes:
        instructions: What to rate.
        levels: Each level label mapped to its description, or `None`, lowest first.

    Example:
        >>> Score("How urgent is the message?", {"Not urgent": None, "Somewhat urgent": None, "Blocking": None})
    """

    instructions: str
    levels: Mapping[str, str | None]


type Question = Binary | Label | Score


@dataclass(frozen=True, slots=True)
class BinaryAnswer:
    """The answer to a `Binary` question.

    Attributes:
        p_yes: Probability the answer is yes.
        confidence: `abs(2 * p_yes - 1)`, 0 at a coin flip and 1 at certainty.
    """

    p_yes: float
    confidence: float


@dataclass(frozen=True, slots=True)
class LabelAnswer:
    """The answer to a `Label` question.

    Attributes:
        choice: The most likely option value.
        probabilities: Every option value mapped to its probability, in declared order.
        confidence: How far the distribution sits from uniform, 0 to 1.
    """

    choice: str
    probabilities: Mapping[str, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """The answer to a `Score` question.

    Attributes:
        score: The probability-weighted level index; it can land between levels.
        probabilities: Every level label mapped to its probability, lowest first.
        confidence: How concentrated the distribution is around its top level, 0 to 1.
    """

    score: float
    probabilities: Mapping[str, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class Refused:
    """A question the provider declined to answer while it answered the others."""


type Answer = BinaryAnswer | LabelAnswer | ScoreAnswer | Refused


@dataclass(frozen=True, slots=True)
class Decision:
    """One provider response: an answer per question id, in the order asked.

    Attributes:
        answers: Each question id mapped to its answer.
        model: The model that answered, as the provider reports it.
        input_tokens: Billed input tokens.
        latency_ms: Wall time of the attempt that answered, request to parsed response.
    """

    answers: Mapping[str, Answer]
    model: str
    input_tokens: int
    latency_ms: float


class DecideError(Exception):
    """The provider rejected a decision request, or answered one with a body that matches no question.

    Attributes:
        status: The HTTP status of the failed response.
        message: The provider's error body, or what about its answer failed to match.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"decision failed with HTTP {status}: {message}")
        self.status = status
        self.message = message


class DecideKeyMissing(LookupError):
    """No API key for the provider: none was passed, its variable is unset, and the Keychain holds none."""


def keychain_item(name: TDecideProvider) -> tuple[str, str]:
    return f"spawnllm-{name}-api-key", name


def api_key(provider: Provider, *, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Return the provider's API key from its environment variable, else from the macOS Keychain.

    Jev reads `TYPESAFE_API_KEY` and OpenAI reads `OPENAI_API_KEY`; on macOS an unset variable falls
    through to the Keychain item `store_key` wrote, read when the call is made.

    Raises:
        DecideKeyMissing: When neither source holds a key.
        TimeoutError: When the Keychain read outlasts `timeout`.
    """
    if key := os.environ.get(KEY_ENV[provider.name]):
        return key
    service, account = keychain_item(provider.name)
    if sys.platform == "darwin":
        try:
            found = subprocess.run(
                ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"the Keychain read for the {provider.name} key outlasted {timeout:g}s") from exc
        if found.returncode == 0:
            return found.stdout.strip()
    raise DecideKeyMissing(
        f"no {provider.name} API key: set {KEY_ENV[provider.name]} or run `spawnllm key set {provider.name}`"
    )


def store_key(provider: Provider, key: str) -> None:
    """Store `key` as the provider's macOS Keychain item, replacing any earlier one.

    The key reaches `security` over stdin, never its argv, so no process listing shows it.

    Raises:
        ValueError: When the key holds characters outside `[A-Za-z0-9._-]`.
        subprocess.CalledProcessError: When `security` refuses the write.
    """
    if not KEY_PATTERN.fullmatch(key):
        raise ValueError(f"a {provider.name} API key holds only letters, digits, '.', '_', and '-'")
    service, account = keychain_item(provider.name)
    subprocess.run(
        ["security", "-i"],
        input=f"add-generic-password -U -s {service} -a {account} -w {key}\n",
        capture_output=True,
        text=True,
        check=True,
    )


def sync_client() -> httpx.Client:
    global CLIENT
    with CLIENT_LOCK:
        if CLIENT is None:
            CLIENT = httpx.Client()
        return CLIENT


def fork_reset() -> None:
    global CLIENT, CLIENT_LOCK
    CLIENT_LOCK = threading.Lock()
    CLIENT = None


os.register_at_fork(after_in_child=fork_reset)


def wire_question(id: str, question: Question) -> dict[str, Any]:
    match question:
        case Binary(instructions=instructions, yes=yes, no=no):
            return {"type": "binary", "id": id, "instructions": instructions, "yes": yes, "no": no}
        case Label(instructions=instructions, options=options):
            return {
                "type": "label",
                "id": id,
                "instructions": instructions,
                "options": [{"value": value, "description": text} for value, text in options.items()],
            }
        case Score(instructions=instructions, levels=levels):
            return {
                "type": "score",
                "id": id,
                "instructions": instructions,
                "levels": [{"label": label, "description": text} for label, text in levels.items()],
            }


def keys(question: Question) -> Sequence[str]:
    match question:
        case Label(options=options):
            return tuple(options)
        case Score(levels=levels):
            return tuple(levels)
        case Binary():
            return ()


def answer(question: Question, wire: dict[str, Any]) -> Answer:
    match wire:
        case {"type": "binary", "p_yes": p_yes, "confidence": confidence}:
            return BinaryAnswer(p_yes, confidence)
        case {"type": "label", "choice": choice, "probabilities": probabilities, "confidence": confidence}:
            return LabelAnswer(choice, dict(zip(keys(question), probabilities, strict=True)), confidence)
        case {"type": "score", "score": score, "probabilities": probabilities, "confidence": confidence}:
            return ScoreAnswer(score, dict(zip(keys(question), probabilities, strict=True)), confidence)
        case {"type": "refused"}:
            return Refused()
    raise ValueError(f"the core resolved an unknown answer: {wire}")


@dataclass(frozen=True, slots=True)
class Call:
    provider: Provider
    questions: Mapping[str, Question]
    wired: list[dict[str, Any]]
    request: dict[str, Any]
    deadline: float
    timeout: float

    @classmethod
    def plan(
        cls,
        state: str | Mapping[str, Any] | Sequence[Any],
        questions: Mapping[str, Question],
        provider: Provider,
        timeout: float,
        key: str | None,
    ) -> Call:
        deadline = time.monotonic() + timeout
        wired = [wire_question(id, question) for id, question in questions.items()]
        planned = {
            "provider": {"name": provider.name, "model": provider.model},
            "api_key": key or api_key(provider, timeout=timeout),
            "state": state,
            "questions": wired,
        }
        try:
            request = _core.dispatch("decide_plan", planned)
        except _core.CoreError as error:
            raise ValueError(error.msg) from error
        return cls(provider, questions, wired, request, deadline, timeout)

    def remaining(self) -> float:
        if (left := self.deadline - time.monotonic()) <= 0:
            raise TimeoutError(f"no {self.provider.name} decision within {self.timeout:g}s")
        return left

    def outcome(self, response: httpx.Response | None, attempt: int, started: float) -> Decision | float:
        resolved = _core.dispatch(
            "decide_resolve",
            {
                "provider": {"name": self.provider.name, "model": self.provider.model},
                "questions": self.wired,
                "status": None if response is None else response.status_code,
                "body": "" if response is None else response.text,
                "retry_after": None if response is None else response.headers.get("retry-after"),
                "retry_after_ms": None if response is None else response.headers.get("retry-after-ms"),
                "attempt": attempt,
            },
        )
        match resolved:
            case {"kind": "decision", "model": model, "input_tokens": tokens, "answers": answers}:
                return Decision(
                    {id: answer(self.questions[id], wire) for id, wire in answers.items()},
                    model,
                    tokens,
                    (time.monotonic() - started) * 1000,
                )
            case {"kind": "retry", "sleep_s": sleep_s} if time.monotonic() + sleep_s < self.deadline:
                return sleep_s
            case {"kind": "retry"}:
                last = "no response" if response is None else f"HTTP {response.status_code}"
                raise TimeoutError(
                    f"no {self.provider.name} decision within {self.timeout:g}s; the last try got {last}"
                )
            case {"kind": "failed", "status": status, "message": message}:
                raise DecideError(status, message)
        raise ValueError(f"the core resolved an unknown outcome: {resolved}")


def decide_sync(
    state: str | Mapping[str, Any] | Sequence[Any],
    questions: Mapping[str, Question],
    *,
    provider: Provider = JEV,
    timeout: float = DEFAULT_TIMEOUT,
    api_key: str | None = None,
) -> Decision:
    """Ask every question about `state` in one request and return the typed answers.

    Keep each question to one observable fact and combine the answers in code; put counting,
    dates, and arithmetic in code too. Retries 408, 429, 5xx, 529, and a lost connection with
    backoff from 0.5s doubling to 5s, honoring `retry-after`, and only while the retry still fits
    inside `timeout`. The Keychain read counts against `timeout` too, and an answer that arrives
    after it raises `TimeoutError` rather than returning late.

    Args:
        state: The text to judge, or JSON data; Jev reads structure natively and OpenAI receives
            it as compact JSON text.
        questions: Each question id mapped to its question, in the order to ask them.
        provider: `JEV` or `OPENAI`, or a `Provider` pinning another model.
        timeout: Seconds for the whole call, retries included.
        api_key: The provider key; `None` reads it through `api_key`.

    Returns:
        The decision, with a `Refused` answer for any question the provider declined.

    Raises:
        TimeoutError: When no answer arrives within `timeout`.
        DecideError: When the provider rejects the request.
        DecideKeyMissing: When no key is passed or found.
        ValueError: When the question set breaks a limit both providers share, such as a label
            with fewer than 2 options or a score with more than 10 levels.

    Example:
        >>> decision = decide_sync("Please roll back last night's release.", {
        ...     "rollback": Binary("Does the message ask to roll back a release?"),
        ... })
        >>> decision.answers["rollback"].p_yes > 0.5
        True
    """
    call = Call.plan(state, questions, provider, timeout, api_key)
    tries = 0
    while True:
        started = time.monotonic()
        try:
            response = sync_client().post(
                call.request["url"],
                headers=call.request["headers"],
                json=call.request["body"],
                timeout=call.remaining(),
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"no {provider.name} decision within {timeout:g}s") from exc
        except httpx.TransportError:
            response = None
        call.remaining()
        match call.outcome(response, tries, started):
            case Decision() as decision:
                return decision
            case float() as sleep_s:
                time.sleep(sleep_s)
                tries += 1


async def decide(
    state: str | Mapping[str, Any] | Sequence[Any],
    questions: Mapping[str, Question],
    *,
    provider: Provider = JEV,
    timeout: float = DEFAULT_TIMEOUT,
    api_key: str | None = None,
) -> Decision:
    """Ask every question about `state` in one request, asynchronously; `decide_sync` documents the contract.

    The call runs `decide_sync` on a worker thread, so it shares the process's keep-alive client and
    holds no connection to the event loop.

    Example:
        >>> decision = await decide({"title": "Revert the release"}, {
        ...     "kind": Label("What does the title ask for?", {"other": None, "rollback": None}),
        ... }, provider=OPENAI)
    """
    return await asyncio.to_thread(decide_sync, state, questions, provider=provider, timeout=timeout, api_key=api_key)

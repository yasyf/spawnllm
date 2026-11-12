from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from typing import Any

import click
from loguru import logger

from spawnllm import _core
from spawnllm.backends import (
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    BackendUnavailable,
)
from spawnllm.backends.registry import BACKENDS_BY_NAME, PRIORITY, select_backend
from spawnllm.call import DEFAULT_MODEL, call_sync
from spawnllm.decide import (
    DEFAULT_TIMEOUT,
    JEV,
    OPENAI,
    Answer,
    Binary,
    BinaryAnswer,
    DecideError,
    DecideKeyMissing,
    Label,
    LabelAnswer,
    Provider,
    Question,
    Refused,
    Score,
    ScoreAnswer,
    decide_sync,
    store_key,
)

BACKEND_NAMES = tuple(BACKENDS_BY_NAME)
DECIDE_PROVIDERS: dict[str, Provider] = {provider.name: provider for provider in (JEV, OPENAI)}
QUESTION_TYPES: dict[str, type[Binary | Label | Score]] = {"binary": Binary, "label": Label, "score": Score}
ANSWER_TYPES: dict[type[Answer], str] = {
    BinaryAnswer: "binary",
    LabelAnswer: "label",
    ScoreAnswer: "score",
    Refused: "refused",
}


@click.group()
@click.version_option(package_name="spawnllm")
def main() -> None:
    """Subshell + MLX LLM-calling backends (Claude/Codex CLI, local MLX) shared across tools."""


@main.command()
def backends() -> None:
    """List the LLM backends spawnllm can drive."""
    logger.debug("backends invoked")
    for name in (*BACKEND_NAMES, "mlx"):
        click.echo(name)


@main.command()
@click.option("--backend", type=click.Choice(list(BACKEND_NAMES)), required=True)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    help="Abstract tier (small/medium/large) or a concrete provider model id, e.g. claude-fable-5.",
)
@click.option("--agent", is_flag=True, help="Allow tools / agent capabilities.")
@click.option("--api-auth", is_flag=True, help="Inherit provider API-key environment variables.")
@click.argument("prompt", required=False)
def call(backend: str, model: str, agent: bool, api_auth: bool, prompt: str | None) -> None:
    """Make a one-off LLM call (reads PROMPT or stdin) and print the response."""
    text = prompt if prompt is not None else sys.stdin.read()
    result = call_sync(text, backend=BACKENDS_BY_NAME[backend], model=model, agent=agent, api_auth=api_auth)
    click.echo(result)


@main.command()
def status() -> None:
    """Report install/auth status for each backend and which one auto-selection would pick."""
    for backend in PRIORITY:
        try:
            match backend.check_status():
                case BackendReady():
                    click.echo(f"{backend.binary}: ready")
                case BackendNotInstalled(install_hint=hint):
                    click.echo(f"{backend.binary}: not installed — install with: {hint}")
                case BackendNotAuthenticated():
                    click.echo(f"{backend.binary}: not authenticated")
        except subprocess.TimeoutExpired:
            click.echo(f"{backend.binary}: timed out")
    try:
        click.echo(f"selected: {select_backend(model=DEFAULT_MODEL).binary}")
    except BackendUnavailable:
        click.echo("selected: none available")
    click.echo(f"core: {(v := _core.version())['core_version']}@{v['source_hash'][:12]}")


@main.command("decide")
@click.option("--provider", type=click.Choice(list(DECIDE_PROVIDERS)), default=JEV.name, show_default=True)
@click.option(
    "--timeout",
    type=float,
    default=DEFAULT_TIMEOUT,
    show_default=True,
    help="Seconds for the whole call, retries included.",
)
def decide_questions(provider: str, timeout: float) -> None:
    """Ask the questions a JSON object on stdin holds about its state, and print the answers as JSON.

    The object holds `state`, the text or JSON to judge, and `questions`, each id mapped to
    `{"binary": {...}}`, `{"label": {...}}` or `{"score": {...}}` holding that question's fields,
    such as `{"label": {"instructions": "...", "options": {"none": "...", "a1": null}}}`. It prints
    `{"model": ..., "answers": {<id>: {"type": "label", "choice": ..., "probabilities": {...}, "confidence": ...}}}`.
    """
    try:
        request = json.loads(sys.stdin.read())
        questions = {id: question(spec) for id, spec in request["questions"].items()}
        decision = decide_sync(request["state"], questions, provider=DECIDE_PROVIDERS[provider], timeout=timeout)
    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        TimeoutError,
        DecideError,
        DecideKeyMissing,
    ) as error:
        raise click.ClickException(str(error)) from error
    answers = {
        id: {"type": ANSWER_TYPES[type(answer)], **dataclasses.asdict(answer)}
        for id, answer in decision.answers.items()
    }
    click.echo(json.dumps({"model": decision.model, "answers": answers}))


def question(spec: dict[str, Any]) -> Question:
    [(kind, fields)] = spec.items()
    if kind not in QUESTION_TYPES:
        raise ValueError(f"unknown question type {kind!r}: use {', '.join(QUESTION_TYPES)}")
    return QUESTION_TYPES[kind](**fields)


@main.group()
def key() -> None:
    """Manage the decision providers' API keys in the macOS Keychain."""


@key.command("set")
@click.argument("provider", type=click.Choice(list(DECIDE_PROVIDERS)))
def key_set(provider: str) -> None:
    """Store PROVIDER's API key, read from stdin, in the macOS Keychain, where `decide` reads it."""
    store_key(DECIDE_PROVIDERS[provider], sys.stdin.read().strip())
    click.echo(f"stored the {provider} key in the Keychain item spawnllm-{provider}-api-key")

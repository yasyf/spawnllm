from __future__ import annotations

import subprocess
import sys
from typing import cast

import click
from loguru import logger

from spawnllm.backends import (
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    BackendUnavailable,
)
from spawnllm.backends.registry import BACKENDS_BY_NAME, PRIORITY, select_backend
from spawnllm.call import call_sync
from spawnllm.types import TModel

BACKEND_NAMES = tuple(BACKENDS_BY_NAME)


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
@click.option("--model", type=click.Choice(["small", "medium", "large"]), default="small")
@click.option("--agent", is_flag=True, help="Allow tools / agent capabilities.")
@click.argument("prompt", required=False)
def call(backend: str, model: str, agent: bool, prompt: str | None) -> None:
    """Make a one-off LLM call (reads PROMPT or stdin) and print the response."""
    text = prompt if prompt is not None else sys.stdin.read()
    result = call_sync(text, backend=BACKENDS_BY_NAME[backend], model=cast(TModel, model), agent=agent)
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
        click.echo(f"selected: {select_backend().binary}")
    except BackendUnavailable:
        click.echo("selected: none available")

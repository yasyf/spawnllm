from __future__ import annotations

import sys
from typing import cast

import click
from loguru import logger

from subllm.backends import ClaudeCliBackend, CodexCliBackend
from subllm.call import call as call_backend
from subllm.types import TModel

BACKENDS = ("claude", "codex", "mlx")
CLI_BACKENDS = {"claude": ClaudeCliBackend, "codex": CodexCliBackend}


@click.group()
@click.version_option(package_name="subllm-py")
def main() -> None:
    """Subshell + MLX LLM-calling backends (Claude/Codex CLI, local MLX) shared across tools."""


@main.command()
def backends() -> None:
    """List the LLM backends subllm can drive."""
    logger.debug("backends invoked")
    for name in BACKENDS:
        click.echo(name)


@main.command()
@click.option("--backend", type=click.Choice(["claude", "codex"]), required=True)
@click.option("--model", type=click.Choice(["small", "medium", "large"]), default="small")
@click.option("--agent", is_flag=True, help="Allow tools / agent capabilities.")
@click.argument("prompt", required=False)
def call(backend: str, model: str, agent: bool, prompt: str | None) -> None:
    """Make a one-off LLM call (reads PROMPT or stdin) and print the response."""
    text = prompt if prompt is not None else sys.stdin.read()
    result = call_backend(text, backend=CLI_BACKENDS[backend](), model=cast(TModel, model), agent=agent)
    click.echo(result)

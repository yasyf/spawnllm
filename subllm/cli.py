from __future__ import annotations

import click
from loguru import logger

BACKENDS = ("claude", "codex", "mlx")


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

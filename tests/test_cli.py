from __future__ import annotations

from click.testing import CliRunner

from subllm.cli import main


def test_help_exits_cleanly() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert result.output.startswith("Usage: main")


def test_backends_lists_available() -> None:
    result = CliRunner().invoke(main, ["backends"])
    assert result.exit_code == 0
    assert result.output.splitlines() == ["claude", "codex", "mlx"]

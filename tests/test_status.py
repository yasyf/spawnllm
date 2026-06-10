from __future__ import annotations

import subprocess
from unittest.mock import patch

from spawnllm import ClaudeNotAuthenticated, ClaudeNotInstalled, ClaudeReady, check_status


class TestCheckStatus:
    def test_not_installed_with_brew(self) -> None:
        def which(name: str) -> str | None:
            return "/opt/homebrew/bin/brew" if name == "brew" else None

        with patch("spawnllm.backends.claude.shutil.which", side_effect=which):
            assert check_status() == ClaudeNotInstalled(brew_available=True)

    def test_not_installed_without_brew(self) -> None:
        with patch("spawnllm.backends.claude.shutil.which", return_value=None):
            assert check_status() == ClaudeNotInstalled(brew_available=False)

    def test_ready_when_auth_status_zero(self) -> None:
        done = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        with (
            patch("spawnllm.backends.claude.shutil.which", return_value="/usr/bin/claude"),
            patch("spawnllm.backends.claude.subprocess.run", return_value=done),
        ):
            assert check_status() == ClaudeReady()

    def test_not_authenticated_when_auth_status_nonzero(self) -> None:
        done = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="no")
        with (
            patch("spawnllm.backends.claude.shutil.which", return_value="/usr/bin/claude"),
            patch("spawnllm.backends.claude.subprocess.run", return_value=done),
        ):
            assert check_status() == ClaudeNotAuthenticated()

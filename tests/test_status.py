from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from spawnllm import (
    AntigravityCliBackend,
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    BackendUnavailable,
    ClaudeCliBackend,
    ClaudeSdkBackend,
    CodexCliBackend,
    GeminiCliBackend,
    select_backend,
)

if TYPE_CHECKING:
    from pathlib import Path

    from spawnllm.backends.base import LlmBackend

CLAUDE_HINT = "curl -fsSL https://claude.ai/install.sh | bash"
CODEX_HINT = "npm install -g @openai/codex"
GEMINI_HINT = "npm install -g @google/gemini-cli"
AGY_HINT = "curl -fsSL https://antigravity.google/cli/install.sh | bash"

NOT_INSTALLED = [
    pytest.param(ClaudeCliBackend, "claude", CLAUDE_HINT, id="claude"),
    pytest.param(CodexCliBackend, "codex", CODEX_HINT, id="codex"),
    pytest.param(GeminiCliBackend, "gemini", GEMINI_HINT, id="gemini"),
    pytest.param(AntigravityCliBackend, "agy", AGY_HINT, id="agy"),
]

# The auth-probe executor runs every subprocess-backed probe through the host
# module, so one patch point serves each exec_exit0 (claude/codex) probe.
SUBPROCESS_AUTH = [
    pytest.param(ClaudeCliBackend, "claude", id="claude"),
    pytest.param(CodexCliBackend, "codex", id="codex"),
]


def installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("spawnllm.backends.base.shutil.which", lambda name: f"/usr/bin/{name}")


def gemini_creds_absent(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


class TestCheckStatus:
    @pytest.mark.parametrize(("backend_cls", "binary", "install_hint"), NOT_INSTALLED)
    def test_not_installed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_cls: type[LlmBackend],
        binary: str,
        install_hint: str,
    ) -> None:
        monkeypatch.setattr("spawnllm.backends.base.shutil.which", lambda name: None)
        assert backend_cls().check_status() == BackendNotInstalled(binary=binary, install_hint=install_hint)

    @pytest.mark.parametrize(("backend_cls", "binary"), SUBPROCESS_AUTH)
    @pytest.mark.parametrize(
        ("returncode", "make_expected"),
        [
            pytest.param(0, lambda binary: BackendReady(binary=binary), id="ready"),
            pytest.param(1, lambda binary: BackendNotAuthenticated(binary=binary), id="not_authenticated"),
        ],
    )
    def test_subprocess_auth(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend_cls: type[LlmBackend],
        binary: str,
        returncode: int,
        make_expected,
    ) -> None:
        installed(monkeypatch)
        monkeypatch.setattr(
            "spawnllm.backends.base.subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=returncode),
        )
        assert backend_cls().check_status() == make_expected(binary)

    def test_gemini_ready_via_api_key(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        installed(monkeypatch)
        gemini_creds_absent(monkeypatch, tmp_path)
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        assert GeminiCliBackend().check_status() == BackendReady(binary="gemini")

    def test_gemini_ready_via_cached_creds(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        installed(monkeypatch)
        (tmp_path / ".gemini").mkdir()
        (tmp_path / ".gemini" / "oauth_creds.json").write_text("{}")
        gemini_creds_absent(monkeypatch, tmp_path)
        assert GeminiCliBackend().check_status() == BackendReady(binary="gemini")

    def test_gemini_not_authenticated(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        installed(monkeypatch)
        gemini_creds_absent(monkeypatch, tmp_path)
        assert GeminiCliBackend().check_status() == BackendNotAuthenticated(binary="gemini")

    def test_agy_ready_via_keychain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        installed(monkeypatch)
        monkeypatch.setattr("spawnllm.backends.base.sys.platform", "darwin")
        monkeypatch.setattr(
            "spawnllm.backends.base.subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=0),
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("ANTIGRAVITY_API_KEY", raising=False)
        assert AntigravityCliBackend().check_status() == BackendReady(binary="agy")

    def test_agy_ready_via_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "spawnllm.backends.base.shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None
        )
        monkeypatch.setattr("spawnllm.backends.base.sys.platform", "darwin")
        # A Keychain miss leaves the env-key probe to authenticate.
        monkeypatch.setattr(
            "spawnllm.backends.base.subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=1),
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("ANTIGRAVITY_API_KEY", "k")
        assert AntigravityCliBackend().check_status() == BackendReady(binary="agy")

    def test_agy_not_authenticated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        installed(monkeypatch)
        monkeypatch.setattr("spawnllm.backends.base.sys.platform", "darwin")
        monkeypatch.setattr(
            "spawnllm.backends.base.subprocess.run",
            lambda *a, **k: subprocess.CompletedProcess(args=[], returncode=1),
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("ANTIGRAVITY_API_KEY", raising=False)
        assert AntigravityCliBackend().check_status() == BackendNotAuthenticated(binary="agy")


def ready(self: LlmBackend, *, timeout: int = 10) -> BackendReady:
    return BackendReady(self.binary)


def unauthenticated(self: LlmBackend, *, timeout: int = 10) -> BackendNotAuthenticated:
    return BackendNotAuthenticated(self.binary)


def not_installed(self: LlmBackend, *, timeout: int = 10) -> BackendNotInstalled:
    return BackendNotInstalled(self.binary, "install")


def timed_out(self: LlmBackend, *, timeout: int = 10):
    raise subprocess.TimeoutExpired(cmd=[self.binary], timeout=timeout)


class TestSelectBackend:
    def test_all_ready_returns_first_in_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for cls in (ClaudeSdkBackend, ClaudeCliBackend, CodexCliBackend, AntigravityCliBackend, GeminiCliBackend):
            monkeypatch.setattr(cls, "check_status", ready)
        assert isinstance(select_backend(), ClaudeSdkBackend)

    def test_falls_through_to_antigravity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ClaudeSdkBackend, "check_status", unauthenticated)
        monkeypatch.setattr(ClaudeCliBackend, "check_status", unauthenticated)
        monkeypatch.setattr(CodexCliBackend, "check_status", unauthenticated)
        monkeypatch.setattr(AntigravityCliBackend, "check_status", ready)
        monkeypatch.setattr(GeminiCliBackend, "check_status", ready)
        assert isinstance(select_backend(), AntigravityCliBackend)

    def test_none_ready_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for cls in (ClaudeSdkBackend, ClaudeCliBackend, CodexCliBackend, AntigravityCliBackend, GeminiCliBackend):
            monkeypatch.setattr(cls, "check_status", unauthenticated)
        with pytest.raises(BackendUnavailable):
            select_backend()

    def test_specialty_promotes_codex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ClaudeSdkBackend, "check_status", ready)
        monkeypatch.setattr(ClaudeCliBackend, "check_status", ready)
        monkeypatch.setattr(CodexCliBackend, "check_status", ready)
        assert isinstance(select_backend(specialty="debugging"), CodexCliBackend)

    def test_timeout_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ClaudeSdkBackend, "check_status", unauthenticated)
        monkeypatch.setattr(ClaudeCliBackend, "check_status", timed_out)
        monkeypatch.setattr(CodexCliBackend, "check_status", ready)
        assert isinstance(select_backend(), CodexCliBackend)

    def test_sdk_not_installed_falls_through_to_claude(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ClaudeSdkBackend, "check_status", not_installed)
        monkeypatch.setattr(ClaudeCliBackend, "check_status", ready)
        assert isinstance(select_backend(), ClaudeCliBackend)

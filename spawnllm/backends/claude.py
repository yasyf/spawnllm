"""CliBackend for the Anthropic `claude` CLI, plus its keychain-sourced config isolation."""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from spawnllm import _core
from spawnllm.backends.base import ClaudeIsolation, CliBackend

if TYPE_CHECKING:
    from spawnllm.response import Response
    from spawnllm.spec import RunSpec
    from spawnllm.types import ProviderName, TModel

CLAUDE_MODELS: dict[TModel, str] = {"small": "haiku", "medium": "sonnet", "large": "opus"}


def keychain_credentials(service: str) -> str | None:
    """Return the claude.ai OAuth credentials stored under `service` in the macOS Keychain, or `None` on a miss."""
    proc = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-w"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def read_file_opt(path: str) -> str | None:
    """Return the text at `path`, or `None` when it does not exist."""
    file = Path(path)
    return file.read_text() if file.exists() else None


class ClaudeCliBackend(CliBackend):
    """`CliBackend` for the Anthropic `claude` CLI.

    The core plans the `claude -p` argv (prompt delivered over stdin, result read
    from a stdout file) and lays out the host-free config home this backend seeds
    with only the active-account pointer; the claude.ai access token reaches the
    child as `CLAUDE_CODE_OAUTH_TOKEN`, never as a file.

    Attributes:
        models: Mapping from abstract model size to a Claude model alias
            (`haiku`/`sonnet`/`opus`).

    Example:
        >>> from spawnllm.spec import RunSpec
        >>> ClaudeCliBackend().invocation(RunSpec(prompt="hi", model="haiku")).argv[:5]
        ['claude', '-p', '--no-session-persistence', '--model', 'haiku']
    """

    models: ClassVar[dict[TModel, str]] = CLAUDE_MODELS
    provider: ClassVar[ProviderName] = "claude"
    binary: ClassVar[str] = "claude"
    install_hint: ClassVar[str] = "curl -fsSL https://claude.ai/install.sh | bash"
    schema_dialect: ClassVar[str | None] = "anthropic"

    _isolated_config_dir: str | None = None
    _api_config_dir: str | None = None
    _rejected_tokens: frozenset[str] = frozenset()

    def claude_isolation(self, api_auth: bool) -> ClaudeIsolation:
        """Return the isolation for one run: the process-lifetime config home and the env resolved now.

        The core's `claude_isolation_sources` op resolves the account pointer,
        credentials file, and Keychain service from the caller's effective config
        home, naming no credential source when the process already carries
        `CLAUDE_CODE_OAUTH_TOKEN`; this host reads those sources (the Keychain
        before the credentials file, the order Claude Code itself reads them) and
        hands them, with every token a run saw rejected, to `claude_isolation_seed`
        for the exact files-and-modes to write and the env to set. The files land
        in a fresh temp dir removed at interpreter exit, created on the first
        call and cached on the backend; the env is resolved on every call so a
        renewed Keychain token reaches the next run, and it only ever lives in
        memory. An `api_auth` run names no source at all, so it
        reads no account, credentials file, or Keychain item and gets an empty home
        of its own.
        """
        sources = _core.dispatch(
            "claude_isolation_sources",
            {
                "host": {
                    "platform": sys.platform,
                    "home": str(Path.home()),
                    "claude_config_dir_env": os.environ.get("CLAUDE_CONFIG_DIR") or None,
                    "claude_securestorage_config_dir_env": os.environ.get("CLAUDE_SECURESTORAGE_CONFIG_DIR"),
                    "claude_code_custom_oauth_url_env": os.environ.get("CLAUDE_CODE_CUSTOM_OAUTH_URL"),
                    "claude_code_oauth_token_env": os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"),
                },
                "api_auth": api_auth,
            },
        )
        account_json = read_file_opt(sources["account_path"]) if sources["account_path"] else None
        credentials_json = [
            credentials
            for credentials in (
                keychain_credentials(sources["keychain_service"]) if sources["keychain_service"] else None,
                read_file_opt(sources["credentials_path"]) if sources["credentials_path"] else None,
            )
            if credentials is not None
        ]
        seed = _core.dispatch(
            "claude_isolation_seed",
            {
                "account_json": account_json,
                "credentials_json": credentials_json,
                "rejected_tokens": sorted(self._rejected_tokens),
            },
        )
        cached = self._api_config_dir if api_auth else self._isolated_config_dir
        if cached is None:
            config_dir = Path(tempfile.mkdtemp(prefix="spawnllm-claude-config-"))
            for file in seed["files"]:
                fd = os.open(config_dir / file["name"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, int(file["mode"], 8))
                with os.fdopen(fd, "w") as handle:
                    handle.write(file["content"])
            atexit.register(shutil.rmtree, config_dir, ignore_errors=True)
            cached = str(config_dir)
            if api_auth:
                self._api_config_dir = cached
            else:
                self._isolated_config_dir = cached
        return ClaudeIsolation(cached, seed["env"])

    def reject_credentials(self, spec: RunSpec, env: dict[str, str], response: Response) -> bool:
        """Record the OAuth token in a run's `env` as rejected when the core reads `response` as an auth failure.

        Returns:
            `True` when another credential source remains to retry the run with.
        """
        token = env.get("CLAUDE_CODE_OAUTH_TOKEN")
        if token is None or response.error is None:
            return False
        if not _core.dispatch("claude_auth_rejected", {"error_msg": response.error.msg})["rejected"]:
            return False
        self._rejected_tokens |= {token}
        return self.claude_isolation(spec.api_auth).env.get("CLAUDE_CODE_OAUTH_TOKEN") not in self._rejected_tokens

    async def aexecute(self, spec: RunSpec) -> Response:
        env = self.env(spec)
        response = await self.aexecute_invocation(spec, self.invocation(spec), env)
        while self.reject_credentials(spec, env, response):
            env = self.env(spec)
            response = await self.aexecute_invocation(spec, self.invocation(spec), env)
        return response

    def execute(self, spec: RunSpec) -> Response:
        env = self.env(spec)
        response = self.execute_invocation(spec, self.invocation(spec), env)
        while self.reject_credentials(spec, env, response):
            env = self.env(spec)
            response = self.execute_invocation(spec, self.invocation(spec), env)
        return response

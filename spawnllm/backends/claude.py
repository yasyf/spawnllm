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
from spawnllm.backends.base import CliBackend

if TYPE_CHECKING:
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
    with only the active-account pointer and claude.ai OAuth token.

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

    def claude_isolation(self) -> str:
        """Return the process-lifetime isolated config home, creating and seeding it once.

        The core's `claude_isolation_sources` op resolves the account pointer,
        credentials file, and Keychain service from the caller's effective config
        home; this host reads those sources (falling back to the Keychain when the
        credentials file is absent), hands them to `claude_isolation_seed` for the
        exact files-and-modes to write, and materializes them into a fresh temp
        dir removed at interpreter exit. The dir is cached on the backend.
        """
        if self._isolated_config_dir is not None:
            return self._isolated_config_dir
        sources = _core.dispatch(
            "claude_isolation_sources",
            {
                "host": {
                    "platform": sys.platform,
                    "home": str(Path.home()),
                    "claude_config_dir_env": os.environ.get("CLAUDE_CONFIG_DIR") or None,
                }
            },
        )
        account_json = read_file_opt(sources["account_path"])
        credentials_json = read_file_opt(sources["credentials_path"])
        if credentials_json is None and sources["keychain_service"] is not None:
            credentials_json = keychain_credentials(sources["keychain_service"])
        seed = _core.dispatch(
            "claude_isolation_seed", {"account_json": account_json, "credentials_json": credentials_json}
        )
        config_dir = Path(tempfile.mkdtemp(prefix="spawnllm-claude-config-"))
        for file in seed["files"]:
            path = config_dir / file["name"]
            path.write_text(file["content"])
            path.chmod(int(file["mode"], 8))
        atexit.register(shutil.rmtree, config_dir, ignore_errors=True)
        self._isolated_config_dir = str(config_dir)
        return self._isolated_config_dir

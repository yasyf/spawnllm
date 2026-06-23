"""Abstract execution contract for an LLM backend and its subprocess family."""

from __future__ import annotations

import json
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from spawnllm.proc import RunResult, acapture_cli, capture_cli

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.spec import RunSpec
    from spawnllm.types import ProviderName, TModel


@dataclass(frozen=True)
class BackendReady:
    """A backend whose CLI is installed and authenticated.

    Attributes:
        binary: Name of the backend's CLI executable on PATH.
    """

    binary: str


@dataclass(frozen=True)
class BackendNotInstalled:
    """A backend whose CLI is not on PATH.

    Attributes:
        binary: Name of the backend's CLI executable.
        install_hint: Suggested shell command to install the CLI.
    """

    binary: str
    install_hint: str


@dataclass(frozen=True)
class BackendNotAuthenticated:
    """A backend whose CLI is installed but not authenticated.

    Attributes:
        binary: Name of the backend's CLI executable on PATH.
    """

    binary: str


BackendStatus = BackendReady | BackendNotInstalled | BackendNotAuthenticated
"""Result of `LlmBackend.check_status`: `BackendReady`, `BackendNotInstalled`, or `BackendNotAuthenticated`."""


class BackendUnavailable(RuntimeError):
    """Raised when no backend is ready (installed and authenticated)."""


@dataclass(frozen=True)
class Invocation:
    """A built CLI invocation: argv, optional stdin, and where to read the result.

    Attributes:
        argv: The argv list to execute.
        stdin: Prompt text delivered over stdin, or `None` when delivered inline.
        result_path: File the backend writes its final message to; when set, the
            result is read from this file instead of stdout.
        cleanup_paths: Temp files to remove once the invocation completes.
    """

    argv: list[str]
    stdin: str | None = None
    result_path: str | None = None
    cleanup_paths: tuple[str, ...] = ()


class LlmBackend(ABC):
    """Abstract execution contract for an LLM backend.

    Concrete backends map abstract model sizes to provider-specific model names
    and encapsulate how to execute a `RunSpec` and parse the raw response.

    Attributes:
        models: Mapping from abstract model size to the provider's model name.
        provider: Provider identifier keying a `RunSpec`'s `provider_configs`.
    """

    models: ClassVar[dict[TModel, str]]
    provider: ClassVar[ProviderName]

    @abstractmethod
    async def aexecute(self, spec: RunSpec) -> RunResult:
        """Execute a single run asynchronously and capture its raw outcome.

        Args:
            spec: The configured run to execute.

        Returns:
            The captured stdout, stderr, and exit code.
        """

    @abstractmethod
    def execute(self, spec: RunSpec) -> RunResult:
        """Execute a single run synchronously and capture its raw outcome.

        Args:
            spec: The configured run to execute.

        Returns:
            The captured stdout, stderr, and exit code.
        """

    @abstractmethod
    def parse_response(self, raw: str, response_model: type[BaseModel] | None) -> str | BaseModel:
        """Parse raw stdout into text or a validated model.

        Args:
            raw: Raw stdout from the backend.
            response_model: Model to validate against, or `None` for raw text.

        Returns:
            `raw` when `response_model` is `None`, else a validated instance.
        """

    @abstractmethod
    def env(self) -> dict[str, str]:
        """Return extra environment variables for the invocation, merged over the inherited environment."""

    @abstractmethod
    def is_authenticated(self, *, timeout: int) -> bool:
        """Probe whether the backend holds valid credentials for its provider.

        "Authenticated" means the backend reports an active login session for the
        provider, not merely that an executable is present on PATH.

        Args:
            timeout: Seconds to wait for the credential probe.

        Returns:
            `True` when the backend reports an authenticated session.
        """

    @abstractmethod
    def check_status(self, *, timeout: int = 10) -> BackendStatus:
        """Check whether this backend is installed and authenticated.

        Args:
            timeout: Seconds to wait for the authentication probe.

        Returns:
            `BackendReady` when authenticated, `BackendNotInstalled` when the
            backend is not available, else `BackendNotAuthenticated`.
        """

    def schema_for(self, model: type[BaseModel]) -> str:
        """Serialize a Pydantic model into the JSON-schema string this backend expects.

        The default emits the model's plain JSON schema; provider backends
        override to apply their SDK's strict-schema transform.

        Args:
            model: The Pydantic model describing the structured output.

        Returns:
            A JSON-schema string suitable for this backend's structured-output argument.
        """
        return json.dumps(model.model_json_schema())


class CliBackend(LlmBackend):
    """Execution contract for the subprocess-backed LLM family.

    Concrete CLI backends build an argv from a `RunSpec`; `aexecute`/`execute`
    run it, merge environment overrides, and resolve the result from stdout or a
    designated result file.

    Attributes:
        binary: Name of the backend's CLI executable on PATH.
        install_hint: Suggested shell command to install the CLI.
    """

    binary: ClassVar[str]
    install_hint: ClassVar[str]

    @abstractmethod
    def build_command(self, spec: RunSpec) -> list[str]:
        """Build the CLI argv for a single invocation.

        Args:
            spec: The configured run to translate into argv.

        Returns:
            The argv list to execute.
        """

    def invocation(self, spec: RunSpec) -> Invocation:
        """Build the argv, stdin, and result source for a single invocation.

        The default delivers the prompt over stdin and reads the result from
        stdout; subclasses override to deliver the prompt inline or to read the
        result from a file.

        Args:
            spec: The configured run to translate into an invocation.

        Returns:
            An `Invocation` carrying the argv, stdin text, and result source.
        """
        return Invocation(self.build_command(spec), spec.prompt)

    async def aexecute(self, spec: RunSpec) -> RunResult:
        inv = self.invocation(spec)
        try:
            rr = await acapture_cli(
                inv.argv,
                input=inv.stdin,
                env=os.environ | self.env() | (spec.env or {}),
                cwd=spec.cwd,
                timeout=spec.timeout,
            )
            stdout = Path(inv.result_path).read_text() if inv.result_path else rr.stdout
        finally:
            for path in inv.cleanup_paths:
                Path(path).unlink(missing_ok=True)
        return RunResult(stdout, rr.stderr, rr.returncode)

    def execute(self, spec: RunSpec) -> RunResult:
        inv = self.invocation(spec)
        try:
            rr = capture_cli(
                inv.argv,
                input=inv.stdin,
                env=os.environ | self.env() | (spec.env or {}),
                cwd=spec.cwd,
                timeout=spec.timeout,
            )
            stdout = Path(inv.result_path).read_text() if inv.result_path else rr.stdout
        finally:
            for path in inv.cleanup_paths:
                Path(path).unlink(missing_ok=True)
        return RunResult(stdout, rr.stderr, rr.returncode)

    def check_status(self, *, timeout: int = 10) -> BackendStatus:
        """Check whether this backend's CLI is installed and authenticated.

        Args:
            timeout: Seconds to wait for the authentication probe.

        Returns:
            `BackendReady` when authenticated, `BackendNotInstalled` when the CLI
            is not on PATH, else `BackendNotAuthenticated`.

        Raises:
            subprocess.TimeoutExpired: If `is_authenticated` exceeds `timeout`.
        """
        if not shutil.which(self.binary):
            return BackendNotInstalled(binary=self.binary, install_hint=self.install_hint)
        if self.is_authenticated(timeout=timeout):
            return BackendReady(binary=self.binary)
        return BackendNotAuthenticated(binary=self.binary)

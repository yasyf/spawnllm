"""Abstract interface for an LLM CLI backend."""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.types import TModel


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
    """Abstract interface for an LLM CLI backend.

    Concrete backends map abstract model sizes to provider-specific model names
    and encapsulate how to invoke the provider's CLI and parse the raw response.

    Attributes:
        models: Mapping from abstract model size to the provider's model name.
    """

    models: ClassVar[dict[TModel, str]]
    binary: ClassVar[str]
    install_hint: ClassVar[str]

    @abstractmethod
    def build_command(self, model: str, schema_path: str | None, agent: bool) -> list[str]:
        """Build the CLI argv for a single invocation (prompt delivered via stdin).

        Args:
            model: Provider-specific model name.
            schema_path: Schema argument for structured output, or `None`.
            agent: Whether the invocation may use tools / agent capabilities.

        Returns:
            The argv list to execute.
        """

    @abstractmethod
    def parse_response(self, raw: str, response_model: type[BaseModel] | None) -> str | BaseModel:
        """Parse raw CLI stdout into text or a validated model.

        Args:
            raw: Raw stdout from the backend CLI.
            response_model: Model to validate against, or `None` for raw text.

        Returns:
            `raw` when `response_model` is `None`, else a validated instance.
        """

    @abstractmethod
    def env(self) -> dict[str, str]:
        """Return extra environment variables for the CLI invocation, merged over the inherited environment."""

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

    @abstractmethod
    def is_authenticated(self, *, timeout: int) -> bool:
        """Probe whether the CLI holds valid credentials for its provider.

        "Authenticated" means the CLI reports an active login session for the
        provider, not merely that the executable is present on PATH.

        Args:
            timeout: Seconds to wait for the credential probe.

        Returns:
            `True` when the CLI reports an authenticated session.
        """

    def schema_for(self, model: type[BaseModel]) -> str:
        """Serialize a Pydantic model into the JSON-schema string this backend's CLI expects.

        The default emits the model's plain JSON schema; provider backends
        override to apply their SDK's strict-schema transform.

        Args:
            model: The Pydantic model describing the structured output.

        Returns:
            A JSON-schema string suitable for this backend's structured-output argument.
        """
        return json.dumps(model.model_json_schema())

    def invocation(self, prompt: str, *, model: str, schema_path: str | None, agent: bool) -> Invocation:
        """Build the argv, stdin, and result source for a single invocation.

        The default delivers the prompt over stdin and reads the result from
        stdout; subclasses override to deliver the prompt inline or to read the
        result from a file.

        Args:
            prompt: The prompt text to deliver to the CLI.
            model: Provider-specific model name.
            schema_path: Schema argument for structured output, or `None`.
            agent: Whether the invocation may use tools / agent capabilities.

        Returns:
            An `Invocation` carrying the argv, stdin text, and result source.
        """
        return Invocation(self.build_command(model, schema_path, agent), prompt)

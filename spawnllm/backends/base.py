"""Abstract interface for an LLM CLI backend."""

from __future__ import annotations

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

    def invocation(
        self, prompt: str, *, model: str, schema_path: str | None, agent: bool
    ) -> tuple[list[str], str | None]:
        """Build the argv and stdin text for a single invocation.

        The default delivers the prompt over stdin; subclasses override to
        deliver it inline within the argv.

        Args:
            prompt: The prompt text to deliver to the CLI.
            model: Provider-specific model name.
            schema_path: Schema argument for structured output, or `None`.
            agent: Whether the invocation may use tools / agent capabilities.

        Returns:
            A `(argv, stdin_text)` pair; `stdin_text` is `None` when the prompt is delivered inline.
        """
        return self.build_command(model, schema_path, agent), prompt

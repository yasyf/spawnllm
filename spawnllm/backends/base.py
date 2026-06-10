"""Abstract interface for an LLM CLI backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.types import TModel


class LlmBackend(ABC):
    """Abstract interface for an LLM CLI backend.

    Concrete backends map abstract model sizes to provider-specific model names
    and encapsulate how to invoke the provider's CLI and parse the raw response.

    Attributes:
        models: Mapping from abstract model size to the provider's model name.
    """

    models: ClassVar[dict[TModel, str]]

    @abstractmethod
    def build_command(self, model: str, schema_path: str | None, agent: bool) -> list[str]:
        """Build the CLI argv for a single invocation (prompt delivered via stdin).

        Args:
            model: Provider-specific model name.
            schema_path: Schema argument for structured output, or ``None``.
            agent: Whether the invocation may use tools / agent capabilities.

        Returns:
            The argv list to execute.
        """

    @abstractmethod
    def parse_response(self, raw: str, response_model: type[BaseModel] | None) -> str | BaseModel:
        """Parse raw CLI stdout into text or a validated model.

        Args:
            raw: Raw stdout from the backend CLI.
            response_model: Model to validate against, or ``None`` for raw text.

        Returns:
            ``raw`` when ``response_model`` is ``None``, else a validated instance.
        """

    @abstractmethod
    def env(self) -> dict[str, str]:
        """Return extra environment variables to set for the CLI invocation."""

"""Specialty-to-backend registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.claude import ClaudeCliBackend
from spawnllm.backends.codex import CodexCliBackend

if TYPE_CHECKING:
    from spawnllm.backends.base import LlmBackend
    from spawnllm.types import TSpecialty


class LlmBackends:
    """Registry mapping each specialty to the `LlmBackend` that serves it.

    `debugging` and `review` route to `CodexCliBackend`; `general` routes to
    `ClaudeCliBackend`.

    Attributes:
        LLM_BACKENDS: Mapping from specialty to its backend instance.
    """

    LLM_BACKENDS: ClassVar[dict[TSpecialty, LlmBackend]] = {
        "debugging": CodexCliBackend(),
        "review": CodexCliBackend(),
        "general": ClaudeCliBackend(),
    }

    @classmethod
    def for_specialty(cls, specialty: TSpecialty) -> LlmBackend:
        """Return the backend registered for a specialty.

        Args:
            specialty: One of `debugging`, `review`, or `general`.

        Returns:
            The `LlmBackend` instance that serves `specialty`.
        """
        return cls.LLM_BACKENDS[specialty]

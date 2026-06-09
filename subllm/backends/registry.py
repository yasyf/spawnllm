"""Specialty → backend registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from subllm.backends.claude import ClaudeCliBackend
from subllm.backends.codex import CodexCliBackend

if TYPE_CHECKING:
    from subllm.backends.base import LlmBackend
    from subllm.types import TSpecialty


class LlmBackends:
    """Registry mapping each specialty to the :class:`LlmBackend` that serves it."""

    LLM_BACKENDS: ClassVar[dict[TSpecialty, LlmBackend]] = {
        "debugging": CodexCliBackend(),
        "review": CodexCliBackend(),
        "general": ClaudeCliBackend(),
    }

    @classmethod
    def for_specialty(cls, specialty: TSpecialty) -> LlmBackend:
        """Return the backend registered for ``specialty``."""
        return cls.LLM_BACKENDS[specialty]

"""Specialty registry and selector, prioritizing Claude SDK before native CLIs."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import BackendReady, BackendUnavailable, LlmBackend
from spawnllm.backends.claude import ClaudeCliBackend
from spawnllm.backends.claude_sdk import ClaudeSdkBackend
from spawnllm.backends.codex import CodexCliBackend
from spawnllm.backends.gemini import AntigravityCliBackend, GeminiCliBackend

if TYPE_CHECKING:
    from spawnllm.types import TSpecialty

BACKENDS_BY_NAME: dict[str, LlmBackend] = {
    "claude-sdk": ClaudeSdkBackend(),
    "claude": ClaudeCliBackend(),
    "codex": CodexCliBackend(),
    "antigravity": AntigravityCliBackend(),
    "gemini": GeminiCliBackend(),
}
PRIORITY: tuple[LlmBackend, ...] = tuple(BACKENDS_BY_NAME.values())


class LlmBackends:
    """Registry mapping each specialty to the `LlmBackend` that serves it.

    `debugging` and `review` route to `CodexCliBackend`; `general` routes to the
    first-priority `ClaudeSdkBackend`.

    Attributes:
        LLM_BACKENDS: Mapping from specialty to its backend instance.
    """

    LLM_BACKENDS: ClassVar[dict[TSpecialty, LlmBackend]] = {
        "debugging": CodexCliBackend(),
        "review": CodexCliBackend(),
        "general": ClaudeSdkBackend(),
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


def select_backend(*, specialty: TSpecialty | None = None, timeout: int = 10) -> LlmBackend:
    """Return the first installed, authenticated backend in priority order.

    A `specialty`, when given, promotes its registered backend to the front of
    the chain; the chain otherwise follows `PRIORITY`, minus `GeminiCliBackend`
    (its Code Assist OAuth tier is retired, so it reports ready yet fails at call
    time — reach it only via an explicit `backend=`). The first backend whose
    `check_status` reports `BackendReady` wins, short-circuiting the rest;
    backends that time out are skipped.

    Args:
        specialty: Specialty whose backend is tried first, or `None`.
        timeout: Seconds to wait for each backend's authentication probe.

    Returns:
        The first ready `LlmBackend`.

    Raises:
        BackendUnavailable: When no backend is installed and authenticated.
    """
    preferred = [LlmBackends.LLM_BACKENDS[specialty]] if specialty else []
    seen = {type(b) for b in preferred}
    auto = (b for b in PRIORITY if type(b) not in seen and not isinstance(b, GeminiCliBackend))
    for backend in (*preferred, *auto):
        try:
            if isinstance(backend.check_status(timeout=timeout), BackendReady):
                return backend
        except subprocess.TimeoutExpired:
            continue
    raise BackendUnavailable("no installed, authenticated LLM backend found")

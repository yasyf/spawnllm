"""LLM CLI backends (Claude/Codex/Gemini family) and the specialty registry."""

from __future__ import annotations

from spawnllm.backends.base import (
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    BackendStatus,
    BackendUnavailable,
    LlmBackend,
)
from spawnllm.backends.claude import ClaudeCliBackend
from spawnllm.backends.codex import CodexCliBackend
from spawnllm.backends.gemini import AntigravityCliBackend, GeminiCliBackend
from spawnllm.backends.registry import LlmBackends, select_backend

__all__ = [
    "AntigravityCliBackend",
    "BackendNotAuthenticated",
    "BackendNotInstalled",
    "BackendReady",
    "BackendStatus",
    "BackendUnavailable",
    "ClaudeCliBackend",
    "CodexCliBackend",
    "GeminiCliBackend",
    "LlmBackend",
    "LlmBackends",
    "select_backend",
]

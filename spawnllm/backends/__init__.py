"""LLM CLI backends (Claude/Codex/Gemini family) and the specialty registry."""

from __future__ import annotations

from spawnllm.backends.base import (
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    BackendStatus,
    BackendUnavailable,
    CliBackend,
    Invocation,
    LlmBackend,
)
from spawnllm.backends.claude import ClaudeCliBackend
from spawnllm.backends.codex import CodexCliBackend
from spawnllm.backends.gemini import AntigravityCliBackend, GeminiCliBackend
from spawnllm.backends.mlx import MlxBackend
from spawnllm.backends.registry import LlmBackends, select_backend

__all__ = [
    "AntigravityCliBackend",
    "BackendNotAuthenticated",
    "BackendNotInstalled",
    "BackendReady",
    "BackendStatus",
    "BackendUnavailable",
    "ClaudeCliBackend",
    "CliBackend",
    "CodexCliBackend",
    "GeminiCliBackend",
    "Invocation",
    "LlmBackend",
    "LlmBackends",
    "MlxBackend",
    "select_backend",
]

"""LLM CLI backends (Claude/Codex/Gemini family) and the specialty registry."""

from __future__ import annotations

from spawnllm.backends.base import (
    BackendCallError,
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    BackendStatus,
    BackendUnavailable,
    CliBackend,
    LlmBackend,
)
from spawnllm.backends.claude import ClaudeCliBackend
from spawnllm.backends.claude_sdk import ClaudeSdkBackend
from spawnllm.backends.codex import CodexCliBackend
from spawnllm.backends.gemini import AntigravityCliBackend, GeminiCliBackend
from spawnllm.backends.mlx import MlxBackend
from spawnllm.backends.openai_endpoint import OpenAiEndpointBackend
from spawnllm.backends.registry import LlmBackends, select_backend

__all__ = [
    "AntigravityCliBackend",
    "BackendCallError",
    "BackendNotAuthenticated",
    "BackendNotInstalled",
    "BackendReady",
    "BackendStatus",
    "BackendUnavailable",
    "ClaudeCliBackend",
    "ClaudeSdkBackend",
    "CliBackend",
    "CodexCliBackend",
    "GeminiCliBackend",
    "LlmBackend",
    "LlmBackends",
    "MlxBackend",
    "OpenAiEndpointBackend",
    "select_backend",
]

"""LLM CLI backends (Claude/Codex) and the specialty registry."""

from __future__ import annotations

from spawnllm.backends.base import LlmBackend
from spawnllm.backends.claude import (
    ClaudeCliBackend,
    ClaudeNotAuthenticated,
    ClaudeNotInstalled,
    ClaudeReady,
    ClaudeStatus,
    check_status,
)
from spawnllm.backends.codex import CodexCliBackend
from spawnllm.backends.registry import LlmBackends

__all__ = [
    "ClaudeCliBackend",
    "ClaudeNotAuthenticated",
    "ClaudeNotInstalled",
    "ClaudeReady",
    "ClaudeStatus",
    "CodexCliBackend",
    "LlmBackend",
    "LlmBackends",
    "check_status",
]

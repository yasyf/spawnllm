"""Shared type aliases for the LLM-calling surface."""

from __future__ import annotations

from typing import Literal

__all__ = ["ProviderName", "TModel", "TReasoningEffort", "TSettingSource", "TSpecialty"]

TSpecialty = Literal["debugging", "review", "general"]
"""Task specialty; `LlmBackends.for_specialty` maps each to its registered backend."""

TSettingSource = Literal["user", "project", "local"]
"""A settings layer the Claude CLI loads; `ClaudeConfig.setting_sources` selects them."""

TModel = Literal["small", "medium", "large"]
"""Abstract model tier; each backend maps it to a provider-specific model name."""

TReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
"""A `reasoning_effort` an OpenAI-compatible endpoint accepts; `OpenAiEndpointBackend` sends it with every request."""

ProviderName = Literal["claude", "claude-sdk", "codex", "gemini", "antigravity", "mlx", "apple", "openai_endpoint"]
"""Backend provider identifier; keys the per-backend `provider_configs` on a `RunSpec`."""

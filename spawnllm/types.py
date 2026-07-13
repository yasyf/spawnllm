"""Shared type aliases for the LLM-calling surface."""

from __future__ import annotations

from typing import Literal

__all__ = ["ProviderName", "TModel", "TSpecialty"]

TSpecialty = Literal["debugging", "review", "general"]
"""Task specialty; `LlmBackends.for_specialty` maps each to its registered backend."""

TModel = Literal["small", "medium", "large"]
"""Abstract model tier; each backend maps it to a provider-specific model name."""

ProviderName = Literal["claude", "codex", "gemini", "antigravity", "mlx", "openai_endpoint"]
"""Backend provider identifier; keys the per-backend `provider_configs` on a `RunSpec`."""

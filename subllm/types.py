"""Shared type aliases for the LLM-calling surface."""

from __future__ import annotations

from typing import Literal

__all__ = ["TModel", "TSpecialty"]

TSpecialty = Literal["debugging", "review", "general"]
TModel = Literal["small", "medium", "large"]

"""Subshell + MLX LLM-calling backends (Claude/Codex CLI, local MLX) shared across tools.

The top-level namespace exposes the CLI backends, subprocess transport, and
structured-output helpers. The MLX engine lives under :mod:`subllm.mlx` and is
imported lazily so that ``import subllm`` never pulls ``mlx_lm``/``zstandard``.
"""

from __future__ import annotations

from subllm.backends import (
    ClaudeCliBackend,
    ClaudeNotAuthenticated,
    ClaudeNotInstalled,
    ClaudeReady,
    ClaudeStatus,
    CodexCliBackend,
    LlmBackend,
    LlmBackends,
    check_status,
)
from subllm.call import call
from subllm.proc import arun_cli, collect_process, map_concurrent, run_cli
from subllm.structured import (
    extract_structured,
    parse_result_envelope,
    parse_structured_output,
    resolve_schema_path,
    schema_for,
)
from subllm.types import TModel, TSpecialty

__all__ = [
    "ClaudeCliBackend",
    "ClaudeNotAuthenticated",
    "ClaudeNotInstalled",
    "ClaudeReady",
    "ClaudeStatus",
    "CodexCliBackend",
    "LlmBackend",
    "LlmBackends",
    "TModel",
    "TSpecialty",
    "arun_cli",
    "call",
    "check_status",
    "collect_process",
    "extract_structured",
    "map_concurrent",
    "parse_result_envelope",
    "parse_structured_output",
    "resolve_schema_path",
    "run_cli",
    "schema_for",
]

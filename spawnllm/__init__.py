"""Subshell + MLX LLM-calling backends (Claude/Codex CLI, local MLX) shared across tools.

The top-level namespace exposes the CLI backends, subprocess transport, and
structured-output helpers. The MLX engine lives under `spawnllm.mlx`, whose
imports are lazy so that `import spawnllm` never pulls `mlx_lm`/`zstandard`.
"""

from __future__ import annotations

from spawnllm.backends import (
    AntigravityCliBackend,
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    BackendStatus,
    BackendUnavailable,
    ClaudeCliBackend,
    CodexCliBackend,
    GeminiCliBackend,
    Invocation,
    LlmBackend,
    LlmBackends,
    select_backend,
)
from spawnllm.call import call
from spawnllm.proc import arun_cli, collect_process, map_concurrent, run_cli
from spawnllm.structured import (
    extract_structured,
    parse_result_envelope,
    parse_structured_output,
    resolve_schema_path,
)
from spawnllm.types import TModel, TSpecialty

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
    "Invocation",
    "LlmBackend",
    "LlmBackends",
    "TModel",
    "TSpecialty",
    "arun_cli",
    "call",
    "collect_process",
    "extract_structured",
    "map_concurrent",
    "parse_result_envelope",
    "parse_structured_output",
    "resolve_schema_path",
    "run_cli",
    "select_backend",
]

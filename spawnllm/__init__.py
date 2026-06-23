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
    CliBackend,
    CodexCliBackend,
    GeminiCliBackend,
    Invocation,
    LlmBackend,
    LlmBackends,
    MlxBackend,
    select_backend,
)
from spawnllm.call import call, call_sync
from spawnllm.proc import RunResult, arun_cli, collect_process, map_concurrent, run_cli
from spawnllm.run import run, run_sync
from spawnllm.spec import ClaudeConfig, CodexConfig, GeminiConfig, RunSpec
from spawnllm.structured import (
    extract_structured,
    parse_result_envelope,
    parse_structured_output,
    resolve_schema_path,
)
from spawnllm.types import ProviderName, TModel, TSpecialty

__all__ = [
    "AntigravityCliBackend",
    "BackendNotAuthenticated",
    "BackendNotInstalled",
    "BackendReady",
    "BackendStatus",
    "BackendUnavailable",
    "ClaudeCliBackend",
    "ClaudeConfig",
    "CliBackend",
    "CodexCliBackend",
    "CodexConfig",
    "GeminiCliBackend",
    "GeminiConfig",
    "Invocation",
    "LlmBackend",
    "LlmBackends",
    "MlxBackend",
    "ProviderName",
    "RunResult",
    "RunSpec",
    "TModel",
    "TSpecialty",
    "arun_cli",
    "call",
    "call_sync",
    "collect_process",
    "extract_structured",
    "map_concurrent",
    "parse_result_envelope",
    "parse_structured_output",
    "resolve_schema_path",
    "run",
    "run_cli",
    "run_sync",
    "select_backend",
]

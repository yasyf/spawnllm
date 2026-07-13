"""Subshell + MLX LLM-calling backends (Claude/Codex CLI, local MLX) shared across tools.

The top-level namespace exposes the three primitives — `run`/`call`/`extract`
and their `_sync` companions — over a `Backend` family that fully encapsulates
execution and returns one shared `Response`. The MLX engine lives under
`spawnllm.mlx`, whose imports are lazy so that `import spawnllm` never pulls
`mlx_lm`/`zstandard`.
"""

from __future__ import annotations

from spawnllm.backends import (
    AntigravityCliBackend,
    BackendCallError,
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    BackendStatus,
    BackendUnavailable,
    ClaudeCliBackend,
    CliBackend,
    CodexCliBackend,
    GeminiCliBackend,
    LlmBackend,
    LlmBackends,
    MlxBackend,
    OpenAiEndpointBackend,
    select_backend,
)
from spawnllm.call import call, call_sync
from spawnllm.extract import extract, extract_sync
from spawnllm.response import DiscardedAttempt, Error, Output, Response, Result
from spawnllm.run import run, run_sync
from spawnllm.spec import ClaudeConfig, CodexConfig, GeminiConfig, RunSpec
from spawnllm.types import ProviderName, TModel, TSpecialty

__all__ = [
    "AntigravityCliBackend",
    "BackendCallError",
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
    "DiscardedAttempt",
    "Error",
    "GeminiCliBackend",
    "GeminiConfig",
    "LlmBackend",
    "LlmBackends",
    "MlxBackend",
    "OpenAiEndpointBackend",
    "Output",
    "ProviderName",
    "Response",
    "Result",
    "RunSpec",
    "TModel",
    "TSpecialty",
    "call",
    "call_sync",
    "extract",
    "extract_sync",
    "run",
    "run_sync",
    "select_backend",
]

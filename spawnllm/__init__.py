"""Subshell + MLX LLM-calling backends (Claude/Codex CLI, local MLX) shared across tools.

The top-level namespace exposes the three primitives — `run`/`call`/`extract`
and their `_sync` companions — over a `Backend` family that fully encapsulates
execution and returns one shared `Response`, plus `decide`/`decide_sync` for
typed yes/no, label, and score decisions over TypeSafe Jev and OpenAI
Decisions. The MLX engine lives under `spawnllm.mlx`, whose imports are lazy so
that `import spawnllm` never pulls `mlx_lm`/`zstandard`.
"""

from __future__ import annotations

from spawnllm.backends import (
    AntigravityCliBackend,
    AppleBackend,
    BackendCallError,
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    BackendStatus,
    BackendUnavailable,
    ClaudeCliBackend,
    ClaudeSdkBackend,
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
from spawnllm.decide import (
    JEV,
    OPENAI,
    Binary,
    BinaryAnswer,
    DecideError,
    DecideKeyMissing,
    Decision,
    Label,
    LabelAnswer,
    Provider,
    Refused,
    Score,
    ScoreAnswer,
    decide,
    decide_sync,
)
from spawnllm.extract import extract, extract_sync
from spawnllm.response import DiscardedAttempt, Error, Output, Response, Result
from spawnllm.run import run, run_sync
from spawnllm.spec import AppleConfig, ClaudeConfig, CodexConfig, GeminiConfig, RunSpec
from spawnllm.types import ProviderName, TModel, TReasoningEffort, TSpecialty

__all__ = [
    "JEV",
    "OPENAI",
    "AntigravityCliBackend",
    "AppleBackend",
    "AppleConfig",
    "BackendCallError",
    "BackendNotAuthenticated",
    "BackendNotInstalled",
    "BackendReady",
    "BackendStatus",
    "BackendUnavailable",
    "Binary",
    "BinaryAnswer",
    "ClaudeCliBackend",
    "ClaudeConfig",
    "ClaudeSdkBackend",
    "CliBackend",
    "CodexCliBackend",
    "CodexConfig",
    "DecideError",
    "DecideKeyMissing",
    "Decision",
    "DiscardedAttempt",
    "Error",
    "GeminiCliBackend",
    "GeminiConfig",
    "Label",
    "LabelAnswer",
    "LlmBackend",
    "LlmBackends",
    "MlxBackend",
    "OpenAiEndpointBackend",
    "Output",
    "Provider",
    "ProviderName",
    "Refused",
    "Response",
    "Result",
    "RunSpec",
    "Score",
    "ScoreAnswer",
    "TModel",
    "TReasoningEffort",
    "TSpecialty",
    "call",
    "call_sync",
    "decide",
    "decide_sync",
    "extract",
    "extract_sync",
    "run",
    "run_sync",
    "select_backend",
]

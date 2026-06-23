# spawnllm

![spawnllm banner](https://github.com/yasyf/spawnllm/raw/main/docs/assets/readme-banner.webp)

[![PyPI](https://img.shields.io/pypi/v/spawnllm.svg)](https://pypi.org/project/spawnllm/)
[![Python](https://img.shields.io/pypi/pyversions/spawnllm.svg)](https://pypi.org/project/spawnllm/)
[![Docs](https://img.shields.io/github/actions/workflow/status/yasyf/spawnllm/docs.yml?branch=main&label=docs)](https://yasyf.github.io/spawnllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/yasyf/spawnllm/blob/main/LICENSE)

Subshell + MLX LLM-calling backends (Claude/Codex CLI, local MLX) shared across tools.

spawnllm centralizes the LLM-calling plumbing that small tools keep re-inventing: driving the
`claude` and `codex` CLIs as subshells — with structured Pydantic output, model tiers, and
faithful error capture — and running local Apple-Silicon MLX models with adapter fusion,
prompt-cache reuse, and batched generation. Depend on it once and each tool keeps only its
domain logic instead of its own copy of the backends.

## Install

Run the CLI with [uvx](https://docs.astral.sh/uv/):

```bash
uvx spawnllm --help
```

For the local MLX engine (Apple Silicon only), pull the extra: `uv add "spawnllm[mlx]"`.

## Quickstart

See which backends are installed and authenticated, and which one auto-selection picks:

```bash
uvx spawnllm status
```

```
claude: ready
codex: ready
selected: claude
```

Make a request by passing a prompt as the argument, or piping it over stdin:

```bash
uvx spawnllm call --backend claude "What is 2+2? Reply with just the number."
```

```
4
```

`--model small|medium|large` swaps the tier, which each backend maps to a concrete model — the
`claude` backend resolves `small` to Haiku, `medium` to Sonnet, and `large` to Opus. Add
`--agent` to let the call use tools. Run `uvx spawnllm --help` for the full flag list.

### From Python

`call_sync` runs one request and returns the response. With no `backend`, it auto-selects
the first installed, authenticated CLI (its async companion `call` mirrors the same
signature):

```python
from spawnllm import call_sync

print(call_sync("Reply with just the word: pong"))
# pong
```

Pin a backend and tier explicitly, or pass a Pydantic model to get a validated object back
instead of text:

```python
from pydantic import BaseModel

from spawnllm import call_sync, ClaudeCliBackend


class Capital(BaseModel):
    country: str
    capital: str


result = call_sync(
    "What is the capital of France?",
    backend=ClaudeCliBackend(),
    model="large",
    response_model=Capital,
)
print(result.capital)  # Paris
```

When you don't pin a backend, set `specialty=` to scope auto-selection by task. The
`debugging` and `review` specialties route to Codex, and `general` routes to Claude.

### Spec-driven runs

For full control, build a `RunSpec` and execute it with `run_sync` (or its async companion
`run`). A `RunSpec` takes a literal provider model id — no tier mapping — and per-provider
flag passthrough via `provider_configs`. The call returns a `RunResult` with raw stdout,
stderr, and exit code, retrying transient `529`/overloaded/rate-limit failures with backoff:

```python
from spawnllm import run_sync, RunSpec, ClaudeConfig, ClaudeCliBackend

result = run_sync(
    RunSpec(
        prompt="What is 2+2? Reply with just the number.",
        model="opus",
        provider_configs={"claude": ClaudeConfig(permission_mode="bypassPermissions")},
    ),
    backend=ClaudeCliBackend(),
)
print(result.stdout)  # 4
```

## How it works

Each backend holds plumbing that consumers would otherwise rebuild: the CLI backends own argv
construction, stdin/stdout piping, stderr teeing, and turning non-zero exits into useful errors,
and they turn a Pydantic model into a JSON-schema constraint plus a parsed, validated result. The
MLX engine wraps adapter fusion, prompt-cache reuse, worker-thread lifecycle, and batched
single-token generation. Tools that share the layer stay byte-for-byte consistent instead of
drifting across diverging copies.

## Docs

[Read the docs](https://yasyf.github.io/spawnllm/) for the full guide and API reference.

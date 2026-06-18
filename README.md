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

No install needed — run everything through [uvx](https://docs.astral.sh/uv/):

```bash
uvx spawnllm --help
```

`uvx` fetches spawnllm into a throwaway environment and runs it. To add it
to a project instead:

```bash
uv add spawnllm
```

For the local MLX engine (Apple Silicon only), pull the extra:

```bash
uv add "spawnllm[mlx]"
```

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

`--model small|medium|large` swaps the tier, which each backend maps to a concrete model.
The `claude` backend resolves `small` to Haiku, `medium` to Sonnet, and `large` to Opus. Add
`--agent` to let the call use tools.

### From Python

`call` runs one request and returns the response. With no `backend`, it auto-selects the
first installed, authenticated CLI:

```python
from spawnllm import call

print(call("Reply with just the word: pong"))
# pong
```

Pin a backend and tier explicitly, or pass a Pydantic model to get a validated object back
instead of text:

```python
from pydantic import BaseModel

from spawnllm import call, ClaudeCliBackend


class Capital(BaseModel):
    country: str
    capital: str


result = call(
    "What is the capital of France?",
    backend=ClaudeCliBackend(),
    model="large",
    response_model=Capital,
)
print(result.capital)  # Paris
```

When you don't pin a backend, set `specialty=` to scope auto-selection by task. The
`debugging` and `review` specialties route to Codex, and `general` routes to Claude.

## What problems does this solve?

Every tool that shells out to `claude` or `codex` rebuilds the same plumbing: argv
construction, stdin/stdout piping, stderr teeing, and turning non-zero exits into useful
errors. spawnllm holds it once.

Structured output is boilerplate too. A Pydantic model becomes a JSON-schema constraint
and a parsed, validated result, identically for both CLI backends.

Local MLX is fiddly. Adapter fusion, prompt-cache reuse, worker-thread lifecycle, and
batched single-token generation live behind one engine instead of in every consumer.

Behavior drift goes away with the duplication: two tools that call the same models stay
byte-for-byte consistent because they share the backend layer, not a pair of diverging
copies.

## Docs

[Read the docs](https://yasyf.github.io/spawnllm/) for the full guide and API reference.

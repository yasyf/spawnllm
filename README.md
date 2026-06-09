# subllm

[![PyPI](https://img.shields.io/pypi/v/subllm-py.svg)](https://pypi.org/project/subllm-py/)
[![Python](https://img.shields.io/pypi/pyversions/subllm-py.svg)](https://pypi.org/project/subllm-py/)
[![Docs](https://img.shields.io/github/actions/workflow/status/yasyf/subllm/docs.yml?branch=main&label=docs)](https://yasyf.github.io/subllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/yasyf/subllm/blob/main/LICENSE)

Subshell + MLX LLM-calling backends (Claude/Codex CLI, local MLX) shared across tools.

subllm centralizes the LLM-calling plumbing that small tools keep re-inventing: driving the
`claude` and `codex` CLIs as subshells — with structured Pydantic output, model tiers, and
faithful error capture — and running local Apple-Silicon MLX models with adapter fusion,
prompt-cache reuse, and batched generation. Depend on it once and each tool keeps only its
domain logic instead of its own copy of the backends.

## Install

No install needed — run everything through [uvx](https://docs.astral.sh/uv/):

```bash
uvx subllm-py --help
```

`uvx` fetches subllm into a throwaway environment and runs it. To add it
to a project instead:

```bash
uv add subllm-py
```

For the local MLX engine (Apple Silicon only), pull the extra:

```bash
uv add "subllm-py[mlx]"
```

## Quickstart

List the backends subllm can drive:

```bash
uvx subllm-py backends
```

```
claude
codex
mlx
```

## What problems does this solve?

- **Duplicate subshell plumbing.** Building `claude`/`codex` argv, piping stdin/stdout, teeing
  stderr, and turning non-zero exits into useful errors — written once, not re-derived per tool.
- **Structured-output boilerplate.** A Pydantic model becomes a JSON-schema constraint and a
  parsed, validated result the same way for every backend.
- **Local MLX is fiddly.** Adapter fusion, prompt-cache reuse, worker-thread lifecycle, and
  batched single-token generation live behind one engine instead of in every consumer.
- **Behavior drift.** Two tools that call the same models stay byte-for-byte consistent because
  they share the backend layer rather than each maintaining a copy.

## Docs

[Read the docs](https://yasyf.github.io/subllm/) for the full guide and API reference.

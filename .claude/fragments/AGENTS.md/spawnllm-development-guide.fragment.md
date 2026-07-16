# spawnllm Development Guide

Subshell + MLX LLM-calling backends (Claude/Codex CLI, local MLX) shared across tools. Published to PyPI as `spawnllm`; the CLI is `spawnllm`, run as `uvx spawnllm`.

## Repository Structure

```
spawnllm/
├── spawnllm/            # The package — Claude/Codex CLI backends, MLX engine, transport, Click CLI
├── tests/            # Pytest suite
├── .github/          # GitHub Actions workflows
├── AGENTS.md         # This file — shared conventions
└── README.md         # Project overview
```

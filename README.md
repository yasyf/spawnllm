# ![spawnllm](https://github.com/yasyf/spawnllm/raw/main/docs/assets/readme-banner.webp)

**Delete your subprocess wrappers around claude, codex, and gemini.** spawnllm subshells all three CLIs plus local MLX and returns one Pydantic-validated Response, so the per-model plumbing you hand-rolled goes away.

[![CI](https://github.com/yasyf/spawnllm/actions/workflows/ci.yml/badge.svg)](https://github.com/yasyf/spawnllm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/spawnllm)](https://pypi.org/project/spawnllm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/yasyf/spawnllm/blob/main/LICENSE)

## Get started

```bash
uvx spawnllm status
```

<img src="https://github.com/yasyf/spawnllm/raw/main/docs/assets/demo.png" alt="Terminal running 'uvx spawnllm status' — every backend reports ready and auto-selection picks claude" width="700">

Driving with an agent? Paste this:

```text
Run `uv add spawnllm` in this project.
Replace our hand-rolled claude/codex subprocess code with spawnllm's `call_sync`,
or `extract_sync` with a Pydantic response model for structured output.
Verify available backends with `uvx spawnllm status`.
Docs: https://yasyf.github.io/spawnllm/
```

---

## Use cases

### Delete your hand-rolled claude/codex subprocess plumbing

Every small tool grows its own `subprocess.run(["claude", "-p", ...])` — argv quirks, stdin piping, exit-code guesswork — and each copy drifts. One call replaces all of it:

```python
from spawnllm import call_sync

print(call_sync("Reply with just the word: pong"))
```

Prints `pong`. With no `backend=`, spawnllm auto-selects the first installed, authenticated CLI, pipes the prompt over stdin, and retries transient 529/overloaded/rate-limit failures with capped backoff.

### Get a validated Pydantic object back, not a string to parse

Scraping JSON out of a model's stdout means regexes, code fences, and silent schema drift. `extract_sync` validates instead:

```python
from pydantic import BaseModel

from spawnllm import extract_sync


class Capital(BaseModel):
    country: str
    capital: str


result = extract_sync("What is the capital of France?", Capital)
print(result.capital)  # Paris
```

The backend turns `Capital` into a JSON-schema constraint on the call itself, and a non-conforming reply raises `pydantic.ValidationError` instead of sneaking downstream.

### Run Apple-Silicon MLX models with fused adapters and prompt-cache reuse

Shipping a LoRA-tuned local model means hand-rolling adapter fusion, model caching, and worker-thread lifecycle. The MLX extra owns all three:

```bash
uv add "spawnllm[mlx]"
```

`AdapterFuser.ensure_fused` fuses your compressed adapter into the base model once and caches the result in the Hugging Face hub layout; `MlxEngine` loads it on a dedicated worker thread, precomputes a prompt cache for your shared prefix messages, and batches generation. Wrap the engine in an `MlxBackend` and the same `run_sync` call works.

### Call the same backends from Go or Rust

The bindings ship the identical engine: argv planning, output parsing, schema strictification, and retry policy compile from one Rust core, pinned to the Python behavior by a shared golden-vector suite and released in lockstep.

```bash
go get github.com/yasyf/spawnllm/go   # pure Go, no cgo — the core embeds as WASM
cargo add spawnllm                    # async-first, with a blocking mirror
```

Both expose `Call`/`call` and typed `Extract`/`extract` against your existing CLI logins — see the [Go README](https://github.com/yasyf/spawnllm/tree/main/go) and the [Rust README](https://github.com/yasyf/spawnllm/tree/main/rust/spawnllm). MLX stays Python-only.

## More in the docs

- **Spec-driven runs** — a literal model id, per-provider flag passthrough, and envelope-aware retry via `RunSpec` — [Running reference](https://yasyf.github.io/spawnllm/reference/#running)
- **Backend selection** — the priority chain, plus `specialty=` routing (`debugging` and `review` go to Codex, `general` to Claude) — [Backends reference](https://yasyf.github.io/spawnllm/reference/#backends)
- **Transport helpers** — `run_cli`, `collect_process`, and `map_concurrent`, the subprocess plumbing shared by every CLI backend — [Transport reference](https://yasyf.github.io/spawnllm/reference/#transport)
- **The CLI** — `spawnllm call`, `status`, and `backends` from any shell — [CLI reference](https://yasyf.github.io/spawnllm/reference/cli/)
- **MLX internals** — the adapter codec, fuser, and runtime patches behind the local engine — [MLX reference](https://yasyf.github.io/spawnllm/reference/#mlx)

Read the [docs](https://yasyf.github.io/spawnllm/) for the full guide and API reference. Licensed under [MIT](https://github.com/yasyf/spawnllm/blob/main/LICENSE).

# ![spawnllm](https://github.com/yasyf/spawnllm/raw/main/docs/assets/readme-banner.webp)

**Delete your subprocess wrappers around claude, codex, and gemini.** spawnllm subshells all three CLIs — or drives Claude in-process through the bundled Agent SDK — plus local MLX and Apple's on-device Foundation Models, and returns one Pydantic-validated Response, so the per-model plumbing you hand-rolled goes away.

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

Prints `pong`. With no `backend=`, spawnllm auto-selects the first installed, authenticated backend — a CLI backend gets the prompt over stdin — and retries transient 529/overloaded/rate-limit failures with capped backoff.

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

### Keep billing on your subscription, not a stray API key

An `ANTHROPIC_API_KEY` left in your shell silently flips the `claude` CLI from your logged-in plan to per-token API billing. spawnllm strips each provider's key vars from the child environment by default, so every run bills the login:

```python
from spawnllm import call_sync

print(call_sync("Reply with just the word: pong"))
```

Prints `pong`, billed to your Claude plan even with `ANTHROPIC_API_KEY` exported. Pass `api_auth=True` to opt back into key auth. The same guard covers codex (`OPENAI_API_KEY`/`CODEX_API_KEY`) and the Gemini family, in Python, Go, and Rust alike; an explicit `RunSpec.env` entry always wins.

### Call Claude with zero installs

The `sdk` extra adds a backend over the Claude Agent SDK, whose wheel bundles the Claude Code CLI — no separate `claude` install:

```bash
uv add "spawnllm[sdk]"
```

`claude-sdk` registers first in the auto-selection chain and signs in with your existing subscription credentials (keychain login or `CLAUDE_CODE_OAUTH_TOKEN`), so the `call_sync` above works on a machine that has never installed the CLI.

### Run Apple-Silicon MLX models with fused adapters and prompt-cache reuse

Shipping a LoRA-tuned local model means hand-rolling adapter fusion, model caching, and worker-thread lifecycle. The MLX extra owns all three:

```bash
uv add "spawnllm[mlx]"
```

`AdapterFuser.ensure_fused` fuses your compressed adapter into the base model once and caches the result in the Hugging Face hub layout; `MlxEngine` loads it on a dedicated worker thread, precomputes a prompt cache for your shared prefix messages, and batches generation. Wrap the engine in an `MlxBackend` and the same `run_sync` call works.

### Call Apple's on-device model with zero downloads

Even local MLX starts with a multi-gigabyte model fetch. On a Mac with Apple Intelligence, `AppleBackend` skips that too: a prebuilt Swift sidecar inside the macOS wheel drives Apple's Foundation Models framework against the model already resident on the device. No extra, no compiler, no credentials, no network — installing spawnllm is the whole setup, `uvx spawnllm` included:

```python
from spawnllm import AppleBackend, call_sync

print(call_sync("Reply with just the word: pong", backend=AppleBackend()))
```

Auto-selection tries this backend last and only for `model="small"`; an explicit `backend=AppleBackend()` always reaches it. Session and decoding knobs (`use_case`, `guardrails`, `instructions`, `temperature`, sampling) ride in via `RunSpec(provider_configs={"apple": AppleConfig(...)})`. Structured `extract_sync` works too, nested models included, and schema constraints now bind during decoding: `minimum`/`maximum`, `minItems`/`maxItems`, and string-valued `enum`s are enforced exactly (a non-string `enum` such as `Literal[1, 2]` fails before generation, `extract_sync` raising `BackendCallError`), and a `Field(pattern=...)` constrains the value's shape — length, separators, and character families. Apple's decoder rejects bracket character classes, so the sidecar widens each to the narrowest escape it accepts (`^[A-Z]{3}-\d{4}$` decodes as `\w{3}-\d{4}`), and pydantic stays the exact validator: a value that fits the widened shape but violates your regex raises a plain `ValidationError`. Self-referential models extract cleanly; a mutually recursive pair (A referencing B referencing A) fails cleanly instead, `extract_sync` raising `BackendCallError` and `run` returning an error `Response`. Requires macOS 26+ on Apple Silicon with Apple Intelligence enabled — every other platform gets the pure-Python wheel and reports the backend as not installed.

### Call the same backends from Go or Rust

All three languages run the identical engine: argv planning, output parsing, schema strictification, and retry policy live once in a Rust core — linked natively by the Rust crate, embedded as WASM by the Go module and the Python package — pinned by a shared golden-vector suite and released in lockstep.

```bash
go get github.com/yasyf/spawnllm/go   # pure Go, no cgo — the core embeds as WASM
cargo add spawnllm                    # async-first, with a blocking mirror
```

Both expose `Call`/`call` and typed `Extract`/`extract` against your existing CLI logins, and both reach Apple's on-device model on a capable Mac with [binrun](https://github.com/yasyf/binrun) installed to fetch the digest-pinned sidecar — see the [Go README](https://github.com/yasyf/spawnllm/tree/main/go) and the [Rust README](https://github.com/yasyf/spawnllm/tree/main/rust/spawnllm). MLX stays Python-only.

## More in the docs

- **Spec-driven runs** — a literal model id, per-provider flag passthrough, and envelope-aware retry via `RunSpec` — [Running reference](https://yasyf.github.io/spawnllm/reference/#running)
- **Backend selection** — the priority chain, plus `specialty=` routing (`debugging` and `review` go to Codex, `general` to the Claude Agent SDK backend) — [Backends reference](https://yasyf.github.io/spawnllm/reference/#backends)
- **Transport helpers** — `run_cli`, `collect_process`, and `map_concurrent`, the subprocess plumbing shared by every CLI backend — [Transport reference](https://yasyf.github.io/spawnllm/reference/#transport)
- **The CLI** — `spawnllm call`, `status`, and `backends` from any shell — [CLI reference](https://yasyf.github.io/spawnllm/reference/cli/)
- **MLX internals** — the adapter codec, fuser, and runtime patches behind the local engine — [MLX reference](https://yasyf.github.io/spawnllm/reference/#mlx)

Read the [docs](https://yasyf.github.io/spawnllm/) for the full guide and API reference. Licensed under [MIT](https://github.com/yasyf/spawnllm/blob/main/LICENSE).

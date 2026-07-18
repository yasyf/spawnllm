# spawnllm for Go

**Delete your `exec.Command("claude", "-p", ...)` plumbing.** This module subshells the claude, codex, gemini, and agy CLIs (or POSTs to an OpenAI-compatible endpoint) and returns one typed `Response` — the same engine as the [Python package](https://github.com/yasyf/spawnllm), compiled from the shared Rust core to WASM and run under [wazero](https://github.com/tetratelabs/wazero), so `go get` needs no cgo and no extra toolchain.

[![Go Reference](https://pkg.go.dev/badge/github.com/yasyf/spawnllm/go.svg)](https://pkg.go.dev/github.com/yasyf/spawnllm/go)

## Get started

```bash
go get github.com/yasyf/spawnllm/go
```

Use a tagged version. The embedded WASM engine ships only on release tags (`go/vX.Y.Z`), so `@main` and commit pseudo-versions fail to compile.

```go
package main

import (
	"context"
	"fmt"
	"time"

	spawnllm "github.com/yasyf/spawnllm/go"
)

type Capital struct {
	City string `json:"city"`
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	text, err := spawnllm.Call(ctx, "Reply with exactly: pong", spawnllm.CallOpts{})
	if err != nil {
		panic(err)
	}
	fmt.Println("call:", text)

	capital, err := spawnllm.Extract[Capital](ctx, "What is the capital of France?", spawnllm.CallOpts{})
	if err != nil {
		panic(err)
	}
	fmt.Println("extract:", capital.City)
}
```

```
call: pong
extract: Paris
```

No API keys: the CLI backends drive the logins you already have, auto-selecting the first ready backend (claude, then codex, then agy). `Extract` turns your struct into a strict JSON-schema constraint on the call itself and unmarshals the validated reply.

## The contract

- `Run` and `RunOn` never surface provider failures as Go errors. Every outcome — nonzero exit, error envelope, per-attempt timeout — lands in `Response.Err`, with exactly one of `Response.Result` and `Response.Err` set. The `error` return is reserved for caller faults and `ctx` cancellation; `Call` and `Extract` unwrap the outcome for you and do return errors.
- `ctx` bounds the whole call across retries; `RunSpec.Timeout` bounds each attempt (default 180s). Transient failures — 529, overloaded, rate limits, 5xx — retry with capped backoff up to `MaxAttempts`.
- Runs are isolated by default: claude executes against a seeded throwaway `CLAUDE_CONFIG_DIR`, keeping your session config out of it. Set `UseHostConfig: true` to opt out.

## Backends

| Constructor | Binary | Auth |
|---|---|---|
| `ClaudeBackend()` | `claude` | `claude` login (keychain-aware on macOS) |
| `CodexBackend()` | `codex` | `codex login` |
| `AntigravityBackend()` | `agy` | keychain, or `GEMINI_API_KEY`/`ANTIGRAVITY_API_KEY` |
| `GeminiBackend()` | `gemini` | OAuth or `GEMINI_API_KEY`; never auto-selected |
| `OpenAIEndpoint(url, model, opts)` | — | any OpenAI-compatible `/chat/completions` server |

Local MLX inference stays Python-only — use the [Python package](https://github.com/yasyf/spawnllm) for that.

## Parity

Behavior lives once in the `spawnllm-core` Rust engine all three languages share — embedded here as a WASM blob (`CGO_ENABLED=0` throughout), linked natively by the Rust crate, and embedded via wasmtime by the Python package — with every vector of a shared golden suite replayed in CI. Versions release in lockstep; identical versions implement identical semantics.

Full API: [pkg.go.dev](https://pkg.go.dev/github.com/yasyf/spawnllm/go). Licensed under [MIT](https://github.com/yasyf/spawnllm/blob/main/LICENSE).

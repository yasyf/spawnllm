# spawnllm

**Delete your `Command::new("claude")` plumbing.** This crate subshells the claude, codex, gemini, and agy CLIs (or POSTs to an OpenAI-compatible endpoint) and returns one typed `Response` — the same engine as the [Python package](https://github.com/yasyf/spawnllm), sharing the sans-io [`spawnllm-core`](https://crates.io/crates/spawnllm-core) logic and its conformance contract.

## Get started

```bash
cargo add spawnllm
```

The crate ships this as `examples/call.rs` (`cargo run --example call`):

```rust
use spawnllm::{CallOpts, blocking};

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct Capital {
    city: String,
}

fn main() {
    let text = blocking::call("Reply with exactly: pong", CallOpts::default()).expect("call");
    println!("call: {text}");
    let capital: Capital =
        blocking::extract("What is the capital of France?", CallOpts::default()).expect("extract");
    println!("extract: {}", capital.city);
}
```

```
call: pong
extract: Paris
```

The API is async-first — `call`, `extract`, `run`, and `run_on` are `async fn`s on tokio, and the `blocking` module mirrors them for sync contexts. No API keys: the CLI backends drive the logins you already have, auto-selecting the first ready backend (claude, then codex, then agy). `extract` derives a strict JSON schema from your type via `schemars` and deserializes the validated reply.

## The contract

- `run(spec).await` is infallible at the boundary: every provider outcome — nonzero exit, error envelope, per-attempt timeout — lands in `Response::outcome` as a `Result<RunResult, RunError>`, with the raw transport stream preserved in `Response::output`. `call` and `extract` return `Result<_, Error>`.
- `RunSpec::new(prompt, model)` is a consuming builder. `.timeout(..)` bounds each attempt (default 180s), and transient failures — 529, overloaded, rate limits, 5xx — retry with capped backoff up to `.max_attempts(..)`.
- Runs are isolated by default: claude executes against a seeded throwaway `CLAUDE_CONFIG_DIR`, keeping your session config out of it. Call `.isolated(false)` to opt out.

## Features

`openai` (default) enables the `OpenAiEndpoint` backend for any OpenAI-compatible `/chat/completions` server, via `reqwest` with rustls. Build with `default-features = false` for a CLI-only crate with no HTTP stack.

## Parity

Behavior is pinned to the Python implementation by a shared golden-vector suite: argv construction, output parsing, schema strictification, and retry policy live once in `spawnllm-core`, linked natively here and embedded as WASM by the [Go module](https://pkg.go.dev/github.com/yasyf/spawnllm/go). Versions release in lockstep with the Python package; identical versions implement identical semantics. Local MLX inference stays Python-only.

Full API: [docs.rs/spawnllm](https://docs.rs/spawnllm). Licensed under [MIT](https://github.com/yasyf/spawnllm/blob/main/LICENSE).

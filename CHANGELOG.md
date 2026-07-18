# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Go (`github.com/yasyf/spawnllm/go`) and Rust (`spawnllm` on crates.io) bindings
  covering the full portable surface — `run`/`call`/`extract`, backend selection,
  and status — against the claude/codex/gemini/agy CLIs and OpenAI-compatible
  endpoints. All drift-prone logic lives once in the new `spawnllm-core` Rust
  crate: linked natively by the Rust crate, embedded as a wazero-run WASM blob by
  the Go module (no cgo), and pinned to the Python implementation by 117 shared
  conformance vectors generated from it (`tests/conformance/`). Versions release
  in lockstep across all three languages.

### Changed
- **Breaking:** `ClaudeConfig.tools` is now `tuple[str, ...] | None` instead of a
  raw flag string: `None` keeps the CLI's default toolset, `()` disables every
  built-in tool (emitted as `--tools ""`), and names spread variadically
  (`("Bash", "Read")` → `--tools Bash Read`). The old empty-string sentinel
  (`tools=""`) becomes `tools=()`.

## [0.6.2] - 2026-07-14

### Added
- `CodexConfig.service_tier` (default `"fast"`), emitted as `-c service_tier=<value>`
  on every `codex exec` invocation; `None` drops the flag. Isolated runs pass
  `--ignore-user-config`, which silently dropped a user's `service_tier = "fast"`
  pin in `~/.codex/config.toml` and turned xhigh/long prompts into 10-30+ minute runs.
- `CodexConfig.developer_instructions`, emitted as `-c developer_instructions=<value>`
  and serialized as a TOML string so the value always arrives as a string (codex
  parses `-c` values as TOML; a bare `true` would otherwise coerce to a boolean).
- The codex argv always passes `--color never`, keeping ANSI codes out of the streamed log.

## [0.6.1] - 2026-07-13

### Added
- `OpenAiEndpointBackend` — an `LlmBackend` that POSTs to any OpenAI-compatible
  `/chat/completions` endpoint over raw `httpx` (no subprocess), with structured
  output via a strict `response_format` json_schema. Constructed explicitly with
  a `base_url` + literal `model`; never auto-selected. Its `transport` param
  injects an `httpx.AsyncBaseTransport` into the async client, so a caller can
  supply a record/replay caching transport.

## [0.5.5] - 2026-07-05

### Fixed
- The `codex` backend passes `--skip-git-repo-check`, so `codex exec` no longer
  refuses to run when `cwd` is an untrusted or non-git directory (it exited with
  "Not inside a trusted directory and `--skip-git-repo-check` was not specified").
  The read-only sandbox already confines the run, so the trust gate added nothing
  and only silenced structured verdicts made from scratch directories.

## [0.5.4] - 2026-07-02

### Fixed
- Isolated `claude` runs seed auth from the caller's effective config home
  (`$CLAUDE_CONFIG_DIR` when set, else `~/.claude`) instead of hard-coded `~/.claude`
  paths, and fall back to the home's macOS Keychain item
  (`Claude Code-credentials-<sha256(home)[:8]>`) when the home has no
  `.credentials.json` file. Sessions under a relocated config home (cc-pool) no longer
  run isolated spawns with stale or missing credentials.

## [0.4.0] - 2026-06-22

### Added

- `RunSpec` and `RunResult`: a spec-driven execution contract. `RunSpec` carries
  the common fields spawnllm translates per backend (`prompt`, literal `model`,
  `schema`, `agent`, `cwd`, `env`, `timeout`, `max_attempts`) plus
  `provider_configs`, and `RunResult` returns raw `stdout`/`stderr`/`returncode`.
- `ClaudeConfig`, `CodexConfig`, and `GeminiConfig`: typed provider configs that
  the matching backend applies, exposing each CLI's real flags (Claude's
  permission/mcp/system-prompt/settings/disallowed-tools/budget/turns zoo,
  Codex's sandbox knobs, Gemini's approval mode and extensions).
- `run`/`run_sync`: async-first raw entries that select a backend, retry the
  transient `529 Overloaded` envelope with backoff, and return a `RunResult`.
- `call`/`call_sync`: prompt-ergonomic entries that map tiers to literal models,
  serialize the response schema, and parse the structured result.
- `CliBackend`: the subprocess mid-class carrying the argv/`Invocation` machinery
  and spec-driven `build_command`.
- `MlxBackend`: an opt-in local backend adapting `MlxEngine` to the execution
  contract. It is constructed explicitly and never auto-selected.

### Changed

- The backend contract is now spec-driven: `LlmBackend` exposes
  `aexecute`/`execute` taking a `RunSpec`, and `CliBackend.build_command(spec)`
  builds the invocation from the spec's common fields and provider config.
- `call` is now `call`/`call_sync`: the canonical entry is async with the bare
  name, and the sync companion takes the `_sync` suffix.

### Removed

- BREAKING: `ClaudeCliBackend.cc_sentiment`, `build_argv`, `inline_system_prompt`,
  `verbose`, and the static `parse_result_envelope` are gone, along with the old
  synchronous `call` signature. Drive the backend through `RunSpec` +
  `ClaudeConfig` and `run`/`run_sync`/`call`/`call_sync` instead.

## [0.3.1] - 2026-06-19

### Added

- `call` accepts `cwd` and `timeout`, forwarding both to the spawned CLI so
  callers can run the backend in a chosen working directory and bound its
  runtime. `timeout` keeps the previous fixed value of 180 seconds as its default.

## [0.3.0] - 2026-06-18

### Changed

- `schema_for` is now a method on `LlmBackend`, so each backend emits the strict
  JSON schema its CLI expects. The Claude backend applies the Anthropic SDK's
  `transform_schema` and the Codex backend the OpenAI SDK's
  `to_strict_json_schema`. `openai` and `anthropic` join the runtime dependencies.
- Backends build their invocation through `backend.invocation`, a new
  `Invocation` seam carrying argv, stdin, an optional result file, and any temp
  files to clean up. The Codex backend now reads its final message from a `-o`
  file instead of its interactive stdout log.

### Removed

- The top-level `schema_for` function is gone; call `backend.schema_for(model)`
  instead. This is a breaking change to the public API.

## [0.2.0] - 2026-06-18

### Added

- Gemini-family fallback backends for when Claude and Codex are unavailable.
  `GeminiCliBackend` drives the `gemini` CLI and `AntigravityCliBackend` drives
  its `agy` successor, both authenticating OAuth-first with no injected API keys.
  `gemini` reads its credentials from `~/.gemini/oauth_creds.json` while `agy`
  reads the macOS keychain, and both produce structured output by prompt-injecting
  the JSON schema since neither CLI exposes a schema flag.
- `select_backend` returns the first installed and authenticated backend, trying
  claude, then codex, then agy, then gemini, and accepts an optional specialty to
  reorder that chain. It raises `BackendUnavailable` when none are ready. `call`
  now auto-selects a backend when none is passed.
- `spawnllm status` reports each backend's install and auth state and the
  selected backend. `gemini` and `antigravity` are now valid `spawnllm call
  --backend` choices and appear in `spawnllm backends`.

### Changed

- Status checking now spans every backend. `check_status` is a method on each
  `LlmBackend` and returns the provider-neutral `BackendReady`,
  `BackendNotInstalled`, or `BackendNotAuthenticated`.

### Removed

- The Claude-specific `ClaudeStatus`, `ClaudeReady`, `ClaudeNotInstalled`, and
  `ClaudeNotAuthenticated` types and the standalone `check_status` function are
  gone. Use `BackendStatus` and `backend.check_status` instead. This is a
  breaking change to the public API.

## [0.1.3] - 2026-06-10

### Fixed

- `parse_structured_output` now unwraps the single result-envelope object the
  Claude CLI emits with `--output-format json`. Previously only stream-json
  event lists were handled, so the envelope failed validation instead of
  yielding its `structured_output`.

## [0.1.2] - 2026-06-10

### Fixed

- The Claude backend no longer sets `CLAUDE_CODE_SIMPLE=1` in the subprocess
  environment. On current Claude CLIs the flag breaks claude.ai keychain auth,
  so every spawned call failed with "Not logged in".

## [0.1.1] - 2026-06-10

### Changed

- `parse_structured_output` and `extract_structured` are typed generically
  over the response model. Passing `response_model=SomeModel` returns
  `SomeModel`, not `str | BaseModel`, so consumers no longer need a cast.

## [0.1.0] - 2026-06-10

First release, published to PyPI as `spawnllm`.

### Added

- Claude and Codex CLI backends with structured Pydantic output and
  small/medium/large model tiers.
- Subprocess transport: `run_cli`, `arun_cli`, `collect_process`, and
  `map_concurrent`.
- Local MLX engine with adapter fusion, prompt-cache reuse, and batched
  generation.
- Click CLI: `spawnllm backends` and `spawnllm call`.

[Unreleased]: https://github.com/yasyf/spawnllm/compare/v0.6.2...HEAD
[0.6.2]: https://github.com/yasyf/spawnllm/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/yasyf/spawnllm/compare/v0.6.0...v0.6.1
[0.5.5]: https://github.com/yasyf/spawnllm/compare/v0.5.4...v0.5.5
[0.5.4]: https://github.com/yasyf/spawnllm/compare/v0.5.3...v0.5.4
[0.4.0]: https://github.com/yasyf/spawnllm/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/yasyf/spawnllm/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/yasyf/spawnllm/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yasyf/spawnllm/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/yasyf/spawnllm/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/yasyf/spawnllm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/yasyf/spawnllm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/yasyf/spawnllm/releases/tag/v0.1.0

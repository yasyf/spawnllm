# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.0] - 2026-07-28

### Changed
- **The Apple backend no longer needs the `apple` extra, `apple-fm-sdk`, or
  Xcode.** A prebuilt Swift sidecar ships inside the macOS platform wheel
  (`macosx_26_0_arm64`), so installing spawnllm on an Apple-Intelligence-capable
  Mac carries the backend with no extra, no compiler, and no network fetch at
  runtime — `uvx spawnllm` now reaches it too, where the sdist previously
  compiled a Swift package into every ephemeral environment. Linux and
  pre-macOS-26 machines get the pure-Python `py3-none-any` wheel and report the
  backend as not installed, exactly as before.

### Added
- Structured output on the Apple backend now enforces schema constraints
  during generation: the sidecar compiles `minimum`/`maximum`,
  `minItems`/`maxItems`, and string-valued `enum` into the framework's native
  constraints exactly (a non-string `enum` such as `Literal[1, 2]` fails the
  call before generation rather than dropping the constraint), and compiles
  `pattern` — stripped entirely in 0.11.0 — into a shape
  constraint covering length, separators, and character families, widening the
  bracket character classes Apple's decoder rejects to the narrowest escape it
  accepts (`^[A-Z]{3}-\d{4}$` decodes as `\w{3}-\d{4}`, measured at full
  extraction accuracy). pydantic remains the exact validator: a value that
  fits the widened shape but violates the regex raises `ValidationError`.
- The Go module and Rust crate gain the Apple backend. Previously the one
  in-process, Python-only backend, it is now an ordinary CLI-style provider
  planned by the shared core, so all three hosts drive Apple's on-device
  Foundation Models through the same sidecar. With no wheel to carry the
  binary, both hosts embed a binrun descriptor pinning the sidecar release
  build's version, size, and sha256: when `spawnllm-apple` is not on `PATH`,
  the host hands the descriptor to `binrun`, which fetches the archive from
  the GitHub release, verifies it against the pinned digest, caches it, and
  runs it. binrun is the one prerequisite — `brew install yasyf/tap/binrun`,
  or `go install github.com/yasyf/binrun/cmd/binrun@latest` — and without it
  the backend reports not installed. The platform requirement is Python's,
  unchanged: macOS 26+ on Apple Silicon with Apple Intelligence enabled.
- The backend behaves the same in all three hosts. Auto-selection skips it
  unless the request asks for the `small` tier, since the device hosts one
  small model; off macOS every host reports it not installed rather than
  fetching a binary the platform cannot run; and the `AppleConfig` sampling
  combinations Python has always rejected — `sampling_top`,
  `sampling_probability_threshold`, or `sampling_seed` set without
  `sampling="random"`, and `sampling_top` set together with
  `sampling_probability_threshold` — are now rejected by the shared core, so Go
  and Rust refuse them with the message Python raises.

### Fixed
- `AppleConfig.sampling_seed` is honored: a fixed seed now reproduces the same
  generation run over run. The old backend dropped the seed in `apple-fm-sdk`'s
  C bridge, so seeded runs still sampled freely; the sidecar calls Swift's
  `SamplingMode` directly. A negative seed now raises `ValueError` — the
  framework takes a `UInt64`.
- Concurrent Apple runs no longer contend: each request is its own sidecar
  subprocess with no shared session lock, so parallel `asyncio.gather` calls
  all succeed where the in-process SDK raised `ConcurrentRequestsError`.

### Removed
- The `apple` extra and the `apple-fm-sdk` dependency.

## [0.11.0] - 2026-07-27

### Added
- `AppleBackend` (`apple`): an in-process backend over Apple's on-device
  Foundation Models framework, via the optional `apple-fm-sdk` package
  (`uv pip install 'spawnllm[apple]'`). No credential, no network, no model
  download — generation runs against the model Apple Intelligence keeps
  resident on the device, so it needs macOS 26+ on Apple Silicon with Apple
  Intelligence enabled; the SDK is sdist-only and compiles a Swift dylib at
  install time, and its build backend rejects the Command Line Tools, so
  installing needs full Xcode 26+. Auto-selection tries it last, after every
  CLI backend, and only for `model="small"` — `medium`, `large`, a concrete
  provider model id, or no model never reach it automatically; an explicit
  `backend=AppleBackend()` always does. Generation is Python-only, like MLX.
- `AppleConfig`: the session and decoding knobs the Apple backend applies —
  `use_case` (`"general"`/`"content_tagging"`), `guardrails`
  (`"default"`/`"permissive_content_transformations"`), `instructions`,
  `temperature`, `maximum_response_tokens`, and the flat sampling knobs
  (`sampling` as `"greedy"`/`"random"` plus `sampling_top`,
  `sampling_probability_threshold`, `sampling_seed`). Passed via
  `RunSpec(provider_configs={"apple": AppleConfig(...)})`.
- Structured output on the Apple backend dispatches through the core's new
  `apple` strict-schema dialect, nested response models included. Apple's
  schema importer rejects JSON Schema `pattern`, so the dialect strips it: a
  `Field(pattern=...)` constraint is not enforced during generation and can
  still fail `model_validate` afterward. Self-referential models extract
  cleanly; only a mutually recursive pair (A referencing B referencing A)
  degrades, coming back as an error `Response` rather than a crash.

## [0.10.0] - 2026-07-19

### Added
- `ClaudeSdkBackend` (`claude-sdk`): an in-process Claude backend over the
  `claude-agent-sdk` package, whose wheel bundles the Claude Code CLI — no
  install beyond `uv add "spawnllm[sdk]"`. It authenticates with the same
  ambient subscription OAuth as the standalone CLI (keychain login or
  `CLAUDE_CODE_OAUTH_TOKEN`), registers first in the auto-selection chain, and
  takes the `general` specialty; schema strictification and result resolution
  still dispatch through the core's claude dialect. Python-only, like MLX —
  the core and the Go/Rust bindings never learn the provider.
- `RunSpec.api_auth` (default `False`), threaded through `call`/`extract`, the
  `spawnllm call --api-auth` flag, the Rust builder, and the Go spec.
- The `capabilities` op exposes each provider's API-key env vars as
  `api_key_vars`.

### Changed
- **Child processes no longer inherit provider API-key env vars by default**,
  in all three languages: every exec plan carries `env_unset` (claude:
  `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`; codex:
  `OPENAI_API_KEY`/`CODEX_API_KEY`; gemini: `GEMINI_API_KEY`/`GOOGLE_API_KEY`;
  antigravity: `GEMINI_API_KEY`/`ANTIGRAVITY_API_KEY`), and each host strips
  those keys from the inherited environment before the plan and spec overlays.
  A stray exported key can no longer silently switch billing from your CLI
  login to API credits; opt back in with `api_auth=True`. An explicit
  `RunSpec.env` entry always survives the strip. Auth probes still read the
  ambient environment, so a Gemini-family backend that reports ready via an
  API key alone needs `api_auth=True` at run time.

## [0.9.1] - 2026-07-19

### Fixed
- A nonzero-exit `claude` run now surfaces the CLI's own failure reason: the
  core's `resolve` op reads the error envelope claude writes to *stdout* on
  failure ("Failed to authenticate: OAuth session expired and could not be
  refreshed", "You've hit your session limit · resets 9pm") instead of the
  usually-empty stderr, which rendered every such failure as a blank
  `claude exited 1: `. Envelope cost/usage accounting on the exit path is
  unchanged. When no error envelope is present the stderr tail is used as
  before, now falling back to the raw stdout tail when stderr is empty (all
  providers).

## [0.9.0] - 2026-07-19

### Changed
- **The Python package now runs on `spawnllm-core`**, the same Rust engine the Go
  and Rust bindings use: argv planning, output parsing, strict-schema transforms,
  JSON extraction, retry policy, auth probes, and Claude isolation seeding all
  dispatch through the core's wasm32-wasip1 build, loaded via `wasmtime`. The
  hand-written Python duplicates are gone; Python owns only I/O — subprocess
  spawning, temp files, the keychain, HTTP, and the MLX engine. Public API is
  unchanged.
- The conformance oracle inverted: `rust/conformance-gen` now generates the
  golden vectors from the core (byte-identical to the retired Python generator
  on the whole committed corpus), and Python replays them through its wasm glue
  exactly like Go does. A behavior change now lands core-first: edit the Rust
  core, regenerate vectors, and every host inherits it.
- The wasm blob is no longer committed; every surface builds it from source
  (`bash scripts/build_wasm.sh`, CI artifact job, release workflows). Wheels and
  sdists ship the freshly built blob and stay `py3-none-any`.
- Edge-case output now matches the core's serialization: large-exponent floats
  format as `1e20` (not `1e+20`), the Claude isolation seed writes sorted-key
  raw-UTF-8 JSON, a `RunSpec.schema` given as a pre-serialized *string* is
  parsed and re-serialized rather than passed through byte-for-byte, and JSON
  extraction recognizes only objects and arrays — a bare scalar reply no longer
  parses as a structured value (it never could through Go or Rust).
- Tests that patched per-backend auth internals patch
  `spawnllm.backends.base.subprocess.run` instead.

### Removed
- `openai` and `anthropic` are no longer runtime dependencies — the core's
  `strict_schema` op replaces their private transform functions. They remain in
  the `dev` extra, where a cross-check test pins the core against the live SDK
  output.
- The Python conformance generator (`tests/conformance/`) and the per-backend
  argv/parse internals it snapshotted (`build_command` bodies, `result_text`,
  `result_value`, `envelope_error`, `schema_arg`, `structured.first_json_value`,
  `is_transient`, `backoff`).

## [0.8.0] - 2026-07-19

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

[Unreleased]: https://github.com/yasyf/spawnllm/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/yasyf/spawnllm/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/yasyf/spawnllm/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/yasyf/spawnllm/compare/v0.9.1...v0.10.0
[0.9.1]: https://github.com/yasyf/spawnllm/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/yasyf/spawnllm/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/yasyf/spawnllm/compare/v0.6.2...v0.8.0
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

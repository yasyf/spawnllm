# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/yasyf/spawnllm/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/yasyf/spawnllm/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/yasyf/spawnllm/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yasyf/spawnllm/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/yasyf/spawnllm/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/yasyf/spawnllm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/yasyf/spawnllm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/yasyf/spawnllm/releases/tag/v0.1.0

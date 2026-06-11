# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/yasyf/spawnllm/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/yasyf/spawnllm/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/yasyf/spawnllm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/yasyf/spawnllm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/yasyf/spawnllm/releases/tag/v0.1.0

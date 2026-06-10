# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2]

### Fixed
- `ClaudeCliBackend.env()` no longer sets `CLAUDE_CODE_SIMPLE=1`: on current
  Claude CLIs the flag breaks claude.ai keychain auth (every spawned call fails
  with "Not logged in"). The argv already trims startup via
  `--setting-sources ""` and `--strict-mcp-config`.

## [0.1.1]

### Changed
- `parse_structured_output` and `extract_structured` are now typed generically
  over the pydantic model: passing `response_model=SomeModel` returns
  `SomeModel`, not `str | BaseModel`, so consumers no longer need a cast.

## [0.1.0]

### Added
- Initial scaffolding.

[0.1.2]: https://github.com/yasyf/spawnllm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/yasyf/spawnllm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/yasyf/spawnllm/commits/v0.1.0

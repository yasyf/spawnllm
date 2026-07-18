"""The single conformance case table.

Every permutation the oracle snapshots lives here as data — `generate.py` reads
these lists, drives the real `spawnllm` implementation, and writes the golden
vectors. Cases are grouped by wire op; each carries a stable `name` that becomes
its vector filename.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from spawnllm.spec import ClaudeConfig, CodexConfig, GeminiConfig, RunSpec

if TYPE_CHECKING:
    from spawnllm.types import ProviderName


@dataclass(frozen=True)
class PlanCase:
    name: str
    provider: ProviderName
    spec: RunSpec


@dataclass(frozen=True)
class EndpointPlanCase:
    name: str
    base_url: str
    model: str
    api_key: str
    spec: RunSpec


@dataclass(frozen=True)
class ResolveCase:
    name: str
    provider: ProviderName
    raw: str
    returncode: int
    stderr: str
    wants_value: bool


@dataclass(frozen=True)
class StrictSchemaCase:
    name: str
    model: type[BaseModel]


@dataclass(frozen=True)
class TextCase:
    name: str
    text: str


@dataclass(frozen=True)
class RetryCase:
    name: str
    attempt: int
    max_attempts: int
    error_msg: str | None


@dataclass(frozen=True)
class AuthProbeCase:
    name: str
    provider: ProviderName
    platform: str
    home: str


@dataclass(frozen=True)
class IsolationSourcesCase:
    name: str
    platform: str
    home: str
    claude_config_dir_env: str | None


@dataclass(frozen=True)
class IsolationSeedCase:
    name: str
    account_json: str | None
    credentials_json: str | None


SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
HOME_DARWIN = "/Users/testuser"
HOME_LINUX = "/home/testuser"


def _claude(name: str, cfg: ClaudeConfig, **spec_kwargs: object) -> PlanCase:
    spec = RunSpec(prompt="hi", model="haiku", provider_configs={"claude": cfg}, **spec_kwargs)
    return PlanCase(name, "claude", spec)


PLAN_CASES: tuple[PlanCase, ...] = (
    # --- claude: three-way flag resolution, isolation, extras ---
    PlanCase("claude-default", "claude", RunSpec(prompt="hi", model="haiku")),
    PlanCase("claude-isolated-false", "claude", RunSpec(prompt="hi", model="haiku", isolated=False)),
    PlanCase("claude-agent", "claude", RunSpec(prompt="hi", model="opus", agent=True)),
    _claude("claude-permission-mode", ClaudeConfig(permission_mode="bypassPermissions")),
    _claude("claude-mcp-config", ClaudeConfig(mcp_config='{"mcpServers":{}}')),
    _claude("claude-append-system-prompt", ClaudeConfig(append_system_prompt="extra instructions")),
    _claude("claude-system-prompt", ClaudeConfig(system_prompt="You are terse.")),
    _claude("claude-settings", ClaudeConfig(settings='{"model":"opus"}')),
    _claude("claude-disallowed-tools", ClaudeConfig(disallowed_tools=("Bash", "Write"))),
    PlanCase(
        "claude-strict-mcp",
        "claude",
        RunSpec(prompt="hi", model="haiku", isolated=False, provider_configs={"claude": ClaudeConfig(strict_mcp=True)}),
    ),
    _claude("claude-max-turns", ClaudeConfig(max_turns=3)),
    # max_budget_usd only emits inside the explicit branch, so it needs a co-triggering explicit field.
    _claude("claude-explicit-max-budget", ClaudeConfig(permission_mode="acceptEdits", max_budget_usd=2.5)),
    _claude("claude-tools-empty", ClaudeConfig(tools=())),
    _claude("claude-tools-list", ClaudeConfig(tools=("Bash", "Read"))),
    _claude("claude-disable-slash-commands", ClaudeConfig(disable_slash_commands=True)),
    _claude("claude-output-format", ClaudeConfig(output_format="stream-json")),
    _claude("claude-verbose", ClaudeConfig(verbose=True)),
    PlanCase("claude-schema", "claude", RunSpec(prompt="hi", model="haiku", schema=SCHEMA)),
    PlanCase(
        "claude-full-explicit-agent",
        "claude",
        RunSpec(
            prompt="hi",
            model="opus",
            agent=True,
            schema=SCHEMA,
            provider_configs={
                "claude": ClaudeConfig(
                    permission_mode="bypassPermissions",
                    mcp_config='{"mcpServers":{}}',
                    strict_mcp=True,
                    disallowed_tools=("Bash",),
                    append_system_prompt="extra",
                    settings='{"model":"opus"}',
                    max_budget_usd=5.0,
                    system_prompt="terse",
                    max_turns=2,
                    tools=("Read",),
                    disable_slash_commands=True,
                    verbose=True,
                )
            },
        ),
    ),
    # --- codex: service tier, effort suffix, feature toggles, TOML developer_instructions traps ---
    PlanCase("codex-default", "codex", RunSpec(prompt="hi", model="gpt-5.5")),
    PlanCase("codex-isolated-false", "codex", RunSpec(prompt="hi", model="gpt-5.5", isolated=False)),
    PlanCase("codex-agent", "codex", RunSpec(prompt="hi", model="gpt-5.4-mini", agent=True)),
    PlanCase("codex-effort-suffix", "codex", RunSpec(prompt="hi", model="gpt-5.4-mini:medium")),
    PlanCase(
        "codex-service-tier-none",
        "codex",
        RunSpec(prompt="hi", model="gpt-5.5", provider_configs={"codex": CodexConfig(service_tier=None)}),
    ),
    PlanCase(
        "codex-sandbox-override",
        "codex",
        RunSpec(prompt="hi", model="gpt-5.5", provider_configs={"codex": CodexConfig(sandbox="workspace-write")}),
    ),
    PlanCase(
        "codex-enable-hooks",
        "codex",
        RunSpec(prompt="hi", model="gpt-5.5", provider_configs={"codex": CodexConfig(enable_hooks=True)}),
    ),
    PlanCase(
        "codex-enable-mcp",
        "codex",
        RunSpec(prompt="hi", model="gpt-5.5", provider_configs={"codex": CodexConfig(enable_mcp=True)}),
    ),
    PlanCase("codex-schema", "codex", RunSpec(prompt="hi", model="gpt-5.5", schema=SCHEMA)),
    PlanCase(
        "codex-dev-instructions-multiline",
        "codex",
        RunSpec(
            prompt="hi",
            model="gpt-5.5",
            provider_configs={"codex": CodexConfig(developer_instructions="Be terse.\nCite sources.")},
        ),
    ),
    PlanCase(
        "codex-dev-instructions-literal-true",
        "codex",
        RunSpec(prompt="hi", model="gpt-5.5", provider_configs={"codex": CodexConfig(developer_instructions="true")}),
    ),
    PlanCase(
        "codex-dev-instructions-unicode",
        "codex",
        RunSpec(
            prompt="hi",
            model="gpt-5.5",
            provider_configs={"codex": CodexConfig(developer_instructions="café ☕ 你好")},
        ),
    ),
    PlanCase(
        "codex-dev-instructions-control-char",
        "codex",
        RunSpec(
            prompt="hi",
            model="gpt-5.5",
            provider_configs={"codex": CodexConfig(developer_instructions="a\x7fb")},
        ),
    ),
    # --- gemini: approval mode, extension arms, agent, schema ---
    PlanCase("gemini-default", "gemini", RunSpec(prompt="hi", model="gemini-2.5-flash")),
    PlanCase("gemini-agent", "gemini", RunSpec(prompt="hi", model="gemini-2.5-flash", agent=True)),
    PlanCase(
        "gemini-approval-mode",
        "gemini",
        RunSpec(prompt="hi", model="gemini-2.5-flash", provider_configs={"gemini": GeminiConfig(approval_mode="auto")}),
    ),
    PlanCase(
        "gemini-extensions-list",
        "gemini",
        RunSpec(
            prompt="hi",
            model="gemini-2.5-flash",
            provider_configs={"gemini": GeminiConfig(extensions=("search", "fs"))},
        ),
    ),
    PlanCase(
        "gemini-extensions-empty",
        "gemini",
        RunSpec(prompt="hi", model="gemini-2.5-flash", provider_configs={"gemini": GeminiConfig(extensions=())}),
    ),
    PlanCase("gemini-schema", "gemini", RunSpec(prompt="hi", model="gemini-2.5-flash", schema=SCHEMA)),
    # --- antigravity ---
    PlanCase("antigravity-default", "antigravity", RunSpec(prompt="hi", model="gemini-3.5")),
    PlanCase("antigravity-agent", "antigravity", RunSpec(prompt="hi", model="gemini-3.5", agent=True)),
    PlanCase("antigravity-schema", "antigravity", RunSpec(prompt="hi", model="gemini-3.5", schema=SCHEMA)),
)


ENDPOINT_PLAN_CASES: tuple[EndpointPlanCase, ...] = (
    EndpointPlanCase(
        "openai-endpoint-plain", "http://local.test/v1", "qwen3", "sk-test", RunSpec(prompt="ping", model="qwen3")
    ),
    EndpointPlanCase(
        "openai-endpoint-schema",
        "http://local.test/v1",
        "qwen3",
        "sk-test",
        RunSpec(prompt="ping", model="qwen3", schema=SCHEMA),
    ),
)


def _claude_result(**fields: object) -> str:
    import json

    return json.dumps({"type": "result", **fields})


RESOLVE_CASES: tuple[ResolveCase, ...] = (
    # --- claude ---
    ResolveCase("claude-ok-dict", "claude", _claude_result(is_error=False, result="hello world"), 0, "", False),
    ResolveCase(
        "claude-ok-stream-list",
        "claude",
        '[{"type": "system"}, {"type": "result", "is_error": false, "result": "answer", '
        '"structured_output": {"answer": "42"}}]',
        0,
        "",
        True,
    ),
    ResolveCase("claude-is-error", "claude", _claude_result(is_error=True, result="Overloaded"), 0, "", False),
    ResolveCase("claude-truncated-garbage", "claude", '{"type": "result", "resu', 0, "", False),
    ResolveCase("claude-exit-nonzero", "claude", "", 1, "claude: fatal boom", False),
    ResolveCase(
        "claude-float-cost",
        "claude",
        _claude_result(
            is_error=False, result="hi", total_cost_usd=0.0123, usage={"input_tokens": 12, "output_tokens": 7}
        ),
        0,
        "",
        False,
    ),
    ResolveCase(
        "claude-huge-int-usage",
        "claude",
        '{"type": "result", "is_error": false, "result": "hi", '
        '"usage": {"input_tokens": 999999999999999999999, "output_tokens": 7}}',
        0,
        "",
        False,
    ),
    # --- codex ---
    ResolveCase("codex-ok-text", "codex", "plain answer text", 0, "", False),
    ResolveCase("codex-ok-value", "codex", '{"block": true, "reason": "policy"}', 0, "", True),
    ResolveCase("codex-empty", "codex", "", 0, "", False),
    ResolveCase("codex-exit-nonzero", "codex", "", 42, "codex exec failed", False),
    # --- gemini ---
    ResolveCase(
        "gemini-ok",
        "gemini",
        '{"response": "hi there", "stats": {"models": {"g": {"api": {"totalErrors": 0}}}}}',
        0,
        "",
        False,
    ),
    ResolveCase(
        "gemini-total-errors",
        "gemini",
        '{"response": "", "stats": {"models": {"g": {"api": {"totalErrors": 1}}}}}',
        0,
        "",
        False,
    ),
    ResolveCase(
        "gemini-ok-value",
        "gemini",
        '{"response": "```json\\n{\\"answer\\": \\"7\\"}\\n```", '
        '"stats": {"models": {"g": {"api": {"totalErrors": 0}}}}}',
        0,
        "",
        True,
    ),
    # --- openai endpoint ---
    ResolveCase(
        "openai-ok",
        "openai_endpoint",
        '{"choices": [{"message": {"role": "assistant", "content": "pong"}}]}',
        0,
        "",
        False,
    ),
    ResolveCase("openai-http-error", "openai_endpoint", "service overloaded", 503, "service overloaded", False),
    ResolveCase("openai-2xx-error-body", "openai_endpoint", '{"error": {"message": "no such model"}}', 0, "", False),
    ResolveCase(
        "openai-ok-value",
        "openai_endpoint",
        '{"choices": [{"message": {"role": "assistant", "content": "{\\"answer\\": \\"9\\"}"}}]}',
        0,
        "",
        True,
    ),
)


class Color(StrEnum):
    red = "red"
    blue = "blue"


class Flat(BaseModel):
    x: int


class OptionalField(BaseModel):
    a: int
    b: str | None = None


class Inner(BaseModel):
    a: int
    b: str = "default"


class NestedRefs(BaseModel):
    name: str
    inner: Inner
    tags: list[str]
    color: Color
    opt: int | None = None


class Arrays(BaseModel):
    items: list[int]
    matrix: list[list[str]]


class Enums(BaseModel):
    color: Color


class Defaults(BaseModel):
    a: int = 1
    b: str = "hi"
    c: bool = False


class ListOfModels(BaseModel):
    rows: list[Inner]


class Union(BaseModel):
    val: int | str


class Constrained(BaseModel):
    n: int = Field(ge=1, le=10)
    label: str = Field(min_length=2)


STRICT_SCHEMA_CASES: tuple[StrictSchemaCase, ...] = (
    StrictSchemaCase("flat", Flat),
    StrictSchemaCase("optional-field", OptionalField),
    StrictSchemaCase("nested-refs", NestedRefs),
    StrictSchemaCase("arrays", Arrays),
    StrictSchemaCase("enums", Enums),
    StrictSchemaCase("defaults", Defaults),
    StrictSchemaCase("list-of-models", ListOfModels),
    StrictSchemaCase("union", Union),
    StrictSchemaCase("constrained", Constrained),
)


EXTRACT_JSON_CASES: tuple[TextCase, ...] = (
    TextCase("fenced-json", '```json\n{"x": 1}\n```'),
    TextCase("fenced-no-tag", '```\n{"x": 2}\n```'),
    TextCase("bare-object", '{"x": 3}'),
    TextCase("bare-array", "[1, 2, 3]"),
    TextCase("leading-prose", 'Here is the result: {"x": 4} — done.'),
    TextCase("trailing-prose", '{"x": 5}\n\nHope that helps!'),
    TextCase("first-value-wins", '{"first": 1} and then {"second": 2}'),
    TextCase("nested-braces-in-string", '{"path": "a{b}c", "n": 6}'),
    TextCase("no-json", "just some plain text with no json at all"),
)


RETRY_CASES: tuple[RetryCase, ...] = (
    RetryCase("transient-529-attempt-0", 0, 5, "Error 529 overloaded"),
    RetryCase("transient-rate-limit-attempt-1", 1, 5, "hit rate limit, retry later"),
    RetryCase("transient-503-attempt-2", 2, 5, "upstream returned 503"),
    RetryCase("transient-500-attempt-3-caps-at-60", 3, 5, "internal 500 error"),
    RetryCase("transient-overloaded-attempt-0", 0, 5, "the service is overloaded"),
    RetryCase("transient-last-attempt-no-retry", 4, 5, "529 overloaded"),
    RetryCase("transient-max-attempts-one", 0, 1, "529 overloaded"),
    RetryCase("non-transient-attempt-0", 0, 5, "invalid request: bad schema"),
    RetryCase("no-error-msg", 0, 5, None),
)


AUTH_PROBE_CASES: tuple[AuthProbeCase, ...] = (
    AuthProbeCase("claude-darwin", "claude", "darwin", HOME_DARWIN),
    AuthProbeCase("codex-darwin", "codex", "darwin", HOME_DARWIN),
    AuthProbeCase("gemini-darwin", "gemini", "darwin", HOME_DARWIN),
    AuthProbeCase("gemini-linux", "gemini", "linux", HOME_LINUX),
    AuthProbeCase("antigravity-darwin", "antigravity", "darwin", HOME_DARWIN),
    AuthProbeCase("antigravity-linux", "antigravity", "linux", HOME_LINUX),
    AuthProbeCase("openai-endpoint", "openai_endpoint", "darwin", HOME_DARWIN),
)


ISOLATION_SOURCES_CASES: tuple[IsolationSourcesCase, ...] = (
    IsolationSourcesCase("default-home-darwin", "darwin", HOME_DARWIN, None),
    IsolationSourcesCase("default-home-linux", "linux", HOME_LINUX, None),
    IsolationSourcesCase("config-dir-env-darwin", "darwin", HOME_DARWIN, "/Users/testuser/.acct"),
    IsolationSourcesCase("config-dir-env-linux", "linux", HOME_LINUX, "/home/testuser/.acct"),
)


ISOLATION_SEED_CASES: tuple[IsolationSeedCase, ...] = (
    IsolationSeedCase(
        "both-files-mcp-popped",
        '{"oauthAccount": {"accountUuid": "a"}, "mcpServers": {"semble": {"command": "x"}}}',
        '{"claudeAiOauth": {"accessToken": "tok"}}',
    ),
    IsolationSeedCase("account-only", '{"oauthAccount": {"accountUuid": "b"}, "mcpServers": {"s": {}}}', None),
    IsolationSeedCase("credentials-only", None, '{"claudeAiOauth": {"accessToken": "kc-tok"}}'),
    IsolationSeedCase("both-null", None, None),
    IsolationSeedCase("account-without-mcp-servers", '{"oauthAccount": {"accountUuid": "c"}}', None),
)

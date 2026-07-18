"""Oracle generator: snapshot the real `spawnllm` implementation into golden vectors.

Run as `uv run python -m tests.conformance.generate`. Each wire op is produced by
driving the actual package code (backend `build_command`/`invocation`, the private
SDK strict-schema transforms, the structured-output helpers, the registry tables),
never by re-deriving it — so a behavior change in `spawnllm` shows up as a vector
diff. Temp-file creation is intercepted so plans are deterministic: real paths
become `${file:ID}` semantic tokens.

The committed artifacts (`conformance/vectors/<op>/<case>.json` and
`conformance/schema/*.schema.json`) are exactly what `build_all()` returns;
`test_vectors_fresh.py` regenerates in memory and diffs.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import spawnllm.backends.claude as claude_mod
import spawnllm.backends.codex as codex_mod
import spawnllm.structured as structured_mod
from spawnllm.backends.base import CliBackend, LlmBackend
from spawnllm.backends.claude import ClaudeCliBackend, keychain_credentials
from spawnllm.backends.codex import CodexCliBackend
from spawnllm.backends.gemini import AntigravityCliBackend, GeminiCliBackend
from spawnllm.backends.openai_endpoint import OpenAiEndpointBackend
from spawnllm.backends.registry import BACKENDS_BY_NAME, PRIORITY, LlmBackends
from spawnllm.spec import ClaudeConfig, CodexConfig, GeminiConfig, RunSpec
from spawnllm.structured import TRANSIENT, backoff, extract_json_block
from tests.conformance import cases

if TYPE_CHECKING:
    from spawnllm.types import ProviderName

REPO_ROOT = Path(__file__).resolve().parents[2]
VECTORS_DIR = REPO_ROOT / "conformance" / "vectors"
SCHEMA_DIR = REPO_ROOT / "conformance" / "schema"

ISOLATED_CONFIG_DIR_TOKEN = "${isolated_config_dir}"

Vector = tuple[str, str, dict[str, object], dict[str, object]]

RESOLVE_BACKENDS: dict[ProviderName, LlmBackend] = {
    "claude": ClaudeCliBackend(),
    "codex": CodexCliBackend(),
    "gemini": GeminiCliBackend(),
    "antigravity": AntigravityCliBackend(),
    "openai_endpoint": OpenAiEndpointBackend("http://local.test/v1", "qwen3"),
}

CLI_BACKENDS: dict[ProviderName, CliBackend] = {
    "claude": ClaudeCliBackend(),
    "codex": CodexCliBackend(),
    "gemini": GeminiCliBackend(),
    "antigravity": AntigravityCliBackend(),
}


class TempInterceptor:
    """Deterministic `tempfile.mkstemp` stub that maps real paths to `${file:ID}` tokens.

    Each interception mints a counter-based token path and a real backing file
    (so the backend's `os.write`/`os.close` still work); the token is later mapped
    to a semantic id (`schema`/`result`/`stdout`) and its captured content read
    back for the plan's `files` entry.
    """

    def __init__(self) -> None:
        self.tokens: dict[str, tuple[str, str]] = {}
        self.counter = 0

    def mkstemp(
        self, suffix: str = "", prefix: str = "tmp", dir: str | None = None, text: bool = False
    ) -> tuple[int, str]:
        fd, real = _ORIG_MKSTEMP(suffix=suffix)
        token = f"/conformance/tmp{self.counter}{suffix}"
        self.counter += 1
        self.tokens[token] = (real, suffix)
        return fd, token

    def content(self, token: str) -> str:
        return Path(self.tokens[token][0]).read_text()

    def suffix(self, token: str) -> str:
        return self.tokens[token][1]

    def cleanup(self) -> None:
        for real, _ in self.tokens.values():
            Path(real).unlink(missing_ok=True)


_ORIG_MKSTEMP = tempfile.mkstemp


@contextlib.contextmanager
def _intercept_tempfiles() -> Iterator[TempInterceptor]:
    interceptor = TempInterceptor()
    saved_mkstemp = [mod.tempfile.mkstemp for mod in (claude_mod, codex_mod, structured_mod)]
    saved_isolated = ClaudeCliBackend._isolated_dir
    for mod in (claude_mod, codex_mod, structured_mod):
        mod.tempfile.mkstemp = interceptor.mkstemp
    ClaudeCliBackend._isolated_dir = lambda self: ISOLATED_CONFIG_DIR_TOKEN
    try:
        yield interceptor
    finally:
        for mod, mkstemp in zip((claude_mod, codex_mod, structured_mod), saved_mkstemp, strict=True):
            mod.tempfile.mkstemp = mkstemp
        ClaudeCliBackend._isolated_dir = saved_isolated
        interceptor.cleanup()


def _config_dict(cfg: object) -> dict[str, object] | None:
    return dataclasses.asdict(cfg) if cfg is not None else None


def to_portable(spec: RunSpec, *, openai_endpoint: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "prompt": spec.prompt,
        "model": spec.model,
        "schema": spec.schema if isinstance(spec.schema, dict) else None,
        "agent": spec.agent,
        "isolated": spec.isolated,
        "timeout": spec.timeout,
        "max_attempts": spec.max_attempts,
        "claude": _config_dict(spec.config_for(ClaudeConfig)),
        "codex": _config_dict(spec.config_for(CodexConfig)),
        "gemini": _config_dict(spec.config_for(GeminiConfig)),
        "openai_endpoint": openai_endpoint,
    }


def _semantic_id(token: str, stdout_path: str | None, result_path: str | None) -> str:
    match token:
        case _ if token == stdout_path:
            return "stdout"
        case _ if token == result_path:
            return "result"
        case _:
            return "schema"


def _cli_plan(provider: ProviderName, spec: RunSpec) -> dict[str, object]:
    backend = CLI_BACKENDS[provider]
    with _intercept_tempfiles() as interceptor:
        inv = backend.invocation(spec)
        env = backend.env(spec)
        ids = {token: _semantic_id(token, inv.stdout_path, inv.result_path) for token in interceptor.tokens}
        files = [
            {
                "id": ids[t],
                "suffix": interceptor.suffix(t),
                "content": interceptor.content(t) if ids[t] == "schema" else None,
            }
            for t in interceptor.tokens
        ]
        return {
            "kind": "exec",
            "argv": [f"${{file:{ids[arg]}}}" if arg in ids else arg for arg in inv.argv],
            "stdin": inv.stdin,
            "files": files,
            "stdout_to_file": inv.stdout_path is not None,
            "read_result_from": "file:result" if inv.result_path is not None else "stdout",
            "env": env,
            "needs_claude_isolation": provider == "claude" and spec.isolated,
        }


def _endpoint_plan(case: cases.EndpointPlanCase) -> dict[str, object]:
    backend = OpenAiEndpointBackend(case.base_url, case.model, api_key=case.api_key)
    return {
        "kind": "http",
        "method": "POST",
        "url": backend.url,
        "headers": backend.headers(),
        "body": backend.payload(case.spec),
    }


def plan_vectors() -> Iterator[Vector]:
    for case in cases.PLAN_CASES:
        yield (
            "plan",
            case.name,
            {"provider": case.provider, "spec": to_portable(case.spec), "host": {"platform": "darwin"}},
            _cli_plan(case.provider, case.spec),
        )
    for case in cases.ENDPOINT_PLAN_CASES:
        endpoint = {"base_url": case.base_url, "model": case.model, "api_key": case.api_key}
        yield (
            "plan",
            case.name,
            {
                "provider": "openai_endpoint",
                "spec": to_portable(case.spec, openai_endpoint=endpoint),
                "host": {"platform": "darwin"},
            },
            _endpoint_plan(case),
        )


def _resolve_error(kind: str, msg: str, cost_usd: float | None, usage: dict[str, object] | None) -> dict[str, object]:
    return {
        "status": "error",
        "kind": kind,
        "msg": msg,
        "transient": bool(TRANSIENT.search(msg)),
        "cost_usd": cost_usd,
        "usage": usage,
    }


def _resolve(case: cases.ResolveCase) -> dict[str, object]:
    backend = RESOLVE_BACKENDS[case.provider]
    cost_usd, usage = backend.accounting(case.raw)
    if case.returncode != 0:
        return _resolve_error(
            "exit", f"{backend.provider} exited {case.returncode}: {case.stderr.strip()[-2000:]}", cost_usd, usage
        )
    if (err := backend.envelope_error(case.raw)) is not None:
        return _resolve_error("envelope", err, cost_usd, usage)
    return {
        "status": "ok",
        "text": backend.result_text(case.raw),
        "value": backend.result_value(case.raw) if case.wants_value else None,
        "cost_usd": cost_usd,
        "usage": usage,
    }


def resolve_vectors() -> Iterator[Vector]:
    for case in cases.RESOLVE_CASES:
        yield (
            "resolve",
            case.name,
            {
                "provider": case.provider,
                "raw": case.raw,
                "returncode": case.returncode,
                "stderr": case.stderr,
                "wants_value": case.wants_value,
            },
            _resolve(case),
        )


def strict_schema_vectors() -> Iterator[Vector]:
    from anthropic.lib._parse._transform import transform_schema
    from openai.lib._pydantic import to_strict_json_schema

    for case in cases.STRICT_SCHEMA_CASES:
        yield (
            "strict_schema",
            f"{case.name}-anthropic",
            {"dialect": "anthropic", "schema": case.model.model_json_schema()},
            {"schema": transform_schema(case.model)},
        )
        yield (
            "strict_schema",
            f"{case.name}-openai",
            {"dialect": "openai", "schema": case.model.model_json_schema()},
            {"schema": to_strict_json_schema(case.model)},
        )


def _extract_json(text: str) -> object:
    try:
        return json.loads(extract_json_block(text))
    except ValueError:
        return None


def extract_json_vectors() -> Iterator[Vector]:
    for case in cases.EXTRACT_JSON_CASES:
        yield ("extract_json", case.name, {"text": case.text}, {"value": _extract_json(case.text)})


def _retry_decision(case: cases.RetryCase) -> dict[str, object]:
    transient = case.error_msg is not None and bool(TRANSIENT.search(case.error_msg))
    retry = transient and case.attempt + 1 < case.max_attempts
    return {"retry": retry, "sleep_s": float(backoff(case.attempt)) if retry else 0.0}


def retry_vectors() -> Iterator[Vector]:
    for case in cases.RETRY_CASES:
        yield (
            "retry_decision",
            case.name,
            {"attempt": case.attempt, "max_attempts": case.max_attempts, "error_msg": case.error_msg},
            _retry_decision(case),
        )


INSTALL_HINTS: dict[ProviderName, str] = {
    "claude": ClaudeCliBackend.install_hint,
    "codex": CodexCliBackend.install_hint,
    "gemini": GeminiCliBackend.install_hint,
    "antigravity": AntigravityCliBackend.install_hint,
}


def _auth_probes(case: cases.AuthProbeCase) -> dict[str, object]:
    match case.provider:
        case "claude":
            return {
                "binary": "claude",
                "install_hint": INSTALL_HINTS["claude"],
                "probes": [{"kind": "exec_exit0", "argv": ["claude", "auth", "status"]}],
            }
        case "codex":
            return {
                "binary": "codex",
                "install_hint": INSTALL_HINTS["codex"],
                "probes": [{"kind": "exec_exit0", "argv": ["codex", "login", "status"]}],
            }
        case "gemini":
            return {
                "binary": "gemini",
                "install_hint": INSTALL_HINTS["gemini"],
                "probes": [
                    {"kind": "file_exists", "path": f"{case.home}/.gemini/oauth_creds.json"},
                    {"kind": "env_any", "vars": list(GeminiCliBackend.api_key_envs)},
                ],
            }
        case "antigravity":
            keychain = (
                [{"kind": "keychain_exists", "service": "gemini", "account": "antigravity"}]
                if case.platform == "darwin"
                else []
            )
            return {
                "binary": "agy",
                "install_hint": INSTALL_HINTS["antigravity"],
                "probes": [*keychain, {"kind": "env_any", "vars": list(AntigravityCliBackend.api_key_envs)}],
            }
        case "openai_endpoint":
            return {"binary": "openai_endpoint", "install_hint": None, "probes": []}
        case _:
            raise ValueError(f"no auth probe mapping for {case.provider}")


def auth_probe_vectors() -> Iterator[Vector]:
    for case in cases.AUTH_PROBE_CASES:
        yield (
            "auth_probes",
            case.name,
            {"provider": case.provider, "host": {"platform": case.platform, "home": case.home}},
            _auth_probes(case),
        )


PROVIDER_BINARIES: dict[str, str] = {"claude": "claude", "codex": "codex", "antigravity": "agy", "gemini": "gemini"}


def _capabilities() -> dict[str, object]:
    name_by_type = {type(b): n for n, b in BACKENDS_BY_NAME.items()}
    return {
        "providers": list(BACKENDS_BY_NAME),
        "priority": [name_by_type[type(b)] for b in PRIORITY],
        "auto_select_excludes": ["gemini"],
        "specialties": {s: name_by_type[type(b)] for s, b in LlmBackends.LLM_BACKENDS.items()},
        "models": {name: dict(backend.models) for name, backend in BACKENDS_BY_NAME.items()},
        "binaries": PROVIDER_BINARIES,
        "install_hints": dict(INSTALL_HINTS),
    }


def capabilities_vectors() -> Iterator[Vector]:
    yield ("capabilities", "capabilities", {}, _capabilities())


def _keychain_service(home: str) -> str:
    """Capture the exact service name the real `keychain_credentials` derives."""
    captured: dict[str, list[str]] = {}
    saved = claude_mod.subprocess.run

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        return subprocess.CompletedProcess(args=argv, returncode=1, stdout="", stderr="")

    saved_platform = sys.platform
    claude_mod.subprocess.run = fake_run
    sys.platform = "darwin"
    try:
        keychain_credentials(Path(home))
    finally:
        claude_mod.subprocess.run = saved
        sys.platform = saved_platform
    argv = captured["argv"]
    return argv[argv.index("-s") + 1]


def _isolation_sources(case: cases.IsolationSourcesCase) -> dict[str, object]:
    # Mirrors ClaudeBase._isolated_dir path derivation; keychain service hashes str(config_home).
    env = case.claude_config_dir_env
    config_home = env if env is not None else f"{case.home}/.claude"
    return {
        "account_path": f"{env}/.claude.json" if env is not None else f"{case.home}/.claude.json",
        "credentials_path": f"{config_home}/.credentials.json",
        "keychain_service": _keychain_service(config_home) if case.platform == "darwin" else None,
    }


def isolation_sources_vectors() -> Iterator[Vector]:
    for case in cases.ISOLATION_SOURCES_CASES:
        yield (
            "claude_isolation_sources",
            case.name,
            {
                "host": {
                    "platform": case.platform,
                    "home": case.home,
                    "claude_config_dir_env": case.claude_config_dir_env,
                }
            },
            _isolation_sources(case),
        )


def _isolation_seed(case: cases.IsolationSeedCase) -> dict[str, object]:
    # Mirrors ClaudeBase._isolated_dir seeding: account minus mcpServers at 0644, token verbatim at 0600.
    files: list[dict[str, object]] = []
    if case.account_json is not None:
        account = json.loads(case.account_json)
        account.pop("mcpServers", None)
        files.append({"name": ".claude.json", "content": json.dumps(account), "mode": "0644"})
    if case.credentials_json is not None:
        files.append({"name": ".credentials.json", "content": case.credentials_json, "mode": "0600"})
    return {"files": files}


def isolation_seed_vectors() -> Iterator[Vector]:
    for case in cases.ISOLATION_SEED_CASES:
        yield (
            "claude_isolation_seed",
            case.name,
            {"account_json": case.account_json, "credentials_json": case.credentials_json},
            _isolation_seed(case),
        )


DRAFT = "https://json-schema.org/draft/2020-12/schema"

RUN_SPEC_SCHEMA: dict[str, object] = {
    "$schema": DRAFT,
    "title": "RunSpec",
    "description": "Portable run configuration (snake_case; response_model is host-side sugar and absent).",
    "type": "object",
    "required": ["prompt", "model", "agent", "isolated", "timeout", "max_attempts"],
    "properties": {
        "prompt": {"type": "string"},
        "model": {"type": "string"},
        "schema": {"type": ["object", "null"]},
        "agent": {"type": "boolean"},
        "isolated": {"type": "boolean"},
        "timeout": {"type": "integer"},
        "max_attempts": {"type": "integer"},
        "claude": {"type": ["object", "null"]},
        "codex": {"type": ["object", "null"]},
        "gemini": {"type": ["object", "null"]},
        "openai_endpoint": {
            "type": ["object", "null"],
            "properties": {
                "base_url": {"type": "string"},
                "model": {"type": "string"},
                "api_key": {"type": "string"},
            },
        },
    },
}

INVOCATION_PLAN_SCHEMA: dict[str, object] = {
    "$schema": DRAFT,
    "title": "InvocationPlan",
    "description": "A host-executable plan: a subprocess exec or an HTTP request.",
    "oneOf": [
        {
            "type": "object",
            "required": [
                "kind",
                "argv",
                "stdin",
                "files",
                "stdout_to_file",
                "read_result_from",
                "env",
                "needs_claude_isolation",
            ],
            "properties": {
                "kind": {"const": "exec"},
                "argv": {"type": "array", "items": {"type": "string"}},
                "stdin": {"type": "string"},
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "suffix", "content"],
                        "properties": {
                            "id": {"enum": ["schema", "result", "stdout"]},
                            "suffix": {"type": "string"},
                            "content": {"type": ["string", "null"]},
                        },
                    },
                },
                "stdout_to_file": {"type": "boolean"},
                "read_result_from": {"enum": ["stdout", "file:result"]},
                "env": {"type": "object", "additionalProperties": {"type": "string"}},
                "needs_claude_isolation": {"type": "boolean"},
            },
        },
        {
            "type": "object",
            "required": ["kind", "method", "url", "headers", "body"],
            "properties": {
                "kind": {"const": "http"},
                "method": {"type": "string"},
                "url": {"type": "string"},
                "headers": {"type": "object", "additionalProperties": {"type": "string"}},
                "body": {"type": "object"},
            },
        },
    ],
}

RESOLVED_SCHEMA: dict[str, object] = {
    "$schema": DRAFT,
    "title": "Resolved",
    "description": "A resolved provider outcome: exactly one of ok / error.",
    "oneOf": [
        {
            "type": "object",
            "required": ["status", "text", "value", "cost_usd", "usage"],
            "properties": {
                "status": {"const": "ok"},
                "text": {"type": "string"},
                "value": {},
                "cost_usd": {"type": ["number", "null"]},
                "usage": {"type": ["object", "null"]},
            },
        },
        {
            "type": "object",
            "required": ["status", "kind", "msg", "transient", "cost_usd", "usage"],
            "properties": {
                "status": {"const": "error"},
                "kind": {"enum": ["exit", "envelope", "parse"]},
                "msg": {"type": "string"},
                "transient": {"type": "boolean"},
                "cost_usd": {"type": ["number", "null"]},
                "usage": {"type": ["object", "null"]},
            },
        },
    ],
}

SCHEMAS: dict[str, dict[str, object]] = {
    "run_spec": RUN_SPEC_SCHEMA,
    "invocation_plan": INVOCATION_PLAN_SCHEMA,
    "resolved": RESOLVED_SCHEMA,
}

VECTOR_SOURCES = (
    plan_vectors,
    resolve_vectors,
    strict_schema_vectors,
    extract_json_vectors,
    retry_vectors,
    auth_probe_vectors,
    capabilities_vectors,
    isolation_sources_vectors,
    isolation_seed_vectors,
)


def render(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def build_all() -> dict[Path, str]:
    artifacts: dict[Path, str] = {}
    for source in VECTOR_SOURCES:
        for op, name, input_payload, expected in source():
            vector = {"name": name, "op": op, "input": input_payload, "expected": expected}
            artifacts[VECTORS_DIR / op / f"{name}.json"] = render(vector)
    for name, schema in SCHEMAS.items():
        artifacts[SCHEMA_DIR / f"{name}.schema.json"] = render(schema)
    return artifacts


def main() -> None:
    for stale in VECTORS_DIR.rglob("*.json"):
        stale.unlink()
    artifacts = build_all()
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(f"wrote {len(artifacts)} conformance artifacts under {REPO_ROOT}/conformance")


if __name__ == "__main__":
    main()

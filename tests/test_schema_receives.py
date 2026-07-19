"""Assert the schema a backend's CLI actually receives is strict.

Mirrors `go/schema_test.go`'s `TestExtractNestedReceivesStrictSchema`: fake
`claude` and `codex` executables on `PATH` capture the `--json-schema` /
`--output-schema` argument the backend hands them, then emit a canned success
payload the host validates. The captured schema — anthropic's `transform_schema`
for `claude`, openai's `to_strict_json_schema` for `codex` — must carry
`additionalProperties: false` on the root and every `$defs` object, with the
dialect's own `required` semantics (openai forces every property required;
anthropic preserves the model's optionality).
"""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from spawnllm import ClaudeCliBackend, CliBackend, CodexCliBackend, RunSpec, extract_sync

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class Address(BaseModel):
    city: str
    postcode: str


class Person(BaseModel):
    name: str
    address: Address
    neighbors: list[Address]
    note: str | None = None


PAYLOAD_JSON = json.dumps(
    {
        "name": "Ada",
        "address": {"city": "London", "postcode": "SW1A"},
        "neighbors": [{"city": "Paris", "postcode": "75001"}],
        "note": None,
    }
)


def write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)


def write_fake_claude(path: Path, sink: Path) -> None:
    write_executable(
        path,
        f"""\
import json, sys
argv = sys.argv[1:]
open({str(sink)!r}, "w").write(argv[argv.index("--json-schema") + 1])
print(json.dumps([
    {{"type": "system"}},
    {{"type": "result", "is_error": False, "result": "ok", "structured_output": json.loads({PAYLOAD_JSON!r})}},
]))
""",
    )


def write_fake_claude_env(path: Path, sink: Path) -> None:
    write_executable(
        path,
        f"""\
import json, os
open({str(sink)!r}, "w").write(
    "ANTHROPIC_API_KEY=" + os.environ.get("ANTHROPIC_API_KEY", "<unset>") + "\\n"
    "ANTHROPIC_AUTH_TOKEN=" + os.environ.get("ANTHROPIC_AUTH_TOKEN", "<unset>") + "\\n"
)
print(json.dumps([
    {{"type": "system"}},
    {{"type": "result", "is_error": False, "result": "ok", "structured_output": json.loads({PAYLOAD_JSON!r})}},
]))
""",
    )


def write_fake_codex(path: Path, sink: Path) -> None:
    write_executable(
        path,
        f"""\
import json, sys
argv = sys.argv[1:]
open({str(sink)!r}, "w").write(open(argv[argv.index("--output-schema") + 1]).read())
open(argv[argv.index("-o") + 1], "w").write({PAYLOAD_JSON!r})
print("interactive log line the host must ignore")
""",
    )


def object_nodes(node: object) -> Iterator[dict[str, object]]:
    match node:
        case dict():
            if node.get("type") == "object":
                yield node
            for value in node.values():
                yield from object_nodes(value)
        case list():
            for value in node:
                yield from object_nodes(value)


CASES = (
    pytest.param(ClaudeCliBackend, "anthropic", id="claude-receives-anthropic-strict-schema"),
    pytest.param(CodexCliBackend, "openai", id="codex-receives-openai-strict-schema"),
)

AUTH_ENV_CASES = (
    pytest.param(
        False,
        None,
        "ANTHROPIC_API_KEY=<unset>\nANTHROPIC_AUTH_TOKEN=<unset>\n",
        id="default-strips-parent-api-auth-env",
    ),
    pytest.param(
        True,
        None,
        "ANTHROPIC_API_KEY=parent-api-key\nANTHROPIC_AUTH_TOKEN=parent-auth-token\n",
        id="api-auth-keeps-parent-api-auth-env",
    ),
    pytest.param(
        False,
        {"ANTHROPIC_API_KEY": "spec-api-key"},
        "ANTHROPIC_API_KEY=spec-api-key\nANTHROPIC_AUTH_TOKEN=<unset>\n",
        id="explicit-spec-env-restores-stripped-api-auth-key",
    ),
)


@pytest.mark.parametrize("api_auth,spec_env,expected", AUTH_ENV_CASES)
def test_claude_child_process_api_auth_env(
    api_auth: bool,
    spec_env: dict[str, str] | None,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sink = tmp_path / "received-env.txt"
    write_fake_claude_env(bindir / "claude", sink)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    (isolated_home := tmp_path / "config").mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(isolated_home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "parent-api-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "parent-auth-token")

    response = ClaudeCliBackend().execute(
        RunSpec(prompt="hi", model="haiku", isolated=False, env=spec_env, api_auth=api_auth)
    )

    assert response.error is None
    assert response.result is not None
    assert response.result.raw == "ok"
    assert sink.read_text() == expected


@pytest.mark.parametrize("backend_cls,dialect", CASES)
def test_backend_receives_strict_schema(
    backend_cls: type[CliBackend],
    dialect: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    sink = tmp_path / "received-schema.json"
    write_fake_claude(bindir / "claude", sink)
    write_fake_codex(bindir / "codex", sink)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    (isolated_home := tmp_path / "config").mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(isolated_home))

    assert isinstance(extract_sync("extract the person", Person, backend=backend_cls()), Person)

    schema = json.loads(sink.read_text())
    assert schema.get("$defs"), "outer model must reference the inner model through $defs, not inline it"
    objects = list(object_nodes(schema))
    assert all(node.get("additionalProperties") is False for node in objects), schema
    match dialect:
        case "openai":
            assert all(set(node["required"]) == set(node["properties"]) for node in objects if "properties" in node)
        case "anthropic":
            assert set(schema["required"]) == {"name", "address", "neighbors"}
            assert "note" not in schema["required"]

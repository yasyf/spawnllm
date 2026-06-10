from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from spawnllm import ClaudeCliBackend, CodexCliBackend
from spawnllm.structured import parse_result_envelope, parse_structured_output, resolve_schema_path, schema_for


class Verdict(BaseModel):
    should_block: bool
    reason: str


class TestSchemaFor:
    def test_byte_identical_to_inline_build(self) -> None:
        assert schema_for(Verdict) == json.dumps(Verdict.model_json_schema() | {"additionalProperties": False})

    def test_additional_properties_false(self) -> None:
        assert json.loads(schema_for(Verdict))["additionalProperties"] is False


class TestResolveSchemaPath:
    def test_none_when_no_schema(self) -> None:
        assert resolve_schema_path(ClaudeCliBackend(), None) is None

    def test_claude_returns_schema_verbatim(self) -> None:
        assert resolve_schema_path(ClaudeCliBackend(), '{"x":1}') == '{"x":1}'

    def test_codex_writes_tempfile(self) -> None:
        path = resolve_schema_path(CodexCliBackend(), '{"x":1}')
        assert path is not None
        assert Path(path).read_text() == '{"x":1}'


class TestParseStructuredOutput:
    def test_text_passthrough_when_no_model(self) -> None:
        assert parse_structured_output("hello", None) == "hello"

    def test_event_list_structured_output(self) -> None:
        events = json.dumps([{"type": "result", "structured_output": {"should_block": True, "reason": "x"}}])
        result = parse_structured_output(events, Verdict)
        assert isinstance(result, Verdict)
        assert result.should_block is True

    def test_falls_back_to_model_validate_json(self) -> None:
        result = parse_structured_output('{"should_block": false, "reason": "ok"}', Verdict)
        assert isinstance(result, Verdict)
        assert result.should_block is False

    def test_single_result_envelope_structured_output(self) -> None:
        envelope = json.dumps(
            {
                "type": "result",
                "is_error": False,
                "result": "",
                "structured_output": {"should_block": True, "reason": "x"},
            }
        )
        result = parse_structured_output(envelope, Verdict)
        assert isinstance(result, Verdict)
        assert result.should_block is True


class TestParseResultEnvelope:
    def test_returns_result(self) -> None:
        assert parse_result_envelope(b'{"is_error": false, "result": "4"}', argv=["claude"], stderr=b"") == "4"

    def test_raises_zero_returncode_with_bytes(self) -> None:
        raw = b'{"is_error": true, "result": "x"}'
        with pytest.raises(subprocess.CalledProcessError) as exc:
            parse_result_envelope(raw, argv=["claude", "-p"], stderr=b"e")
        assert exc.value.returncode == 0
        assert exc.value.cmd == ["claude", "-p"]
        assert exc.value.output == raw
        assert exc.value.stderr == b"e"

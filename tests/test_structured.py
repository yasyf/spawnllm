from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import BaseModel

from spawnllm import ClaudeCliBackend, CodexCliBackend, GeminiCliBackend, LlmBackend
from spawnllm.structured import parse_result_envelope, parse_structured_output, resolve_schema_path


class Verdict(BaseModel):
    should_block: bool
    reason: str


class Leg(BaseModel):
    price: float
    note: str | None = None


class Window(BaseModel):
    name: str
    leg: Leg


class TestSchemaFor:
    @pytest.mark.parametrize("backend", [CodexCliBackend(), ClaudeCliBackend()], ids=["codex", "claude"])
    def test_strict_backends_set_additional_properties_false_recursively(self, backend: LlmBackend) -> None:
        schema = json.loads(backend.schema_for(Window))
        assert schema["additionalProperties"] is False
        assert schema["$defs"]["Leg"]["additionalProperties"] is False

    def test_codex_forces_all_properties_required(self) -> None:
        schema = json.loads(CodexCliBackend().schema_for(Window))
        assert set(schema["required"]) == {"name", "leg"}
        assert set(schema["$defs"]["Leg"]["required"]) == {"price", "note"}

    def test_claude_preserves_optional_fields(self) -> None:
        schema = json.loads(ClaudeCliBackend().schema_for(Window))
        assert set(schema["required"]) == {"name", "leg"}
        assert schema["$defs"]["Leg"]["required"] == ["price"]

    def test_gemini_emits_plain_schema(self) -> None:
        schema = json.loads(GeminiCliBackend().schema_for(Window))
        assert "additionalProperties" not in schema
        assert "additionalProperties" not in schema["$defs"]["Leg"]
        assert schema["$defs"]["Leg"]["properties"]["price"]["type"] == "number"


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

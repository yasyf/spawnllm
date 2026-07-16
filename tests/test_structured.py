from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from spawnllm import (
    ClaudeCliBackend,
    CodexCliBackend,
    Error,
    GeminiCliBackend,
    LlmBackend,
    Output,
    Response,
    Result,
    RunSpec,
)
from spawnllm.structured import (
    extract_json_block,
    is_transient,
    resolve_schema_path,
    structured_value,
)

SPEC = RunSpec(prompt="hi", model="haiku")


def err_response(msg: str) -> Response:
    return Response(spec=SPEC, output=Output(msg), error=Error(msg, RuntimeError(msg)))


def ok_response(text: str) -> Response:
    return Response(spec=SPEC, output=Output(text), result=Result(raw=text))


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


class TestStructuredValue:
    def test_event_list_structured_output(self) -> None:
        events = json.dumps([{"type": "result", "structured_output": {"should_block": True, "reason": "x"}}])
        assert structured_value(events) == {"should_block": True, "reason": "x"}

    def test_single_result_envelope_structured_output(self) -> None:
        envelope = json.dumps(
            {
                "type": "result",
                "is_error": False,
                "result": "",
                "structured_output": {"should_block": True, "reason": "x"},
            }
        )
        assert structured_value(envelope) == {"should_block": True, "reason": "x"}

    def test_falls_back_to_parsed_json(self) -> None:
        assert structured_value('{"should_block": false, "reason": "ok"}') == {"should_block": False, "reason": "ok"}

    def test_validates_through_model(self) -> None:
        events = json.dumps([{"type": "result", "structured_output": {"should_block": True, "reason": "x"}}])
        result = Verdict.model_validate(structured_value(events))
        assert result.should_block is True


class TestIsTransient:
    @pytest.mark.parametrize(
        "resp, expected",
        [
            (err_response("codex exited 1: API Error: 529 Overloaded"), True),
            (err_response("claude reported an error: Overloaded"), True),
            (err_response("gemini call failed: rate limit"), True),
            (ok_response("ok"), False),
            (err_response("codex exited 127: codex: not found"), False),
            (err_response("boom"), False),
        ],
        ids=[
            "exit-529-error",
            "overloaded-error",
            "rate-limit-error",
            "no-error",
            "nonzero-no-transient-text",
            "plain-error",
        ],
    )
    def test_classifies_by_error_text(self, resp: Response, expected: bool) -> None:
        assert is_transient(resp) is expected


class TestExtractJsonBlock:
    def test_fenced_block(self) -> None:
        assert json.loads(extract_json_block('```json\n{"x": 1}\n```')) == {"x": 1}

    def test_object_amid_prose(self) -> None:
        assert json.loads(extract_json_block('Sure, here you go: {"x": 1} hope that helps!')) == {"x": 1}

    def test_skips_invalid_brace_in_leading_prose(self) -> None:
        assert json.loads(extract_json_block('I considered {a, b} then chose {"x": 1}')) == {"x": 1}

    def test_ignores_trailing_noise(self) -> None:
        assert json.loads(extract_json_block('{"a": {"b": 2}} trailing } noise')) == {"a": {"b": 2}}

    def test_array_value(self) -> None:
        assert json.loads(extract_json_block("result: [1, 2, 3].")) == [1, 2, 3]

    def test_falls_back_to_full_text_when_fence_lacks_json(self) -> None:
        text = 'Here is some code:\n```\nnot json at all\n```\nAnswer: {"x": 1}'
        assert json.loads(extract_json_block(text)) == {"x": 1}

    def test_deeply_nested_raises_valueerror_not_recursionerror(self) -> None:
        with pytest.raises(ValueError, match="no JSON value"):
            extract_json_block("[" * 2000)

    def test_raises_when_no_json(self) -> None:
        with pytest.raises(ValueError, match="no JSON value"):
            extract_json_block("no json here at all")

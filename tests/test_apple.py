from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import spawnllm.backends.apple as apple_module
from spawnllm import (
    AntigravityCliBackend,
    AppleBackend,
    AppleConfig,
    ClaudeCliBackend,
    ClaudeSdkBackend,
    CodexCliBackend,
    GeminiCliBackend,
    RunSpec,
    _core,
)
from spawnllm.backends import base
from spawnllm.backends.apple import BINARY
from spawnllm.backends.base import (
    BackendCallError,
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    BackendUnavailable,
    LlmBackend,
)
from spawnllm.backends.registry import select_backend
from spawnllm.proc import RunResult

OTHER_BACKENDS = (ClaudeSdkBackend, ClaudeCliBackend, CodexCliBackend, AntigravityCliBackend, GeminiCliBackend)
INSTALL_HINT = "swift build -c release --package-path swift/spawnllm-apple"


class Answer(BaseModel):
    answer: int


def spec(*, config: AppleConfig | None = None, **changes: object) -> RunSpec:
    return dataclasses.replace(
        RunSpec(prompt="ping", model="small", provider_configs={"apple": config} if config is not None else {}),
        **changes,
    )


def request(run: RunSpec) -> dict[str, Any]:
    return json.loads(AppleBackend().invocation(run).stdin)


def ok(text: str) -> str:
    return json.dumps({"status": "ok", "text": text})


def failed(kind: str, message: str) -> str:
    return json.dumps({"status": "error", "kind": kind, "message": message})


def sidecar(monkeypatch: pytest.MonkeyPatch, raw: str, *, returncode: int = 0, stderr: str = "") -> list[list[str]]:
    argvs: list[list[str]] = []

    def fake_capture_cli(argv: list[str], **_kwargs: object) -> RunResult:
        argvs.append(argv)
        return RunResult(raw, stderr, returncode)

    async def fake_acapture_cli(argv: list[str], **kwargs: object) -> RunResult:
        return fake_capture_cli(argv, **kwargs)

    monkeypatch.setattr(base, "capture_cli", fake_capture_cli)
    monkeypatch.setattr(base, "acapture_cli", fake_acapture_cli)
    return argvs


def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bundled = tmp_path / "_bin" / BINARY
    bundled.parent.mkdir()
    bundled.write_text("#!/bin/sh\nexit 0\n")
    bundled.chmod(0o755)
    monkeypatch.setattr(apple_module, "files", lambda _package: tmp_path)
    return bundled


def unbundled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apple_module, "files", lambda _package: tmp_path)


def probes(monkeypatch: pytest.MonkeyPatch, *, returncode: int) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(args=argv, returncode=returncode)

    monkeypatch.setattr("spawnllm.backends.base.subprocess.run", fake_run)
    return calls


def transient(msg: str) -> bool:
    return _core.dispatch("retry_decision", {"attempt": 0, "max_attempts": 5, "error_msg": msg})["retry"]


def unauthenticated(self: LlmBackend, *, timeout: int = 10) -> BackendNotAuthenticated:
    return BackendNotAuthenticated(self.binary)


class TestWireSpec:
    def test_absent_config_serializes_as_null(self) -> None:
        assert AppleBackend().wire_spec(spec())["apple"] is None

    def test_config_section_is_dataclass_asdict(self) -> None:
        config = AppleConfig(use_case="content_tagging", instructions="Tag it.", sampling="random", sampling_top=20)

        assert AppleBackend().wire_spec(spec(config=config))["apple"] == dataclasses.asdict(config)


class TestInvocation:
    def test_argv_is_the_lone_sidecar_with_no_files_or_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unbundled(tmp_path, monkeypatch)

        inv = AppleBackend().invocation(spec())

        assert inv.argv == ["spawnllm-apple"]
        assert inv.result_path is None
        assert inv.stdout_path is None
        assert inv.cleanup_paths == ()
        assert inv.env == {}
        assert inv.env_unset == ()

    def test_argv_runs_the_bundled_sidecar_when_the_wheel_carries_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundled = bundle(tmp_path, monkeypatch)

        assert AppleBackend().invocation(spec()).argv == [str(bundled)]

    def test_default_request_carries_the_prompt_and_framework_defaults(self) -> None:
        assert request(spec()) == {
            "prompt": "ping",
            "instructions": None,
            "use_case": "general",
            "guardrails": "default",
            "options": None,
            "schema": None,
        }

    def test_session_knobs_reach_the_request(self) -> None:
        config = AppleConfig(
            use_case="content_tagging",
            guardrails="permissive_content_transformations",
            instructions="Tag it.",
        )

        assert request(spec(config=config)) == {
            "prompt": "ping",
            "instructions": "Tag it.",
            "use_case": "content_tagging",
            "guardrails": "permissive_content_transformations",
            "options": None,
            "schema": None,
        }

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            pytest.param(AppleConfig(), None, id="no_knobs"),
            pytest.param(
                AppleConfig(temperature=0.2),
                {"temperature": 0.2, "maximum_response_tokens": None, "sampling": None},
                id="temperature_only",
            ),
            pytest.param(
                AppleConfig(maximum_response_tokens=64),
                {"temperature": None, "maximum_response_tokens": 64, "sampling": None},
                id="max_tokens_only",
            ),
            pytest.param(
                AppleConfig(sampling="greedy"),
                {"temperature": None, "maximum_response_tokens": None, "sampling": {"mode": "greedy"}},
                id="greedy",
            ),
            pytest.param(
                AppleConfig(sampling="random", sampling_top=20, sampling_seed=7, temperature=0.9),
                {
                    "temperature": 0.9,
                    "maximum_response_tokens": None,
                    "sampling": {"mode": "random", "top": 20, "probability_threshold": None, "seed": 7},
                },
                id="random_top_k",
            ),
            pytest.param(
                AppleConfig(sampling="random", sampling_probability_threshold=0.8),
                {
                    "temperature": None,
                    "maximum_response_tokens": None,
                    "sampling": {"mode": "random", "top": None, "probability_threshold": 0.8, "seed": None},
                },
                id="random_nucleus",
            ),
        ],
    )
    def test_options_carry_only_the_configured_knobs(
        self, config: AppleConfig, expected: dict[str, Any] | None
    ) -> None:
        assert request(spec(config=config))["options"] == expected

    def test_response_model_reaches_the_request_in_the_apple_dialect(self) -> None:
        schema = request(spec(response_model=Answer))["schema"]

        assert schema["properties"]["answer"]["type"] == "integer"
        assert json.dumps(schema) == AppleBackend().schema_for(Answer)

    def test_raw_schema_reaches_the_request_verbatim(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}

        assert request(spec(schema=schema))["schema"] == schema


class TestResolution:
    def test_ok_envelope_resolves_to_text(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        unbundled(tmp_path, monkeypatch)
        argvs = sidecar(monkeypatch, ok("echo: ping"))

        response = AppleBackend().execute(spec())

        assert response.error is None
        assert response.result.raw == "echo: ping"
        assert response.result.parsed is None
        assert argvs == [["spawnllm-apple"]]

    async def test_aexecute_resolves_the_same_envelope(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sidecar(monkeypatch, ok("echo: ping"))

        assert (await AppleBackend().aexecute(spec())).result.raw == "echo: ping"

    def test_written_refusal_is_a_successful_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sidecar(monkeypatch, ok("I can't help with that."))

        assert AppleBackend().execute(spec()).result.raw == "I can't help with that."

    def test_structured_envelope_validates_into_the_response_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sidecar(monkeypatch, ok('{"answer": 7}'))

        response = AppleBackend().execute(spec(response_model=Answer))

        assert response.error is None
        assert response.result.parsed == Answer(answer=7)
        assert response.result.raw == '{"answer": 7}'

    def test_non_conforming_structured_output_becomes_a_validation_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pydantic

        sidecar(monkeypatch, ok('{"answer": "seven"}'))

        response = AppleBackend().execute(spec(response_model=Answer))

        assert response.result is None
        assert isinstance(response.error.ex, pydantic.ValidationError)

    @pytest.mark.parametrize(
        "kind",
        [
            pytest.param("RateLimitedError", id="rate_limited"),
            pytest.param("ConcurrentRequestsError", id="concurrent_requests"),
        ],
    )
    def test_throttling_kinds_resolve_as_transient(self, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
        sidecar(monkeypatch, failed(kind, "busy"))

        response = AppleBackend().execute(spec())

        assert response.result is None
        assert isinstance(response.error.ex, BackendCallError)
        assert response.error.msg == f"apple hit a rate limit ({kind}): busy"
        assert transient(response.error.msg) is True

    @pytest.mark.parametrize(
        ("kind", "message"),
        [
            pytest.param("RefusalError", "the model declined to answer", id="refusal"),
            pytest.param("GuardrailViolationError", "blocked", id="guardrail_violation"),
            pytest.param("DecodingFailureError", "the output was not valid JSON", id="decoding_failure"),
            pytest.param("ExceededContextWindowSizeError", "prompt too long", id="context_window"),
        ],
    )
    def test_terminal_kinds_resolve_as_terminal(self, monkeypatch: pytest.MonkeyPatch, kind: str, message: str) -> None:
        sidecar(monkeypatch, failed(kind, message))

        response = AppleBackend().execute(spec())

        assert response.error.msg == f"apple generation failed ({kind}): {message}"
        assert transient(response.error.msg) is False

    def test_a_malformed_envelope_is_a_terminal_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sidecar(monkeypatch, "not json")

        response = AppleBackend().execute(spec())

        assert response.error.msg == "apple returned an invalid response envelope: not json"
        assert transient(response.error.msg) is False

    def test_a_nonzero_exit_reports_stderr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sidecar(monkeypatch, "", returncode=1, stderr="boom")

        response = AppleBackend().execute(spec())

        assert response.error.msg == "apple exited 1: boom"
        assert transient(response.error.msg) is False

    def test_a_timeout_is_terminal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def timing_out(argv: list[str], **_kwargs: object) -> RunResult:
            raise subprocess.TimeoutExpired(argv, 5)

        monkeypatch.setattr(base, "capture_cli", timing_out)
        response = AppleBackend().execute(spec(timeout=5))

        assert response.output.raw == ""
        assert isinstance(response.error.ex, TimeoutError)
        assert response.error.msg == "apple timed out after 5s"
        assert transient(response.error.msg) is False


class TestConfigValidation:
    @pytest.mark.parametrize(
        "knobs",
        [
            pytest.param({"sampling_top": 20}, id="top_without_sampling"),
            pytest.param({"sampling_probability_threshold": 0.8}, id="threshold_without_sampling"),
            pytest.param({"sampling_seed": 7}, id="seed_without_sampling"),
            pytest.param({"sampling": "greedy", "sampling_top": 20}, id="top_with_greedy"),
            pytest.param({"sampling": "greedy", "sampling_seed": 7}, id="seed_with_greedy"),
        ],
    )
    def test_random_only_knobs_are_rejected_without_random_sampling(self, knobs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="require sampling='random'"):
            AppleConfig(**knobs)

    def test_top_and_probability_threshold_together_are_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="either sampling_top or sampling_probability_threshold"):
            AppleConfig(sampling="random", sampling_top=20, sampling_probability_threshold=0.5)

    def test_a_negative_seed_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="sampling_seed is unsigned"):
            AppleConfig(sampling="random", sampling_seed=-1)

    def test_random_sampling_keeps_every_compatible_knob(self) -> None:
        config = AppleConfig(sampling="random", sampling_top=20, sampling_seed=7)

        assert (config.sampling, config.sampling_top, config.sampling_seed) == ("random", 20, 7)

    @pytest.mark.parametrize(
        "knobs",
        [
            pytest.param({"sampling_seed": 7}, id="seed_without_sampling"),
            pytest.param({"sampling": "random", "sampling_top": 20, "sampling_probability_threshold": 0.5}, id="both"),
        ],
    )
    def test_the_core_rejects_what_construction_rejects_with_the_same_message(self, knobs: dict[str, Any]) -> None:
        with pytest.raises(ValueError) as raised:
            AppleConfig(**knobs)
        wire = AppleBackend().wire_spec(RunSpec(prompt="hi", model="small", provider_configs={"apple": AppleConfig()}))

        with pytest.raises(_core.CoreError) as rejected:
            _core.dispatch("validate_spec", {"spec": wire | {"apple": wire["apple"] | knobs}})

        assert (rejected.value.kind, rejected.value.msg) == ("invalid_spec", str(raised.value))


class TestStatus:
    def test_binary_path_prefers_the_wheel_bundled_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundled = bundle(tmp_path, monkeypatch)

        assert AppleBackend().binary_path() == str(bundled)

    def test_binary_path_falls_back_to_a_path_lookup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(apple_module, "files", lambda _package: tmp_path)

        assert AppleBackend().binary_path() == BINARY

    def test_not_installed_when_no_sidecar_resolves(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(apple_module, "files", lambda _package: tmp_path)
        monkeypatch.setattr("spawnllm.backends.base.shutil.which", lambda _name: None)

        assert AppleBackend().check_status() == BackendNotInstalled(binary=BINARY, install_hint=INSTALL_HINT)

    def test_ready_probes_the_bundled_sidecar(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bundled = bundle(tmp_path, monkeypatch)
        calls = probes(monkeypatch, returncode=0)

        assert AppleBackend().check_status() == BackendReady(binary=BINARY)
        assert calls == [[str(bundled), "--probe"]]

    def test_not_authenticated_when_the_probe_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle(tmp_path, monkeypatch)
        probes(monkeypatch, returncode=1)

        assert AppleBackend().check_status() == BackendNotAuthenticated(binary=BINARY)


class TestTierGate:
    def test_auto_select_tiers_is_small_only(self) -> None:
        assert AppleBackend.auto_select_tiers == frozenset({"small"})

    @pytest.mark.parametrize(
        ("model", "reachable"),
        [
            pytest.param("small", True, id="small"),
            pytest.param(None, False, id="unspecified"),
            pytest.param("medium", False, id="medium"),
            pytest.param("large", False, id="large"),
            pytest.param("claude-fable-5", False, id="pinned_model_id"),
        ],
    )
    def test_apple_is_auto_selected_only_for_the_small_tier(
        self, monkeypatch: pytest.MonkeyPatch, model: str | None, reachable: bool
    ) -> None:
        for cls in OTHER_BACKENDS:
            monkeypatch.setattr(cls, "check_status", unauthenticated)
        monkeypatch.setattr(AppleBackend, "check_status", lambda self, *, timeout=10: BackendReady(binary=self.binary))

        if reachable:
            assert isinstance(select_backend(model=model), AppleBackend)
            return
        with pytest.raises(BackendUnavailable):
            select_backend(model=model)


class Capital(BaseModel):
    city: str
    country: str


@pytest.mark.integration
@pytest.mark.skipif(
    not isinstance(AppleBackend().check_status(), BackendReady),
    reason="the spawnllm-apple sidecar is not installed, or it reports Apple Intelligence unavailable on this device",
)
async def test_structured_round_trip_on_device() -> None:
    response = await AppleBackend().aexecute(
        spec(
            prompt="What is the capital of France? Answer with the city and its country.",
            response_model=Capital,
            config=AppleConfig(sampling="greedy"),
        )
    )

    assert response.error is None
    assert response.result.parsed.city == "Paris"
    assert response.result.parsed.country == "France"

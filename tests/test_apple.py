from __future__ import annotations

import asyncio
import dataclasses
import importlib.machinery
import importlib.util
import inspect
import sys
import types
from enum import IntEnum, StrEnum
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
from spawnllm.backends.base import (
    BackendCallError,
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    BackendUnavailable,
    LlmBackend,
)
from spawnllm.backends.registry import select_backend

CLI_BACKENDS = (ClaudeSdkBackend, ClaudeCliBackend, CodexCliBackend, AntigravityCliBackend, GeminiCliBackend)


class FoundationModelsError(Exception):
    pass


class GenerationError(FoundationModelsError):
    pass


class RateLimitedError(GenerationError):
    pass


class ConcurrentRequestsError(GenerationError):
    pass


class RefusalError(GenerationError):
    pass


class GuardrailViolationError(GenerationError):
    pass


class SystemLanguageModelUseCase(IntEnum):
    GENERAL = 0
    CONTENT_TAGGING = 1


class SystemLanguageModelGuardrails(IntEnum):
    DEFAULT = 0
    PERMISSIVE_CONTENT_TRANSFORMATIONS = 1


class SystemLanguageModelUnavailableReason(IntEnum):
    APPLE_INTELLIGENCE_NOT_ENABLED = 0
    DEVICE_NOT_ELIGIBLE = 1
    MODEL_NOT_READY = 2


class SamplingModeType(StrEnum):
    GREEDY = "greedy"
    RANDOM = "random"


@dataclasses.dataclass(frozen=True)
class SamplingMode:
    mode_type: SamplingModeType
    top: int | None = None
    probability_threshold: float | None = None
    seed: int | None = None

    @classmethod
    def greedy(cls) -> SamplingMode:
        return cls(mode_type=SamplingModeType.GREEDY)

    @classmethod
    def random(
        cls,
        top: int | None = None,
        probability_threshold: float | None = None,
        seed: int | None = None,
    ) -> SamplingMode:
        if top is not None and probability_threshold is not None:
            raise ValueError("Cannot specify both 'top' and 'probability_threshold'. Choose one sampling constraint.")
        return cls(mode_type=SamplingModeType.RANDOM, top=top, probability_threshold=probability_threshold, seed=seed)


@dataclasses.dataclass(frozen=True)
class GenerationOptions:
    sampling: SamplingMode | None = None
    temperature: float | None = None
    maximum_response_tokens: int | None = None


@dataclasses.dataclass(frozen=True)
class GeneratedContent:
    json_text: str

    def to_json(self) -> str:
        return self.json_text


@dataclasses.dataclass(frozen=True)
class Call:
    prompt: str
    json_schema: dict[str, Any] | None
    options: GenerationOptions | None


@dataclasses.dataclass
class Recorder:
    models: list[Any] = dataclasses.field(default_factory=list)
    sessions: list[Any] = dataclasses.field(default_factory=list)

    @property
    def calls(self) -> list[Call]:
        return [call for session in self.sessions for call in session.calls]


def install_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    respond: Any = lambda call: "hello",
    available: tuple[bool, SystemLanguageModelUnavailableReason | None] = (True, None),
) -> Recorder:
    recorder = Recorder()

    class FakeSystemLanguageModel:
        def __init__(
            self,
            use_case: SystemLanguageModelUseCase = SystemLanguageModelUseCase.GENERAL,
            guardrails: SystemLanguageModelGuardrails = SystemLanguageModelGuardrails.DEFAULT,
        ) -> None:
            self.use_case = use_case
            self.guardrails = guardrails
            recorder.models.append(self)

        def is_available(self) -> tuple[bool, SystemLanguageModelUnavailableReason | None]:
            return available

    class FakeLanguageModelSession:
        def __init__(self, instructions: str | None = None, model: Any = None) -> None:
            self.instructions = instructions
            self.model = model
            self.calls: list[Call] = []
            recorder.sessions.append(self)

        async def respond(
            self,
            prompt: str,
            *,
            json_schema: dict[str, Any] | None = None,
            options: GenerationOptions | None = None,
        ) -> Any:
            self.calls.append(call := Call(prompt=prompt, json_schema=json_schema, options=options))
            outcome = respond(call)
            return await outcome if inspect.isawaitable(outcome) else outcome

    module = types.ModuleType("apple_fm_sdk")
    module.__spec__ = importlib.machinery.ModuleSpec("apple_fm_sdk", loader=None)
    vars(module).update(
        FoundationModelsError=FoundationModelsError,
        GenerationError=GenerationError,
        RateLimitedError=RateLimitedError,
        ConcurrentRequestsError=ConcurrentRequestsError,
        RefusalError=RefusalError,
        GuardrailViolationError=GuardrailViolationError,
        SamplingMode=SamplingMode,
        GenerationOptions=GenerationOptions,
        GeneratedContent=GeneratedContent,
        SystemLanguageModelUseCase=SystemLanguageModelUseCase,
        SystemLanguageModelGuardrails=SystemLanguageModelGuardrails,
        SystemLanguageModelUnavailableReason=SystemLanguageModelUnavailableReason,
        SystemLanguageModel=FakeSystemLanguageModel,
        LanguageModelSession=FakeLanguageModelSession,
    )
    monkeypatch.setitem(sys.modules, "apple_fm_sdk", module)
    return recorder


def raising(error: Exception) -> Any:
    def respond(_call: Call) -> Any:
        raise error

    return respond


def hanging() -> Any:
    def respond(_call: Call) -> Any:
        return asyncio.Event().wait()

    return respond


def spec(*, config: AppleConfig | None = None, **changes: object) -> RunSpec:
    base = RunSpec(prompt="ping", model="small", provider_configs={"apple": config} if config is not None else {})
    return dataclasses.replace(base, **changes)


def transient(msg: str) -> bool:
    return _core.dispatch("retry_decision", {"attempt": 0, "max_attempts": 5, "error_msg": msg})["retry"]


def unauthenticated(self: LlmBackend, *, timeout: int = 10) -> BackendNotAuthenticated:
    return BackendNotAuthenticated(self.binary)


class Answer(BaseModel):
    answer: int


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
        self,
        monkeypatch: pytest.MonkeyPatch,
        model: str | None,
        reachable: bool,
    ) -> None:
        for cls in CLI_BACKENDS:
            monkeypatch.setattr(cls, "check_status", unauthenticated)
        monkeypatch.setattr(AppleBackend, "check_status", lambda self, *, timeout=10: BackendReady(binary="apple"))

        if reachable:
            assert isinstance(select_backend(model=model), AppleBackend)
            return
        with pytest.raises(BackendUnavailable):
            select_backend(model=model)


class TestRetryCoupling:
    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(RateLimitedError("busy"), id="rate_limited"),
            pytest.param(ConcurrentRequestsError("busy"), id="concurrent_requests"),
        ],
    )
    async def test_throttling_message_is_transient_to_the_core(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
    ) -> None:
        install_sdk(monkeypatch, respond=raising(error))

        response = await AppleBackend().aexecute(spec())

        assert response.result is None
        assert isinstance(response.error.ex, BackendCallError)
        assert "rate limit" in response.error.msg
        assert transient(response.error.msg) is True

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(RefusalError("512 of 4096 tokens were unusable"), id="digits_that_look_like_a_5xx"),
            pytest.param(GuardrailViolationError("blocked"), id="guardrail_violation"),
            pytest.param(FoundationModelsError("529 overloaded"), id="text_that_looks_transient"),
        ],
    )
    async def test_terminal_message_is_not_transient_to_the_core(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
    ) -> None:
        install_sdk(monkeypatch, respond=raising(error))

        response = await AppleBackend().aexecute(spec())

        assert response.result is None
        assert isinstance(response.error.ex, BackendCallError)
        assert str(error) not in response.error.msg
        assert transient(response.error.msg) is False

    async def test_sdk_errors_surface_as_the_documented_public_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        install_sdk(monkeypatch, respond=raising(RefusalError("no")))

        response = await AppleBackend().aexecute(spec())

        assert type(response.error.ex) is BackendCallError
        assert str(response.error.ex) == response.error.msg == "apple generation failed (RefusalError)"


class TestTimeout:
    async def test_a_hanging_generation_resolves_to_a_timeout_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        install_sdk(monkeypatch, respond=hanging())

        response = await AppleBackend().aexecute(spec(timeout=0))

        assert response.result is None
        assert response.output.raw == ""
        assert isinstance(response.error.ex, TimeoutError)
        assert response.error.msg == "apple timed out after 0s"

    @pytest.mark.parametrize("timeout", [0, 5, 180, 500, 529])
    async def test_a_timeout_is_terminal_to_the_core(self, monkeypatch: pytest.MonkeyPatch, timeout: int) -> None:
        install_sdk(monkeypatch, respond=raising(TimeoutError()))

        response = await AppleBackend().aexecute(spec(timeout=timeout))

        assert response.error.msg == f"apple timed out after {timeout}s"
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

    def test_random_sampling_keeps_every_knob(self) -> None:
        config = AppleConfig(sampling="random", sampling_top=20, sampling_seed=7)

        assert (config.sampling, config.sampling_top, config.sampling_seed) == ("random", 20, 7)


class TestGeneration:
    async def test_plain_prompt_resolves_to_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = install_sdk(monkeypatch, respond=lambda call: f"echo: {call.prompt}")

        response = await AppleBackend().aexecute(spec())

        assert response.error is None
        assert response.result.raw == "echo: ping"
        assert response.result.parsed is None
        assert response.output.raw == "echo: ping"
        assert recorder.calls == [Call(prompt="ping", json_schema=None, options=None)]

    async def test_written_refusal_is_a_successful_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_sdk(monkeypatch, respond=lambda call: "I can't help with that.")

        response = await AppleBackend().aexecute(spec())

        assert response.error is None
        assert response.result.raw == "I can't help with that."

    async def test_every_call_builds_a_fresh_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = install_sdk(monkeypatch)
        backend = AppleBackend()

        await backend.aexecute(spec())
        await backend.aexecute(spec())

        assert len(recorder.sessions) == 2
        assert recorder.sessions[0] is not recorder.sessions[1]
        assert [len(session.calls) for session in recorder.sessions] == [1, 1]

    async def test_config_selects_use_case_guardrails_and_instructions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = install_sdk(monkeypatch)
        config = AppleConfig(
            use_case="content_tagging",
            guardrails="permissive_content_transformations",
            instructions="Tag the content.",
        )

        await AppleBackend().aexecute(spec(config=config))

        model = recorder.models[0]
        assert model.use_case is SystemLanguageModelUseCase.CONTENT_TAGGING
        assert model.guardrails is SystemLanguageModelGuardrails.PERMISSIVE_CONTENT_TRANSFORMATIONS
        assert recorder.sessions[0].instructions == "Tag the content."
        assert recorder.sessions[0].model is model

    async def test_default_config_uses_general_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = install_sdk(monkeypatch)

        await AppleBackend().aexecute(spec())

        assert recorder.models[0].use_case is SystemLanguageModelUseCase.GENERAL
        assert recorder.models[0].guardrails is SystemLanguageModelGuardrails.DEFAULT
        assert recorder.sessions[0].instructions is None

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            pytest.param(AppleConfig(), None, id="no_knobs"),
            pytest.param(
                AppleConfig(temperature=0.2),
                GenerationOptions(temperature=0.2),
                id="temperature_only",
            ),
            pytest.param(
                AppleConfig(maximum_response_tokens=64),
                GenerationOptions(maximum_response_tokens=64),
                id="max_tokens_only",
            ),
            pytest.param(
                AppleConfig(sampling="greedy"),
                GenerationOptions(sampling=SamplingMode(mode_type=SamplingModeType.GREEDY)),
                id="greedy",
            ),
            pytest.param(
                AppleConfig(sampling="random", sampling_top=20, sampling_seed=7, temperature=0.9),
                GenerationOptions(
                    sampling=SamplingMode(mode_type=SamplingModeType.RANDOM, top=20, seed=7),
                    temperature=0.9,
                ),
                id="random_top_k",
            ),
            pytest.param(
                AppleConfig(sampling="random", sampling_probability_threshold=0.8),
                GenerationOptions(sampling=SamplingMode(mode_type=SamplingModeType.RANDOM, probability_threshold=0.8)),
                id="random_nucleus",
            ),
        ],
    )
    async def test_generation_options_carry_only_the_configured_knobs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: AppleConfig,
        expected: GenerationOptions | None,
    ) -> None:
        recorder = install_sdk(monkeypatch)

        await AppleBackend().aexecute(spec(config=config))

        assert recorder.calls[0].options == expected

    async def test_conflicting_sampling_constraints_raise_the_framework_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        install_sdk(monkeypatch)
        config = AppleConfig(sampling="random", sampling_top=20, sampling_probability_threshold=0.8)

        with pytest.raises(ValueError, match="Cannot specify both"):
            await AppleBackend().aexecute(spec(config=config))

    async def test_response_model_round_trips_through_the_apple_dialect(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recorder = install_sdk(monkeypatch, respond=lambda call: GeneratedContent('{"answer": 7}'))

        response = await AppleBackend().aexecute(spec(response_model=Answer))

        assert response.error is None
        assert response.result.parsed == Answer(answer=7)
        assert response.result.raw == '{"answer": 7}'
        assert recorder.calls[0].json_schema["properties"]["answer"]["type"] == "integer"

    async def test_non_conforming_structured_output_becomes_a_validation_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pydantic

        install_sdk(monkeypatch, respond=lambda call: GeneratedContent('{"answer": "seven"}'))

        response = await AppleBackend().aexecute(spec(response_model=Answer))

        assert response.result is None
        assert isinstance(response.error.ex, pydantic.ValidationError)
        assert response.output.raw == '{"answer": "seven"}'

    def test_wire_schema_passes_a_raw_schema_through(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}

        assert AppleBackend().wire_schema(spec(schema=schema)) == schema

    async def test_raw_schema_reaches_the_framework_in_the_apple_dialect(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recorder = install_sdk(monkeypatch, respond=lambda call: GeneratedContent('{"answer": 7}'))
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}

        response = await AppleBackend().aexecute(spec(schema=schema))

        assert response.error is None
        assert response.result.parsed is None
        assert response.result.raw == '{"answer": 7}'
        assert (
            recorder.calls[0].json_schema
            == _core.dispatch("strict_schema", {"dialect": "apple", "schema": schema})["schema"]
        )

    def test_execute_bridges_to_async_execution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_sdk(monkeypatch, respond=lambda call: "sync result")

        assert AppleBackend().execute(spec()).result.raw == "sync result"


class TestStatus:
    def test_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(apple_module.importlib.util, "find_spec", lambda name: None)

        assert AppleBackend().check_status() == BackendNotInstalled(
            binary="apple",
            install_hint="uv pip install 'spawnllm[apple]'",
        )

    def test_ready_when_the_device_model_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_sdk(monkeypatch)

        assert AppleBackend().check_status() == BackendReady(binary="apple")
        assert AppleBackend().is_authenticated(timeout=1) is True

    def test_unavailable_model_keeps_the_backend_identity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_sdk(monkeypatch, available=(False, SystemLanguageModelUnavailableReason.MODEL_NOT_READY))

        assert AppleBackend().check_status() == BackendNotAuthenticated(binary="apple")
        assert AppleBackend().check_status().binary == AppleBackend.binary
        assert AppleBackend().availability() == (False, "MODEL_NOT_READY")
        assert AppleBackend().is_authenticated(timeout=1) is False

    def test_env_is_empty(self) -> None:
        assert AppleBackend().env(spec()) == {}


def device_model_available() -> bool:
    if importlib.util.find_spec("apple_fm_sdk") is None:
        return False
    import apple_fm_sdk as fm

    return fm.SystemLanguageModel().is_available()[0]


class Capital(BaseModel):
    city: str
    country: str


@pytest.mark.integration
@pytest.mark.skipif(not device_model_available(), reason="Apple Foundation Models unavailable on this device")
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

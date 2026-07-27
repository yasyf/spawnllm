"""In-process backend over Apple's on-device Foundation Models framework."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from typing import TYPE_CHECKING, Any, ClassVar

from spawnllm import _core
from spawnllm.backends.base import (
    BackendCallError,
    BackendNotAuthenticated,
    BackendNotInstalled,
    BackendReady,
    LlmBackend,
)
from spawnllm.response import Error, Output, Response, Result
from spawnllm.spec import AppleConfig

if TYPE_CHECKING:
    from apple_fm_sdk import GenerationOptions

    from spawnllm.backends.base import BackendStatus
    from spawnllm.spec import RunSpec
    from spawnllm.types import ProviderName, TModel


class AppleBackend(LlmBackend):
    """Backend running prompts through Apple's on-device Foundation Models.

    Generation happens in-process against the system language model, so there is
    no subprocess and no credential: `RunSpec`'s `model`, `isolated`, `api_auth`,
    `env`, and `cwd` are all inert, and "authenticated" means the framework
    reports the model available on this device. `AppleConfig` carries the knobs
    that do apply — the use case and guardrails the model is built with, the
    session instructions, and the decoding options.

    The optional `apple-fm-sdk` package is imported lazily, so importing
    `spawnllm` stays free of it on every platform Apple Intelligence does not
    reach. Each call builds a fresh session: the SDK's session holds a lock and
    accumulates a transcript, neither of which survives reuse across runs.

    Attributes:
        models: Empty identity mapping; the device hosts exactly one model.
        provider: Provider identifier keying `AppleConfig` on a `RunSpec`.
        binary: Name this backend reports in status output.
        install_hint: Suggested shell command to install the optional SDK.
        schema_dialect: Apple strict-schema dialect the core applies.
        auto_select_tiers: Only `small`; the on-device model is a small model.

    Example:
        >>> AppleBackend().execute(RunSpec(prompt="ping", model="small"))
    """

    models: ClassVar[dict[TModel, str]] = {}
    provider: ClassVar[ProviderName] = "apple"
    binary: ClassVar[str] = "apple"
    install_hint: ClassVar[str] = "uv pip install 'spawnllm[apple]'"
    schema_dialect: ClassVar[str | None] = "apple"
    auto_select_tiers: ClassVar[frozenset[TModel] | None] = frozenset({"small"})

    def generation_options(self, cfg: AppleConfig) -> GenerationOptions | None:
        """Translate an `AppleConfig`'s decoding knobs into framework options.

        Args:
            cfg: The Apple provider config carrying the decoding knobs.

        Returns:
            The options for this request, or `None` when no knob is set and
            every framework default applies.

        Raises:
            ValueError: When the config asks for random sampling with both
                `sampling_top` and `sampling_probability_threshold`, which the
                framework rejects.
        """
        import apple_fm_sdk as fm

        match cfg.sampling:
            case "greedy":
                sampling = fm.SamplingMode.greedy()
            case "random":
                sampling = fm.SamplingMode.random(
                    top=cfg.sampling_top,
                    probability_threshold=cfg.sampling_probability_threshold,
                    seed=cfg.sampling_seed,
                )
            case None:
                sampling = None
        if (sampling, cfg.temperature, cfg.maximum_response_tokens) == (None, None, None):
            return None
        return fm.GenerationOptions(
            sampling=sampling,
            temperature=cfg.temperature,
            maximum_response_tokens=cfg.maximum_response_tokens,
        )

    def apple_schema(self, spec: RunSpec) -> dict[str, Any] | None:
        """Return `spec`'s wire schema in Apple's strict dialect, or `None` when the run is unstructured.

        A raw `RunSpec.schema` reaches the wire without passing through
        `schema_for`, so the dialect transform runs here over whichever source
        `wire_schema` resolved; it is idempotent over a `response_model`'s
        already-transformed schema.

        Args:
            spec: The configured run carrying a `response_model` or a raw `schema`.

        Returns:
            The Apple-dialect JSON schema, or `None` for an unstructured run.
        """
        schema = self.wire_schema(spec)
        if schema is None:
            return None
        return _core.dispatch("strict_schema", {"dialect": self.schema_dialect, "schema": schema})["schema"]

    async def aexecute(self, spec: RunSpec) -> Response:
        """Run one prompt through a fresh on-device session and resolve its response.

        Args:
            spec: The configured run to execute.

        Returns:
            The resolved `Response`. A framework error becomes a `Response.error`
            carrying a `BackendCallError` whose message marks rate limiting and
            concurrency as transient and everything else — a generation outliving
            `spec.timeout` included — as terminal; a refusal the model writes as
            prose is a successful generation like any other.
        """
        import apple_fm_sdk as fm
        import pydantic

        cfg = spec.config_for(AppleConfig) or AppleConfig()
        options = self.generation_options(cfg)
        schema = self.apple_schema(spec)
        try:
            async with asyncio.timeout(spec.timeout):
                session = fm.LanguageModelSession(
                    instructions=cfg.instructions,
                    model=fm.SystemLanguageModel(
                        use_case=fm.SystemLanguageModelUseCase[cfg.use_case.upper()],
                        guardrails=fm.SystemLanguageModelGuardrails[cfg.guardrails.upper()],
                    ),
                )
                match schema:
                    case None:
                        raw = await session.respond(spec.prompt, options=options)
                    case _:
                        raw = (await session.respond(spec.prompt, json_schema=schema, options=options)).to_json()
        except TimeoutError:
            msg = f"{self.provider} timed out after {spec.timeout}s"
            return Response(spec=spec, output=Output(""), error=Error(msg, TimeoutError(msg)))
        except fm.FoundationModelsError as e:
            match e:
                case fm.RateLimitedError() | fm.ConcurrentRequestsError():
                    msg = f"{self.provider} hit a rate limit ({type(e).__name__})"
                case _:
                    msg = f"{self.provider} generation failed ({type(e).__name__})"
            return Response(spec=spec, output=Output(""), error=Error(msg, BackendCallError(msg)))
        output = Output(raw)
        if spec.response_model is None:
            return Response(spec=spec, output=output, result=Result(raw=raw))
        try:
            parsed = spec.response_model.model_validate(json.loads(raw))
        except pydantic.ValidationError as e:
            return Response(spec=spec, output=output, error=Error(str(e), e))
        return Response(spec=spec, output=output, result=Result(raw=raw, parsed=parsed))

    def execute(self, spec: RunSpec) -> Response:
        """Run one prompt synchronously via `asyncio.run`.

        Args:
            spec: The configured run to execute.

        Returns:
            The resolved `Response`.

        Raises:
            RuntimeError: If called from a thread already running an event loop.
        """
        return asyncio.run(self.aexecute(spec))

    def availability(self) -> tuple[bool, str | None]:
        """Return whether the system model is available and, when it is not, the framework's reason."""
        import apple_fm_sdk as fm

        available, reason = fm.SystemLanguageModel().is_available()
        return available, None if reason is None else reason.name

    def env(self, _spec: RunSpec) -> dict[str, str]:
        """Return no extra environment variables; generation runs in-process with nothing to isolate."""
        return {}

    def is_authenticated(self, *, timeout: int) -> bool:
        """Report whether the system model is available on this device.

        Args:
            timeout: Unused; the availability check is a local framework call.

        Returns:
            `True` when Apple Intelligence reports the model ready.
        """
        return self.availability()[0]

    def check_status(self, *, timeout: int = 10) -> BackendStatus:
        """Check whether the optional SDK is installed and the device model available.

        Args:
            timeout: Unused; the availability check is a local framework call.

        Returns:
            `BackendReady` when the model is available, `BackendNotInstalled`
            without the `apple` extra, else `BackendNotAuthenticated`; the
            framework's unavailability reason is available from `availability`.
        """
        if importlib.util.find_spec("apple_fm_sdk") is None:
            return BackendNotInstalled(binary=self.binary, install_hint=self.install_hint)
        if self.is_authenticated(timeout=timeout):
            return BackendReady(binary=self.binary)
        return BackendNotAuthenticated(binary=self.binary)

"""Abstract execution contract for an LLM backend and its subprocess family."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from spawnllm.proc import acapture_cli, capture_cli
from spawnllm.response import Error, Output, Response, Result

if TYPE_CHECKING:
    from pydantic import BaseModel

    from spawnllm.spec import RunSpec
    from spawnllm.types import ProviderName, TModel


@dataclass(frozen=True)
class BackendReady:
    """A backend whose CLI is installed and authenticated.

    Attributes:
        binary: Name of the backend's CLI executable on PATH.
    """

    binary: str


@dataclass(frozen=True)
class BackendNotInstalled:
    """A backend whose CLI is not on PATH.

    Attributes:
        binary: Name of the backend's CLI executable.
        install_hint: Suggested shell command to install the CLI.
    """

    binary: str
    install_hint: str


@dataclass(frozen=True)
class BackendNotAuthenticated:
    """A backend whose CLI is installed but not authenticated.

    Attributes:
        binary: Name of the backend's CLI executable on PATH.
    """

    binary: str


BackendStatus = BackendReady | BackendNotInstalled | BackendNotAuthenticated
"""Result of `LlmBackend.check_status`: `BackendReady`, `BackendNotInstalled`, or `BackendNotAuthenticated`."""


class BackendUnavailable(RuntimeError):
    """Raised when no backend is ready (installed and authenticated)."""


class BackendCallError(RuntimeError):
    """Raised by `call`/`extract` when a backend returns a provider error.

    Carries the backend's error string: a nonzero exit with stderr, or an error envelope.
    """


@dataclass(frozen=True)
class Invocation:
    """A built CLI invocation: argv, optional stdin, and where to read the result.

    Attributes:
        argv: The argv list to execute.
        stdin: Prompt text delivered over stdin, or `None` when delivered inline.
        result_path: File the backend writes its final message to; when set, the
            result is read from this file instead of stdout.
        cleanup_paths: Temp files to remove once the invocation completes.
    """

    argv: list[str]
    stdin: str | None = None
    result_path: str | None = None
    cleanup_paths: tuple[str, ...] = ()


class LlmBackend(ABC):
    """Abstract execution contract for an LLM backend.

    Concrete backends map abstract model sizes to provider-specific model names
    and encapsulate how to execute a `RunSpec` and parse the raw response.

    Attributes:
        models: Mapping from abstract model size to the provider's model name.
        provider: Provider identifier keying a `RunSpec`'s `provider_configs`.
    """

    models: ClassVar[dict[TModel, str]]
    provider: ClassVar[ProviderName]

    @abstractmethod
    async def aexecute(self, spec: RunSpec) -> Response:
        """Execute a single run asynchronously and resolve it to a `Response`.

        The backend runs the process, reads its output wherever the provider
        writes it, detects failure, and validates against `spec.response_model`.

        Args:
            spec: The configured run to execute.

        Returns:
            The resolved `Response`.
        """

    @abstractmethod
    def execute(self, spec: RunSpec) -> Response:
        """Execute a single run synchronously and resolve it to a `Response`.

        Args:
            spec: The configured run to execute.

        Returns:
            The resolved `Response`.
        """

    @abstractmethod
    def env(self) -> dict[str, str]:
        """Return extra environment variables for the invocation, merged over the inherited environment."""

    @abstractmethod
    def is_authenticated(self, *, timeout: int) -> bool:
        """Probe whether the backend holds valid credentials for its provider.

        "Authenticated" means the backend reports an active login session for the
        provider, not merely that an executable is present on PATH.

        Args:
            timeout: Seconds to wait for the credential probe.

        Returns:
            `True` when the backend reports an authenticated session.
        """

    @abstractmethod
    def check_status(self, *, timeout: int = 10) -> BackendStatus:
        """Check whether this backend is installed and authenticated.

        Args:
            timeout: Seconds to wait for the authentication probe.

        Returns:
            `BackendReady` when authenticated, `BackendNotInstalled` when the
            backend is not available, else `BackendNotAuthenticated`.
        """

    def schema_for(self, model: type[BaseModel]) -> str:
        """Serialize a Pydantic model into the JSON-schema string this backend expects.

        The default emits the model's plain JSON schema; provider backends
        override to apply their SDK's strict-schema transform.

        Args:
            model: The Pydantic model describing the structured output.

        Returns:
            A JSON-schema string suitable for this backend's structured-output argument.
        """
        return json.dumps(model.model_json_schema())

    def schema_arg(self, spec: RunSpec) -> str | None:
        """Return the JSON-schema string for `spec`, from a `response_model` or a raw `schema`.

        A `response_model` is run through `schema_for` (the provider's
        strict-schema transform); a raw `schema` passes verbatim — a dict is
        `json.dumps`'d, a string is returned unchanged. Returns `None` when
        neither is set.

        Args:
            spec: The configured run, carrying the optional `response_model` or `schema`.

        Returns:
            The JSON-schema string for this backend's structured-output argument, or `None`.
        """
        if spec.response_model is not None:
            return self.schema_for(spec.response_model)
        if spec.schema is not None:
            return json.dumps(spec.schema) if isinstance(spec.schema, dict) else spec.schema
        return None

    def to_response(self, raw: str, *, returncode: int, stderr: str, spec: RunSpec) -> Response:
        """Resolve a raw capture into a structured `Response`: detect failure, extract text, validate.

        `output` always carries the full raw stream. A nonzero exit, an error
        envelope, or a `pydantic.ValidationError` from a non-conforming model all
        route through `error` (with the underlying exception preserved in
        `error.ex`) and leave `result` as `None`; a success yields `result` (text
        from `result_text`, plus the validated model from `result_value` when
        `spec.response_model` is set) and `error` as `None`.

        Args:
            raw: The raw output read wherever the provider wrote it.
            returncode: The process exit code.
            stderr: The captured stderr.
            spec: The configured run, carrying the optional `response_model` or `schema`.

        Returns:
            The resolved `Response`.
        """
        import pydantic

        output = Output(raw)
        if returncode != 0:
            msg = f"{self.provider} exited {returncode}: {stderr.strip()[-2000:]}"
            return Response(spec=spec, output=output, error=Error(msg, BackendCallError(msg)))
        if (err := self.envelope_error(raw)) is not None:
            return Response(spec=spec, output=output, error=Error(err, BackendCallError(err)))
        if spec.response_model is None:
            return Response(spec=spec, output=output, result=Result(raw=self.result_text(raw)))
        try:
            parsed = spec.response_model.model_validate(self.result_value(raw))
        except pydantic.ValidationError as e:
            return Response(spec=spec, output=output, error=Error(str(e), e))
        return Response(spec=spec, output=output, result=Result(raw=self.result_text(raw), parsed=parsed))

    def result_text(self, raw: str) -> str:
        """Return the final text output from a raw capture; the default is `raw` unchanged."""
        return raw

    def result_value(self, raw: str) -> object:
        """Return the JSON value to validate from a raw capture; the default parses `raw` as JSON."""
        return json.loads(raw)

    def envelope_error(self, raw: str) -> str | None:
        """Return the provider's error message from an error envelope, or `None` on success."""
        return None


class CliBackend(LlmBackend):
    """Execution contract for the subprocess-backed LLM family.

    Concrete CLI backends build an argv from a `RunSpec`; `aexecute`/`execute`
    run it, merge environment overrides, and resolve the result from stdout or a
    designated result file.

    Attributes:
        binary: Name of the backend's CLI executable on PATH.
        install_hint: Suggested shell command to install the CLI.
    """

    binary: ClassVar[str]
    install_hint: ClassVar[str]

    @abstractmethod
    def build_command(self, spec: RunSpec) -> list[str]:
        """Build the CLI argv for a single invocation.

        Args:
            spec: The configured run to translate into argv.

        Returns:
            The argv list to execute.
        """

    def invocation(self, spec: RunSpec) -> Invocation:
        """Build the argv, stdin, and result source for a single invocation.

        The default delivers the prompt over stdin and reads the result from
        stdout; subclasses override to deliver the prompt inline or to read the
        result from a file.

        Args:
            spec: The configured run to translate into an invocation.

        Returns:
            An `Invocation` carrying the argv, stdin text, and result source.
        """
        return Invocation(self.build_command(spec), spec.prompt)

    def timed_out(self, spec: RunSpec) -> Response:
        msg = f"{self.provider} timed out after {spec.timeout}s"
        return Response(spec=spec, output=Output(""), error=Error(msg, TimeoutError(msg)))

    async def aexecute(self, spec: RunSpec) -> Response:
        inv = self.invocation(spec)
        try:
            try:
                rr = await acapture_cli(
                    inv.argv,
                    input=inv.stdin,
                    env=os.environ | self.env() | (spec.env or {}),
                    cwd=spec.cwd,
                    timeout=spec.timeout,
                )
            except TimeoutError:
                return self.timed_out(spec)
            raw = Path(inv.result_path).read_text() if inv.result_path else rr.stdout
        finally:
            for path in inv.cleanup_paths:
                Path(path).unlink(missing_ok=True)
        return self.to_response(raw, returncode=rr.returncode, stderr=rr.stderr, spec=spec)

    def execute(self, spec: RunSpec) -> Response:
        inv = self.invocation(spec)
        try:
            try:
                rr = capture_cli(
                    inv.argv,
                    input=inv.stdin,
                    env=os.environ | self.env() | (spec.env or {}),
                    cwd=spec.cwd,
                    timeout=spec.timeout,
                )
            except subprocess.TimeoutExpired:
                return self.timed_out(spec)
            raw = Path(inv.result_path).read_text() if inv.result_path else rr.stdout
        finally:
            for path in inv.cleanup_paths:
                Path(path).unlink(missing_ok=True)
        return self.to_response(raw, returncode=rr.returncode, stderr=rr.stderr, spec=spec)

    def check_status(self, *, timeout: int = 10) -> BackendStatus:
        """Check whether this backend's CLI is installed and authenticated.

        Args:
            timeout: Seconds to wait for the authentication probe.

        Returns:
            `BackendReady` when authenticated, `BackendNotInstalled` when the CLI
            is not on PATH, else `BackendNotAuthenticated`.

        Raises:
            subprocess.TimeoutExpired: If `is_authenticated` exceeds `timeout`.
        """
        if not shutil.which(self.binary):
            return BackendNotInstalled(binary=self.binary, install_hint=self.install_hint)
        if self.is_authenticated(timeout=timeout):
            return BackendReady(binary=self.binary)
        return BackendNotAuthenticated(binary=self.binary)

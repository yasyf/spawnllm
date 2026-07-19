"""Abstract execution contract for an LLM backend and its subprocess family.

Every drift-prone decision — argv planning, output resolution, schema
strictification, and auth-probe layout — executes in the embedded `spawnllm-core`
wasm engine via `spawnllm._core`. This module is the I/O host: it spawns
processes, mints and cleans temp files, seeds claude isolation, and runs the
auth probes the core lays out.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from spawnllm import _core
from spawnllm.proc import acapture_cli, capture_cli
from spawnllm.response import Error, Output, Response, Result
from spawnllm.spec import ClaudeConfig, CodexConfig, GeminiConfig

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
        stdout_path: File the capture machinery redirects the child's stdout to;
            when set, stdout goes to this regular file instead of a pipe and is
            read back as the capture, dodging a Node child's async-pipe truncation.
        cleanup_paths: Temp files to remove once the invocation completes.
        env: Environment entries supplied by the core plan.
        env_unset: Parent environment keys to omit before applying explicit entries.
    """

    argv: list[str]
    stdin: str | None = None
    result_path: str | None = None
    stdout_path: str | None = None
    cleanup_paths: tuple[str, ...] = ()
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    env_unset: tuple[str, ...] = ()


def run_probe(probe: dict[str, Any], *, timeout: int) -> bool:
    """Execute one auth probe the core laid out, mirroring the reference host's probe kinds."""
    match probe:
        case {"kind": "exec_exit0", "argv": [*argv]}:
            try:
                return (
                    subprocess.run(
                        [*map(str, argv)], capture_output=True, text=True, timeout=timeout, check=False
                    ).returncode
                    == 0
                )
            except FileNotFoundError:
                return False
        case {"kind": "keychain_exists", "service": service, "account": account}:
            try:
                return (
                    subprocess.run(
                        ["security", "find-generic-password", "-s", str(service), "-a", str(account)],
                        capture_output=True,
                        timeout=timeout,
                        check=False,
                    ).returncode
                    == 0
                )
            except FileNotFoundError:
                return False
        case {"kind": "env_any", "vars": [*names]}:
            return any(os.environ.get(str(name)) for name in names)
        case {"kind": "file_exists", "path": path}:
            return os.path.exists(str(path))
        case _:
            return False


class LlmBackend(ABC):
    """Abstract execution contract for an LLM backend.

    Concrete backends map abstract model sizes to provider-specific model names
    and encapsulate how to execute a `RunSpec` and parse the raw response. The
    portable decisions — argv, output resolution, schema strictification — run in
    the shared wasm core; a backend supplies only its `provider` and the I/O.

    Attributes:
        models: Mapping from abstract model size to the provider's model name.
        provider: Provider identifier keying a `RunSpec`'s `provider_configs`.
        schema_dialect: Strict-schema dialect the core applies to a `response_model`
            (`"anthropic"`, `"openai"`, or `None` to emit the plain JSON schema).
    """

    models: ClassVar[dict[TModel, str]]
    provider: ClassVar[ProviderName]
    schema_dialect: ClassVar[str | None] = None

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
    def env(self, spec: RunSpec) -> dict[str, str]:
        """Return extra environment variables for the invocation, merged over the inherited environment.

        Args:
            spec: The configured run, so a backend can scope env overrides to
                `spec.isolated` (e.g. a fresh config home only when isolating).
        """

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

        The core's `strict_schema` op applies the backend's `schema_dialect`
        transform over the model's plain JSON schema; a `None` dialect emits the
        schema unchanged.

        Args:
            model: The Pydantic model describing the structured output.

        Returns:
            A JSON-schema string suitable for this backend's structured-output argument.
        """
        schema = model.model_json_schema()
        if self.schema_dialect is None:
            return json.dumps(schema)
        return json.dumps(_core.dispatch("strict_schema", {"dialect": self.schema_dialect, "schema": schema})["schema"])

    def wire_schema(self, spec: RunSpec) -> dict[str, Any] | None:
        """Return the portable schema object for `spec`, from a `response_model` or a raw `schema`."""
        if spec.response_model is not None:
            return json.loads(self.schema_for(spec.response_model))
        if spec.schema is not None:
            return spec.schema if isinstance(spec.schema, dict) else json.loads(spec.schema)
        return None

    def openai_section(self) -> dict[str, Any] | None:
        """Return the `openai_endpoint` wire section; `None` for every backend but the endpoint backend."""
        return None

    def wire_spec(self, spec: RunSpec) -> dict[str, Any]:
        """Serialize a `RunSpec` into the portable wire spec the core `plan`/`resolve` ops consume."""
        return {
            "prompt": spec.prompt,
            "model": spec.model,
            "schema": self.wire_schema(spec),
            "agent": spec.agent,
            "isolated": spec.isolated,
            "api_auth": spec.api_auth,
            "timeout": spec.timeout,
            "max_attempts": spec.max_attempts,
            "claude": dataclasses.asdict(c) if (c := spec.config_for(ClaudeConfig)) is not None else None,
            "codex": dataclasses.asdict(x) if (x := spec.config_for(CodexConfig)) is not None else None,
            "gemini": dataclasses.asdict(g) if (g := spec.config_for(GeminiConfig)) is not None else None,
            "openai_endpoint": self.openai_section(),
        }

    def core_plan(self, spec: RunSpec) -> dict[str, Any]:
        """Ask the core to plan this run's invocation (an exec argv or an HTTP request)."""
        return _core.dispatch(
            "plan", {"host": {"platform": sys.platform}, "provider": self.provider, "spec": self.wire_spec(spec)}
        )

    def to_response(self, raw: str, *, returncode: int, stderr: str, spec: RunSpec) -> Response:
        """Resolve a raw capture into a structured `Response` via the core `resolve` op.

        `output` always carries the full raw stream. The core detects failure (a
        nonzero exit or an error envelope) and extracts the final text and, when
        `spec.response_model` is set, the structured value; a `pydantic.ValidationError`
        from a non-conforming model routes through `error`.

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
        resolved = _core.dispatch(
            "resolve",
            {
                "provider": self.provider,
                "raw": raw,
                "returncode": returncode,
                "stderr": stderr,
                "wants_value": spec.response_model is not None,
            },
        )
        if resolved["status"] != "ok":
            return Response(spec=spec, output=output, error=Error(resolved["msg"], BackendCallError(resolved["msg"])))
        if spec.response_model is None:
            return Response(spec=spec, output=output, result=Result(raw=resolved["text"]))
        try:
            parsed = spec.response_model.model_validate(resolved["value"])
        except pydantic.ValidationError as e:
            return Response(spec=spec, output=output, error=Error(str(e), e))
        return Response(spec=spec, output=output, result=Result(raw=resolved["text"], parsed=parsed))

    def accounting(self, raw: str) -> tuple[float | None, dict[str, object] | None]:
        """Return the `(cost_usd, usage)` an attempt's output reports, or `(None, None)` when it carries neither.

        The retry loop calls this on each transient failure it discards, so a
        caller reconciling spend can still see the cost. The default parses
        nothing; the CLI family reads the core's `resolve` accounting.

        Args:
            raw: The raw output read wherever the provider wrote it.

        Returns:
            A `(cost_usd, usage)` pair, each `None` when the output does not carry it.
        """
        return None, None


class CliBackend(LlmBackend):
    """Execution contract for the subprocess-backed LLM family.

    The core plans each invocation's argv, files, env, and result source; this
    class materializes that plan into temp files and a subprocess, runs it, and
    resolves the result. Concrete CLI backends supply only ClassVars.

    Attributes:
        binary: Name of the backend's CLI executable on PATH.
        install_hint: Suggested shell command to install the CLI.
    """

    binary: ClassVar[str]
    install_hint: ClassVar[str]

    def claude_isolation(self) -> str:
        """Return the isolated config home a claude run substitutes into `${isolated_config_dir}`."""
        raise NotImplementedError

    def build_command(self, spec: RunSpec) -> list[str]:
        """Return the core-planned argv (with `${file:id}` placeholders) for a single invocation."""
        return self.core_plan(spec)["argv"]

    def invocation(self, spec: RunSpec) -> Invocation:
        """Materialize the core's exec plan into a runnable `Invocation`.

        Mints a temp file per `files[]` entry (writing its content when non-null),
        substitutes `${file:id}` placeholders in the argv and — for a claude run —
        `${isolated_config_dir}`, and wires the plan's stdout/result source and the
        minted paths into the returned `Invocation` for cleanup.

        Args:
            spec: The configured run to translate into an invocation.

        Returns:
            An `Invocation` carrying the materialized argv, stdin, and result source.
        """
        plan = self.core_plan(spec)
        paths: dict[str, str] = {}
        try:
            for entry in plan["files"]:
                fd, path = tempfile.mkstemp(suffix=entry["suffix"])
                paths[entry["id"]] = path
                try:
                    if entry["content"] is not None:
                        os.write(fd, entry["content"].encode())
                finally:
                    os.close(fd)
            tokens = {f"${{file:{file_id}}}": path for file_id, path in paths.items()}
            argv = [tokens.get(arg, arg) for arg in plan["argv"]]
            env = plan["env"]
            if plan["needs_claude_isolation"]:
                directory = self.claude_isolation()
                argv = [arg.replace("${isolated_config_dir}", directory) for arg in argv]
                env = {key: value.replace("${isolated_config_dir}", directory) for key, value in env.items()}
        except BaseException:
            for path in paths.values():
                Path(path).unlink(missing_ok=True)
            raise
        return Invocation(
            argv,
            plan["stdin"],
            result_path=paths.get("result") if plan["read_result_from"] == "file:result" else None,
            stdout_path=paths.get("stdout") if plan["stdout_to_file"] else None,
            cleanup_paths=tuple(paths.values()),
            env=env,
            env_unset=tuple(plan["env_unset"]),
        )

    def env(self, spec: RunSpec) -> dict[str, str]:
        """Return the core-planned env, substituting the isolated config home into a claude run's values.

        Args:
            spec: The configured run; the plan gates isolation on `spec.isolated`.

        Returns:
            The plan's env map with `${isolated_config_dir}` resolved, or `{}`.
        """
        plan = self.core_plan(spec)
        if not plan["needs_claude_isolation"]:
            return plan["env"]
        directory = self.claude_isolation()
        return {key: value.replace("${isolated_config_dir}", directory) for key, value in plan["env"].items()}

    def accounting(self, raw: str) -> tuple[float | None, dict[str, object] | None]:
        """Return the `(cost_usd, usage)` the core's `resolve` op reads from `raw`."""
        resolved = _core.dispatch(
            "resolve", {"provider": self.provider, "raw": raw, "returncode": 0, "stderr": "", "wants_value": False}
        )
        return resolved["cost_usd"], resolved["usage"]

    def is_authenticated(self, *, timeout: int) -> bool:
        """Report whether any of the core's auth probes for this provider succeeds.

        Args:
            timeout: Seconds to wait for each subprocess-backed probe.

        Returns:
            `True` when a probe reports an authenticated session.
        """
        probes = _core.dispatch(
            "auth_probes",
            {"provider": self.provider, "host": {"platform": sys.platform, "home": str(Path.home())}},
        )
        return any(run_probe(probe, timeout=timeout) for probe in probes["probes"])

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
                    env={key: value for key, value in os.environ.items() if key not in inv.env_unset}
                    | self.env(spec)
                    | (spec.env or {}),
                    cwd=spec.cwd,
                    timeout=spec.timeout,
                    stdout_path=inv.stdout_path,
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
                    env={key: value for key, value in os.environ.items() if key not in inv.env_unset}
                    | self.env(spec)
                    | (spec.env or {}),
                    cwd=spec.cwd,
                    timeout=spec.timeout,
                    stdout_path=inv.stdout_path,
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

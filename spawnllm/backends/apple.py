"""CliBackend for the `spawnllm-apple` Foundation Models sidecar."""

from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import CliBackend

if TYPE_CHECKING:
    from spawnllm.types import ProviderName, TModel

BINARY = "spawnllm-apple"


class AppleBackend(CliBackend):
    """`CliBackend` for the `spawnllm-apple` on-device Foundation Models sidecar.

    The core plans a single `spawnllm-apple` invocation carrying the whole
    request as one JSON object on stdin and reads one JSON envelope back from
    stdout. Generation is local to the device, so there is no credential:
    `RunSpec`'s `model`, `isolated`, `api_auth`, `env`, and `cwd` are all inert,
    and "authenticated" means the sidecar's `--probe` reports Apple Intelligence
    available. `AppleConfig` carries the knobs that do apply — the use case and
    guardrails the session is built with, its instructions, and the decoding
    options.

    The macOS wheel bundles the sidecar, so `binary_path` prefers it and falls
    back to `PATH`; where neither resolves — Linux, or a wheel built without it —
    the backend reports `BackendNotInstalled` and auto-selection passes it by.

    Attributes:
        models: Empty identity mapping; the device hosts exactly one model.
        provider: Provider identifier keying `AppleConfig` on a `RunSpec`.
        binary: Name of the sidecar executable.
        install_hint: Suggested shell command to build the sidecar.
        schema_dialect: Apple strict-schema dialect the core applies.
        auto_select_tiers: Only `small`; the on-device model is a small model.

    Example:
        >>> from spawnllm.spec import RunSpec
        >>> AppleBackend().invocation(RunSpec(prompt="ping", model="small")).stdin[:16]
        '{"prompt":"ping"'
    """

    models: ClassVar[dict[TModel, str]] = {}
    provider: ClassVar[ProviderName] = "apple"
    binary: ClassVar[str] = BINARY
    install_hint: ClassVar[str] = "swift build -c release --package-path swift/spawnllm-apple"
    schema_dialect: ClassVar[str | None] = "apple"
    auto_select_tiers: ClassVar[frozenset[TModel] | None] = frozenset({"small"})

    def binary_path(self) -> str:
        """Return the wheel-bundled sidecar's path, or the bare binary name for a `PATH` lookup."""
        bundled = files("spawnllm").joinpath("_bin", BINARY)
        return str(bundled) if bundled.is_file() else BINARY

"""CliBackends for the Gemini CLI family (`gemini` and the `agy`/Antigravity successor)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import CliBackend

if TYPE_CHECKING:
    from spawnllm.types import ProviderName, TModel


class GeminiCliBackend(CliBackend):
    """`CliBackend` for Google's `gemini` CLI.

    The core plans a `gemini --model … -o json` invocation with the prompt
    delivered inline via `-p`; structured output appends the JSON schema and an
    instruction to emit only conforming JSON. Authentication prefers cached OAuth
    credentials and falls back to a `GEMINI_API_KEY`/`GOOGLE_API_KEY` env key.

    Attributes:
        models: Mapping from abstract model size to a Gemini model name.

    Example:
        >>> from spawnllm.spec import RunSpec
        >>> GeminiCliBackend().invocation(RunSpec(prompt="hi", model="gemini-2.5-flash")).argv[:5]
        ['gemini', '--model', 'gemini-2.5-flash', '-o', 'json']
    """

    models: ClassVar[dict[TModel, str]] = {
        "small": "gemini-2.5-flash-lite",
        "medium": "gemini-2.5-flash",
        "large": "gemini-3-pro-preview",
    }
    provider: ClassVar[ProviderName] = "gemini"
    binary: ClassVar[str] = "gemini"
    install_hint: ClassVar[str] = "npm install -g @google/gemini-cli"


class AntigravityCliBackend(CliBackend):
    """`CliBackend` for the Antigravity `agy` CLI, a Gemini-family successor.

    The core plans an `agy --model … -p` invocation and reads its plain-text
    stdout. Authentication prefers an Antigravity login stored in the macOS
    keychain and falls back to a `GEMINI_API_KEY`/`ANTIGRAVITY_API_KEY` env key.

    Attributes:
        models: Mapping from abstract model size to an Antigravity model name.

    Example:
        >>> from spawnllm.spec import RunSpec
        >>> AntigravityCliBackend().invocation(RunSpec(prompt="hi", model="gemini-3.5")).argv[:3]
        ['agy', '--model', 'gemini-3.5']
    """

    models: ClassVar[dict[TModel, str]] = {
        "small": "gemini-3.5-flash",
        "medium": "gemini-3.5",
        "large": "gemini-3.5-pro",
    }
    provider: ClassVar[ProviderName] = "antigravity"
    binary: ClassVar[str] = "agy"
    install_hint: ClassVar[str] = "curl -fsSL https://antigravity.google/cli/install.sh | bash"

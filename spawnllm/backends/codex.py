"""CliBackend for the OpenAI `codex` CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from spawnllm.backends.base import CliBackend

if TYPE_CHECKING:
    from spawnllm.types import ProviderName, TModel


class CodexCliBackend(CliBackend):
    """`CliBackend` for the OpenAI `codex` CLI.

    The core plans a `codex exec` argv that runs an ephemeral session in a
    read-only sandbox, resolving the schema to an `--output-schema` file and the
    final message to an `-o` file. It pins `service_tier=fast` by default (an
    isolated run passes `--ignore-user-config`, dropping a user-level tier pin, and
    the standard tier turns long prompts into multi-minute runs).

    Attributes:
        models: Mapping from abstract model size to an OpenAI model name.

    Example:
        >>> from spawnllm.spec import RunSpec
        >>> CodexCliBackend().invocation(RunSpec(prompt="hi", model="gpt-5.5")).argv[:4]
        ['codex', 'exec', '--ephemeral', '--sandbox']
    """

    models: ClassVar[dict[TModel, str]] = {
        "small": "gpt-5.4-mini:low",
        "medium": "gpt-5.4-mini:medium",
        "large": "gpt-5.5:medium",
    }
    provider: ClassVar[ProviderName] = "codex"
    binary: ClassVar[str] = "codex"
    install_hint: ClassVar[str] = "npm install -g @openai/codex"
    schema_dialect: ClassVar[str | None] = "openai"

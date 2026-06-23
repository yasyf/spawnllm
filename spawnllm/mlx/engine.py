"""Domain-agnostic MLX batch-inference engine running on a dedicated worker thread."""

from __future__ import annotations

import asyncio
import copy
import platform
import queue
import sys
import threading
from typing import TYPE_CHECKING, Any

import anyio.to_thread

from spawnllm.mlx.patches import MLXPatches

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

WORKER_STOP = object()


class MlxEngine:
    """Runs batched MLX inference on a dedicated worker thread.

    The constructor starts a daemon worker thread that loads the model from
    `fused_dir`, applies `MLXPatches`, and precomputes a prompt cache for
    `prefix_messages`. Await `ensure_loaded` to surface load failures before
    generating.

    Args:
        fused_dir: Directory of the MLX model to load, such as the path
            returned by `AdapterFuser.ensure_fused`.
        logits_processor_factory: Builds the logits processor from the loaded
            tokenizer; the processor is applied to every generation call.
        prefix_messages: Chat messages shared by every conversation; their
            prompt cache is computed once and reused across batches. Each
            conversation passed to `generate` must start with these messages.
        batch_size: Number of conversations generated per batch.
        worker_name: Name of the worker thread.

    Raises:
        RuntimeError: When not running on macOS with Apple Silicon.
    """

    def __init__(
        self,
        fused_dir: Path,
        *,
        logits_processor_factory: Callable[[Any], Callable[..., Any]],
        prefix_messages: list[dict[str, str]],
        batch_size: int,
        worker_name: str = "mlx",
    ) -> None:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise RuntimeError("The MLX engine requires macOS on Apple Silicon. Use a CLI backend on this platform.")
        self._fused_dir = fused_dir
        self._logits_processor_factory = logits_processor_factory
        self._prefix_messages = prefix_messages
        self._batch_size = batch_size
        self._inbox: queue.SimpleQueue = queue.SimpleQueue()
        self._loaded = threading.Event()
        self._init_error: BaseException | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name=worker_name)
        self._thread.start()

    def _worker(self) -> None:
        try:
            self._load()
        except BaseException as exc:
            self._init_error = exc
            self._loaded.set()
            return
        self._loaded.set()
        while True:
            job = self._inbox.get()
            if job is WORKER_STOP:
                return
            fn, args, on_result, on_error = job
            try:
                on_result(fn(*args))
            except BaseException as exc:
                on_error(exc)

    def _load(self) -> None:
        from mlx_lm import batch_generate, load

        MLXPatches.apply()
        self.model, self.tokenizer = load(str(self._fused_dir))
        self.logit_processor = self._logits_processor_factory(self.tokenizer)
        self.prefix_messages = self._prefix_messages
        self.prefix_tokens = self.tokenizer.apply_chat_template(
            self.prefix_messages, tokenize=True, add_generation_prompt=False
        )
        self.base_cache = batch_generate(
            self.model,
            self.tokenizer,
            [self.prefix_tokens],
            max_tokens=1,
            logits_processors=[self.logit_processor],
            return_prompt_caches=True,
        ).caches[0]

    async def ensure_loaded(self) -> None:
        """Wait for the worker thread to finish loading, re-raising the error if loading failed."""
        await anyio.to_thread.run_sync(self._loaded.wait)
        if self._init_error is not None:
            raise self._init_error

    async def submit[R](self, fn: Callable[..., R], *args: Any) -> R:
        """Run `fn(*args)` on the worker thread and await its result.

        Args:
            fn: Callable to execute on the worker thread.
            *args: Positional arguments passed to `fn`.

        Returns:
            The value returned by `fn`; an exception raised by `fn` propagates to the awaiter.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._inbox.put(
            (
                fn,
                args,
                lambda value: loop.call_soon_threadsafe(fut.set_result, value),
                lambda exc: loop.call_soon_threadsafe(fut.set_exception, exc),
            )
        )
        return await fut

    def _generate_chunk(self, chunk: list[list[dict[str, str]]], max_tokens: int) -> list[str]:
        from mlx_lm import batch_generate

        suffixes = [
            self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)[
                len(self.prefix_tokens) :
            ]
            for messages in chunk
        ]
        return batch_generate(
            self.model,
            self.tokenizer,
            suffixes,
            max_tokens=max_tokens,
            logits_processors=[self.logit_processor],
            prompt_caches=[copy.deepcopy(self.base_cache) for _ in suffixes],
        ).texts

    async def generate(
        self,
        message_lists: list[list[dict[str, str]]],
        on_progress: Callable[[int], None],
        *,
        max_tokens: int = 1,
    ) -> list[str]:
        """Generate a completion per conversation, defaulting to a single token.

        Conversations are processed in chunks of `batch_size`, ordered by the
        length of each conversation's final message so similar-length prompts
        batch together. Every conversation must start with the engine's
        `prefix_messages`; the shared prefix is served from the precomputed
        prompt cache.

        Args:
            message_lists: Conversations, each a list of chat messages with
                `role` and `content` keys.
            on_progress: Called after each chunk with the number of
                conversations completed in that chunk.
            max_tokens: Maximum tokens generated per conversation; the default
                of `1` yields single-token classification output.

        Returns:
            Generated texts, in the same order as `message_lists`.
        """
        order = sorted(range(len(message_lists)), key=lambda i: len(message_lists[i][-1]["content"]))
        responses: list[str] = [""] * len(message_lists)
        for start in range(0, len(order), self._batch_size):
            slice_ = order[start : start + self._batch_size]
            chunk = [message_lists[i] for i in slice_]
            chunk_responses = await self.submit(self._generate_chunk, chunk, max_tokens)
            for i, r in zip(slice_, chunk_responses, strict=True):
                responses[i] = r
            on_progress(len(chunk))
        return responses

    def peak_memory_gb(self) -> float:
        """Return the process's peak resident set size in GiB."""
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)

    async def close(self) -> None:
        """Signal the worker thread to exit after completing already-queued jobs."""
        self._inbox.put(WORKER_STOP)

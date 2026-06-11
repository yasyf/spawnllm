"""Runtime patches applied in-worker before the first `batch_generate`."""

from __future__ import annotations

import contextlib
import time


class MLXPatches:
    """Idempotent runtime patches for `mlx_lm`.

    `MlxEngine` applies these on its worker thread before the first
    `batch_generate` call. The single current patch replaces
    `BatchGenerator.stats` with a version that clamps the elapsed-time
    denominators when computing tokens-per-second, preventing a
    `ZeroDivisionError`.
    """

    applied: bool = False

    @classmethod
    def apply(cls) -> None:
        """Apply all patches once; subsequent calls are no-ops."""
        if cls.applied:
            return
        cls.applied = True
        cls._apply_batchstats_zerodiv_guard()

    @staticmethod
    def _apply_batchstats_zerodiv_guard() -> None:
        import mlx.core as mx
        from mlx_lm.generate import BatchGenerator, BatchStats

        @contextlib.contextmanager
        def stats(self, stats: BatchStats | None = None):
            stats = stats or BatchStats()
            self._prompt_tokens_counter = 0
            self._prompt_time_counter = 0
            self._gen_tokens_counter = 0
            tic = time.perf_counter()
            try:
                yield stats
            finally:
                total_time = time.perf_counter() - tic
                stats.prompt_tokens += self._prompt_tokens_counter
                stats.prompt_time += self._prompt_time_counter
                stats.prompt_tps = stats.prompt_tokens / max(stats.prompt_time, 1e-9)
                stats.generation_tokens += self._gen_tokens_counter
                stats.generation_time += total_time - self._prompt_time_counter
                stats.generation_tps = stats.generation_tokens / max(stats.generation_time, 1e-9)
                stats.peak_memory = max(stats.peak_memory, mx.get_peak_memory() / 1e9)

        BatchGenerator.stats = stats

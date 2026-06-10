from __future__ import annotations

import sys
import threading
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spawnllm.mlx import MlxEngine

MLX_AVAILABLE: bool = find_spec("mlx_lm") is not None and sys.platform == "darwin"


class TestGenerateSorting:
    async def test_sorts_by_last_content_length(self) -> None:
        message_lists = [
            [{"role": "user", "content": "longest" * 100}],
            [{"role": "user", "content": "short"}],
            [{"role": "user", "content": "medium" * 30}],
        ]
        seen_chunks: list[list[str]] = []

        def fake_chunk(self: MlxEngine, chunk: list[list[dict[str, str]]]) -> list[str]:
            seen_chunks.append([m[-1]["content"] for m in chunk])
            return [str(len(m[-1]["content"])) for m in chunk]

        async def fake_submit(self: MlxEngine, fn, *args):
            return fn(*args)

        engine = MlxEngine.__new__(MlxEngine)
        engine._batch_size = 2
        with (
            patch.object(MlxEngine, "_generate_chunk", fake_chunk),
            patch.object(MlxEngine, "submit", fake_submit),
        ):
            responses = await engine.generate(message_lists, on_progress=lambda n: None)

        flat_seen = [content for chunk in seen_chunks for content in chunk]
        assert flat_seen == sorted(flat_seen, key=len)
        assert responses == [str(len(m[-1]["content"])) for m in message_lists]

    async def test_preserves_original_order(self) -> None:
        message_lists = [[{"role": "user", "content": "x" * n}] for n in (50, 5, 200, 1, 30)]

        def fake_chunk(self: MlxEngine, chunk: list[list[dict[str, str]]]) -> list[str]:
            return [str(len(m[-1]["content"])) for m in chunk]

        async def fake_submit(self: MlxEngine, fn, *args):
            return fn(*args)

        engine = MlxEngine.__new__(MlxEngine)
        engine._batch_size = 2
        with (
            patch.object(MlxEngine, "_generate_chunk", fake_chunk),
            patch.object(MlxEngine, "submit", fake_submit),
        ):
            responses = await engine.generate(message_lists, on_progress=lambda n: None)
        assert responses == ["50", "5", "200", "1", "30"]


class TestGenerateChunk:
    def test_strips_prefix_and_deepcopies_cache_per_suffix(self) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.batch_generate.return_value = MagicMock(texts=["3", "4"])

        engine = MlxEngine.__new__(MlxEngine)
        engine.model = MagicMock()
        engine.tokenizer = mock_tokenizer
        engine.logit_processor = MagicMock()
        engine.prefix_tokens = [1, 2]
        engine.base_cache = object()

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx_lm}):
            out = engine._generate_chunk([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]])

        assert out == ["3", "4"]
        args, kwargs = mock_mlx_lm.batch_generate.call_args
        assert args[2] == [[3, 4, 5], [3, 4, 5]]  # suffix = template[len(prefix_tokens):]
        assert kwargs["max_tokens"] == 1
        assert len(kwargs["prompt_caches"]) == 2


@pytest.mark.skipif(not MLX_AVAILABLE, reason="requires mlx-lm (Apple Silicon)")
class TestThreadAffinity:
    async def test_load_and_inference_run_on_same_worker_thread(self) -> None:
        observed: list[int] = []

        def fake_load(path: str):
            observed.append(threading.get_ident())
            return MagicMock(), MagicMock(apply_chat_template=lambda *a, **k: [1, 2, 3])

        def fake_batch_generate(*args, **kwargs):
            observed.append(threading.get_ident())
            return MagicMock(caches=[MagicMock()], texts=["3"])

        mock_mlx_lm = MagicMock(load=fake_load, batch_generate=fake_batch_generate)

        with (
            patch.dict(sys.modules, {"mlx_lm": mock_mlx_lm}),
            patch("spawnllm.mlx.engine.MLXPatches.apply"),
        ):
            engine = MlxEngine(
                Path("/tmp/fake-fused"),
                logits_processor_factory=lambda tok: MagicMock(),
                prefix_messages=[],
                batch_size=2,
            )
            await engine.ensure_loaded()
            for _ in range(3):
                await engine.submit(engine._generate_chunk, [[{"role": "user", "content": "hi"}]])

        assert len(set(observed)) == 1, f"MLX work spread across threads: {observed}"
        assert observed[0] != threading.get_ident()

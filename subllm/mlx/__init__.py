"""Local MLX engine, adapter codec, fuser, and runtime patches.

Imports here are lazy so that ``import subllm`` never pulls ``mlx_lm``/``zstandard``;
only consumers that touch ``subllm.mlx`` attributes load the heavy dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subllm.mlx.codec import AdapterCodec
    from subllm.mlx.engine import MlxEngine
    from subllm.mlx.fuse import AdapterFuser
    from subllm.mlx.patches import MLXPatches

__all__ = ["AdapterCodec", "AdapterFuser", "MLXPatches", "MlxEngine"]


def __getattr__(name: str) -> object:
    match name:
        case "AdapterCodec":
            from subllm.mlx.codec import AdapterCodec

            return AdapterCodec
        case "AdapterFuser":
            from subllm.mlx.fuse import AdapterFuser

            return AdapterFuser
        case "MlxEngine":
            from subllm.mlx.engine import MlxEngine

            return MlxEngine
        case "MLXPatches":
            from subllm.mlx.patches import MLXPatches

            return MLXPatches
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

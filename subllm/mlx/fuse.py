"""Fuse a shipped LoRA adapter into a base MLX model, cached in the HF-hub layout."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from subllm.mlx.codec import AdapterCodec


class AdapterFuser:
    @classmethod
    def ensure_fused(
        cls,
        model_repo: str,
        *,
        codec: AdapterCodec,
        cache_namespace: str,
        tqdm_class: type | None = None,
    ) -> Path:
        from huggingface_hub.constants import HF_HUB_CACHE

        digest = codec.digest()
        repo_dir = Path(HF_HUB_CACHE) / f"models--{cache_namespace}-{digest}"
        fused_dir = repo_dir / "snapshots" / digest
        if (fused_dir / "config.json").exists():
            return fused_dir

        from huggingface_hub import snapshot_download
        from mlx.utils import tree_unflatten
        from mlx_lm.utils import load_adapters, load_model, load_tokenizer, save

        src_path = Path(snapshot_download(model_repo, tqdm_class=tqdm_class))
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            (staging / "adapter_config.json").write_bytes(codec.CONFIG.read_bytes())
            codec.decode(staging / "adapters.safetensors")
            model, config = load_model(src_path, lazy=False, strict=False)
            model = load_adapters(model, str(staging))
            model.eval()
            tokenizer = load_tokenizer(src_path, eos_token_ids=config.get("eos_token_id"))
            model.update_modules(
                tree_unflatten([(n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")])
            )
            fused_dir.mkdir(parents=True, exist_ok=True)
            save(fused_dir, src_path, model, tokenizer, config, donate_model=True)

        (refs := repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(digest)
        return fused_dir

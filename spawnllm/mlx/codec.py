"""Cross-platform LoRA adapter codec (byte-shuffle + zstd; imports without `mlx_lm`)."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import ClassVar

import orjson


class AdapterCodec:
    """Compresses and decompresses a homogeneous-dtype safetensors LoRA adapter.

    Each tensor's bytes are grouped by byte position within the dtype
    (byte-shuffle) before zstd compression. Subclasses override `DIR`, `ZST`,
    and `CONFIG` to point at their shipped package data.

    Example:
        >>> class ShippedCodec(AdapterCodec):
        ...     DIR = Path(__file__).parent
        ...     ZST = DIR / "adapters.safetensors.zst"
        ...     CONFIG = DIR / "adapter_config.json"
        >>> ShippedCodec.encode(Path("trained/adapters.safetensors"))
    """

    DIR: ClassVar[Path] = Path(__file__).parent
    ZST: ClassVar[Path] = DIR / "adapters.safetensors.zst"
    CONFIG: ClassVar[Path] = DIR / "adapter_config.json"
    TYPESIZES: ClassVar[dict[str, int]] = {"F32": 4, "BF16": 2, "F16": 2}
    COMPRESSION_LEVEL: ClassVar[int] = 19

    @classmethod
    def digest(cls) -> str:
        """Return the first 16 hex digits of the SHA-256 of the compressed adapter `ZST`."""
        return hashlib.sha256(cls.ZST.read_bytes()).hexdigest()[:16]

    @classmethod
    def encode(cls, src: Path) -> None:
        """Compress the safetensors adapter at `src` into `ZST`.

        Byte-shuffles each tensor's data, then compresses with zstd at
        `COMPRESSION_LEVEL`. All tensors must share one dtype drawn from
        `TYPESIZES`.

        Args:
            src: Path to the adapter's safetensors file.
        """
        import zstandard as zstd

        raw = src.read_bytes()
        cls._assert_homogeneous_dtype(raw)
        cls.ZST.write_bytes(zstd.ZstdCompressor(level=cls.COMPRESSION_LEVEL).compress(cls._walk(raw, shuffle=True)))

    @classmethod
    def decode(cls, dst: Path) -> None:
        """Decompress `ZST` and write the restored safetensors adapter to `dst`.

        Args:
            dst: Destination path for the safetensors file.
        """
        import zstandard as zstd

        dst.write_bytes(cls._walk(zstd.ZstdDecompressor().decompress(cls.ZST.read_bytes()), shuffle=False))

    @classmethod
    def dtype(cls) -> str:
        """Return the safetensors dtype shared by the compressed adapter's tensors, e.g. `"BF16"`."""
        import zstandard as zstd

        raw = zstd.ZstdDecompressor().decompress(cls.ZST.read_bytes())
        cls._assert_homogeneous_dtype(raw)
        header_end = 8 + struct.unpack("<Q", raw[:8])[0]
        return next(v["dtype"] for k, v in orjson.loads(raw[8:header_end]).items() if k != "__metadata__")

    @classmethod
    def _walk(cls, raw: bytes, *, shuffle: bool) -> bytes:
        header_end = 8 + struct.unpack("<Q", raw[:8])[0]
        body = raw[header_end:]
        out = bytearray(raw[:header_end])
        cursor = 0
        for name, meta in sorted(
            ((k, v) for k, v in orjson.loads(raw[8:header_end]).items() if k != "__metadata__"),
            key=lambda kv: kv[1]["data_offsets"][0],
        ):
            assert meta["dtype"] in cls.TYPESIZES, f"{name}: unsupported dtype {meta['dtype']}"
            typesize = cls.TYPESIZES[meta["dtype"]]
            nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
            chunk = body[cursor : cursor + nbytes]
            out.extend(cls._shuffle(chunk, typesize) if shuffle else cls._unshuffle(chunk, typesize))
            cursor += nbytes
        return bytes(out)

    @classmethod
    def _assert_homogeneous_dtype(cls, raw: bytes) -> None:
        header_end = 8 + struct.unpack("<Q", raw[:8])[0]
        dtypes = {v["dtype"] for k, v in orjson.loads(raw[8:header_end]).items() if k != "__metadata__"}
        assert len(dtypes) == 1, f"adapter must be homogeneous-dtype, got {dtypes}"
        assert dtypes.issubset(cls.TYPESIZES.keys()), f"unsupported dtype: {dtypes}"

    @classmethod
    def _shuffle(cls, chunk: bytes, typesize: int) -> bytes:
        import numpy as np

        return np.frombuffer(chunk, dtype=np.uint8).reshape(-1, typesize).T.tobytes()

    @classmethod
    def _unshuffle(cls, chunk: bytes, typesize: int) -> bytes:
        import numpy as np

        return np.frombuffer(chunk, dtype=np.uint8).reshape(typesize, -1).T.tobytes()

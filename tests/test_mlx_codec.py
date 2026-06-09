from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from subllm.mlx import AdapterCodec


def make_safetensors(tensors: dict[str, tuple[str, bytes]]) -> bytes:
    header: dict[str, object] = {}
    body = bytearray()
    for name, (dtype, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": [len(data)], "data_offsets": [len(body), len(body) + len(data)]}
        body.extend(data)
    header_json = json.dumps(header).encode()
    return struct.pack("<Q", len(header_json)) + header_json + bytes(body)


F32_BLOB = make_safetensors({"w": ("F32", bytes(range(16)) * 4)})
BF16_BLOB = make_safetensors({"w": ("BF16", bytes(range(16)) * 2)})


class TestAdapterCodecRoundtrip:
    @pytest.mark.parametrize("blob", [F32_BLOB, BF16_BLOB], ids=["f32", "bf16"])
    def test_byte_exact(self, blob: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        src = tmp_path / "src.safetensors"
        src.write_bytes(blob)
        monkeypatch.setattr(AdapterCodec, "ZST", tmp_path / "adapters.safetensors.zst")
        AdapterCodec.encode(src)
        decoded = tmp_path / "decoded.safetensors"
        AdapterCodec.decode(decoded)
        assert hashlib.sha256(decoded.read_bytes()).hexdigest() == hashlib.sha256(blob).hexdigest()

    def test_dtype_reports_correctly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        src = tmp_path / "src.safetensors"
        monkeypatch.setattr(AdapterCodec, "ZST", tmp_path / "a.zst")
        src.write_bytes(F32_BLOB)
        AdapterCodec.encode(src)
        assert AdapterCodec.dtype() == "F32"
        src.write_bytes(BF16_BLOB)
        AdapterCodec.encode(src)
        assert AdapterCodec.dtype() == "BF16"

    def test_digest_is_16_hex_chars(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        src = tmp_path / "src.safetensors"
        src.write_bytes(F32_BLOB)
        monkeypatch.setattr(AdapterCodec, "ZST", tmp_path / "a.zst")
        AdapterCodec.encode(src)
        digest = AdapterCodec.digest()
        assert len(digest) == 16
        assert int(digest, 16) >= 0

    def test_rejects_mixed_dtype(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mixed = make_safetensors({"a": ("F32", b"\x00" * 16), "b": ("BF16", b"\x00" * 8)})
        src = tmp_path / "mixed.safetensors"
        src.write_bytes(mixed)
        monkeypatch.setattr(AdapterCodec, "ZST", tmp_path / "a.zst")
        with pytest.raises(AssertionError, match="homogeneous"):
            AdapterCodec.encode(src)

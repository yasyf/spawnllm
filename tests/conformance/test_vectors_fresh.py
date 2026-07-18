"""Freshness gate: the committed vectors must equal a fresh in-memory regeneration.

The generator drives the real `spawnllm` code, so any behavior change — a backend
flag tweak, a private SDK-transform bump, a registry edit — makes a committed
vector diverge and fails here. Regenerate with the printed command.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conformance import generate

REGEN = "uv run python -m tests.conformance.generate"


def _committed() -> set[Path]:
    return set(generate.VECTORS_DIR.rglob("*.json")) | set(generate.SCHEMA_DIR.glob("*.json"))


@pytest.mark.parametrize("path", sorted(generate.build_all()), ids=lambda p: str(p.relative_to(generate.REPO_ROOT)))
def test_committed_vector_matches_fresh(path: Path) -> None:
    expected = generate.build_all()[path]
    assert path.exists(), f"missing committed artifact {path}; regenerate with:\n  {REGEN}"
    assert path.read_text() == expected, f"stale committed artifact {path}; regenerate with:\n  {REGEN}"


def test_no_orphan_committed_vectors() -> None:
    orphans = _committed() - set(generate.build_all())
    assert not orphans, f"committed artifacts no longer generated; regenerate with:\n  {REGEN}\n{sorted(orphans)}"

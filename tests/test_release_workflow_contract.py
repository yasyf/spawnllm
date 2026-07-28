import re
from pathlib import Path

import pytest

from spawnllm.backends.apple import BINARY

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
HOME_BREW_TAP_SHA = "2281a3ea884422db190de44fad65ce9bc08b19c4"
PYPI_PUBLISH_SHA = "ba38be9e461d3875417946c167d0b5f3d385a247"
PYPI_BUILD_SHA = "41f8de6765b3b833ef333b0b98f5683f0e46685b"
USES = re.compile(r"uses:\s*(\S+)")
SHA_PINNED = re.compile(r"[^@\s]+@[0-9a-f]{40}")


def workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text()


def jobs(name: str) -> dict[str, str]:
    body = workflow(name).split("\njobs:\n", 1)[1]
    starts = [(m.group(1), m.start()) for m in re.finditer(r"^  (\w[\w-]*):$", body, re.MULTILINE)]
    bounds = [*(start for _, start in starts[1:]), len(body)]
    return {job: body[start:end] for (job, start), end in zip(starts, bounds, strict=True)}


@pytest.mark.parametrize("name", ["release.yml", "ci.yml"], ids=["release", "ci"])
def test_every_action_is_sha_pinned(name: str) -> None:
    unpinned = {
        f"{job}: {ref}"
        for job, body in jobs(name).items()
        for ref in USES.findall(body)
        if not SHA_PINNED.fullmatch(ref)
    }

    assert unpinned == set(), f"{name} runs unpinned actions — no exemption exists, not even for yasyf/*"


def test_release_pins_the_audited_publish_action() -> None:
    assert "pypa/gh-action-pypi-publish@" + PYPI_PUBLISH_SHA in workflow("release.yml")


def test_binrun_is_installed_at_a_pinned_module_version() -> None:
    installs = {
        m.group(1)
        for name in ("release.yml", "ci.yml")
        for m in re.finditer(r"go install github\.com/yasyf/binrun/cmd/binrun@(\S+)", workflow(name))
    }

    assert installs == {"v0.2.0"}


def test_every_job_that_publishes_verifies_the_tag_is_on_main() -> None:
    gate = "yasyf/homebrew-tap/.github/actions/verify-tag-on-main@" + HOME_BREW_TAP_SHA
    gated = {job for job, body in jobs("release.yml").items() if gate in body}

    assert gated == {"blob", "macos-wheel", "go", "crates"}


def test_pypi_build_contract_is_preserved() -> None:
    release = workflow("release.yml")
    assert ("uses: yasyf/homebrew-tap/.github/workflows/release-pypi-build.yml@" + PYPI_BUILD_SHA) in release
    assert "      dist-name: spawnllm" in release
    assert '      python-version: "3.14"' in release
    assert "      pre-build: rustup target add wasm32-wasip1 && bash scripts/build_wasm.sh" in release


def test_the_blob_is_built_once_and_never_overwritten() -> None:
    body = jobs("release.yml")["blob"]
    assert 'SPAWNLLM_BLOB_NO_FETCH: "1"' in body, "the blob job is the producer — it must build, not fetch"
    assert 'if gh release view "$TAG" --json assets --jq \'.assets[].name\' | grep -qxF "$ASSET"' in body
    assert "--clobber" not in body, "re-uploading the blob strands every consumer pinned to the old sha256"


def test_consuming_lanes_never_build_the_blob_themselves() -> None:
    consumers = {job: body for job, body in jobs("release.yml").items() if job != "blob"}
    builders = {job for job, body in consumers.items() if "SPAWNLLM_BLOB_NO_FETCH" in body}

    assert builders == set(), "only the blob job may build; every other lane fetches the published asset"
    assert 'if [[ "${GITHUB_REF_TYPE:-}" == tag ]]; then' in (ROOT / "scripts" / "build_wasm.sh").read_text(), (
        "a tagged run must fail on a fetch miss rather than silently building divergent bytes"
    )


def test_the_versioned_sidecar_archive_is_write_once() -> None:
    body = jobs("release.yml")["macos-wheel"]
    assert 'if gh release view "$TAG" --json assets --jq \'.assets[].name\' | grep -qxF "$ASSET"' in body
    assert 'gh release upload "$TAG" "$ASSET"\n' in body
    assert '"$ASSET" --clobber' not in body


def test_github_release_waits_for_every_language_artifact() -> None:
    needs = re.search(r"^  github-release:\n    needs: \[(.+)\]$", workflow("release.yml"), re.MULTILINE)
    assert needs

    assert set(needs.group(1).split(", ")) == {"build", "macos-wheel", "publish", "descriptor", "go", "crates"}


def test_macos_wheel_stages_the_sidecar_where_the_backend_looks_for_it() -> None:
    release = workflow("release.yml")
    assert f"cp swift/spawnllm-apple/.build/release/{BINARY} spawnllm/_bin/{BINARY}" in release
    assert f"chmod +x spawnllm/_bin/{BINARY}" in release
    assert "run: uv build --wheel" in release

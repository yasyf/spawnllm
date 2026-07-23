import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
HOME_BREW_TAP_SHA = "4afbb78f9e1814af04f9686ccf101ecafd5aa295"
PYPI_PUBLISH_SHA = "ba38be9e461d3875417946c167d0b5f3d385a247"
PYPI_BUILD_SHA = "8f422c652d836c40f9cc5a9d893d4120b26bc681"


def workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text()


def test_release_actions_are_immutable() -> None:
    releases = "\n".join(workflow(name) for name in ("release-crates.yml", "release-go.yml", "release-pypi.yml"))
    assert releases.count("yasyf/homebrew-tap/.github/actions/verify-tag-on-main@" + HOME_BREW_TAP_SHA) == 2
    assert "pypa/gh-action-pypi-publish@" + PYPI_PUBLISH_SHA in releases
    assert not re.search(
        r"(?:verify-tag-on-main|gh-action-pypi-publish)@(?![0-9a-f]{40}(?:\s|$))[^\s]+",
        releases,
    )


def test_pypi_build_contract_is_preserved() -> None:
    release = workflow("release-pypi.yml")
    assert ("uses: yasyf/homebrew-tap/.github/workflows/release-pypi-build.yml@" + PYPI_BUILD_SHA) in release
    assert "      dist-name: spawnllm" in release
    assert '      python-version: "3.14"' in release
    assert "      pre-build: rustup target add wasm32-wasip1 && bash scripts/build_wasm.sh" in release

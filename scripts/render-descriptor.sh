#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:?usage: render-descriptor.sh <archive> <version>}"
VERSION="${2:?usage: render-descriptor.sh <archive> <version>}"
REPO="${GITHUB_REPOSITORY:-yasyf/spawnllm}"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum
  else
    shasum -a 256
  fi
}

DIGEST="$(sha256 <"$ARCHIVE" | cut -d' ' -f1)"
SIZE="$(wc -c <"$ARCHIVE" | tr -d ' ')"

cat <<JSON
#!/usr/bin/env binrun
{
  "schema": 1,
  "name": "spawnllm-apple",
  "kind": "release-binary",
  "version": {"static": "$VERSION"},
  "platforms": {
    "macos-aarch64": {
      "size": $SIZE,
      "hash": "sha256",
      "digest": "$DIGEST",
      "format": "tar.gz",
      "path": "spawnllm-apple",
      "providers": [
        {"type": "github-release", "repo": "$REPO", "tag": "v$VERSION", "name": "$(basename "$ARCHIVE")"}
      ]
    }
  }
}
JSON

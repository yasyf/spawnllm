#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BLOB_REPO="yasyf/spawnllm"
GO_DEST="$REPO_ROOT/go/internal/core/spawnllm_core.wasm"
PYTHON_DEST="$REPO_ROOT/spawnllm/_core/spawnllm_core.wasm"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum
  else
    shasum -a 256
  fi
}

source_hash() {
  cd "$REPO_ROOT"
  find rust/spawnllm-core rust/spawnllm-wasm rust/Cargo.lock rust/Cargo.toml rust/rust-toolchain.toml scripts/build_wasm.sh -type f \
    | LC_ALL=C sort \
    | while IFS= read -r f; do
        printf '%s  %s\n' "$(sha256 <"$f" | cut -d' ' -f1)" "$f"
      done \
    | sha256 | cut -d' ' -f1
}

SRC_HASH="$(source_hash)"
ASSET="spawnllm_core-${SRC_HASH}.wasm"

if [[ "${1:-}" == "--hash-only" ]]; then
  printf '%s\n' "$SRC_HASH"
  exit 0
fi

CURL=(curl -fsSL --retry 2)
if [[ -n "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ]]; then
  CURL+=(-H "Authorization: Bearer ${GH_TOKEN:-${GITHUB_TOKEN}}")
fi

install_blob() {
  cp "$1" "$GO_DEST"
  cp "$1" "$PYTHON_DEST"
  printf 'blob: %s, %s (%s bytes each), source_hash=%s, via=%s\n' \
    "$GO_DEST" "$PYTHON_DEST" "$(wc -c < "$GO_DEST" | tr -d ' ')" "$SRC_HASH" "$2"
}

# The release lane hits the CDN, not the API: its pre-build gets no token, and anonymous
# api.github.com is rate-limited to 60/hour per address.
tag_urls() {
  [[ "${GITHUB_REF_TYPE:-}" == tag ]] || return 0
  local base="https://github.com/$BLOB_REPO/releases/download/$GITHUB_REF_NAME"
  printf '%s/%s %s/%s.sha256\n' "$base" "$ASSET" "$base" "$ASSET"
}

searched_urls() {
  "${CURL[@]}" -H 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/$BLOB_REPO/releases?per_page=100" \
    | python3 -c '
import json, sys
name = sys.argv[1]
assets = {a["name"]: a["url"] for r in json.load(sys.stdin) for a in r["assets"]}
if {name, name + ".sha256"} <= assets.keys():
    print(assets[name], assets[name + ".sha256"])
' "$ASSET" 2>/dev/null
}

fetch_from() {
  [[ -n "$1" ]] || return 1
  read -r blob_url sha_url <<<"$1"
  "${CURL[@]}" -H 'Accept: application/octet-stream' "$blob_url" -o "$FETCH_DIR/blob.wasm" \
    && "${CURL[@]}" -H 'Accept: application/octet-stream' "$sha_url" -o "$FETCH_DIR/blob.sha256"
}

if [[ "${SPAWNLLM_BLOB_NO_FETCH:-}" != "1" ]]; then
  FETCH_DIR="$(mktemp -d)"
  trap 'rm -rf "$FETCH_DIR"' EXIT
  if fetch_from "$(tag_urls)" || fetch_from "$(searched_urls)"; then
    [[ "$(sha256 <"$FETCH_DIR/blob.wasm" | cut -d' ' -f1)" == "$(cut -d' ' -f1 <"$FETCH_DIR/blob.sha256")" ]] \
      || { printf 'blob: %s failed its published sha256\n' "$ASSET" >&2; exit 1; }
    install_blob "$FETCH_DIR/blob.wasm" "release asset"
    exit 0
  fi
  if [[ "${GITHUB_REF_TYPE:-}" == tag ]]; then
    printf 'blob: %s is not published; building it here would ship bytes that differ from the release asset\n' "$ASSET" >&2
    exit 1
  fi
  printf 'blob: %s not published; building from source\n' "$ASSET" >&2
fi

export SPAWNLLM_CORE_SRC_HASH="$SRC_HASH"

cd "$REPO_ROOT/rust"
cargo build -p spawnllm-wasm --target wasm32-wasip1 --profile wasm-release

TARGET_DIR="$(cargo metadata --format-version 1 --no-deps \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"

install_blob "$TARGET_DIR/wasm32-wasip1/wasm-release/spawnllm_wasm.wasm" "cargo build"

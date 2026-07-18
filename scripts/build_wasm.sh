#!/usr/bin/env bash
# Builds the source-hash-stamped wasm blob for Go and Python; --hash-only prints just the hash.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum
  else
    shasum -a 256
  fi
}

# Metadata-free content manifest: identical across checkouts and macOS/Linux hashers.
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

if [[ "${1:-}" == "--hash-only" ]]; then
  printf '%s\n' "$SRC_HASH"
  exit 0
fi

export SPAWNLLM_CORE_SRC_HASH="$SRC_HASH"

cd "$REPO_ROOT/rust"
cargo build -p spawnllm-wasm --target wasm32-wasip1 --profile wasm-release

TARGET_DIR="$(cargo metadata --format-version 1 --no-deps \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"

GO_DEST="$REPO_ROOT/go/internal/core/spawnllm_core.wasm"
PYTHON_DEST="$REPO_ROOT/spawnllm/_core/spawnllm_core.wasm"
cp "$TARGET_DIR/wasm32-wasip1/wasm-release/spawnllm_wasm.wasm" "$GO_DEST"
cp "$TARGET_DIR/wasm32-wasip1/wasm-release/spawnllm_wasm.wasm" "$PYTHON_DEST"

printf 'blob: %s, %s (%s bytes each), source_hash=%s\n' \
  "$GO_DEST" "$PYTHON_DEST" "$(wc -c < "$GO_DEST" | tr -d ' ')" "$SRC_HASH"

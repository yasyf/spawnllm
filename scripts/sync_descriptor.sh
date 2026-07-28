#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE="$REPO_ROOT/descriptor/spawnllm-apple.binrun"
GO_DEST="$REPO_ROOT/go/internal/sidecar/spawnllm-apple.binrun"
RUST_DEST="$REPO_ROOT/rust/spawnllm/spawnllm-apple.binrun"

if [[ "${1:-}" == "--check" ]]; then
  status=0
  for dest in "$GO_DEST" "$RUST_DEST"; do
    cmp -s "$SOURCE" "$dest" || {
      printf 'descriptor drift: %s differs from %s — run scripts/sync_descriptor.sh\n' \
        "$dest" "$SOURCE" >&2
      diff -u "$SOURCE" "$dest" >&2 || true
      status=1
    }
  done
  ((status == 0)) && printf 'descriptor: all three copies match\n'
  exit "$status"
fi

cp "$SOURCE" "$GO_DEST"
cp "$SOURCE" "$RUST_DEST"
chmod +x "$GO_DEST" "$RUST_DEST"

printf 'descriptor: %s, %s (%s bytes each)\n' \
  "$GO_DEST" "$RUST_DEST" "$(wc -c <"$GO_DEST" | tr -d ' ')"

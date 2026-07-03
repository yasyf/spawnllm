#!/usr/bin/env bash
# Regenerate docs/assets/demo.png from a real run of `uvx spawnllm status`.
# Requires freeze (https://github.com/charmbracelet/freeze): brew install freeze
set -euo pipefail
cd "$(dirname "$0")/../.."

out="$(mktemp)"
trap 'rm -f "$out"' EXIT

printf '$ uvx spawnllm status\n' > "$out"
uvx spawnllm status >> "$out"

freeze "$out" \
  --language console \
  --theme github-dark \
  --background "#0d1117" \
  --window \
  --padding 24 \
  --font.size 28 \
  --output docs/assets/demo.png

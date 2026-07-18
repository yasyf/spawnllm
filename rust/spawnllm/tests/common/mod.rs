//! Shared fixtures for the integration tests: fake `claude`/`codex`/`security`
//! CLIs installed on a temp dir that is prepended to `PATH` once per test process.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Serializes tests that mutate the process environment (`CLAUDE_CONFIG_DIR`).
pub static ENV_LOCK: Mutex<()> = Mutex::new(());

static FIXTURES: OnceLock<PathBuf> = OnceLock::new();

const CLAUDE_FAKE: &str = r#"#!/bin/sh
if [ "$1" = "auth" ]; then exit 0; fi
has_schema=0
schema=""
prev=""
for a in "$@"; do
  case "$a" in --json-schema) has_schema=1 ;; esac
  if [ "$prev" = "--json-schema" ]; then schema="$a"; fi
  prev="$a"
done
stdin=$(cat)
if [ -n "$SPAWNLLM_FAKE_STDIN_OUT" ]; then printf '%s' "$stdin" > "$SPAWNLLM_FAKE_STDIN_OUT"; fi
case "$stdin" in DUMP_SCHEMA_TO=*) printf '%s' "$schema" > "${stdin#DUMP_SCHEMA_TO=}" ;; esac
if [ -n "$SPAWNLLM_FAKE_MARKER" ]; then
  if [ -f /dev/stdout ]; then printf 'regular' > "$SPAWNLLM_FAKE_MARKER"; else printf 'pipe' > "$SPAWNLLM_FAKE_MARKER"; fi
fi
if [ -n "$SPAWNLLM_FAKE_CRED_OUT" ]; then cat "$CLAUDE_CONFIG_DIR/.credentials.json" > "$SPAWNLLM_FAKE_CRED_OUT" 2>/dev/null || true; fi
if [ -n "$SPAWNLLM_FAKE_ACCOUNT_OUT" ]; then cat "$CLAUDE_CONFIG_DIR/.claude.json" > "$SPAWNLLM_FAKE_ACCOUNT_OUT" 2>/dev/null || true; fi
if [ -n "$SPAWNLLM_FAKE_EXIT" ]; then printf 'boom' >&2; exit "$SPAWNLLM_FAKE_EXIT"; fi
if [ -n "$SPAWNLLM_FAKE_SLEEP" ]; then sleep "$SPAWNLLM_FAKE_SLEEP"; fi
if [ -n "$SPAWNLLM_FAKE_COUNTER" ]; then
  n=$(cat "$SPAWNLLM_FAKE_COUNTER" 2>/dev/null || printf 0)
  n=$((n + 1))
  printf '%s' "$n" > "$SPAWNLLM_FAKE_COUNTER"
  if [ "$n" -lt 2 ]; then printf '{"is_error": true, "result": "overloaded"}'; exit 0; fi
fi
if [ "$has_schema" = "1" ]; then
  printf '{"result": "ok", "structured_output": {"x": 42}}'
else
  printf '{"result": "hello", "total_cost_usd": 0.01}'
fi
"#;

const CODEX_FAKE: &str = r#"#!/bin/sh
if [ "$1" = "login" ]; then exit 0; fi
out=""
sfile=""
prev=""
has_schema=0
for a in "$@"; do
  if [ "$prev" = "-o" ]; then out="$a"; fi
  if [ "$prev" = "--output-schema" ]; then sfile="$a"; fi
  case "$a" in --output-schema) has_schema=1 ;; esac
  prev="$a"
done
stdin=$(cat)
case "$stdin" in DUMP_SCHEMA_TO=*) cat "$sfile" > "${stdin#DUMP_SCHEMA_TO=}" 2>/dev/null || true ;; esac
if [ -n "$SPAWNLLM_FAKE_SLEEP" ]; then sleep "$SPAWNLLM_FAKE_SLEEP"; fi
printf 'STREAMED LOG NOISE\n'
if [ "$has_schema" = "1" ]; then
  printf '%s' '{"x": 42}' > "$out"
else
  printf '%s' 'codex-hello' > "$out"
fi
"#;

const SECURITY_FAKE: &str = r#"#!/bin/sh
for a in "$@"; do
  if [ "$a" = "-w" ]; then printf 'keychain-token-xyz'; exit 0; fi
done
exit 1
"#;

/// Materialize the fake CLIs once and prepend their dir to `PATH`; returns the dir.
pub fn fixtures() -> PathBuf {
    FIXTURES
        .get_or_init(|| {
            let dir =
                std::env::temp_dir().join(format!("spawnllm-fixtures-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            write_script(&dir, "claude", CLAUDE_FAKE);
            write_script(&dir, "codex", CODEX_FAKE);
            write_script(&dir, "security", SECURITY_FAKE);
            let original = std::env::var("PATH").unwrap_or_default();
            // SAFETY: run once, gated by OnceLock, before any test spawns a child.
            unsafe { std::env::set_var("PATH", format!("{}:{original}", dir.display())) };
            dir
        })
        .clone()
}

fn write_script(dir: &Path, name: &str, body: &str) {
    let path = dir.join(name);
    std::fs::write(&path, body).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
}

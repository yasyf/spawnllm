# spawnllm-core

The sans-io engine behind spawnllm's language bindings. Every drift-prone decision — per-backend argv planning, output-envelope resolution, strict-JSON-schema transforms, retry policy, routing tables, claude isolation seeding — lives here once, behind a JSON `dispatch` boundary, with behavior pinned to the Python reference implementation by a shared golden-vector suite.

This crate does no I/O and is not a user-facing API: it plans invocations and resolves outputs, and a host executes them. Depend on [`spawnllm`](https://crates.io/crates/spawnllm) from Rust, or [`github.com/yasyf/spawnllm/go`](https://pkg.go.dev/github.com/yasyf/spawnllm/go) from Go (which embeds a WASM build of this crate).

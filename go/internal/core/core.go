// Package core embeds the spawnllm-core wasm blob and dispatches ops to it over
// wazero, keeping the Go bindings pure Go (no cgo) and in lockstep with the Rust core.
package core

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

//go:embed spawnllm_core.wasm
var wasmBlob []byte

type engine struct {
	mu     sync.Mutex
	module api.Module
	stderr *bytes.Buffer
	alloc  api.Function
	free   api.Function
	call   api.Function
}

// Version mirrors the core `version` op payload.
type Version struct {
	CoreVersion string `json:"core_version"`
	SourceHash  string `json:"source_hash"`
}

// DispatchError is the `{"err":{...}}` envelope the core returns for a failed op.
type DispatchError struct {
	Kind string
	Msg  string
}

func (e *DispatchError) Error() string {
	return fmt.Sprintf("spawnllm core: %s: %s", e.Kind, e.Msg)
}

var load = sync.OnceValues(loadEngine)

func loadEngine() (*engine, error) {
	ctx := context.Background()
	runtime := wazero.NewRuntime(ctx)
	if _, err := wasi_snapshot_preview1.Instantiate(ctx, runtime); err != nil {
		return nil, fmt.Errorf("spawnllm core: instantiate wasi: %w", err)
	}
	stderr := &bytes.Buffer{}
	config := wazero.NewModuleConfig().WithStderr(stderr).WithStartFunctions()
	module, err := runtime.InstantiateWithConfig(ctx, wasmBlob, config)
	if err != nil {
		return nil, fmt.Errorf("spawnllm core: instantiate module: %w", err)
	}
	if initialize := module.ExportedFunction("_initialize"); initialize != nil {
		if _, err := initialize.Call(ctx); err != nil {
			return nil, fmt.Errorf("spawnllm core: _initialize: %w", err)
		}
	}
	return &engine{
		module: module,
		stderr: stderr,
		alloc:  module.ExportedFunction("sl_alloc"),
		free:   module.ExportedFunction("sl_free"),
		call:   module.ExportedFunction("sl_call"),
	}, nil
}

// Call marshals request to JSON, dispatches it through the wasm core, and returns
// the raw `ok` value. A failed op returns a *DispatchError; a wasm trap wraps the
// captured stderr. The returned bytes are unparsed so callers keep number fidelity.
func Call(request any) (json.RawMessage, error) {
	engine, err := load()
	if err != nil {
		return nil, err
	}
	return engine.dispatch(request)
}

// CoreVersion reports the embedded blob's crate version and stamped source hash.
func CoreVersion() (Version, error) {
	raw, err := Call(struct {
		Op string `json:"op"`
	}{Op: "version"})
	if err != nil {
		return Version{}, err
	}
	var version Version
	if err := json.Unmarshal(raw, &version); err != nil {
		return Version{}, fmt.Errorf("spawnllm core: decode version: %w", err)
	}
	return version, nil
}

func (e *engine) dispatch(request any) (json.RawMessage, error) {
	requestBytes, err := json.Marshal(request)
	if err != nil {
		return nil, err
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	ctx := context.Background()
	inLen := uint64(len(requestBytes))
	in, err := e.alloc.Call(ctx, inLen)
	if err != nil {
		return nil, e.trap("sl_alloc", err)
	}
	inPtr := uint32(in[0])
	if !e.module.Memory().Write(inPtr, requestBytes) {
		return nil, fmt.Errorf("spawnllm core: request write out of bounds")
	}

	packed, err := e.call.Call(ctx, uint64(inPtr), inLen)
	if err != nil {
		return nil, e.trap("sl_call", err)
	}
	if _, err := e.free.Call(ctx, uint64(inPtr), inLen); err != nil {
		return nil, e.trap("sl_free", err)
	}

	outPtr := uint32(packed[0] >> 32)
	outLen := uint32(packed[0])
	view, ok := e.module.Memory().Read(outPtr, outLen)
	if !ok {
		return nil, fmt.Errorf("spawnllm core: response read out of bounds")
	}
	response := make([]byte, len(view))
	copy(response, view)
	if _, err := e.free.Call(ctx, uint64(outPtr), uint64(outLen)); err != nil {
		return nil, e.trap("sl_free", err)
	}

	return parseEnvelope(response)
}

func (e *engine) trap(op string, err error) error {
	if tail := strings.TrimSpace(e.stderr.String()); tail != "" {
		return fmt.Errorf("spawnllm core %s: %w\nwasm stderr: %s", op, err, tail)
	}
	return fmt.Errorf("spawnllm core %s: %w", op, err)
}

func parseEnvelope(response []byte) (json.RawMessage, error) {
	var envelope struct {
		Ok  json.RawMessage `json:"ok"`
		Err *struct {
			Kind string `json:"kind"`
			Msg  string `json:"msg"`
		} `json:"err"`
	}
	if err := json.Unmarshal(response, &envelope); err != nil {
		return nil, fmt.Errorf("spawnllm core: malformed envelope: %w", err)
	}
	if envelope.Err != nil {
		return nil, &DispatchError{Kind: envelope.Err.Kind, Msg: envelope.Err.Msg}
	}
	return envelope.Ok, nil
}

package core

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type wasmFunctionFunc func(context.Context, ...uint64) ([]uint64, error)

func (f wasmFunctionFunc) Call(ctx context.Context, params ...uint64) ([]uint64, error) {
	return f(ctx, params...)
}

func TestDispatchFreesRequestWhenCallTraps(t *testing.T) {
	base, err := loadEngine()
	if err != nil {
		t.Fatal(err)
	}

	freeCalls := 0
	engine := &engine{
		module: base.module,
		stderr: base.stderr,
		alloc:  base.alloc,
		free: wasmFunctionFunc(func(ctx context.Context, params ...uint64) ([]uint64, error) {
			freeCalls++
			return base.free.Call(ctx, params...)
		}),
		call: wasmFunctionFunc(func(context.Context, ...uint64) ([]uint64, error) {
			return nil, errors.New("call trap")
		}),
	}

	if _, err := engine.dispatch(map[string]string{"op": "version"}); err == nil || !strings.Contains(err.Error(), "sl_call") {
		t.Fatalf("dispatch error = %v, want sl_call trap", err)
	}
	if freeCalls != 1 {
		t.Fatalf("free calls = %d, want 1", freeCalls)
	}
}

func TestDispatchClearsResponseWhenRequestFreeTraps(t *testing.T) {
	base, err := loadEngine()
	if err != nil {
		t.Fatal(err)
	}

	freeCalls := 0
	engine := &engine{
		module: base.module,
		stderr: base.stderr,
		alloc:  base.alloc,
		call:   base.call,
		free: wasmFunctionFunc(func(ctx context.Context, params ...uint64) ([]uint64, error) {
			freeCalls++
			if freeCalls == 2 {
				return nil, errors.New("free trap")
			}
			return base.free.Call(ctx, params...)
		}),
	}

	response, err := engine.dispatch(map[string]string{"op": "version"})
	if err == nil || !strings.Contains(err.Error(), "sl_free") {
		t.Fatalf("dispatch error = %v, want sl_free trap", err)
	}
	if response != nil {
		t.Fatalf("dispatch response = %s, want nil", response)
	}
}

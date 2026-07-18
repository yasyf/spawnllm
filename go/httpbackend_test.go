package spawnllm

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestOpenAIEndpointSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer sk-test" {
			t.Errorf("Authorization = %q", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q", got)
		}
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"pong"}}]}`)
	}))
	defer srv.Close()

	b := OpenAIEndpoint(srv.URL+"/v1", "qwen3", OpenAIOpts{APIKey: "sk-test"})
	resp, err := RunOn(context.Background(), b, RunSpec{Prompt: "ping", Model: "qwen3"})
	if err != nil {
		t.Fatalf("RunOn: %v", err)
	}
	if resp.Err != nil {
		t.Fatalf("unexpected provider error: %v", resp.Err)
	}
	if resp.Result.Raw != "pong" {
		t.Fatalf("result = %q, want pong", resp.Result.Raw)
	}
}

func TestOpenAIEndpointErrorBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"error":{"message":"model is on fire"}}`)
	}))
	defer srv.Close()

	b := OpenAIEndpoint(srv.URL, "qwen3", OpenAIOpts{})
	resp, err := RunOn(context.Background(), b, RunSpec{Prompt: "ping"})
	if err != nil {
		t.Fatalf("RunOn: %v", err)
	}
	if resp.Err == nil {
		t.Fatal("expected a provider error from a 2xx error body")
	}
	if resp.Err.Msg != "model is on fire" {
		t.Fatalf("error msg = %q", resp.Err.Msg)
	}
	var callErr *BackendCallError
	if !errors.As(resp.Err, &callErr) {
		t.Fatalf("cause = %T, want *BackendCallError", resp.Err.Cause)
	}
}

func TestOpenAIEndpointTransportFailureLandsInResponse(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	_ = ln.Close() // guarantee a refused connection

	b := OpenAIEndpoint("http://"+addr+"/v1", "qwen3", OpenAIOpts{})
	resp, err := RunOn(context.Background(), b, RunSpec{Prompt: "ping", MaxAttempts: 1})
	if err != nil {
		t.Fatalf("transport failure must not raise a Go error: %v", err)
	}
	if resp.Err == nil {
		t.Fatal("transport failure must land in Response.Err")
	}
	var callErr *BackendCallError
	if !errors.As(resp.Err, &callErr) {
		t.Fatalf("cause = %T, want *BackendCallError", resp.Err.Cause)
	}
	if errors.Is(resp.Err, ErrTimeout) {
		t.Fatal("a refused connection is not a timeout")
	}
}

func TestOpenAIEndpointTransientRetry(t *testing.T) {
	restore := retrySleep
	retrySleep = func(context.Context, float64) error { return nil }
	t.Cleanup(func() { retrySleep = restore })

	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(w, "503 upstream overloaded")
			return
		}
		_, _ = io.WriteString(w, `{"choices":[{"message":{"content":"recovered"}}]}`)
	}))
	defer srv.Close()

	b := OpenAIEndpoint(srv.URL, "qwen3", OpenAIOpts{})
	resp, err := RunOn(context.Background(), b, RunSpec{Prompt: "ping"})
	if err != nil {
		t.Fatalf("RunOn: %v", err)
	}
	if resp.Err != nil {
		t.Fatalf("expected recovery after a 503, got %v", resp.Err)
	}
	if resp.Result.Raw != "recovered" {
		t.Fatalf("result = %q", resp.Result.Raw)
	}
	if len(resp.DiscardedAttempts) != 1 {
		t.Fatalf("expected 1 discarded 503 attempt, got %d", len(resp.DiscardedAttempts))
	}
}

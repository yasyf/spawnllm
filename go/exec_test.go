package spawnllm

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func withFakeBin(t *testing.T) {
	t.Helper()
	bin, err := filepath.Abs(filepath.Join("testdata", "bin"))
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
}

type claudeOutput struct {
	StdoutRegular bool   `json:"stdout_regular"`
	ConfigDir     string `json:"config_dir"`
	Seeded        bool   `json:"seeded"`
	AccountHasMCP bool   `json:"account_has_mcp"`
	CredsPresent  bool   `json:"creds_present"`
}

func TestClaudeStdinAndStdoutFile(t *testing.T) {
	withFakeBin(t)
	resp, err := RunOn(context.Background(), ClaudeBackend(), RunSpec{
		Prompt:        "ping",
		Model:         "haiku",
		UseHostConfig: true,
	})
	if err != nil {
		t.Fatalf("RunOn: %v", err)
	}
	if resp.Err != nil {
		t.Fatalf("unexpected provider error: %v", resp.Err)
	}
	if resp.Result.Raw != "ping" {
		t.Fatalf("stdin not delivered: result = %q", resp.Result.Raw)
	}
	var out claudeOutput
	if err := json.Unmarshal([]byte(resp.Output), &out); err != nil {
		t.Fatalf("decode output %q: %v", resp.Output, err)
	}
	if !out.StdoutRegular {
		t.Fatal("child stdout fd was not a regular file")
	}
}

func TestCodexResultFromFile(t *testing.T) {
	withFakeBin(t)
	resp, err := RunOn(context.Background(), CodexBackend(), RunSpec{
		Prompt:        "read me from the -o file",
		Model:         "gpt-5.5:medium",
		UseHostConfig: true,
	})
	if err != nil {
		t.Fatalf("RunOn: %v", err)
	}
	if resp.Err != nil {
		t.Fatalf("unexpected provider error: %v", resp.Err)
	}
	if resp.Result.Raw != "read me from the -o file" {
		t.Fatalf("result not read from -o file: %q", resp.Result.Raw)
	}
	if resp.Output == "streaming interactive log line to stdout (must be ignored by the host)\n" {
		t.Fatal("host read the stdout log instead of the -o file")
	}
}

func TestTimeoutKillsSleepingFake(t *testing.T) {
	withFakeBin(t)
	start := time.Now()
	resp, err := RunOn(context.Background(), ClaudeBackend(), RunSpec{
		Prompt:        "slow",
		Model:         "haiku",
		UseHostConfig: true,
		Timeout:       300 * time.Millisecond,
		MaxAttempts:   1,
		Env:           map[string]string{"FAKE_SLEEP": "5"},
	})
	if err != nil {
		t.Fatalf("RunOn: %v", err)
	}
	if resp.Err == nil {
		t.Fatalf("expected a timeout error, got result %q", resp.Result.Raw)
	}
	if !errors.Is(resp.Err, ErrTimeout) {
		t.Fatalf("expected ErrTimeout, got %v", resp.Err)
	}
	if elapsed := time.Since(start); elapsed > 4*time.Second {
		t.Fatalf("timeout did not kill the sleeping fake promptly: %s", elapsed)
	}
}

func TestTransientThenSuccessRetry(t *testing.T) {
	withFakeBin(t)
	restore := retrySleep
	retrySleep = func(context.Context, float64) error { return nil }
	t.Cleanup(func() { retrySleep = restore })

	counter := filepath.Join(t.TempDir(), "counter")
	resp, err := RunOn(context.Background(), ClaudeBackend(), RunSpec{
		Prompt:        "retry me",
		Model:         "haiku",
		UseHostConfig: true,
		Env:           map[string]string{"FAKE_TRANSIENT_COUNTER": counter},
	})
	if err != nil {
		t.Fatalf("RunOn: %v", err)
	}
	if resp.Err != nil {
		t.Fatalf("expected success after retry, got %v", resp.Err)
	}
	if resp.Result.Raw != "retry me" {
		t.Fatalf("result = %q", resp.Result.Raw)
	}
	if len(resp.DiscardedAttempts) != 1 {
		t.Fatalf("expected 1 discarded attempt, got %d", len(resp.DiscardedAttempts))
	}
	if got := resp.DiscardedAttempts[0]; got.Attempt != 0 || got.Error != "BackendCallError" {
		t.Fatalf("discarded attempt = %+v", got)
	}
}

func TestClaudeIsolationSeeding(t *testing.T) {
	withFakeBin(t)
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("CLAUDE_CONFIG_DIR", "")
	if err := os.WriteFile(filepath.Join(home, ".claude.json"),
		[]byte(`{"oauthAccount":{"accountUuid":"a"},"mcpServers":{"s":{"command":"x"}}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(home, ".claude"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(home, ".claude", ".credentials.json"),
		[]byte(`{"claudeAiOauth":{"accessToken":"tok"}}`), 0o600); err != nil {
		t.Fatal(err)
	}

	resp, err := RunOn(context.Background(), ClaudeBackend(), RunSpec{Prompt: "iso", Model: "haiku"})
	if err != nil {
		t.Fatalf("RunOn: %v", err)
	}
	if resp.Err != nil {
		t.Fatalf("unexpected provider error: %v", resp.Err)
	}
	var out claudeOutput
	if err := json.Unmarshal([]byte(resp.Output), &out); err != nil {
		t.Fatalf("decode output %q: %v", resp.Output, err)
	}
	if out.ConfigDir == "" {
		t.Fatal("isolated run did not set CLAUDE_CONFIG_DIR")
	}
	if out.ConfigDir == home || out.ConfigDir == filepath.Join(home, ".claude") {
		t.Fatalf("isolated config dir pointed at the host home: %s", out.ConfigDir)
	}
	if !out.Seeded {
		t.Fatal("isolated config dir was not seeded with .claude.json")
	}
	if out.AccountHasMCP {
		t.Fatal("seeded .claude.json still carried mcpServers")
	}
	if !out.CredsPresent {
		t.Fatal("isolated config dir was not seeded with .credentials.json")
	}
	if _, err := os.Stat(out.ConfigDir); !os.IsNotExist(err) {
		t.Fatalf("isolated config dir was not cleaned up: stat err = %v", err)
	}
}

type echoResult struct {
	Echo string `json:"echo"`
}

func TestExtractStructured(t *testing.T) {
	withFakeBin(t)
	got, err := Extract[echoResult](context.Background(), "hello", CallOpts{Backend: CodexBackend()})
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if got.Echo != "hello" {
		t.Fatalf("Extract = %+v, want Echo=hello", got)
	}
}

func TestCallResolvesModelTier(t *testing.T) {
	withFakeBin(t)
	got, err := Call(context.Background(), "ping", CallOpts{Backend: ClaudeBackend(), Model: Medium})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got != "ping" {
		t.Fatalf("Call = %q, want ping", got)
	}
}

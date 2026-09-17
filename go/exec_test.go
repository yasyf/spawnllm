package spawnllm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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
	ConfigDirMode string `json:"config_dir_mode"`
	CredsMode     string `json:"creds_mode"`
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

func TestMissingBinaryLandsInResponse(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	resp, err := RunOn(context.Background(), ClaudeBackend(), RunSpec{
		Prompt:        "ping",
		Model:         "haiku",
		UseHostConfig: true,
		MaxAttempts:   1,
	})
	if err != nil {
		t.Fatalf("setup failure must not raise a Go error: %v", err)
	}
	if resp == nil || resp.Err == nil {
		t.Fatalf("setup failure must land in Response.Err, got %+v", resp)
	}
	var callErr *BackendCallError
	if !errors.As(resp.Err, &callErr) {
		t.Fatalf("cause = %T, want *BackendCallError", resp.Err.Cause)
	}
	if callErr.Provider != ProviderClaude {
		t.Fatalf("provider = %q, want %q", callErr.Provider, ProviderClaude)
	}
}

func TestSubstituteFilesOnlyReplacesWholeArgument(t *testing.T) {
	got := substituteFiles(
		[]string{"${file:schema}", "literal ${file:schema} token", "${file:other}"},
		map[string]string{"schema": "/tmp/schema.json"},
	)
	want := []string{"/tmp/schema.json", "literal ${file:schema} token", "${file:other}"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("argument %d = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestMergeEnv(t *testing.T) {
	const ambientKey = "SPAWNLLM_TEST_ENV_UNSET"
	const foldedAmbientKey = "spawnllm_test_env_unset"
	t.Setenv(ambientKey, "ambient")
	tests := []struct {
		name            string
		goos            string
		planEnv         map[string]string
		specEnv         map[string]string
		envUnset        []string
		wantKey         string
		want            string
		wantPresent     bool
		wantFoldedCount int
	}{
		{name: "strips ambient unset", envUnset: []string{ambientKey}},
		{
			name:            "keeps ambient with nil unset",
			want:            "ambient",
			wantPresent:     true,
			wantFoldedCount: 1,
		},
		{
			name:            "spec override survives unset",
			specEnv:         map[string]string{ambientKey: "explicit"},
			envUnset:        []string{ambientKey},
			want:            "explicit",
			wantPresent:     true,
			wantFoldedCount: 1,
		},
		{
			name:     "windows strips ambient unset with different case",
			goos:     "windows",
			envUnset: []string{foldedAmbientKey},
		},
		{
			name:            "windows plan overlay evicts ambient with different case",
			goos:            "windows",
			planEnv:         map[string]string{foldedAmbientKey: "plan"},
			wantKey:         foldedAmbientKey,
			want:            "plan",
			wantPresent:     true,
			wantFoldedCount: 1,
		},
		{
			name:            "windows spec overlay evicts plan with different case",
			goos:            "windows",
			planEnv:         map[string]string{foldedAmbientKey: "plan"},
			specEnv:         map[string]string{ambientKey: "spec"},
			want:            "spec",
			wantPresent:     true,
			wantFoldedCount: 1,
		},
		{
			name:            "non-windows keeps differently cased keys distinct",
			goos:            "linux",
			planEnv:         map[string]string{foldedAmbientKey: "plan"},
			want:            "ambient",
			wantPresent:     true,
			wantFoldedCount: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got []string
			if tt.goos == "" {
				got = mergeEnv(tt.planEnv, tt.specEnv, tt.envUnset)
			} else {
				got = mergeEnvForOS(tt.planEnv, tt.specEnv, tt.envUnset, tt.goos)
			}
			wantKey := tt.wantKey
			if wantKey == "" {
				wantKey = ambientKey
			}
			var value string
			found := false
			foldedCount := 0
			for _, kv := range got {
				key, candidate, ok := strings.Cut(kv, "=")
				if !ok {
					continue
				}
				if strings.EqualFold(key, ambientKey) {
					foldedCount++
				}
				if key == wantKey {
					value = candidate
					found = true
				}
			}
			if found != tt.wantPresent {
				t.Fatalf("key %q presence = %v, want %v", wantKey, found, tt.wantPresent)
			}
			if found && value != tt.want {
				t.Fatalf("key %q value = %q, want %q", wantKey, value, tt.want)
			}
			if foldedCount != tt.wantFoldedCount {
				t.Fatalf("equal-folded key count = %d, want %d", foldedCount, tt.wantFoldedCount)
			}
		})
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
	if out.ConfigDirMode != "drwx------" || out.CredsMode != "-rw-------" {
		t.Fatalf("isolated config dir mode %q, credentials mode %q; want drwx------ and -rw-------", out.ConfigDirMode, out.CredsMode)
	}
	if _, err := os.Stat(out.ConfigDir); !os.IsNotExist(err) {
		t.Fatalf("isolated config dir was not cleaned up: stat err = %v", err)
	}
}

func suffixedKeychainService(configDirEnv string) string {
	digest := sha256.Sum256([]byte(configDirEnv))
	return "Claude Code-credentials-" + hex.EncodeToString(digest[:])[:8]
}

func keychainSeededRun(t *testing.T, service string) (claudeOutput, string) {
	t.Helper()
	if runtime.GOOS != "darwin" {
		t.Skip("the Keychain fallback runs only on darwin")
	}
	withFakeBin(t)
	argvOut := filepath.Join(t.TempDir(), "argv")
	t.Setenv("FAKE_SECURITY_ARGV_OUT", argvOut)
	t.Setenv("FAKE_KEYCHAIN_SERVICE", service)
	t.Setenv("FAKE_KEYCHAIN_CREDENTIAL", `{"claudeAiOauth":{"accessToken":"kc-tok"}}`)

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
	argv, err := os.ReadFile(argvOut)
	if err != nil {
		t.Fatalf("fake security recorded no argv: %v", err)
	}
	return out, string(argv)
}

func writeAccountPointer(t *testing.T, dir string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, ".claude.json"), []byte(`{"oauthAccount":{"accountUuid":"a"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestClaudeIsolationKeychain(t *testing.T) {
	cases := []struct {
		name    string
		env     func(home, acct string) map[string]string
		service func(home, acct string) string
	}{
		{
			name:    "unset env reads the bare item",
			env:     func(string, string) map[string]string { return nil },
			service: func(string, string) string { return "Claude Code-credentials" },
		},
		{
			name:    "set env reads the suffixed item",
			env:     func(_, acct string) map[string]string { return map[string]string{"CLAUDE_CONFIG_DIR": acct} },
			service: func(_, acct string) string { return suffixedKeychainService(acct) },
		},
		{
			name:    "trailing slash is hashed as set",
			env:     func(_, acct string) map[string]string { return map[string]string{"CLAUDE_CONFIG_DIR": acct + "/"} },
			service: func(_, acct string) string { return suffixedKeychainService(acct + "/") },
		},
		{
			name: "env naming the default path is still suffixed",
			env: func(home, _ string) map[string]string {
				return map[string]string{"CLAUDE_CONFIG_DIR": filepath.Join(home, ".claude")}
			},
			service: func(home, _ string) string { return suffixedKeychainService(filepath.Join(home, ".claude")) },
		},
		{
			name: "empty securestorage env reads the bare item over config dir env",
			env: func(_, acct string) map[string]string {
				return map[string]string{"CLAUDE_CONFIG_DIR": acct, "CLAUDE_SECURESTORAGE_CONFIG_DIR": ""}
			},
			service: func(string, string) string { return "Claude Code-credentials" },
		},
		{
			name: "securestorage env is hashed over config dir env",
			env: func(home, acct string) map[string]string {
				return map[string]string{"CLAUDE_CONFIG_DIR": acct, "CLAUDE_SECURESTORAGE_CONFIG_DIR": home + "/secure"}
			},
			service: func(home, _ string) string { return suffixedKeychainService(home + "/secure") },
		},
		{
			name: "custom oauth url names the custom-oauth item",
			env: func(_, acct string) map[string]string {
				return map[string]string{"CLAUDE_CONFIG_DIR": acct, "CLAUDE_CODE_CUSTOM_OAUTH_URL": "https://oauth.example.test"}
			},
			service: func(_, acct string) string {
				return strings.Replace(suffixedKeychainService(acct), "Claude Code-credentials", "Claude Code-custom-oauth-credentials", 1)
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			home := t.TempDir()
			acct := t.TempDir()
			t.Setenv("HOME", home)
			if err := os.MkdirAll(filepath.Join(home, ".claude"), 0o755); err != nil {
				t.Fatal(err)
			}
			writeAccountPointer(t, home)
			writeAccountPointer(t, acct)
			for _, name := range []string{"CLAUDE_CONFIG_DIR", "CLAUDE_SECURESTORAGE_CONFIG_DIR", "CLAUDE_CODE_CUSTOM_OAUTH_URL"} {
				t.Setenv(name, "")
				if err := os.Unsetenv(name); err != nil {
					t.Fatal(err)
				}
			}
			for name, value := range tc.env(home, acct) {
				t.Setenv(name, value)
			}
			service := tc.service(home, acct)

			out, argv := keychainSeededRun(t, service)

			if want := "find-generic-password\n-s\n" + service + "\n-w\n"; argv != want {
				t.Fatalf("security argv = %q, want %q", argv, want)
			}
			if !out.CredsPresent || out.CredsMode != "-rw-------" {
				t.Fatalf("keychain credentials seeded = %v with mode %q; want seeded at -rw-------", out.CredsPresent, out.CredsMode)
			}
		})
	}
}

func TestClaudeIsolationKeychainMissSeedsNoCredentials(t *testing.T) {
	acct := t.TempDir()
	writeAccountPointer(t, acct)
	t.Setenv("CLAUDE_CONFIG_DIR", acct)

	out, argv := keychainSeededRun(t, "Claude Code-credentials-someone-else")

	if want := "find-generic-password\n-s\n" + suffixedKeychainService(acct) + "\n-w\n"; argv != want {
		t.Fatalf("security argv = %q, want %q", argv, want)
	}
	if out.CredsPresent {
		t.Fatal("a Keychain miss must seed no credentials file")
	}
}

type echoResult struct {
	Echo string `json:"echo"`
}

type preciseNumberResult struct {
	Value any `json:"value"`
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

func TestExtractPreservesLargeIntegerInAny(t *testing.T) {
	withFakeBin(t)
	t.Setenv("FAKE_STRUCTURED_RESULT", `{"value":9007199254740993}`)
	got, err := Extract[preciseNumberResult](context.Background(), "number", CallOpts{Backend: CodexBackend()})
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	number, ok := got.Value.(json.Number)
	if !ok {
		t.Fatalf("Value = %T(%v), want json.Number", got.Value, got.Value)
	}
	if number.String() != "9007199254740993" {
		t.Fatalf("Value = %s, want 9007199254740993", number)
	}
}

func TestRunRejectsNonObjectSchema(t *testing.T) {
	runners := map[string]func(context.Context, RunSpec) (*Response, error){
		"Run":   Run,
		"RunOn": func(ctx context.Context, spec RunSpec) (*Response, error) { return RunOn(ctx, CodexBackend(), spec) },
	}
	for name, run := range runners {
		t.Run(name, func(t *testing.T) {
			resp, err := run(context.Background(), RunSpec{Schema: json.RawMessage(`"string schema"`)})
			if err == nil || !strings.Contains(err.Error(), "schema must be a JSON object") {
				t.Fatalf("error = %v, want JSON object caller fault", err)
			}
			if resp != nil {
				t.Fatalf("response = %+v, want nil on caller fault", resp)
			}
		})
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

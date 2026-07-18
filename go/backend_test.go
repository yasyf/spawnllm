package spawnllm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"testing"
)

func TestSelectBackendPicksFirstReady(t *testing.T) {
	withFakeBin(t)
	b, err := SelectBackend(context.Background(), SelectOpts{})
	if err != nil {
		t.Fatalf("SelectBackend: %v", err)
	}
	if b.Provider() != ProviderClaude {
		t.Fatalf("selected %s, want claude (first in priority)", b.Provider())
	}
}

func TestSelectBackendSpecialtyPromotes(t *testing.T) {
	withFakeBin(t)
	b, err := SelectBackend(context.Background(), SelectOpts{Specialty: SpecialtyDebugging})
	if err != nil {
		t.Fatalf("SelectBackend: %v", err)
	}
	if b.Provider() != ProviderCodex {
		t.Fatalf("debugging selected %s, want codex", b.Provider())
	}
}

func TestSelectBackendUnavailable(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	_, err := SelectBackend(context.Background(), SelectOpts{})
	var unavailable *BackendUnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("expected *BackendUnavailableError, got %v", err)
	}
	if _, ok := unavailable.Statuses[ProviderGemini]; ok {
		t.Fatal("gemini should be excluded from auto-selection")
	}
	if st := unavailable.Statuses[ProviderClaude]; st.State != BackendNotInstalled {
		t.Fatalf("claude status = %v, want NotInstalled", st.State)
	}
}

func TestCoreSpecDefaults(t *testing.T) {
	cs := RunSpec{Prompt: "p", Model: "m"}.core()
	if !cs.Isolated {
		t.Error("zero UseHostConfig should isolate")
	}
	if cs.Timeout != 180 {
		t.Errorf("timeout = %d, want 180", cs.Timeout)
	}
	if cs.MaxAttempts != 5 {
		t.Errorf("max_attempts = %d, want 5", cs.MaxAttempts)
	}
	if cs.Claude != nil || cs.Codex != nil || cs.Gemini != nil {
		t.Error("absent provider configs should map to nil")
	}
	if cs.Schema != nil {
		t.Error("absent schema should map to nil")
	}
}

func TestCoreSpecUseHostConfig(t *testing.T) {
	if (RunSpec{UseHostConfig: true}).core().Isolated {
		t.Fatal("UseHostConfig true should not isolate")
	}
}

func TestCodexServiceTierDefault(t *testing.T) {
	cs := RunSpec{Providers: ProviderConfigs{Codex: &CodexConfig{}}}.core()
	if cs.Codex.ServiceTier == nil || *cs.Codex.ServiceTier != "fast" {
		t.Fatalf("default service tier = %v, want fast", cs.Codex.ServiceTier)
	}
	explicit := "flex"
	cs = RunSpec{Providers: ProviderConfigs{Codex: &CodexConfig{ServiceTier: &explicit}}}.core()
	if *cs.Codex.ServiceTier != "flex" {
		t.Fatalf("explicit service tier = %v, want flex", *cs.Codex.ServiceTier)
	}
}

func TestClaudeToolsNilVsEmpty(t *testing.T) {
	nilTools := RunSpec{Providers: ProviderConfigs{Claude: &ClaudeConfig{}}}.core()
	raw, err := json.Marshal(nilTools.Claude)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`"tools":null`)) {
		t.Fatalf("nil Tools should serialize to null: %s", raw)
	}
	emptyTools := RunSpec{Providers: ProviderConfigs{Claude: &ClaudeConfig{Tools: []string{}}}}.core()
	raw, err = json.Marshal(emptyTools.Claude)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`"tools":[]`)) {
		t.Fatalf("empty Tools should serialize to []: %s", raw)
	}
}

func TestClaudeConfigOptionalsSerializeNull(t *testing.T) {
	raw, err := json.Marshal(RunSpec{Providers: ProviderConfigs{Claude: &ClaudeConfig{}}}.core().Claude)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{
		`"permission_mode":null`, `"mcp_config":null`, `"append_system_prompt":null`,
		`"system_prompt":null`, `"settings":null`, `"output_format":null`,
		`"max_turns":null`, `"max_budget_usd":null`,
	} {
		if !bytes.Contains(raw, []byte(field)) {
			t.Errorf("missing %s in %s", field, raw)
		}
	}
}

func schemaMap(t *testing.T, provider Provider) map[string]any {
	t.Helper()
	raw, err := extractSchema[echoResult](provider)
	if err != nil {
		t.Fatalf("extractSchema: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("decode schema %s: %v", raw, err)
	}
	if _, ok := m["$schema"]; ok {
		t.Errorf("schema still carries $schema draft key: %s", raw)
	}
	return m
}

func TestExtractSchemaAnthropicStrict(t *testing.T) {
	m := schemaMap(t, ProviderClaude)
	if m["additionalProperties"] != false {
		t.Fatalf("anthropic schema missing additionalProperties:false: %v", m)
	}
}

func TestExtractSchemaOpenAIStrict(t *testing.T) {
	m := schemaMap(t, ProviderCodex)
	if m["additionalProperties"] != false {
		t.Fatalf("openai schema missing additionalProperties:false: %v", m)
	}
	required, ok := m["required"].([]any)
	if !ok || len(required) != 1 || required[0] != "echo" {
		t.Fatalf("openai strict schema required = %v, want [echo]", m["required"])
	}
}

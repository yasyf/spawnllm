package spawnllm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type probeAddress struct {
	City string `json:"city"`
	Zip  string `json:"zip"`
}

type probePerson struct {
	Name      string         `json:"name"`
	Address   probeAddress   `json:"address"`
	Neighbors []probeAddress `json:"neighbors"`
}

func assertObjectsStrict(t *testing.T, node any, path string) {
	t.Helper()
	switch n := node.(type) {
	case map[string]any:
		if n["type"] == "object" {
			if strict, ok := n["additionalProperties"].(bool); !ok || strict {
				t.Errorf("object at %s: additionalProperties = %v, want false", path, n["additionalProperties"])
			}
		}
		for k, v := range n {
			assertObjectsStrict(t, v, path+"/"+k)
		}
	case []any:
		for i, v := range n {
			assertObjectsStrict(t, v, fmt.Sprintf("%s[%d]", path, i))
		}
	}
}

func TestExtractSchemaStrictifiesNested(t *testing.T) {
	for _, provider := range []Provider{ProviderClaude, ProviderCodex} {
		raw, err := extractSchema[probePerson](provider)
		if err != nil {
			t.Fatalf("%s: %v", provider, err)
		}
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatalf("%s: decode %s: %v", provider, raw, err)
		}
		assertObjectsStrict(t, m, string(provider))
		if _, ok := m["$id"]; ok {
			t.Errorf("%s: schema carries a leftover $id: %s", provider, raw)
		}
		if _, ok := m["$schema"]; ok {
			t.Errorf("%s: schema carries a leftover $schema: %s", provider, raw)
		}
		if d, ok := m["description"].(string); ok && strings.Contains(d, "$id") {
			t.Errorf("%s: $id folded into a junk description: %q", provider, d)
		}
	}
}

func TestExtractNestedReceivesStrictSchema(t *testing.T) {
	withFakeBin(t)
	sink := filepath.Join(t.TempDir(), "received-schema.json")
	t.Setenv("FAKE_SCHEMA_SINK", sink)

	if _, err := Extract[probePerson](context.Background(), "hello", CallOpts{Backend: CodexBackend()}); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	data, err := os.ReadFile(sink)
	if err != nil {
		t.Fatalf("fake codex did not record the received schema: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("received schema is not JSON: %v", err)
	}
	assertObjectsStrict(t, m, "received")

	props, ok := m["properties"].(map[string]any)
	if !ok {
		t.Fatalf("received schema has no properties: %s", data)
	}
	addr, ok := props["address"].(map[string]any)
	if !ok {
		t.Fatalf("nested address object was inlined away: %s", data)
	}
	if addr["additionalProperties"] != false {
		t.Fatalf("nested address object not strictified: %v", addr)
	}
}

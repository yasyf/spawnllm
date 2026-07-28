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

type probeTreeNode struct {
	Label    string          `json:"label"`
	Children []probeTreeNode `json:"children"`
}

type probeTree struct {
	Root probeTreeNode `json:"root"`
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

func assertDefsStrict(t *testing.T, schema map[string]any, label string) {
	t.Helper()
	defs, ok := schema["$defs"].(map[string]any)
	if !ok {
		t.Fatalf("%s: schema has no $defs (nested types were not referenced): %v", label, schema)
	}
	if _, isDraft07 := schema["definitions"]; isDraft07 {
		t.Errorf("%s: schema uses draft-07 \"definitions\" the anthropic transform does not recurse", label)
	}
	for name, def := range defs {
		obj, ok := def.(map[string]any)
		if !ok {
			continue
		}
		if obj["type"] == "object" && obj["additionalProperties"] != false {
			t.Errorf("%s: $defs/%s missing additionalProperties:false: %v", label, name, obj)
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
		assertDefsStrict(t, m, string(provider))
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

func TestExtractSchemaAppliesTheAppleDialect(t *testing.T) {
	raw, err := extractSchema[probePerson](ProviderApple)
	if err != nil {
		t.Fatalf("apple: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("apple: decode %s: %v", raw, err)
	}
	if _, ok := m["x-order"]; !ok {
		t.Errorf("apple schema carries no x-order — the dialect never ran: %s", raw)
	}
	assertObjectsStrict(t, m, "apple")
	assertDefsStrict(t, m, "apple")
}

func TestExtractSchemaHandlesRecursiveType(t *testing.T) {
	// A recursive type nested in the struct resolves through $defs (the previous
	// inline config stack-overflowed here); its self-$ref stays valid and strict.
	raw, err := extractSchema[probeTree](ProviderCodex)
	if err != nil {
		t.Fatalf("recursive type: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("decode %s: %v", raw, err)
	}
	assertDefsStrict(t, m, "recursive")
	defs := m["$defs"].(map[string]any)
	if _, ok := defs["probeTreeNode"]; !ok {
		t.Fatalf("recursive type not captured in $defs: %v", defs)
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
	assertDefsStrict(t, m, "received")
}

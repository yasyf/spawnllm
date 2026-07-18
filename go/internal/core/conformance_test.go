package core

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func repoRoot(t *testing.T) string {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root, err := filepath.Abs(filepath.Join(wd, "..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func decodeValue(t *testing.T, raw []byte) any {
	t.Helper()
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		t.Fatalf("decode %s: %v", raw, err)
	}
	return value
}

func numberEqual(a, b json.Number) bool {
	if a.String() == b.String() {
		return true
	}
	af, aerr := a.Float64()
	bf, berr := b.Float64()
	return aerr == nil && berr == nil && af == bf && !math.IsInf(af, 0) && !math.IsNaN(af)
}

func valueEqual(a, b any) bool {
	switch av := a.(type) {
	case json.Number:
		bv, ok := b.(json.Number)
		return ok && numberEqual(av, bv)
	case []any:
		bv, ok := b.([]any)
		if !ok || len(av) != len(bv) {
			return false
		}
		for i := range av {
			if !valueEqual(av[i], bv[i]) {
				return false
			}
		}
		return true
	case map[string]any:
		bv, ok := b.(map[string]any)
		if !ok || len(av) != len(bv) {
			return false
		}
		for key, value := range av {
			other, present := bv[key]
			if !present || !valueEqual(value, other) {
				return false
			}
		}
		return true
	default:
		return a == b
	}
}

func TestConformanceVectors(t *testing.T) {
	vectorsDir := filepath.Join(repoRoot(t), "conformance", "vectors")
	if _, err := os.Stat(vectorsDir); err != nil {
		t.Skipf("conformance vectors absent: %v", err)
	}

	opDirs, err := os.ReadDir(vectorsDir)
	if err != nil {
		t.Fatal(err)
	}

	replayed := 0
	for _, opDir := range opDirs {
		if !opDir.IsDir() {
			continue
		}
		opPath := filepath.Join(vectorsDir, opDir.Name())
		files, err := os.ReadDir(opPath)
		if err != nil {
			t.Fatal(err)
		}
		for _, file := range files {
			if filepath.Ext(file.Name()) != ".json" {
				continue
			}
			data, err := os.ReadFile(filepath.Join(opPath, file.Name()))
			if err != nil {
				t.Fatal(err)
			}
			var vector struct {
				Name     string          `json:"name"`
				Op       string          `json:"op"`
				Input    json.RawMessage `json:"input"`
				Expected json.RawMessage `json:"expected"`
			}
			if err := json.Unmarshal(data, &vector); err != nil {
				t.Fatalf("%s: %v", file.Name(), err)
			}
			replayed++
			t.Run(opDir.Name()+"/"+vector.Name, func(t *testing.T) {
				raw, err := Call(struct {
					Op    string          `json:"op"`
					Input json.RawMessage `json:"input"`
				}{Op: vector.Op, Input: vector.Input})
				if err != nil {
					t.Fatalf("Call: %v", err)
				}
				if !valueEqual(decodeValue(t, raw), decodeValue(t, vector.Expected)) {
					t.Fatalf("mismatch\n got: %s\nwant: %s", raw, vector.Expected)
				}
			})
		}
	}

	if replayed == 0 {
		t.Fatal("no conformance vectors replayed")
	}
	t.Logf("replayed %d vectors through the wasm core", replayed)
}

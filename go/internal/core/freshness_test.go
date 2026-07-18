package core

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestBlobFreshness(t *testing.T) {
	root := repoRoot(t)
	script := filepath.Join(root, "scripts", "build_wasm.sh")
	if _, err := os.Stat(filepath.Join(root, "rust")); err != nil {
		t.Skipf("rust/ absent (module-cache build): %v", err)
	}
	if _, err := os.Stat(script); err != nil {
		t.Skipf("build_wasm.sh absent: %v", err)
	}

	out, err := exec.Command("bash", script, "--hash-only").Output()
	if err != nil {
		t.Fatalf("build_wasm.sh --hash-only: %v", err)
	}
	want := strings.TrimSpace(string(out))

	version, err := BlobVersion()
	if err != nil {
		t.Fatalf("BlobVersion: %v", err)
	}
	if version.SourceHash != want {
		t.Fatalf("stale blob: embedded source_hash=%s, current sources=%s (run scripts/build_wasm.sh)",
			version.SourceHash, want)
	}
}

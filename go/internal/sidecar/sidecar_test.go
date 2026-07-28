package sidecar

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestLaunchResolvesOnlyOnMacOS(t *testing.T) {
	argv, err := Launch()

	switch {
	case runtime.GOOS == "darwin":
		if errors.Is(err, ErrUnsupportedPlatform) {
			t.Fatalf("the platform gate rejected macOS: %v", err)
		}
	case err == nil:
		t.Fatalf("the sidecar resolved on %s: %v", runtime.GOOS, argv)
	case !errors.Is(err, ErrUnsupportedPlatform):
		t.Fatalf("want ErrUnsupportedPlatform on %s, got %v", runtime.GOOS, err)
	}
}

func TestDescriptorIsMaterializedByteForByte(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, Binary+".binrun")

	if err := materialize(dir, path); err != nil {
		t.Fatalf("materialize: %v", err)
	}

	written, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if string(written) != string(descriptor) {
		t.Fatalf("materialized descriptor differs from the embedded one")
	}
	if entries, err := os.ReadDir(dir); err != nil || len(entries) != 1 {
		t.Fatalf("materialize left temp files behind: %v, %v", entries, err)
	}
}

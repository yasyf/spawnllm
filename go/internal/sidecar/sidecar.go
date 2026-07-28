// Package sidecar resolves the spawnllm-apple Foundation Models sidecar: the
// executable on PATH when one is installed, otherwise binrun against the pinned
// descriptor embedded here, which fetches, verifies, and caches the release build.
package sidecar

import (
	"bytes"
	_ "embed"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

//go:embed spawnllm-apple.binrun
var descriptor []byte

const (
	// Binary is the sidecar executable's name.
	Binary = "spawnllm-apple"

	// InstallHint names binrun, the resolver that fetches the pinned sidecar build.
	InstallHint = "brew install yasyf/tap/binrun"
)

// ErrUnsupportedPlatform reports that the sidecar cannot run on this OS. Apple
// Foundation Models are macOS-only, so binrun would fetch an archive nothing
// here could execute.
var ErrUnsupportedPlatform = errors.New(Binary + " runs only on macOS")

// Launch returns the argv prefix that runs the sidecar. It reports
// [ErrUnsupportedPlatform] off macOS, and the [exec.LookPath] failure when
// neither the sidecar nor binrun is installed.
func Launch() ([]string, error) {
	if runtime.GOOS != "darwin" {
		return nil, ErrUnsupportedPlatform
	}
	if path, err := exec.LookPath(Binary); err == nil {
		return []string{path}, nil
	}
	binrun, err := exec.LookPath("binrun")
	if err != nil {
		return nil, err
	}
	path, err := descriptorPath()
	if err != nil {
		return nil, err
	}
	return []string{binrun, path}, nil
}

func descriptorPath() (string, error) {
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(cache, "spawnllm")
	path := filepath.Join(dir, Binary+".binrun")
	if current, err := os.ReadFile(path); err == nil && bytes.Equal(current, descriptor) {
		return path, nil
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	return path, materialize(dir, path)
}

// materialize renames the descriptor into place so a concurrent binrun never
// reads a half-written file.
func materialize(dir, path string) error {
	tmp, err := os.CreateTemp(dir, ".spawnllm-apple-*.binrun")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	if _, err := tmp.Write(descriptor); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

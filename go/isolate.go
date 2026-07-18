package spawnllm

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

func seedClaudeIsolation() (string, func(), error) {
	sources, err := coreIsolationSources()
	if err != nil {
		return "", nil, err
	}
	accountJSON := readFileOpt(sources.AccountPath)
	credentialsJSON := readFileOpt(sources.CredentialsPath)
	if credentialsJSON == nil && sources.KeychainService != nil {
		credentialsJSON = keychainCredentials(*sources.KeychainService)
	}

	seed, err := coreIsolationSeed(accountJSON, credentialsJSON)
	if err != nil {
		return "", nil, err
	}

	dir, err := os.MkdirTemp("", "spawnllm-claude-config-")
	if err != nil {
		return "", nil, err
	}
	cleanup := func() { _ = os.RemoveAll(dir) }
	for _, f := range seed.Files {
		mode, err := parseMode(f.Mode)
		if err != nil {
			cleanup()
			return "", nil, err
		}
		path := filepath.Join(dir, f.Name)
		if err := os.WriteFile(path, []byte(f.Content), mode); err != nil {
			cleanup()
			return "", nil, err
		}
		if err := os.Chmod(path, mode); err != nil {
			cleanup()
			return "", nil, err
		}
	}
	return dir, cleanup, nil
}

func substituteIsolationDir(env map[string]string, dir string) map[string]string {
	out := make(map[string]string, len(env))
	for k, v := range env {
		out[k] = strings.ReplaceAll(v, "${isolated_config_dir}", dir)
	}
	return out
}

func readFileOpt(path string) *string {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	s := string(data)
	return &s
}

func keychainCredentials(service string) *string {
	out, err := exec.Command("security", "find-generic-password", "-s", service, "-w").Output()
	if err != nil {
		return nil
	}
	s := string(out)
	return &s
}

func parseMode(s string) (os.FileMode, error) {
	n, err := strconv.ParseUint(s, 8, 32)
	if err != nil {
		return 0, fmt.Errorf("spawnllm: bad file mode %q: %w", s, err)
	}
	return os.FileMode(n), nil
}

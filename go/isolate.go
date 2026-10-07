package spawnllm

import (
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
)

const seededAuthEnv = "CLAUDE_CODE_OAUTH_TOKEN"

var rejectedTokens = struct {
	sync.Mutex
	set map[string]bool
}{set: map[string]bool{}}

type claudeIsolation struct {
	dir     string
	env     map[string]string
	cleanup func()
}

func seedClaudeIsolation(apiAuth bool) (*claudeIsolation, error) {
	sources, err := coreIsolationSources(apiAuth)
	if err != nil {
		return nil, err
	}
	var accountJSON *string
	if sources.AccountPath != nil {
		accountJSON = readFileOpt(*sources.AccountPath)
	}
	credentialsJSON := []string{}
	if sources.KeychainService != nil {
		if credentials := keychainCredentials(*sources.KeychainService); credentials != nil {
			credentialsJSON = append(credentialsJSON, *credentials)
		}
	}
	if sources.CredentialsPath != nil {
		if credentials := readFileOpt(*sources.CredentialsPath); credentials != nil {
			credentialsJSON = append(credentialsJSON, *credentials)
		}
	}

	seed, err := coreIsolationSeed(accountJSON, credentialsJSON, rejectedTokenList())
	if err != nil {
		return nil, err
	}

	dir, err := os.MkdirTemp("", "spawnllm-claude-config-")
	if err != nil {
		return nil, err
	}
	cleanup := func() { _ = os.RemoveAll(dir) }
	for _, f := range seed.Files {
		mode, err := parseMode(f.Mode)
		if err != nil {
			cleanup()
			return nil, err
		}
		path := filepath.Join(dir, f.Name)
		if err := os.WriteFile(path, []byte(f.Content), mode); err != nil {
			cleanup()
			return nil, err
		}
		if err := os.Chmod(path, mode); err != nil {
			cleanup()
			return nil, err
		}
	}
	return &claudeIsolation{dir: dir, env: seed.Env, cleanup: cleanup}, nil
}

func rejectedTokenList() []string {
	rejectedTokens.Lock()
	defer rejectedTokens.Unlock()
	tokens := slices.AppendSeq([]string{}, maps.Keys(rejectedTokens.set))
	slices.Sort(tokens)
	return tokens
}

func (i *claudeIsolation) rejectCredentials(errMsg string) (bool, error) {
	token, ok := i.env[seededAuthEnv]
	if !ok {
		return false, nil
	}
	rejected, err := coreAuthRejected(errMsg)
	if err != nil || !rejected {
		return false, err
	}
	rejectedTokens.Lock()
	defer rejectedTokens.Unlock()
	rejectedTokens.set[token] = true
	return true, nil
}

func (i *claudeIsolation) tokenRejected() bool {
	rejectedTokens.Lock()
	defer rejectedTokens.Unlock()
	return rejectedTokens.set[i.env[seededAuthEnv]]
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

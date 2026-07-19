package spawnllm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"syscall"
	"time"
)

func (b *cliBackend) execute(ctx context.Context, spec RunSpec, wantsValue bool) (*attempt, error) {
	kind, plan, _, err := corePlan(b.provider, spec.core())
	if err != nil {
		return nil, err
	}
	if kind != "exec" {
		return nil, fmt.Errorf("spawnllm: provider %q planned a %s invocation, want exec", b.provider, kind)
	}
	output, returncode, stderr, timedOut, err := runExecPlan(ctx, plan, spec)
	if err != nil {
		return nil, err
	}
	if timedOut {
		return timedOutAttempt(spec, b.provider), nil
	}
	return finishAttempt(spec, b.provider, output, returncode, stderr, wantsValue)
}

func runExecPlan(ctx context.Context, plan execPlan, spec RunSpec) (output string, returncode int, stderr string, timedOut bool, err error) {
	var cleanups []func()
	defer func() {
		for _, c := range cleanups {
			c()
		}
	}()

	paths := make(map[string]string, len(plan.Files))
	for _, f := range plan.Files {
		path, e := writeTempFile(f)
		if e != nil {
			return "", 0, "", false, e
		}
		cleanups = append(cleanups, func() { _ = os.Remove(path) })
		paths[f.ID] = path
	}

	env := plan.Env
	if plan.NeedsClaudeIsolation {
		dir, cleanup, e := seedClaudeIsolation()
		if e != nil {
			return "", 0, "", false, e
		}
		cleanups = append(cleanups, cleanup)
		env = substituteIsolationDir(plan.Env, dir)
	}

	argv := substituteFiles(plan.Argv, paths)
	attemptCtx, cancel := context.WithTimeout(ctx, spec.timeout())
	defer cancel()

	cmd := exec.CommandContext(attemptCtx, argv[0], argv[1:]...)
	cmd.Cancel = func() error { return cmd.Process.Signal(syscall.SIGTERM) }
	cmd.WaitDelay = 2 * time.Second
	cmd.Dir = spec.Dir
	cmd.Env = mergeEnv(env, spec.Env, plan.EnvUnset)
	cmd.Stdin = strings.NewReader(plan.Stdin)

	var stderrBuf, stdoutBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	var stdoutFile *os.File
	if plan.StdoutToFile {
		f, e := os.OpenFile(paths["stdout"], os.O_WRONLY|os.O_TRUNC, 0o600)
		if e != nil {
			return "", 0, "", false, e
		}
		stdoutFile = f
		cmd.Stdout = f
	} else {
		cmd.Stdout = &stdoutBuf
	}

	runErr := cmd.Run()
	if stdoutFile != nil {
		_ = stdoutFile.Close()
	}

	if e := ctx.Err(); e != nil {
		return "", 0, "", false, e
	}
	if attemptCtx.Err() == context.DeadlineExceeded {
		return "", 0, "", true, nil
	}

	if runErr != nil {
		var exitErr *exec.ExitError
		if !errors.As(runErr, &exitErr) {
			return "", 0, "", false, runErr
		}
		returncode = exitErr.ExitCode()
	}

	switch plan.ReadResultFrom {
	case "file:result":
		output, err = readFileString(paths["result"])
	case "stdout":
		if plan.StdoutToFile {
			output, err = readFileString(paths["stdout"])
		} else {
			output = stdoutBuf.String()
		}
	default:
		return "", 0, "", false, fmt.Errorf("spawnllm: unknown read_result_from %q", plan.ReadResultFrom)
	}
	if err != nil {
		return "", 0, "", false, err
	}
	return output, returncode, stderrBuf.String(), false, nil
}

func writeTempFile(f planFile) (string, error) {
	tmp, err := os.CreateTemp("", "spawnllm-*"+f.Suffix)
	if err != nil {
		return "", err
	}
	defer func() { _ = tmp.Close() }()
	if f.Content != nil {
		if _, err := tmp.WriteString(*f.Content); err != nil {
			_ = os.Remove(tmp.Name())
			return "", err
		}
	}
	return tmp.Name(), nil
}

func substituteFiles(argv []string, paths map[string]string) []string {
	out := make([]string, len(argv))
	for i, arg := range argv {
		for id, path := range paths {
			if arg == "${file:"+id+"}" {
				arg = path
				break
			}
		}
		out[i] = arg
	}
	return out
}

func mergeEnv(planEnv, specEnv map[string]string, envUnset []string) []string {
	return mergeEnvForOS(planEnv, specEnv, envUnset, runtime.GOOS)
}

func mergeEnvForOS(planEnv, specEnv map[string]string, envUnset []string, goos string) []string {
	unset := make(map[string]struct{}, len(envUnset))
	for _, key := range envUnset {
		unset[key] = struct{}{}
	}
	merged := map[string]string{}
	for _, kv := range os.Environ() {
		if i := strings.IndexByte(kv, '='); i >= 0 {
			key := kv[:i]
			if envKeyIn(key, envUnset, unset, goos) {
				continue
			}
			merged[key] = kv[i+1:]
		}
	}
	for k, v := range planEnv {
		deleteEqualFoldedEnvKey(merged, k, goos)
		merged[k] = v
	}
	for k, v := range specEnv {
		deleteEqualFoldedEnvKey(merged, k, goos)
		merged[k] = v
	}
	out := make([]string, 0, len(merged))
	for k, v := range merged {
		out = append(out, k+"="+v)
	}
	return out
}

func envKeyIn(key string, envUnset []string, unset map[string]struct{}, goos string) bool {
	if goos != "windows" {
		_, ok := unset[key]
		return ok
	}
	for _, candidate := range envUnset {
		if envKeyEquals(key, candidate, goos) {
			return true
		}
	}
	return false
}

func deleteEqualFoldedEnvKey(env map[string]string, key, goos string) {
	if goos != "windows" {
		return
	}
	for candidate := range env {
		if envKeyEquals(candidate, key, goos) {
			delete(env, candidate)
		}
	}
}

func envKeyEquals(a, b, goos string) bool {
	if goos == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}

func readFileString(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

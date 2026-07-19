package spawnllm

import (
	"encoding/json"
	"fmt"

	"github.com/yasyf/spawnllm/go/internal/core"
)

type planFile struct {
	ID      string  `json:"id"`
	Suffix  string  `json:"suffix"`
	Content *string `json:"content"`
}

type execPlan struct {
	Argv                 []string          `json:"argv"`
	Stdin                string            `json:"stdin"`
	Files                []planFile        `json:"files"`
	StdoutToFile         bool              `json:"stdout_to_file"`
	ReadResultFrom       string            `json:"read_result_from"`
	Env                  map[string]string `json:"env"`
	EnvUnset             []string          `json:"env_unset"`
	NeedsClaudeIsolation bool              `json:"needs_claude_isolation"`
}

type httpPlan struct {
	Method  string            `json:"method"`
	URL     string            `json:"url"`
	Headers map[string]string `json:"headers"`
	Body    json.RawMessage   `json:"body"`
}

type resolved struct {
	Status    string          `json:"status"`
	Text      string          `json:"text"`
	Value     json.RawMessage `json:"value"`
	Kind      string          `json:"kind"`
	Msg       string          `json:"msg"`
	Transient bool            `json:"transient"`
	CostUSD   *float64        `json:"cost_usd"`
	Usage     json.RawMessage `json:"usage"`
}

type modelTiers struct {
	Large  string `json:"large"`
	Medium string `json:"medium"`
	Small  string `json:"small"`
}

func (m modelTiers) tier(t ModelTier) (string, error) {
	switch t.orDefault() {
	case Large:
		return m.Large, nil
	case Medium:
		return m.Medium, nil
	case Small:
		return m.Small, nil
	default:
		return "", fmt.Errorf("spawnllm: unknown model tier %q", t)
	}
}

type capabilities struct {
	Providers          []string              `json:"providers"`
	Priority           []string              `json:"priority"`
	AutoSelectExcludes []string              `json:"auto_select_excludes"`
	Specialties        map[string]string     `json:"specialties"`
	Models             map[string]modelTiers `json:"models"`
	Binaries           map[string]string     `json:"binaries"`
	InstallHints       map[string]string     `json:"install_hints"`
}

type retryDecision struct {
	Retry  bool    `json:"retry"`
	SleepS float64 `json:"sleep_s"`
}

type authProbe struct {
	Kind    string   `json:"kind"`
	Argv    []string `json:"argv"`
	Service string   `json:"service"`
	Account string   `json:"account"`
	Vars    []string `json:"vars"`
	Path    string   `json:"path"`
}

type authProbes struct {
	Binary      string      `json:"binary"`
	InstallHint *string     `json:"install_hint"`
	Probes      []authProbe `json:"probes"`
}

type isolationSources struct {
	AccountPath     string  `json:"account_path"`
	CredentialsPath string  `json:"credentials_path"`
	KeychainService *string `json:"keychain_service"`
}

type seedFile struct {
	Name    string `json:"name"`
	Content string `json:"content"`
	Mode    string `json:"mode"`
}

type isolationSeed struct {
	Files []seedFile `json:"files"`
}

func coreCall(op string, input any) (json.RawMessage, error) {
	return core.Call(struct {
		Op    string `json:"op"`
		Input any    `json:"input"`
	}{Op: op, Input: input})
}

func coreInto[T any](op string, input any) (T, error) {
	var out T
	raw, err := coreCall(op, input)
	if err != nil {
		return out, err
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return out, fmt.Errorf("spawnllm: decode %s: %w", op, err)
	}
	return out, nil
}

func coreCapabilities() (capabilities, error) {
	return coreInto[capabilities]("capabilities", struct{}{})
}

func corePlan(provider Provider, spec coreSpec) (string, execPlan, httpPlan, error) {
	raw, err := coreCall("plan", struct {
		Host     map[string]string `json:"host"`
		Provider Provider          `json:"provider"`
		Spec     coreSpec          `json:"spec"`
	}{Host: map[string]string{"platform": platform()}, Provider: provider, Spec: spec})
	if err != nil {
		return "", execPlan{}, httpPlan{}, err
	}
	var kind struct {
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(raw, &kind); err != nil {
		return "", execPlan{}, httpPlan{}, fmt.Errorf("spawnllm: decode plan: %w", err)
	}
	switch kind.Kind {
	case "exec":
		var plan execPlan
		if err := json.Unmarshal(raw, &plan); err != nil {
			return "", execPlan{}, httpPlan{}, fmt.Errorf("spawnllm: decode exec plan: %w", err)
		}
		return kind.Kind, plan, httpPlan{}, nil
	case "http":
		var plan httpPlan
		if err := json.Unmarshal(raw, &plan); err != nil {
			return "", execPlan{}, httpPlan{}, fmt.Errorf("spawnllm: decode http plan: %w", err)
		}
		return kind.Kind, execPlan{}, plan, nil
	default:
		return "", execPlan{}, httpPlan{}, fmt.Errorf("spawnllm: unknown plan kind %q", kind.Kind)
	}
}

func coreResolve(provider Provider, raw string, returncode int, stderr string, wantsValue bool) (resolved, error) {
	return coreInto[resolved]("resolve", struct {
		Provider   Provider `json:"provider"`
		Raw        string   `json:"raw"`
		Returncode int      `json:"returncode"`
		Stderr     string   `json:"stderr"`
		WantsValue bool     `json:"wants_value"`
	}{Provider: provider, Raw: raw, Returncode: returncode, Stderr: stderr, WantsValue: wantsValue})
}

func coreRetryDecision(attempt, maxAttempts int, errorMsg *string) (retryDecision, error) {
	return coreInto[retryDecision]("retry_decision", struct {
		Attempt     int     `json:"attempt"`
		MaxAttempts int     `json:"max_attempts"`
		ErrorMsg    *string `json:"error_msg"`
	}{Attempt: attempt, MaxAttempts: maxAttempts, ErrorMsg: errorMsg})
}

func coreAuthProbes(provider Provider) (authProbes, error) {
	return coreInto[authProbes]("auth_probes", struct {
		Provider Provider          `json:"provider"`
		Host     map[string]string `json:"host"`
	}{Provider: provider, Host: map[string]string{"platform": platform(), "home": home()}})
}

func coreIsolationSources() (isolationSources, error) {
	host := map[string]any{"platform": platform(), "home": home(), "claude_config_dir_env": nil}
	if dir := configDirEnv(); dir != "" {
		host["claude_config_dir_env"] = dir
	}
	return coreInto[isolationSources]("claude_isolation_sources", struct {
		Host map[string]any `json:"host"`
	}{Host: host})
}

func coreIsolationSeed(accountJSON, credentialsJSON *string) (isolationSeed, error) {
	return coreInto[isolationSeed]("claude_isolation_seed", struct {
		AccountJSON     *string `json:"account_json"`
		CredentialsJSON *string `json:"credentials_json"`
	}{AccountJSON: accountJSON, CredentialsJSON: credentialsJSON})
}

func coreStrictSchema(dialect string, schema json.RawMessage) (json.RawMessage, error) {
	out, err := coreInto[struct {
		Schema json.RawMessage `json:"schema"`
	}]("strict_schema", struct {
		Dialect string          `json:"dialect"`
		Schema  json.RawMessage `json:"schema"`
	}{Dialect: dialect, Schema: schema})
	return out.Schema, err
}

package spawnllm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"
)

// Provider identifies an LLM backend's provider.
type Provider string

// The providers a Backend can identify as.
const (
	ProviderClaude         Provider = "claude"
	ProviderCodex          Provider = "codex"
	ProviderGemini         Provider = "gemini"
	ProviderAntigravity    Provider = "antigravity"
	ProviderOpenAIEndpoint Provider = "openai_endpoint"
)

// Specialty scopes backend auto-selection to the backend that serves it.
type Specialty string

// The specialties the routing table maps to a preferred backend.
const (
	SpecialtyDebugging Specialty = "debugging"
	SpecialtyReview    Specialty = "review"
	SpecialtyGeneral   Specialty = "general"
)

// BackendState is the readiness of a backend reported by CheckStatus.
type BackendState int

// The readiness states CheckStatus reports.
const (
	BackendReady BackendState = iota
	BackendNotInstalled
	BackendNotAuthenticated
)

// BackendStatus is a backend's readiness plus its CLI binary and install hint.
type BackendStatus struct {
	State       BackendState
	Binary      string
	InstallHint string
}

// Ready reports whether the backend is installed and authenticated.
func (s BackendStatus) Ready() bool { return s.State == BackendReady }

// Backend executes a RunSpec against one provider. The set is sealed to this
// package's constructors: ClaudeBackend, CodexBackend, GeminiBackend,
// AntigravityBackend, and OpenAIEndpoint.
type Backend interface {
	Provider() Provider
	CheckStatus(ctx context.Context) BackendStatus
	execute(ctx context.Context, spec RunSpec, wantsValue bool) (*attempt, error)
}

// BackendUnavailableError reports that no backend was installed and
// authenticated, carrying each candidate's status for diagnosis.
type BackendUnavailableError struct {
	Specialty Specialty
	Statuses  map[Provider]BackendStatus
}

func (e *BackendUnavailableError) Error() string {
	return "spawnllm: no installed, authenticated LLM backend found"
}

// BackendCallError reports a provider error: a nonzero exit with stderr, or an
// error envelope on a clean exit.
type BackendCallError struct {
	Provider Provider
	ExitCode int
	Stderr   string
	Msg      string
}

func (e *BackendCallError) Error() string { return e.Msg }

// ErrTimeout is the cause of a RunError when an attempt outlives its timeout.
var ErrTimeout = errors.New("spawnllm: backend call timed out")

var capabilitiesOnce = sync.OnceValues(coreCapabilities)

type cliBackend struct {
	provider Provider
}

func (b *cliBackend) Provider() Provider { return b.provider }

func (b *cliBackend) CheckStatus(ctx context.Context) BackendStatus {
	return checkStatusViaProbes(ctx, b.provider)
}

// ClaudeBackend returns a backend for the Anthropic claude CLI.
func ClaudeBackend() Backend { return &cliBackend{provider: ProviderClaude} }

// CodexBackend returns a backend for the OpenAI codex CLI.
func CodexBackend() Backend { return &cliBackend{provider: ProviderCodex} }

// GeminiBackend returns a backend for Google's gemini CLI. It is never
// auto-selected; reach it only by passing it explicitly.
func GeminiBackend() Backend { return &cliBackend{provider: ProviderGemini} }

// AntigravityBackend returns a backend for the Antigravity agy CLI.
func AntigravityBackend() Backend { return &cliBackend{provider: ProviderAntigravity} }

// OpenAIOpts configures an OpenAIEndpoint backend. APIKey "" becomes "local";
// Client nil uses http.DefaultClient.
type OpenAIOpts struct {
	APIKey string
	Client *http.Client
}

// OpenAIEndpoint returns a backend that POSTs to an OpenAI-compatible
// /chat/completions endpoint. It is never auto-selected.
func OpenAIEndpoint(baseURL, model string, opts OpenAIOpts) Backend {
	apiKey := opts.APIKey
	if apiKey == "" {
		apiKey = "local"
	}
	client := opts.Client
	if client == nil {
		client = http.DefaultClient
	}
	return &openaiBackend{baseURL: baseURL, model: model, apiKey: apiKey, client: client}
}

func backendForProvider(p Provider) Backend {
	switch p {
	case ProviderClaude:
		return ClaudeBackend()
	case ProviderCodex:
		return CodexBackend()
	case ProviderGemini:
		return GeminiBackend()
	case ProviderAntigravity:
		return AntigravityBackend()
	default:
		return nil
	}
}

// SelectOpts configures SelectBackend. Specialty promotes its backend to the
// front of the chain; Timeout bounds each readiness probe (0 → 10s).
type SelectOpts struct {
	Specialty Specialty
	Timeout   time.Duration
}

// SelectBackend returns the first installed, authenticated backend in priority
// order, promoting Specialty's backend to the front and skipping the auto-select
// exclusions. It returns a *BackendUnavailableError when none is ready.
func SelectBackend(ctx context.Context, opts SelectOpts) (Backend, error) {
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 10 * time.Second
	}
	caps, err := capabilitiesOnce()
	if err != nil {
		return nil, err
	}
	order, err := selectionOrder(caps, opts.Specialty)
	if err != nil {
		return nil, err
	}
	statuses := make(map[Provider]BackendStatus, len(order))
	for _, p := range order {
		b := backendForProvider(p)
		probeCtx, cancel := context.WithTimeout(ctx, timeout)
		st := b.CheckStatus(probeCtx)
		cancel()
		statuses[p] = st
		if st.Ready() {
			return b, nil
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return nil, &BackendUnavailableError{Specialty: opts.Specialty, Statuses: statuses}
}

func selectionOrder(caps capabilities, specialty Specialty) ([]Provider, error) {
	excluded := make(map[string]bool, len(caps.AutoSelectExcludes))
	for _, e := range caps.AutoSelectExcludes {
		excluded[e] = true
	}
	seen := map[string]bool{}
	var order []Provider
	if specialty != "" {
		p, ok := caps.Specialties[string(specialty)]
		if !ok {
			return nil, fmt.Errorf("spawnllm: unknown specialty %q", specialty)
		}
		order = append(order, Provider(p))
		seen[p] = true
	}
	for _, p := range caps.Priority {
		if seen[p] || excluded[p] {
			continue
		}
		order = append(order, Provider(p))
		seen[p] = true
	}
	return order, nil
}

func modelTier(b Backend, tier ModelTier) (string, error) {
	if ob, ok := b.(*openaiBackend); ok {
		return ob.model, nil
	}
	caps, err := capabilitiesOnce()
	if err != nil {
		return "", err
	}
	tiers, ok := caps.Models[string(b.Provider())]
	if !ok {
		return "", fmt.Errorf("spawnllm: no model tiers for provider %q", b.Provider())
	}
	return tiers.tier(tier)
}

func checkStatusViaProbes(ctx context.Context, provider Provider) BackendStatus {
	probes, err := coreAuthProbes(provider)
	if err != nil {
		return BackendStatus{State: BackendNotAuthenticated}
	}
	if _, err := exec.LookPath(probes.Binary); err != nil {
		hint := ""
		if probes.InstallHint != nil {
			hint = *probes.InstallHint
		}
		return BackendStatus{State: BackendNotInstalled, Binary: probes.Binary, InstallHint: hint}
	}
	for _, p := range probes.Probes {
		if runProbe(ctx, p) {
			return BackendStatus{State: BackendReady, Binary: probes.Binary}
		}
	}
	return BackendStatus{State: BackendNotAuthenticated, Binary: probes.Binary}
}

func runProbe(ctx context.Context, p authProbe) bool {
	switch p.Kind {
	case "exec_exit0":
		if len(p.Argv) == 0 {
			return false
		}
		return exec.CommandContext(ctx, p.Argv[0], p.Argv[1:]...).Run() == nil
	case "keychain_exists":
		return exec.CommandContext(ctx, "security", "find-generic-password", "-s", p.Service, "-a", p.Account).Run() == nil
	case "env_any":
		for _, v := range p.Vars {
			if os.Getenv(v) != "" {
				return true
			}
		}
		return false
	case "file_exists":
		_, err := os.Stat(p.Path)
		return err == nil
	default:
		return false
	}
}

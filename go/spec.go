package spawnllm

import (
	"encoding/json"
	"time"
)

// ModelTier is an abstract model size resolved to a provider model id via core
// capabilities. The zero value is Small.
type ModelTier string

// The abstract model sizes each backend maps to a concrete model id.
const (
	Small  ModelTier = "small"
	Medium ModelTier = "medium"
	Large  ModelTier = "large"
)

func (t ModelTier) orDefault() ModelTier {
	if t == "" {
		return Small
	}
	return t
}

// AppleUseCase names the SystemLanguageModelUseCase the sidecar builds its
// session with. The zero value is AppleUseCaseGeneral.
type AppleUseCase string

// The use cases the on-device model is specialized for.
const (
	AppleUseCaseGeneral        AppleUseCase = "general"
	AppleUseCaseContentTagging AppleUseCase = "content_tagging"
)

func (u AppleUseCase) orDefault() AppleUseCase {
	if u == "" {
		return AppleUseCaseGeneral
	}
	return u
}

// AppleGuardrails names the SystemLanguageModelGuardrails the sidecar builds its
// session with. The zero value is AppleGuardrailsDefault.
type AppleGuardrails string

// The guardrail settings the on-device model runs under.
const (
	AppleGuardrailsDefault                          AppleGuardrails = "default"
	AppleGuardrailsPermissiveContentTransformations AppleGuardrails = "permissive_content_transformations"
)

func (g AppleGuardrails) orDefault() AppleGuardrails {
	if g == "" {
		return AppleGuardrailsDefault
	}
	return g
}

// AppleSampling picks the SamplingMode factory the sidecar calls. The empty
// string leaves the framework default.
type AppleSampling string

// The sampling modes the on-device model decodes with.
const (
	AppleSamplingGreedy AppleSampling = "greedy"
	AppleSamplingRandom AppleSampling = "random"
)

// AppleConfig passes on-device Foundation Models knobs through to the
// spawnllm-apple sidecar. Empty string and zero fields are unset; SamplingTop,
// SamplingProbabilityThreshold, and SamplingSeed apply only to
// AppleSamplingRandom, and Top and ProbabilityThreshold are mutually exclusive —
// Run and RunOn reject either combination rather than dropping the knob.
type AppleConfig struct {
	Instructions                 string
	UseCase                      AppleUseCase
	Guardrails                   AppleGuardrails
	Temperature                  float64
	MaximumResponseTokens        int
	Sampling                     AppleSampling
	SamplingTop                  int
	SamplingProbabilityThreshold float64
	SamplingSeed                 uint64
}

// ClaudeConfig passes flags through to the claude CLI. Empty string and zero
// fields are unset; Tools distinguishes nil (CLI default) from an empty slice
// (disable every built-in tool).
type ClaudeConfig struct {
	PermissionMode       string
	MCPConfig            string
	StrictMCP            bool
	AppendSystemPrompt   string
	SystemPrompt         string
	Settings             string
	DisallowedTools      []string
	MaxTurns             int
	MaxBudgetUSD         float64
	Tools                []string
	DisableSlashCommands bool
	OutputFormat         string
	Verbose              bool
}

// CodexConfig passes knobs through to the codex CLI.
type CodexConfig struct {
	Sandbox     string
	EnableHooks bool
	EnableMCP   bool
	// ServiceTier nil pins "fast"; a pointer to "" drops the flag, and any
	// other value selects that tier.
	ServiceTier           *string
	DeveloperInstructions string
}

// GeminiConfig passes knobs through to the gemini and antigravity CLIs.
// Extensions distinguishes nil (default set) from an empty slice (no extensions).
type GeminiConfig struct {
	ApprovalMode string
	Extensions   []string
}

// ProviderConfigs carries the per-provider flag passthrough a RunSpec applies;
// only the matching backend's config is read.
type ProviderConfigs struct {
	Apple  *AppleConfig
	Claude *ClaudeConfig
	Codex  *CodexConfig
	Gemini *GeminiConfig
}

// RunSpec is a single configured run, translated per backend at execution time.
// Model is a literal provider model id. UseHostConfig false (the zero value) runs
// against a fresh, host-free config home; Timeout is per-attempt (0 → 180s) and
// MaxAttempts bounds the transient-retry loop (0 → 5).
type RunSpec struct {
	Prompt string
	Model  string
	// Schema, when set, must encode a JSON object and is passed to the provider verbatim.
	Schema        json.RawMessage
	Agent         bool
	UseHostConfig bool
	APIAuth       bool
	Dir           string
	Env           map[string]string
	Timeout       time.Duration
	MaxAttempts   int
	Providers     ProviderConfigs
}

func (s RunSpec) timeout() time.Duration {
	if s.Timeout <= 0 {
		return 180 * time.Second
	}
	return s.Timeout
}

func (s RunSpec) maxAttempts() int {
	if s.MaxAttempts <= 0 {
		return 5
	}
	return s.MaxAttempts
}

// coreSpec is the RunSpec in the core wire shape: every optional maps to a
// pointer so an unset field serializes to null, and the config sub-structs carry
// every field the core requires present.
type coreSpec struct {
	Prompt         string          `json:"prompt"`
	Model          string          `json:"model"`
	Agent          bool            `json:"agent"`
	Isolated       bool            `json:"isolated"`
	APIAuth        bool            `json:"api_auth"`
	Timeout        int64           `json:"timeout"`
	MaxAttempts    int64           `json:"max_attempts"`
	Schema         json.RawMessage `json:"schema"`
	Apple          *coreApple      `json:"apple"`
	Claude         *coreClaude     `json:"claude"`
	Codex          *coreCodex      `json:"codex"`
	Gemini         *coreGemini     `json:"gemini"`
	OpenAIEndpoint *coreOpenAI     `json:"openai_endpoint"`
}

type coreApple struct {
	Instructions                 *string         `json:"instructions"`
	UseCase                      AppleUseCase    `json:"use_case"`
	Guardrails                   AppleGuardrails `json:"guardrails"`
	Temperature                  *float64        `json:"temperature"`
	MaximumResponseTokens        *int            `json:"maximum_response_tokens"`
	Sampling                     *AppleSampling  `json:"sampling"`
	SamplingTop                  *int            `json:"sampling_top"`
	SamplingProbabilityThreshold *float64        `json:"sampling_probability_threshold"`
	SamplingSeed                 *uint64         `json:"sampling_seed"`
}

type coreClaude struct {
	AppendSystemPrompt   *string  `json:"append_system_prompt"`
	DisableSlashCommands bool     `json:"disable_slash_commands"`
	DisallowedTools      []string `json:"disallowed_tools"`
	MaxBudgetUSD         *float64 `json:"max_budget_usd"`
	MaxTurns             *int     `json:"max_turns"`
	MCPConfig            *string  `json:"mcp_config"`
	OutputFormat         *string  `json:"output_format"`
	PermissionMode       *string  `json:"permission_mode"`
	Settings             *string  `json:"settings"`
	StrictMCP            bool     `json:"strict_mcp"`
	SystemPrompt         *string  `json:"system_prompt"`
	Tools                []string `json:"tools"`
	Verbose              bool     `json:"verbose"`
}

type coreCodex struct {
	DeveloperInstructions *string `json:"developer_instructions"`
	EnableHooks           bool    `json:"enable_hooks"`
	EnableMCP             bool    `json:"enable_mcp"`
	Sandbox               *string `json:"sandbox"`
	ServiceTier           *string `json:"service_tier"`
}

type coreGemini struct {
	ApprovalMode *string  `json:"approval_mode"`
	Extensions   []string `json:"extensions"`
}

type coreOpenAI struct {
	APIKey  string `json:"api_key"`
	BaseURL string `json:"base_url"`
	Model   string `json:"model"`
}

func optString(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func optInt(n int) *int {
	if n == 0 {
		return nil
	}
	return &n
}

func optUint64(n uint64) *uint64 {
	if n == 0 {
		return nil
	}
	return &n
}

func optFloat(f float64) *float64 {
	if f == 0 {
		return nil
	}
	return &f
}

func (s RunSpec) core() coreSpec {
	schema := s.Schema
	if len(schema) == 0 {
		schema = nil
	}
	return coreSpec{
		Prompt:      s.Prompt,
		Model:       s.Model,
		Agent:       s.Agent,
		Isolated:    !s.UseHostConfig,
		APIAuth:     s.APIAuth,
		Timeout:     int64(s.timeout() / time.Second),
		MaxAttempts: int64(s.maxAttempts()),
		Schema:      schema,
		Apple:       coreAppleOf(s.Providers.Apple),
		Claude:      coreClaudeOf(s.Providers.Claude),
		Codex:       coreCodexOf(s.Providers.Codex),
		Gemini:      coreGeminiOf(s.Providers.Gemini),
	}
}

func coreAppleOf(c *AppleConfig) *coreApple {
	if c == nil {
		return nil
	}
	var sampling *AppleSampling
	if c.Sampling != "" {
		sampling = &c.Sampling
	}
	return &coreApple{
		Instructions:                 optString(c.Instructions),
		UseCase:                      c.UseCase.orDefault(),
		Guardrails:                   c.Guardrails.orDefault(),
		Temperature:                  optFloat(c.Temperature),
		MaximumResponseTokens:        optInt(c.MaximumResponseTokens),
		Sampling:                     sampling,
		SamplingTop:                  optInt(c.SamplingTop),
		SamplingProbabilityThreshold: optFloat(c.SamplingProbabilityThreshold),
		SamplingSeed:                 optUint64(c.SamplingSeed),
	}
}

func coreClaudeOf(c *ClaudeConfig) *coreClaude {
	if c == nil {
		return nil
	}
	return &coreClaude{
		AppendSystemPrompt:   optString(c.AppendSystemPrompt),
		DisableSlashCommands: c.DisableSlashCommands,
		DisallowedTools:      c.DisallowedTools,
		MaxBudgetUSD:         optFloat(c.MaxBudgetUSD),
		MaxTurns:             optInt(c.MaxTurns),
		MCPConfig:            optString(c.MCPConfig),
		OutputFormat:         optString(c.OutputFormat),
		PermissionMode:       optString(c.PermissionMode),
		Settings:             optString(c.Settings),
		StrictMCP:            c.StrictMCP,
		SystemPrompt:         optString(c.SystemPrompt),
		Tools:                c.Tools,
		Verbose:              c.Verbose,
	}
}

func coreCodexOf(c *CodexConfig) *coreCodex {
	if c == nil {
		return nil
	}
	tier := c.ServiceTier
	if tier == nil {
		fast := "fast"
		tier = &fast
	} else if *tier == "" {
		tier = nil
	}
	return &coreCodex{
		DeveloperInstructions: optString(c.DeveloperInstructions),
		EnableHooks:           c.EnableHooks,
		EnableMCP:             c.EnableMCP,
		Sandbox:               optString(c.Sandbox),
		ServiceTier:           tier,
	}
}

func coreGeminiOf(c *GeminiConfig) *coreGemini {
	if c == nil {
		return nil
	}
	return &coreGemini{
		ApprovalMode: optString(c.ApprovalMode),
		Extensions:   c.Extensions,
	}
}

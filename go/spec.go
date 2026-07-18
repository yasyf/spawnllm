package spawnllm

import (
	"encoding/json"
	"time"
)

// ModelTier is an abstract model size resolved to a provider model id via core
// capabilities. The zero value is Small.
type ModelTier string

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

// CodexConfig passes knobs through to the codex CLI. ServiceTier nil pins the
// default "fast"; set it to drop the flag or choose another tier.
type CodexConfig struct {
	Sandbox               string
	EnableHooks           bool
	EnableMCP             bool
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
	Claude *ClaudeConfig
	Codex  *CodexConfig
	Gemini *GeminiConfig
}

// RunSpec is a single configured run, translated per backend at execution time.
// Model is a literal provider model id. UseHostConfig false (the zero value) runs
// against a fresh, host-free config home; Timeout is per-attempt (0 → 180s) and
// MaxAttempts bounds the transient-retry loop (0 → 5). Schema, when set, is passed
// to the provider verbatim.
type RunSpec struct {
	Prompt        string
	Model         string
	Schema        json.RawMessage
	Agent         bool
	UseHostConfig bool
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
	Timeout        int64           `json:"timeout"`
	MaxAttempts    int64           `json:"max_attempts"`
	Schema         json.RawMessage `json:"schema"`
	Claude         *coreClaude     `json:"claude"`
	Codex          *coreCodex      `json:"codex"`
	Gemini         *coreGemini     `json:"gemini"`
	OpenAIEndpoint *coreOpenAI     `json:"openai_endpoint"`
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
		Timeout:     int64(s.timeout() / time.Second),
		MaxAttempts: int64(s.maxAttempts()),
		Schema:      schema,
		Claude:      coreClaudeOf(s.Providers.Claude),
		Codex:       coreCodexOf(s.Providers.Codex),
		Gemini:      coreGeminiOf(s.Providers.Gemini),
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

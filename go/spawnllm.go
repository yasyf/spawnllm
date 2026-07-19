// Package spawnllm runs LLM calls through provider CLIs (claude, codex, gemini,
// agy) or an OpenAI-compatible endpoint, returning one typed Response. Argv
// planning, output resolution, schema strictification, and retry policy execute
// in the embedded spawnllm-core WASM engine shared with the Python and Rust
// implementations; this package is the I/O host: it spawns processes, manages
// temp files and claude isolation, executes auth probes, and drives the retry
// loop.
package spawnllm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/invopop/jsonschema"
)

type attempt struct {
	resp    *Response
	costUSD *float64
	usage   map[string]any
}

// retrySleep waits out the backoff between transient retries; a test seam
// overrides it to skip the wait.
var retrySleep = sleepCtx

// CallOpts configures Call and Extract. Backend overrides auto-selection;
// Specialty scopes auto-selection; Model "" resolves to Small; APIAuth requests
// API-credential authentication; Timeout is per-attempt (0 → 180s).
type CallOpts struct {
	Backend   Backend
	Specialty Specialty
	Model     ModelTier
	Agent     bool
	APIAuth   bool
	Dir       string
	Timeout   time.Duration
}

// Run executes a RunSpec on the first ready backend, retrying transient failures
// with backoff. The returned error is only a caller-side failure (an invalid
// spec, context cancellation, or backend selection); every provider outcome
// lands in Response.Err.
func Run(ctx context.Context, spec RunSpec) (*Response, error) {
	if err := validateRunSpec(spec); err != nil {
		return nil, err
	}
	backend, err := SelectBackend(ctx, SelectOpts{})
	if err != nil {
		return nil, err
	}
	return RunOn(ctx, backend, spec)
}

// RunOn executes a RunSpec on a given backend, retrying transient failures with
// backoff. ctx is the deadline across all retries; RunSpec.Timeout is per-attempt.
func RunOn(ctx context.Context, backend Backend, spec RunSpec) (*Response, error) {
	if err := validateRunSpec(spec); err != nil {
		return nil, err
	}
	return runOn(ctx, backend, spec, false)
}

// Call runs one text LLM call and returns its response, resolving the model tier
// through the selected backend. A provider error returns a *RunError.
func Call(ctx context.Context, prompt string, opts CallOpts) (string, error) {
	backend, err := resolveBackend(ctx, opts)
	if err != nil {
		return "", err
	}
	model, err := modelTier(backend, opts.Model)
	if err != nil {
		return "", err
	}
	spec := RunSpec{Prompt: prompt, Model: model, Agent: opts.Agent, APIAuth: opts.APIAuth, Dir: opts.Dir, Timeout: opts.Timeout}
	resp, err := RunOn(ctx, backend, spec)
	if err != nil {
		return "", err
	}
	if resp.Err != nil {
		return "", resp.Err
	}
	return resp.Result.Raw, nil
}

// Extract runs one structured LLM call and unmarshals the result into T. It
// derives T's JSON schema, applies the selected backend's strict-schema
// transform, runs, and unmarshals the structured output. A provider error
// returns a *RunError.
func Extract[T any](ctx context.Context, prompt string, opts CallOpts) (T, error) {
	var zero T
	backend, err := resolveBackend(ctx, opts)
	if err != nil {
		return zero, err
	}
	model, err := modelTier(backend, opts.Model)
	if err != nil {
		return zero, err
	}
	schema, err := extractSchema[T](backend.Provider())
	if err != nil {
		return zero, err
	}
	spec := RunSpec{Prompt: prompt, Model: model, Schema: schema, Agent: opts.Agent, APIAuth: opts.APIAuth, Dir: opts.Dir, Timeout: opts.Timeout}
	resp, err := runOn(ctx, backend, spec, true)
	if err != nil {
		return zero, err
	}
	if resp.Err != nil {
		return zero, resp.Err
	}
	var out T
	dec := json.NewDecoder(bytes.NewReader(resp.Result.Parsed))
	dec.UseNumber()
	if err := dec.Decode(&out); err != nil {
		return zero, fmt.Errorf("spawnllm: unmarshal structured output: %w", err)
	}
	return out, nil
}

func validateRunSpec(spec RunSpec) error {
	if len(spec.Schema) == 0 {
		return nil
	}
	var schema map[string]json.RawMessage
	if err := json.Unmarshal(spec.Schema, &schema); err != nil {
		return fmt.Errorf("spawnllm: schema must be a JSON object: %w", err)
	}
	if schema == nil {
		return errors.New("spawnllm: schema must be a JSON object")
	}
	return nil
}

func resolveBackend(ctx context.Context, opts CallOpts) (Backend, error) {
	if opts.Backend != nil {
		return opts.Backend, nil
	}
	return SelectBackend(ctx, SelectOpts{Specialty: opts.Specialty})
}

func extractSchema[T any](provider Provider) (json.RawMessage, error) {
	// Referenced ($defs + $ref) like Pydantic, so recursive types don't inline forever.
	reflector := jsonschema.Reflector{ExpandedStruct: true}
	var value T
	schema := reflector.Reflect(&value)
	// $schema/$id trip the anthropic transform into a junk description.
	schema.Version = ""
	schema.ID = ""
	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("spawnllm: marshal schema: %w", err)
	}
	switch provider {
	case ProviderClaude:
		return coreStrictSchema("anthropic", raw)
	case ProviderCodex, ProviderOpenAIEndpoint:
		return coreStrictSchema("openai", raw)
	default:
		return raw, nil
	}
}

func runOn(ctx context.Context, backend Backend, spec RunSpec, wantsValue bool) (*Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	maxAttempts := spec.maxAttempts()
	var discarded []DiscardedAttempt
	for attemptIdx := 0; ; attemptIdx++ {
		att, err := backend.execute(ctx, spec, wantsValue)
		if err != nil {
			if ctxErr := ctx.Err(); ctxErr != nil {
				return nil, ctxErr
			}
			att = transportFailureAttempt(spec, backend.Provider(), err.Error())
		}
		resp := att.resp
		if resp.Err == nil {
			resp.DiscardedAttempts = discarded
			return resp, nil
		}
		decision, err := coreRetryDecision(attemptIdx, maxAttempts, &resp.Err.Msg)
		if err != nil {
			return nil, err
		}
		if !decision.Retry {
			resp.DiscardedAttempts = discarded
			return resp, nil
		}
		discarded = append(discarded, DiscardedAttempt{
			Attempt:  attemptIdx,
			Error:    causeName(resp.Err.Cause),
			CostUSD:  att.costUSD,
			Usage:    att.usage,
			RawBytes: len(resp.Output),
		})
		if err := retrySleep(ctx, decision.SleepS); err != nil {
			return nil, err
		}
	}
}

func finishAttempt(spec RunSpec, provider Provider, output string, returncode int, stderr string, wantsValue bool) (*attempt, error) {
	r, err := coreResolve(provider, output, returncode, stderr, wantsValue)
	if err != nil {
		return nil, err
	}
	resp := &Response{Spec: spec, Output: output}
	if r.Status == "ok" {
		result := &Result{Raw: r.Text}
		if len(r.Value) > 0 && !isJSONNull(r.Value) {
			result.Parsed = r.Value
		}
		resp.Result = result
	} else {
		cause := &BackendCallError{Provider: provider, ExitCode: returncode, Stderr: stderr, Msg: r.Msg}
		resp.Err = &RunError{Msg: r.Msg, Cause: cause}
	}
	return &attempt{resp: resp, costUSD: r.CostUSD, usage: decodeUsage(r.Usage)}, nil
}

func timedOutAttempt(spec RunSpec, provider Provider) *attempt {
	msg := fmt.Sprintf("%s timed out after %s", provider, spec.timeout())
	return &attempt{resp: &Response{Spec: spec, Err: &RunError{Msg: msg, Cause: ErrTimeout}}}
}

func transportFailureAttempt(spec RunSpec, provider Provider, msg string) *attempt {
	cause := &BackendCallError{Provider: provider, Msg: msg}
	return &attempt{resp: &Response{Spec: spec, Err: &RunError{Msg: msg, Cause: cause}}}
}

func causeName(err error) string {
	if errors.Is(err, ErrTimeout) {
		return "TimeoutError"
	}
	var bce *BackendCallError
	if errors.As(err, &bce) {
		return "BackendCallError"
	}
	return "Error"
}

func sleepCtx(ctx context.Context, seconds float64) error {
	if seconds <= 0 {
		return nil
	}
	timer := time.NewTimer(time.Duration(seconds * float64(time.Second)))
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func isJSONNull(raw json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}

func decodeUsage(raw json.RawMessage) map[string]any {
	if len(raw) == 0 || isJSONNull(raw) {
		return nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var usage map[string]any
	if err := dec.Decode(&usage); err != nil {
		return nil
	}
	return usage
}

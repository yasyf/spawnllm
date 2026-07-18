package spawnllm

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
)

type openaiBackend struct {
	baseURL string
	model   string
	apiKey  string
	client  *http.Client
}

func (b *openaiBackend) Provider() Provider { return ProviderOpenAIEndpoint }

func (b *openaiBackend) CheckStatus(_ context.Context) BackendStatus {
	return BackendStatus{State: BackendReady, Binary: string(ProviderOpenAIEndpoint)}
}

func (b *openaiBackend) execute(ctx context.Context, spec RunSpec, wantsValue bool) (*attempt, error) {
	cs := spec.core()
	cs.Model = b.model
	cs.OpenAIEndpoint = &coreOpenAI{APIKey: b.apiKey, BaseURL: b.baseURL, Model: b.model}
	kind, _, plan, err := corePlan(ProviderOpenAIEndpoint, cs)
	if err != nil {
		return nil, err
	}
	if kind != "http" {
		return nil, fmt.Errorf("spawnllm: provider %q planned a %s invocation, want http", ProviderOpenAIEndpoint, kind)
	}
	result, err := runHTTPPlan(ctx, b.client, plan, spec)
	if err != nil {
		return nil, err
	}
	switch {
	case result.timedOut:
		return timedOutAttempt(spec, ProviderOpenAIEndpoint), nil
	case result.transportErr != "":
		return transportFailureAttempt(spec, ProviderOpenAIEndpoint, result.transportErr), nil
	}
	returncode, stderr := 0, ""
	if result.status < 200 || result.status >= 300 {
		returncode, stderr = result.status, result.body
	}
	return finishAttempt(spec, ProviderOpenAIEndpoint, result.body, returncode, stderr, wantsValue)
}

type httpResult struct {
	body         string
	status       int
	timedOut     bool
	transportErr string
}

func runHTTPPlan(ctx context.Context, client *http.Client, plan httpPlan, spec RunSpec) (httpResult, error) {
	attemptCtx, cancel := context.WithTimeout(ctx, spec.timeout())
	defer cancel()

	req, err := http.NewRequestWithContext(attemptCtx, plan.Method, plan.URL, bytes.NewReader(plan.Body))
	if err != nil {
		return httpResult{}, err
	}
	for k, v := range plan.Headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return classifyHTTPError(ctx, attemptCtx, err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return classifyHTTPError(ctx, attemptCtx, err)
	}
	return httpResult{body: string(body), status: resp.StatusCode}, nil
}

// classifyHTTPError routes a transport failure: a parent-context fault is a
// caller error; a per-attempt timeout and a bodyless transport failure are both
// provider outcomes that land in Response.Err.
func classifyHTTPError(ctx, attemptCtx context.Context, err error) (httpResult, error) {
	if e := ctx.Err(); e != nil {
		return httpResult{}, e
	}
	if attemptCtx.Err() == context.DeadlineExceeded {
		return httpResult{timedOut: true}, nil
	}
	return httpResult{transportErr: err.Error()}, nil
}

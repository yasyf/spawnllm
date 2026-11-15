package spawnllm

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"os"
	"os/exec"
	"slices"
	"strings"
	"time"
)

// DecideProvider is a decision provider and the model it runs. Name "jev" posts
// to TypeSafe's System One endpoint and "openai" to OpenAI's /v1/decisions; pin
// a versioned Model once a threshold is tuned against it.
type DecideProvider struct {
	Name  string `json:"name"`
	Model string `json:"model"`
}

var (
	// Jev is TypeSafe Jev pinned to jev-1.13.0, the model jev-latest resolves to today.
	Jev = DecideProvider{Name: "jev", Model: "jev-1.13.0"}
	// OpenAIDecisions is OpenAI Decisions on gpt-6-luna, its only model.
	OpenAIDecisions = DecideProvider{Name: "openai", Model: "gpt-6-luna"}
)

// ErrDecideKeyMissing is the cause when no API key was passed, the provider's
// variable is unset, and the macOS Keychain holds none.
var ErrDecideKeyMissing = errors.New("spawnllm: no decision API key")

// DecideError is a provider's rejection of a decision request, or an answer
// whose body matches no question.
type DecideError struct {
	Status  int
	Message string
}

func (e *DecideError) Error() string {
	return fmt.Sprintf("spawnllm: decision failed with HTTP %d: %s", e.Status, e.Message)
}

// Question is a Binary, Label or Score question.
type Question interface {
	wire(id string) wireQuestion
	keys() []string
}

// Binary is a yes/no question about one observable fact; its answer is the
// probability of yes. Yes and No say what each answer means when the question
// alone leaves the boundary open.
type Binary struct {
	Instructions string
	Yes          string
	No           string
}

// Option is one choice of a Label question.
type Option struct {
	Value       string
	Description string
}

// Label picks one of 2 to 255 options; put the fail-safe option first, since
// Jev leans toward it.
type Label struct {
	Instructions string
	Options      []Option
}

// Level is one rung of a Score question.
type Level struct {
	Label       string
	Description string
}

// Score rates along 2 to 10 ordered levels, lowest first; the score is the
// probability-weighted level index.
type Score struct {
	Instructions string
	Levels       []Level
}

// Answer is a BinaryAnswer, LabelAnswer, ScoreAnswer or Refused.
type Answer interface{ answer() }

// BinaryAnswer answers a Binary question; Confidence is |2·PYes − 1|.
type BinaryAnswer struct {
	PYes       float64
	Confidence float64
}

// LabelAnswer answers a Label question with its most likely option value and
// every option's probability.
type LabelAnswer struct {
	Choice        string
	Probabilities map[string]float64
	Confidence    float64
}

// ScoreAnswer answers a Score question; Score can land between levels.
type ScoreAnswer struct {
	Score         float64
	Probabilities map[string]float64
	Confidence    float64
}

// Refused marks a question the provider declined while it answered the others.
type Refused struct{}

func (BinaryAnswer) answer() {}
func (LabelAnswer) answer()  {}
func (ScoreAnswer) answer()  {}
func (Refused) answer()      {}

// Decision is one provider response: an answer per question id.
type Decision struct {
	Answers     map[string]Answer
	Model       string
	InputTokens int
	Latency     time.Duration
}

// DecideOpts configures Decide. The zero value asks Jev within 10 seconds with
// the key from TYPESAFE_API_KEY or the Keychain over http.DefaultClient.
type DecideOpts struct {
	Provider DecideProvider
	Timeout  time.Duration
	APIKey   string
	Client   *http.Client
}

const defaultDecideTimeout = 10 * time.Second

var decideKeyEnv = map[string]string{"jev": "TYPESAFE_API_KEY", "openai": "OPENAI_API_KEY"}

// Decide asks every question about state in one request and returns the typed
// answers. State is text or JSON data; Jev reads structure natively and OpenAI
// receives it as compact JSON text. Keep each question to one observable fact
// and combine the answers in code.
//
// Timeout bounds the whole call, the Keychain read and retries included: 408,
// 429, 5xx, 529 and a lost connection retry with backoff from 0.5s doubling to
// 5s, honoring retry-after, only while the retry still fits. A call that runs
// out returns an error wrapping ErrTimeout; a rejected request returns a
// *DecideError; a question set breaking a shared limit, such as a label with one
// option, returns the core's error before any request.
func Decide(ctx context.Context, state any, questions map[string]Question, opts DecideOpts) (Decision, error) {
	provider := opts.Provider
	if provider == (DecideProvider{}) {
		provider = Jev
	}
	timeout := opts.Timeout
	if timeout == 0 {
		timeout = defaultDecideTimeout
	}
	client := opts.Client
	if client == nil {
		client = http.DefaultClient
	}
	callCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	outOfTime := func(last string) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		return fmt.Errorf("%w: no %s decision within %s%s", ErrTimeout, provider.Name, timeout, last)
	}

	ids := slices.Sorted(maps.Keys(questions))
	wired := make([]wireQuestion, len(ids))
	for i, id := range ids {
		wired[i] = questions[id].wire(id)
	}
	key := opts.APIKey
	if key == "" {
		var err error
		if key, err = decideKey(callCtx, provider); err != nil {
			if callCtx.Err() != nil {
				return Decision{}, outOfTime("")
			}
			return Decision{}, err
		}
	}
	var baseURL *string
	if url := os.Getenv("TYPESAFE_BASE_URL"); provider.Name == "jev" && url != "" {
		baseURL = &url
	}
	plan, err := coreInto[httpPlan]("decide_plan", struct {
		Provider  DecideProvider `json:"provider"`
		APIKey    string         `json:"api_key"`
		BaseURL   *string        `json:"base_url"`
		State     any            `json:"state"`
		Questions []wireQuestion `json:"questions"`
	}{provider, key, baseURL, state, wired})
	if err != nil {
		return Decision{}, err
	}

	for attempt := 0; ; attempt++ {
		started := time.Now()
		deadline, _ := callCtx.Deadline()
		result, err := runHTTPPlan(callCtx, client, plan, time.Until(deadline))
		if err != nil || result.timedOut {
			return Decision{}, outOfTime("")
		}
		outcome, err := coreInto[decideOutcome]("decide_resolve", resolveInput(provider, wired, result, attempt))
		if err != nil {
			return Decision{}, err
		}
		switch outcome.Kind {
		case "decision":
			return outcome.decision(questions, time.Since(started))
		case "failed":
			return Decision{}, &DecideError{Status: outcome.Status, Message: outcome.Message}
		case "retry":
			sleep := time.Duration(outcome.SleepS * float64(time.Second))
			last := "; the last try got no response"
			if result.transportErr == "" {
				last = fmt.Sprintf("; the last try got HTTP %d", result.status)
			}
			if time.Now().Add(sleep).After(deadline) {
				return Decision{}, outOfTime(last)
			}
			select {
			case <-time.After(sleep):
			case <-callCtx.Done():
				return Decision{}, outOfTime(last)
			}
		default:
			return Decision{}, fmt.Errorf("spawnllm: the core resolved an unknown outcome %q", outcome.Kind)
		}
	}
}

func decideKey(ctx context.Context, provider DecideProvider) (string, error) {
	if key := os.Getenv(decideKeyEnv[provider.Name]); key != "" {
		return key, nil
	}
	if platform() == "darwin" {
		out, err := exec.CommandContext(ctx, "security", "find-generic-password",
			"-s", "spawnllm-"+provider.Name+"-api-key", "-a", provider.Name, "-w").Output()
		if err == nil {
			return strings.TrimSpace(string(out)), nil
		}
	}
	return "", fmt.Errorf("%w for %s: set %s or run `spawnllm key set %s`",
		ErrDecideKeyMissing, provider.Name, decideKeyEnv[provider.Name], provider.Name)
}

type wireQuestion struct {
	Type         string       `json:"type"`
	ID           string       `json:"id"`
	Instructions string       `json:"instructions"`
	Yes          string       `json:"yes,omitempty"`
	No           string       `json:"no,omitempty"`
	Options      []wireOption `json:"options,omitempty"`
	Levels       []wireLevel  `json:"levels,omitempty"`
}

type wireOption struct {
	Value       string `json:"value"`
	Description string `json:"description,omitempty"`
}

type wireLevel struct {
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
}

func (q Binary) wire(id string) wireQuestion {
	return wireQuestion{Type: "binary", ID: id, Instructions: q.Instructions, Yes: q.Yes, No: q.No}
}

func (q Binary) keys() []string { return nil }

func (q Label) wire(id string) wireQuestion {
	options := make([]wireOption, len(q.Options))
	for i, o := range q.Options {
		options[i] = wireOption(o)
	}
	return wireQuestion{Type: "label", ID: id, Instructions: q.Instructions, Options: options}
}

func (q Label) keys() []string {
	keys := make([]string, len(q.Options))
	for i, o := range q.Options {
		keys[i] = o.Value
	}
	return keys
}

func (q Score) wire(id string) wireQuestion {
	levels := make([]wireLevel, len(q.Levels))
	for i, l := range q.Levels {
		levels[i] = wireLevel(l)
	}
	return wireQuestion{Type: "score", ID: id, Instructions: q.Instructions, Levels: levels}
}

func (q Score) keys() []string {
	keys := make([]string, len(q.Levels))
	for i, l := range q.Levels {
		keys[i] = l.Label
	}
	return keys
}

type resolveRequest struct {
	Provider     DecideProvider `json:"provider"`
	Questions    []wireQuestion `json:"questions"`
	Status       *int           `json:"status"`
	Body         string         `json:"body"`
	RetryAfter   *string        `json:"retry_after"`
	RetryAfterMS *string        `json:"retry_after_ms"`
	Attempt      int            `json:"attempt"`
}

func resolveInput(provider DecideProvider, wired []wireQuestion, result httpResult, attempt int) resolveRequest {
	request := resolveRequest{Provider: provider, Questions: wired, Attempt: attempt}
	if result.transportErr != "" {
		return request
	}
	request.Status, request.Body = &result.status, result.body
	header := func(name string) *string {
		if values := result.header.Values(name); len(values) > 0 {
			return &values[0]
		}
		return nil
	}
	request.RetryAfter, request.RetryAfterMS = header("Retry-After"), header("Retry-After-Ms")
	return request
}

type decideOutcome struct {
	Kind        string                `json:"kind"`
	Model       string                `json:"model"`
	InputTokens int                   `json:"input_tokens"`
	Answers     map[string]wireAnswer `json:"answers"`
	SleepS      float64               `json:"sleep_s"`
	Status      int                   `json:"status"`
	Message     string                `json:"message"`
}

type wireAnswer struct {
	Type          string    `json:"type"`
	PYes          float64   `json:"p_yes"`
	Choice        string    `json:"choice"`
	Score         float64   `json:"score"`
	Probabilities []float64 `json:"probabilities"`
	Confidence    float64   `json:"confidence"`
}

func (o decideOutcome) decision(questions map[string]Question, latency time.Duration) (Decision, error) {
	answers := make(map[string]Answer, len(o.Answers))
	for id, wire := range o.Answers {
		probabilities := func() map[string]float64 {
			keyed := make(map[string]float64, len(wire.Probabilities))
			for i, key := range questions[id].keys() {
				keyed[key] = wire.Probabilities[i]
			}
			return keyed
		}
		switch wire.Type {
		case "binary":
			answers[id] = BinaryAnswer{PYes: wire.PYes, Confidence: wire.Confidence}
		case "label":
			answers[id] = LabelAnswer{Choice: wire.Choice, Probabilities: probabilities(), Confidence: wire.Confidence}
		case "score":
			answers[id] = ScoreAnswer{Score: wire.Score, Probabilities: probabilities(), Confidence: wire.Confidence}
		case "refused":
			answers[id] = Refused{}
		default:
			return Decision{}, fmt.Errorf("spawnllm: the core resolved an unknown answer type %q", wire.Type)
		}
	}
	return Decision{Answers: answers, Model: o.Model, InputTokens: o.InputTokens, Latency: latency}, nil
}

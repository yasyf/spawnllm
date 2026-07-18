package spawnllm

import "encoding/json"

// Result is a successful run. Raw is the final text; Parsed is the structured
// JSON value, set only when the run carried a schema and Extract requested it.
type Result struct {
	Raw    string
	Parsed json.RawMessage
}

// RunError is a failed run: a message plus the underlying cause. Cause is a
// *BackendCallError for a nonzero exit or error envelope, ErrTimeout for a
// per-attempt timeout, else the resolving error; Unwrap exposes it.
type RunError struct {
	Msg   string
	Cause error
}

func (e *RunError) Error() string { return e.Msg }

func (e *RunError) Unwrap() error { return e.Cause }

// DiscardedAttempt summarizes a transient failure the retry loop threw away.
// CostUSD and Usage carry the attempt's accounting when its output reported any;
// RawBytes is the UTF-8 length of the discarded output; Error classifies the cause.
type DiscardedAttempt struct {
	Attempt  int
	Error    string
	CostUSD  *float64
	Usage    map[string]any
	RawBytes int
}

// Response is a backend's resolved outcome. Spec and Output are always set;
// exactly one of Result and Err is set. Every provider failure lands in Err, never
// a returned error from Run. DiscardedAttempts holds the transient retries before it.
type Response struct {
	Spec              RunSpec
	Output            string
	Result            *Result
	Err               *RunError
	DiscardedAttempts []DiscardedAttempt
}

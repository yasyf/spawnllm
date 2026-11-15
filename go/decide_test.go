package spawnllm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

const recordedJevBody = `{"model":"jev-1.13.0","answers":{"is_rollback_request":{"type":"noul","noul":0.99},"action":{"type":"choice","choice":"rollback","confidence":1.0,"probabilities":{"release":0.0,"rollback":1.0,"none":0.0}},"urgency":{"type":"score","score":2.0,"confidence":1.0,"legend":{"0":"Not urgent","1":"Somewhat urgent","2":"Blocking, needs action now"},"probabilities":{"0":0.0,"1":0.0,"2":1.0}}},"usage":{"input_tokens":414,"output_tokens":73}}`

var releaseQuestions = map[string]Question{
	"is_rollback_request": Binary{Instructions: "Does the message ask to roll back a release?", Yes: "It asks to undo a release", No: "Anything else"},
	"action": Label{Instructions: "Which release action does the message ask for?", Options: []Option{
		{Value: "none", Description: "No release action"},
		{Value: "release", Description: "Ship a new release"},
		{Value: "rollback", Description: "Roll back a release"},
	}},
	"urgency": Score{Instructions: "How urgent is the message?", Levels: []Level{
		{Label: "Not urgent"}, {Label: "Somewhat urgent"}, {Label: "Blocking, needs action now"},
	}},
}

func jevServer(t *testing.T, handler http.HandlerFunc) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	t.Setenv("TYPESAFE_BASE_URL", srv.URL)
}

func TestDecideJevAnswersEveryQuestionType(t *testing.T) {
	var body struct {
		State     map[string]string         `json:"state"`
		Model     string                    `json:"model"`
		Questions map[string]map[string]any `json:"questions"`
	}
	jevServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/systemone" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q", got)
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		_, _ = io.WriteString(w, recordedJevBody)
	})

	decision, err := Decide(context.Background(), map[string]string{"message": "Please roll back last night's release now."},
		releaseQuestions, DecideOpts{APIKey: "test-key"})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}

	if body.Model != Jev.Model || body.State["message"] == "" {
		t.Fatalf("request model %q state %v", body.Model, body.State)
	}
	if q := body.Questions["is_rollback_request"]; q["type"] != "noul" || q["criteria"].(map[string]any)["true"] != "It asks to undo a release" {
		t.Fatalf("binary question wired as %+v", q)
	}
	if got := decision.Answers["is_rollback_request"].(BinaryAnswer); got.PYes != 0.99 {
		t.Fatalf("binary answer = %+v", got)
	}
	if got := decision.Answers["action"].(LabelAnswer); got.Choice != "rollback" || got.Probabilities["rollback"] != 1 || got.Probabilities["none"] != 0 {
		t.Fatalf("label answer = %+v", got)
	}
	if got := decision.Answers["urgency"].(ScoreAnswer); got.Score != 2 || got.Probabilities["Blocking, needs action now"] != 1 {
		t.Fatalf("score answer = %+v", got)
	}
	if decision.Model != "jev-1.13.0" || decision.InputTokens != 414 {
		t.Fatalf("decision model %q tokens %d", decision.Model, decision.InputTokens)
	}
}

func TestDecideRetriesAnOverloadedProvider(t *testing.T) {
	var calls atomic.Int32
	jevServer(t, func(w http.ResponseWriter, _ *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_, _ = io.WriteString(w, recordedJevBody)
	})

	decision, err := Decide(context.Background(), "Please roll back the release.", releaseQuestions, DecideOpts{APIKey: "k"})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	if calls.Load() != 2 || decision.Answers["action"].(LabelAnswer).Choice != "rollback" {
		t.Fatalf("calls = %d, decision = %+v", calls.Load(), decision)
	}
}

func TestDecideReturnsARejectionAsDecideError(t *testing.T) {
	jevServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = io.WriteString(w, `{"detail":"too short"}`)
	})

	_, err := Decide(context.Background(), "x", releaseQuestions, DecideOpts{APIKey: "k"})
	var rejected *DecideError
	if !errors.As(err, &rejected) || rejected.Status != http.StatusUnprocessableEntity {
		t.Fatalf("err = %v, want a 422 DecideError", err)
	}
}

func TestDecideStopsAtItsTimeout(t *testing.T) {
	jevServer(t, func(_ http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
	})

	started := time.Now()
	_, err := Decide(context.Background(), "x", releaseQuestions, DecideOpts{APIKey: "k", Timeout: 100 * time.Millisecond})
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("err = %v, want ErrTimeout", err)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("Decide returned after %s, past its 100ms timeout", elapsed)
	}
}

func TestDecideRejectsALabelWithOneOption(t *testing.T) {
	_, err := Decide(context.Background(), "x", map[string]Question{
		"only": Label{Instructions: "Pick one", Options: []Option{{Value: "a"}}},
	}, DecideOpts{APIKey: "k"})
	if err == nil {
		t.Fatal("Decide accepted a label with one option")
	}
}

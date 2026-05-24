package llamafit

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"
)

// Runner tests cover the parts that don't require a real GGUF or
// llama-server subprocess — input validation, empty-state behavior,
// concurrent-call serialization. End-to-end startup is exercised by
// the smoke-test binary in cmd/llamafit, not in unit tests.

func TestRunner_EmptyState(t *testing.T) {
	r := NewRunner()
	if r == nil {
		t.Fatal("NewRunner returned nil")
	}
	st := r.Status()
	if st.Running {
		t.Errorf("fresh Runner.Status().Running = true, want false")
	}
	if st.BaseURL != "" || st.ModelPath != "" {
		t.Errorf("fresh Runner.Status() = %+v, want zero value", st)
	}
	// Stop on empty runner must not panic.
	r.Stop()
}

func TestRunner_EmptyModelPath(t *testing.T) {
	r := NewRunner()
	_, err := r.EnsureRunning(context.Background(), Config{})
	if err == nil {
		t.Fatal("EnsureRunning with empty ModelPath returned nil error")
	}
	if !strings.Contains(err.Error(), "ModelPath is empty") {
		t.Errorf("error = %q, want it to mention empty ModelPath", err)
	}
}

func TestRunner_MissingGGUF(t *testing.T) {
	r := NewRunner()
	_, err := r.EnsureRunning(context.Background(), Config{
		ModelPath: "/does/not/exist.gguf",
	})
	if err == nil {
		t.Fatal("EnsureRunning with missing GGUF returned nil error")
	}
	// New(cfg) is what reports the missing-file error; the Runner
	// wraps it. Confirm we surface that path through cleanly.
	if !strings.Contains(err.Error(), "/does/not/exist.gguf") {
		t.Errorf("error = %q, want it to name the missing path", err)
	}
}

// TestRunner_ConcurrentEnsureRunning_Serializes_OnError checks that
// many simultaneous EnsureRunning calls don't race on the mutex
// even when every one returns an error. With -race, this fails
// loudly if the lock around r.current is wrong.
func TestRunner_ConcurrentEnsureRunning_Serializes_OnError(t *testing.T) {
	r := NewRunner()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	const N = 16
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = r.EnsureRunning(ctx, Config{ModelPath: "/missing.gguf"})
		}()
	}
	wg.Wait()
	if r.Status().Running {
		t.Errorf("Status().Running = true after all calls errored")
	}
}

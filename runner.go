package llamafit

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

// Runner owns the lifecycle of a Llamafit Server. It's a thin layer
// on top of `New` + `Start` + `Stop` that handles the embed-in-a-
// long-running-Go-app case: lazy start on first use, idempotent
// across concurrent callers, model swap when the Config changes,
// graceful shutdown.
//
// Use Runner when your app wants Llamafit to feel like a managed
// service ("call EnsureRunning, get a base URL") rather than a
// manually-controlled subprocess ("call New, then Start, hold the
// *Server, call Stop in defer"). The two are equivalent under the
// hood — Runner just removes the bookkeeping.
//
// A Runner is safe for concurrent use. The typical pattern is one
// Runner per process, declared as a package-level var or stored
// on whatever struct owns the LLM dependency.
type Runner struct {
	mu      sync.Mutex
	current *runnerState
}

type runnerState struct {
	server    *Server
	baseURL   string
	cfg       Config
	startedAt time.Time
}

// NewRunner returns a Runner with no server running. The first
// EnsureRunning call starts it.
func NewRunner() *Runner {
	return &Runner{}
}

// EnsureRunning lazily starts the Server for cfg and returns the
// OpenAI-compatible base URL once /health is ready. Behavior:
//
//   - First call: builds the Server (which parses GGUF metadata),
//     starts it, waits for /health, returns the URL.
//   - Subsequent calls with the same ModelPath + DraftModelPath:
//     no-op, returns the existing URL.
//   - Subsequent calls with a different ModelPath or DraftModelPath:
//     stops the running server and starts a new one. Other cfg
//     fields (Parallel, CtxSize, etc.) take effect on swap too,
//     but once a llama-server process is forked its argv is fixed —
//     changing those fields alone does not trigger a swap.
//
// Concurrent callers serialize on the Runner mutex; only one server
// startup is ever in flight. Errors leave no half-started subprocess
// behind.
//
// The passed ctx controls the readiness wait only — NOT the
// subprocess lifetime. The subprocess runs until Stop() is called or
// the host process exits. This is deliberate: callers typically have
// request-scoped ctxs (short timeouts), but the server should
// outlive any single request.
//
// Returns an error when:
//   - cfg.ModelPath is empty
//   - llama-server isn't on PATH (or BinaryPath, if set)
//   - the GGUF can't be read or doesn't fit in available memory
//   - the server fails /health within readinessTimeout (60s)
func (r *Runner) EnsureRunning(ctx context.Context, cfg Config) (string, error) {
	if cfg.ModelPath == "" {
		return "", errors.New("llamafit: ModelPath is empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.current != nil {
		if r.current.cfg.ModelPath == cfg.ModelPath && r.current.cfg.DraftModelPath == cfg.DraftModelPath {
			return r.current.baseURL, nil
		}
		// Model swap — stop the existing server before starting the
		// new one. Holds the mutex through the swap so concurrent
		// callers don't race on a half-stopped server.
		r.current.server.Stop()
		r.current = nil
	}

	srv, err := New(cfg)
	if err != nil {
		return "", fmt.Errorf("llamafit: new server: %w", err)
	}

	// Start in a goroutine with a background context. Server.Start
	// uses exec.CommandContext, which binds the subprocess lifetime
	// to the passed ctx — we want the subprocess to outlive the
	// caller's request, so we deliberately decouple from the caller's
	// ctx here. Stop() is the controlled tear-down path.
	//
	// We discard the goroutine's error: Start blocks until the
	// server is healthy OR the subprocess exits early. The local
	// waitForReady below catches the early-exit case via timeout.
	go func() {
		_ = srv.Start(context.Background())
	}()

	baseURL := srv.OpenAIURL()
	if err := waitForReady(ctx, srv.BaseURL()+"/health", readinessTimeout); err != nil {
		// Don't leave a half-started server behind if readiness
		// failed — the caller has no way to debug it cleanly.
		srv.Stop()
		return "", fmt.Errorf("llamafit: server didn't become ready: %w", err)
	}

	r.current = &runnerState{
		server:    srv,
		baseURL:   baseURL,
		cfg:       cfg,
		startedAt: time.Now(),
	}

	// Persist the (now-proven-stable) Parallel choice into the
	// calibration cache so the next boot with the same (model, host,
	// config) tuple skips the formula and reuses this value. The
	// readiness check above is the success signal — llama-server
	// finished loading the model and reported /health without
	// crashing on KV allocation or Metal command buffer.
	//
	// Persistence failures are non-fatal: we already have a running
	// server, the cache is a perf optimization, not a correctness
	// constraint. Log and continue.
	if cfg.CalibrationPath != "" && srv.cfg.Parallel > 0 {
		if cal, err := LoadCalibration(cfg.CalibrationPath); err == nil {
			if key, kerr := buildCalibrationKey(cfg); kerr == nil {
				key.Parallel = srv.cfg.Parallel
				key.LastBootAt = time.Now().UTC()
				cal.Upsert(key)
				if serr := SaveCalibration(cfg.CalibrationPath, cal); serr != nil {
					cfg.Logger.Warn("calibration save failed",
						slog.String("err", serr.Error()))
				} else {
					cfg.Logger.Info("calibration saved",
						slog.Int("parallel", srv.cfg.Parallel))
				}
			}
		}
	}

	return baseURL, nil
}

// Stop tears down the running server. Idempotent — safe to call
// when nothing is running. Wire into your app's graceful-shutdown
// path so llama-server doesn't outlive the parent process.
func (r *Runner) Stop() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.current == nil {
		return
	}
	r.current.server.Stop()
	r.current = nil
}

// Status snapshots the Runner's current state. Used by health
// surfaces / status commands.
type Status struct {
	Running        bool
	BaseURL        string
	ModelPath      string
	DraftModelPath string
	StartedAt      time.Time
}

// Status returns a snapshot of the runner's current state.
func (r *Runner) Status() Status {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.current == nil {
		return Status{}
	}
	return Status{
		Running:        true,
		BaseURL:        r.current.baseURL,
		ModelPath:      r.current.cfg.ModelPath,
		DraftModelPath: r.current.cfg.DraftModelPath,
		StartedAt:      r.current.startedAt,
	}
}

// readinessTimeout caps how long EnsureRunning will wait for the
// forked llama-server to start serving /health. 60s comfortably
// covers cold model loads on Apple Silicon up to ~30B parameters;
// larger models or slower disks should override via a wrapper that
// adds its own ctx deadline.
const readinessTimeout = 60 * time.Second

// waitForReady polls a /health endpoint until it returns 2xx or
// the timeout expires. 250ms poll interval is fast enough to catch
// a quick startup without hammering the server during slow loads.
func waitForReady(ctx context.Context, healthURL string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("readiness timeout after %s", timeout)
		}
		if probeHealth(healthURL) {
			return nil
		}
		time.Sleep(250 * time.Millisecond)
	}
}

func probeHealth(healthURL string) bool {
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(healthURL)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

package llamafit

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"
)

// Config holds server configuration.
type Config struct {
	ModelPath   string // Path to GGUF file (required)
	Host        string // Listen host (default: 127.0.0.1)
	Port        int    // Listen port (default: 8081)
	CtxSize     int    // Context size per slot (default: 16384)
	Parallel    int    // Concurrent slots (0 = auto from GGUF metadata + available RAM)
	GPULayers   int    // Layers offloaded to GPU (0 = pick a sensible default via autoGPULayers; 99 = all; pass an explicit non-zero count to override)
	KVCacheType string // KV cache quantization type: f16, q8_0, q4_0 (default: q8_0)
	BinaryPath  string // Path to llama-server binary (default: auto-detect)
	// DraftModelPath enables speculative decoding when set. The draft
	// model proposes tokens that the main model verifies in batch — for
	// sequential decoding workloads on the same GPU, this typically
	// gives 1.5-2x throughput at the cost of a small extra resident
	// model in memory. Must share the main model's vocabulary; the
	// natural pairing is a same-family Q4 model 10-50x smaller (e.g.
	// Qwen2.5-0.5B-Instruct as the draft for Qwen2.5-32B-Instruct).
	// When GPULayers > 0, the draft model offloads to the same backend.
	// Default (empty) disables speculative decoding — pure single-model
	// inference, behavior unchanged from older callers.
	DraftModelPath string
	// Embeddings switches the server into embedding-service mode:
	// llama-server is launched with --embeddings, exposes /v1/embeddings,
	// and DOES NOT serve /v1/chat/completions. The model must be a
	// dedicated embedding model (bge-m3, nomic-embed-text, e5-*, etc.) —
	// pointing this at a causal chat model produces unusable vectors
	// (raw final-token hidden states, not pooled embeddings).
	//
	// Mutually exclusive with DraftModelPath: speculative decoding has
	// no meaning for embedding generation. New() returns an error when
	// both are set.
	//
	// Typical setup: one Runner with Embeddings=false on port 8081
	// (chat) + a second Runner with Embeddings=true on port 8082
	// (embeddings). Apple Silicon's unified memory + llama.cpp's Metal
	// backend handle both subprocesses concurrently with no special
	// scheduling.
	Embeddings bool
	Logger     *slog.Logger
}

// autoGPULayers picks a default offload count for the host when the
// caller leaves GPULayers at 0. On macOS, Metal is the only
// acceleration backend that ships with llama-server and Apple Silicon
// uses unified memory, so there's no VRAM ceiling to overflow:
// offloading every layer is always a win and 99 is the llama.cpp
// convention for "all" (the engine clamps to the model's actual layer
// count internally). On other platforms we return 0 — historical
// CPU-only default — until we ship a proper CUDA/ROCm VRAM probe
// that can decide a safe offload count.
func autoGPULayers() int {
	if runtime.GOOS == "darwin" {
		return 99
	}
	return 0
}

func (c *Config) defaults() {
	if c.Host == "" {
		c.Host = "127.0.0.1"
	}
	if c.Port == 0 {
		c.Port = 8081
	}
	if c.CtxSize == 0 {
		c.CtxSize = 16384
	}
	if c.GPULayers == 0 {
		c.GPULayers = autoGPULayers()
	}
	if c.KVCacheType == "" {
		c.KVCacheType = "q8_0"
	}
	if c.Logger == nil {
		c.Logger = slog.Default()
	}
}

// Server manages a llama.cpp server process for serving GGUF models.
type Server struct {
	cfg  Config
	meta *GGUFMetadata
	cmd  *exec.Cmd
	log  *slog.Logger
}

// New creates a server for the given GGUF model.
// Reads model metadata to auto-configure parallel slots if not specified.
func New(cfg Config) (*Server, error) {
	cfg.defaults()

	if cfg.ModelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}
	if cfg.Embeddings && cfg.DraftModelPath != "" {
		return nil, fmt.Errorf("Config.Embeddings and Config.DraftModelPath are mutually exclusive (speculative decoding doesn't apply to embedding generation)")
	}
	if _, err := os.Stat(cfg.ModelPath); err != nil {
		return nil, fmt.Errorf("model not found: %s", cfg.ModelPath)
	}

	// Read GGUF metadata for auto-configuration
	meta, err := ReadGGUFMetadata(cfg.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("read model metadata: %w", err)
	}

	// Auto-calculate parallel slots from model metadata + available RAM
	if cfg.Parallel <= 0 {
		available := AvailableMemory()
		cfg.Parallel = meta.OptimalSlots(cfg.CtxSize, available, cfg.KVCacheType)
	}

	return &Server{
		cfg:  cfg,
		meta: meta,
		log:  cfg.Logger.With("component", "llamafit"),
	}, nil
}

// Metadata returns the GGUF model metadata.
func (s *Server) Metadata() *GGUFMetadata {
	return s.meta
}

// MemoryEstimate returns the memory breakdown for the current configuration.
func (s *Server) MemoryEstimate() MemoryEstimate {
	return s.meta.ModelMemory(s.cfg.CtxSize, s.cfg.KVCacheType)
}

// BaseURL returns the server's API base URL.
func (s *Server) BaseURL() string {
	return fmt.Sprintf("http://%s:%d", s.cfg.Host, s.cfg.Port)
}

// OpenAIURL returns the OpenAI-compatible API URL.
func (s *Server) OpenAIURL() string {
	return s.BaseURL() + "/v1"
}

// Start launches the llama.cpp server and waits for it to be healthy.
func (s *Server) Start(ctx context.Context) error {
	binary := s.resolveBinary()
	args := s.buildArgs()

	mem := s.MemoryEstimate()
	s.log.Info("starting server",
		slog.String("model", filepath.Base(s.cfg.ModelPath)),
		slog.String("arch", s.meta.Architecture),
		slog.Int("layers", s.meta.Layers),
		slog.Int("ctx_size", s.cfg.CtxSize),
		slog.Int("parallel", s.cfg.Parallel),
		slog.Int("gpu_layers", s.cfg.GPULayers),
		slog.String("draft_model", draftLogValue(s.cfg.DraftModelPath)),
		slog.String("model_mem", formatBytes(mem.ModelBytes)),
		slog.String("kv_total", formatBytes(mem.KVPerSlotBytes*int64(s.cfg.Parallel))),
		slog.String("total_mem", formatBytes(mem.TotalBytes(s.cfg.Parallel))),
	)

	s.cmd = exec.CommandContext(ctx, binary, args...)
	s.cmd.Stdout = os.Stdout
	s.cmd.Stderr = os.Stderr

	if err := s.cmd.Start(); err != nil {
		return fmt.Errorf("start llama-server: %w", err)
	}

	if err := s.waitHealthy(ctx); err != nil {
		s.Stop()
		return fmt.Errorf("server failed to become healthy: %w", err)
	}

	s.log.Info("server ready",
		slog.String("url", s.OpenAIURL()),
		slog.Int("parallel_slots", s.cfg.Parallel),
	)
	return nil
}

// buildArgs constructs the llama-server argv from the server's
// config. Extracted from Start so the argv shape is unit-testable
// without spinning up a real subprocess.
func (s *Server) buildArgs() []string {
	// llama-server's --ctx-size is the total context pool, divided
	// across slots.
	totalCtx := s.cfg.CtxSize * s.cfg.Parallel

	args := []string{
		"--model", s.cfg.ModelPath,
		"--host", s.cfg.Host,
		"--port", fmt.Sprintf("%d", s.cfg.Port),
		"--ctx-size", fmt.Sprintf("%d", totalCtx),
		"--parallel", fmt.Sprintf("%d", s.cfg.Parallel),
		"--cont-batching",
		"--flash-attn", "auto",
		"--cache-type-k", s.cfg.KVCacheType,
		"--cache-type-v", s.cfg.KVCacheType,
	}

	if s.cfg.Embeddings {
		// Embedding mode: enable the /v1/embeddings endpoint. llama-server
		// rejects --reasoning* flags in this mode, and chat/completion
		// endpoints aren't served.
		args = append(args, "--embeddings", "--pooling", "mean")
	} else {
		// Chat mode: kill the model's <think> tags from output so
		// downstream OAI clients don't have to strip them.
		args = append(args, "--reasoning-format", "none", "--reasoning", "off")
	}

	// --n-gpu-layers controls offload to Metal/CUDA. Pre-autoGPULayers
	// this code skipped the flag whenever GPULayers was 0 under the
	// (incorrect) belief that llama.cpp would then auto-fit. It does
	// not — n_gpu_layers defaults to 0 inside llama-server, meaning
	// pure CPU inference. defaults() now picks 99 on macOS (Metal /
	// unified memory: offloading every layer is always the right
	// move) and leaves 0 elsewhere as the conservative historical
	// default until we add a VRAM-aware CUDA probe. Skip the flag
	// only when explicitly zero so llama-server's own CPU path is
	// taken cleanly.
	if s.cfg.GPULayers > 0 {
		args = append(args, "--n-gpu-layers", fmt.Sprintf("%d", s.cfg.GPULayers))
	}

	// Speculative decoding: hand llama-server the draft model and
	// mirror the main model's GPU offload so both run on the same
	// backend. llama.cpp's defaults for --draft-max / --draft-min /
	// --draft-p-min (16 / 5 / 0.75) work well for instruction-tuned
	// model pairs; expose them only if tuning becomes load-bearing.
	// Skipped in embedding mode — New() already rejects that combo.
	if !s.cfg.Embeddings && s.cfg.DraftModelPath != "" {
		args = append(args, "--model-draft", s.cfg.DraftModelPath)
		if s.cfg.GPULayers > 0 {
			args = append(args, "--n-gpu-layers-draft", fmt.Sprintf("%d", s.cfg.GPULayers))
		}
	}

	return args
}

// Stop gracefully shuts down the server.
func (s *Server) Stop() {
	if s.cmd == nil || s.cmd.Process == nil {
		return
	}
	s.log.Info("stopping server")
	_ = s.cmd.Process.Signal(os.Interrupt)

	done := make(chan error, 1)
	go func() { done <- s.cmd.Wait() }()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		_ = s.cmd.Process.Kill()
	}
	s.cmd = nil
}

// Wait blocks until the server process exits.
func (s *Server) Wait() error {
	if s.cmd == nil {
		return nil
	}
	err := s.cmd.Wait()
	s.cmd = nil
	return err
}

func (s *Server) waitHealthy(ctx context.Context) error {
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(60 * time.Second)

	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		resp, err := client.Get(s.BaseURL() + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("timeout waiting for server to start")
}

func (s *Server) resolveBinary() string {
	if s.cfg.BinaryPath != "" {
		return s.cfg.BinaryPath
	}
	if path, err := exec.LookPath("llama-server"); err == nil {
		return path
	}
	// Platform-specific defaults
	for _, dir := range []string{"/usr/local/bin", "/opt/homebrew/bin"} {
		path := filepath.Join(dir, "llama-server")
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return "llama-server"
}

// FindFreePort returns an available TCP port.
func FindFreePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

// SystemInfo returns information about the current system for capacity planning.
func SystemInfo() map[string]interface{} {
	return map[string]interface{}{
		"os":               runtime.GOOS,
		"arch":             runtime.GOARCH,
		"cpus":             runtime.NumCPU(),
		"total_memory":     formatBytes(int64(detectTotalMemory())),
		"available_memory": formatBytes(AvailableMemory()),
	}
}

// draftLogValue renders the draft model for structured logs. Empty
// when speculative decoding is disabled (filepath.Base("") returns
// ".", which would log misleadingly).
func draftLogValue(p string) string {
	if p == "" {
		return ""
	}
	return filepath.Base(p)
}

func formatBytes(b int64) string {
	const unit = 1024 * 1024
	if b < unit*1024 {
		return fmt.Sprintf("%.0f MB", float64(b)/float64(unit))
	}
	return fmt.Sprintf("%.1f GB", float64(b)/float64(unit*1024))
}

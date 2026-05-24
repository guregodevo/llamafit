# Llamafit

**Make llama.cpp fit your machine.**

A Go library for programmatically launching tuned llama.cpp servers. Llamafit reads your GGUF, probes your host, and forks `llama-server` with the right flags so you stop hand-tuning context size, slot count, and GPU offload every time you embed a model.

Embedding llama.cpp in a Go application means writing the same hardware-detection, GGUF-parsing, and KV-cache-math boilerplate every time — and getting it subtly wrong every time. Llamafit owns that boilerplate so your application doesn't:

```go
import "github.com/guregodevo/llamafit"

srv, err := llamafit.New(llamafit.Config{
    ModelPath: "/path/to/model.gguf",
})
if err != nil { return err }

if err := srv.Start(ctx); err != nil { return err }
defer srv.Stop()

// srv.OpenAIURL() is a live http://127.0.0.1:8081/v1 endpoint
// — point any OpenAI-compatible client at it.
```

That's the whole startup surface. Llamafit reads the GGUF, probes your host, calculates the optimal slot count + context pool + GPU offload, forks `llama-server` with the right flags, and waits for `/health` to come ready before returning.

## What it does for you

- **Parses GGUF metadata** — architecture, layer count, attention heads, sliding-window detection (Gemma4, Mistral). No need to hardcode model-specific knobs.
- **Probes host memory** — reserves what the OS needs, then sizes parallel slots to fit the rest.
- **Picks the right llama-server flags** — Q8_0 KV cache quantization, flash attention, continuous batching, Metal/CUDA offload tuned to platform.
- **Handles lifecycle** — `Start` blocks until healthy, `Stop` shuts down cleanly, `Wait` joins the subprocess. Safe to embed in a server's graceful-shutdown path.
- **Supports speculative decoding** — set `Config.DraftModelPath` to pair a small draft model with the main; llamafit hands llama-server `--model-draft` and mirrors GPU offload for 1.5-3x decode throughput on long generation.

## Config

```go
type Config struct {
    ModelPath      string        // Path to GGUF file (required)
    Host           string        // Listen host         (default 127.0.0.1)
    Port           int           // Listen port         (default 8081)
    CtxSize        int           // Per-slot context    (default 16384)
    Parallel       int           // Concurrent slots    (0 = auto from GGUF + RAM)
    GPULayers      int           // Layers offloaded    (0 = platform default; 99 = all)
    KVCacheType    string        // f16 | q8_0 | q4_0   (default q8_0)
    DraftModelPath string        // Optional draft GGUF for speculative decoding
    BinaryPath     string        // llama-server path   (default auto-detect)
    Logger         *slog.Logger
}
```

Every field has a sensible auto default. Leave a field zero and llamafit picks for you.

## Server methods

- `New(cfg Config) (*Server, error)` — validate config, parse GGUF, prepare to run.
- `Start(ctx context.Context) error` — fork llama-server, block until `/health` is ready.
- `Stop()` — graceful SIGINT, escalates to SIGKILL after 5s. Idempotent.
- `Wait() error` — block until the subprocess exits (use in long-running embedders).
- `BaseURL() / OpenAIURL()` — endpoints to hand downstream HTTP clients.
- `Metadata() / MemoryEstimate()` — introspect the loaded model without serving.

## Served API

The forked llama-server exposes the standard OpenAI-compatible surface:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`
- `GET  /health`

## Requirements

A `llama-server` binary on `PATH` (or pointed at via `Config.BinaryPath`):

```bash
brew install llama.cpp        # macOS
# or build from https://github.com/ggml-org/llama.cpp
```

## License

Apache 2.0

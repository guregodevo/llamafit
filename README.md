# Spartacus

Running LLMs locally is powerful but configuring llama.cpp is not — you have to figure out how many parallel users your hardware can handle, how much KV cache to allocate, and what context size won't OOM your machine. Get it wrong and you crash. Get it conservative and you waste half your capacity.

Spartacus reads your GGUF model file, detects your available memory, and starts llama.cpp with the optimal configuration. One command, maximum concurrency.

On a 16GB Mac with a 5GB Gemma4 model, Spartacus auto-configures **32 concurrent slots** with full 16K context each — using sliding window detection and Q8_0 KV cache quantization that most manual setups miss.

## Install

```bash
go install github.com/guregodevo/spartacus/cmd/spartacus@latest
```

Requires `llama-server` (from [llama.cpp](https://github.com/ggml-org/llama.cpp)):
```bash
brew install llama.cpp  # macOS
```

## Usage

### CLI

```bash
# Serve a model — everything is auto-configured
spartacus --model model.gguf

# See what Spartacus would do before starting
spartacus --model model.gguf --inspect

# Override if you know better
spartacus --model model.gguf --parallel 8 --ctx-size 16384 --port 8080
```

### Go API

```go
import "github.com/guregodevo/spartacus"

srv, _ := spartacus.New(spartacus.Config{
    ModelPath: "model.gguf",
    Port:      8081,
})

srv.Start(ctx)
defer srv.Stop()
```

## What it does for you

- **Reads the model** — parses GGUF metadata to understand the architecture, layer count, attention heads, and sliding window configuration
- **Understands your hardware** — detects available memory and reserves what the OS needs
- **Maximizes concurrency** — calculates the most parallel slots your system can handle without OOM, with a safety margin
- **Uses the right defaults** — enables Q8_0 KV cache quantization (half the memory, same quality), flash attention, and continuous batching
- **Handles modern architectures** — detects sliding window attention (Gemma4, etc.) where most layers need far less memory than naive calculations assume, unlocking significantly more concurrent users

## API

The server exposes llama.cpp's OpenAI-compatible API:

- `POST /v1/chat/completions` — Chat completions
- `POST /v1/completions` — Text completions
- `POST /v1/embeddings` — Embeddings
- `GET /health` — Health check

## License

Apache 2.0

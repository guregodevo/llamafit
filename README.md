# Spartacus

Go framework for serving GGUF language models via llama.cpp with auto-optimized concurrency.

Reads model metadata from the GGUF file to automatically calculate optimal parallel slots based on available system memory.

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
# Serve a model (auto-detects optimal slots)
spartacus --model model.gguf

# Inspect model and memory estimate
spartacus --model model.gguf --inspect

# Custom configuration
spartacus --model model.gguf --parallel 8 --ctx-size 4096 --port 8080
```

### Go API

```go
import "github.com/guregodevo/spartacus"

// Auto-configure from GGUF metadata
srv, _ := spartacus.New(spartacus.Config{
    ModelPath: "model.gguf",
    Port:      8081,
})

// Start serving (OpenAI-compatible API at /v1)
srv.Start(ctx)
defer srv.Stop()

// Or inspect model without serving
meta, _ := spartacus.ReadGGUFMetadata("model.gguf")
fmt.Println(meta.Architecture)    // "gemma4"
fmt.Println(meta.OptimalSlots(8192, spartacus.AvailableMemory()))  // 16
```

## How it works

1. **Reads GGUF metadata** — layers, embedding dim, KV heads, quantization
2. **Calculates KV cache per slot** — `layers × 2 × ctx × kv_heads × head_dim`
3. **Detects available RAM** — total memory minus OS reservation
4. **Auto-configures parallel slots** — `(available - model_size - overhead) / kv_per_slot`
5. **Starts llama.cpp** with `--parallel N --cont-batching --flash-attn`

## API

The server exposes llama.cpp's OpenAI-compatible API:

- `POST /v1/chat/completions` — Chat completions
- `POST /v1/completions` — Text completions  
- `POST /v1/embeddings` — Embeddings
- `GET /health` — Health check

## License

Apache 2.0

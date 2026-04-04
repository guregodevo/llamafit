package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/guregodevo/spartacus"
)

func main() {
	var (
		modelPath  = flag.String("model", "", "Path to GGUF model file (required)")
		host       = flag.String("host", "127.0.0.1", "Listen host")
		port       = flag.Int("port", 8081, "Listen port")
		parallel   = flag.Int("parallel", 0, "Concurrent slots (0 = auto)")
		ctxSize    = flag.Int("ctx-size", 16384, "Context size per slot")
		gpuLayers  = flag.Int("gpu-layers", 0, "GPU layers (0 = auto-fit, 99 = all)")
		kvCacheType = flag.String("kv-cache-type", "q8_0", "KV cache type: f16, q8_0, q4_0")
		binaryPath  = flag.String("binary", "", "Path to llama-server binary")
		inspect     = flag.Bool("inspect", false, "Print model info and exit")
	)
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintf(os.Stderr, "Usage: spartacus --model <path.gguf>\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  spartacus --model model.gguf                    # auto-detect slots\n")
		fmt.Fprintf(os.Stderr, "  spartacus --model model.gguf --parallel 8        # force 8 slots\n")
		fmt.Fprintf(os.Stderr, "  spartacus --model model.gguf --inspect           # print model info\n")
		os.Exit(1)
	}

	// Read model metadata
	meta, err := spartacus.ReadGGUFMetadata(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading model: %v\n", err)
		os.Exit(1)
	}

	if *inspect {
		printModelInfo(meta, *ctxSize, *kvCacheType)
		return
	}

	// Create server
	srv, err := spartacus.New(spartacus.Config{
		ModelPath:  *modelPath,
		Host:       *host,
		Port:       *port,
		Parallel:   *parallel,
		CtxSize:    *ctxSize,
		GPULayers:   *gpuLayers,
		KVCacheType: *kvCacheType,
		BinaryPath:  *binaryPath,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Graceful shutdown on Ctrl+C
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	if err := srv.Start(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer srv.Stop()

	// Block until signal
	<-ctx.Done()
	fmt.Println("\nShutting down...")
}

func printModelInfo(meta *spartacus.GGUFMetadata, ctxSize int, kvCacheType string) {
	available := spartacus.AvailableMemory()
	slots := meta.OptimalSlots(ctxSize, available, kvCacheType)
	mem := meta.ModelMemory(ctxSize, kvCacheType)

	fmt.Printf("Model: %s\n", meta.Architecture)
	fmt.Printf("Layers: %d\n", meta.Layers)
	fmt.Printf("Embedding: %d\n", meta.EmbeddingDim)
	fmt.Printf("Attention heads: %d (KV: %d)\n", meta.HeadCount, meta.KVHeadCount)
	fmt.Printf("Max context: %d\n", meta.ContextSize)
	fmt.Printf("File size: %.1f GB\n", float64(meta.FileSizeBytes)/(1024*1024*1024))
	fmt.Println()

	sys := spartacus.SystemInfo()
	fmt.Printf("System: %s/%s, %d CPUs, %s RAM (%s available)\n",
		sys["os"], sys["arch"], sys["cpus"], sys["total_memory"], sys["available_memory"])
	fmt.Println()

	fmt.Printf("Context size: %d tokens\n", ctxSize)
	fmt.Printf("KV cache per slot: %.0f MB\n", float64(mem.KVPerSlotBytes)/(1024*1024))
	fmt.Printf("Optimal parallel slots: %d\n", slots)
	fmt.Println()

	fmt.Printf("Memory breakdown (%d slots):\n", slots)
	fmt.Printf("  Model:     %.1f GB\n", float64(mem.ModelBytes)/(1024*1024*1024))
	fmt.Printf("  KV cache:  %.0f MB (%d × %.0f MB)\n",
		float64(mem.KVPerSlotBytes*int64(slots))/(1024*1024),
		slots,
		float64(mem.KVPerSlotBytes)/(1024*1024))
	fmt.Printf("  Overhead:  %.0f MB\n", float64(mem.OverheadBytes)/(1024*1024))
	fmt.Printf("  Total:     %.1f GB\n", float64(mem.TotalBytes(slots))/(1024*1024*1024))

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})))
}

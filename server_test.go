package llamafit

import (
	"slices"
	"strings"
	"testing"
)

// TestBuildArgs_DraftModel covers the speculative-decoding wiring.
// The bare config (no DraftModelPath) must not emit any --model-draft
// arg; setting it must emit --model-draft <path> and, when GPULayers
// is positive, mirror with --n-gpu-layers-draft so both models share
// the same backend.
func TestBuildArgs_DraftModel(t *testing.T) {
	cases := []struct {
		name        string
		draftPath   string
		gpuLayers   int
		wantDraft   bool
		wantGPUMirr bool
	}{
		{"no draft", "", 99, false, false},
		{"draft, GPU", "/models/draft.gguf", 99, true, true},
		{"draft, CPU", "/models/draft.gguf", 0, true, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := &Server{cfg: Config{
				ModelPath:      "/models/main.gguf",
				Host:           "127.0.0.1",
				Port:           8081,
				CtxSize:        4096,
				Parallel:       4,
				GPULayers:      tc.gpuLayers,
				KVCacheType:    "q8_0",
				DraftModelPath: tc.draftPath,
			}}
			args := s.buildArgs()

			gotDraft := slices.Contains(args, "--model-draft")
			gotGPUMirror := slices.Contains(args, "--n-gpu-layers-draft")
			if gotDraft != tc.wantDraft {
				t.Errorf("--model-draft present = %v, want %v (args=%v)", gotDraft, tc.wantDraft, args)
			}
			if gotGPUMirror != tc.wantGPUMirr {
				t.Errorf("--n-gpu-layers-draft present = %v, want %v (args=%v)", gotGPUMirror, tc.wantGPUMirr, args)
			}

			if tc.wantDraft {
				idx := slices.Index(args, "--model-draft")
				if idx+1 >= len(args) || args[idx+1] != tc.draftPath {
					t.Errorf("--model-draft value = %q, want %q", args[idx+1], tc.draftPath)
				}
			}
		})
	}
}

// TestBuildArgs_EmbeddingMode covers the embedding-server argv path:
// --embeddings replaces --reasoning* flags; --model-draft is never
// emitted regardless of DraftModelPath (New() rejects that combo,
// but buildArgs still skips defensively).
func TestBuildArgs_EmbeddingMode(t *testing.T) {
	s := &Server{cfg: Config{
		ModelPath:   "/models/bge-m3.gguf",
		Host:        "127.0.0.1",
		Port:        8082,
		CtxSize:     8192,
		Parallel:    1,
		GPULayers:   99,
		KVCacheType: "q8_0",
		Embeddings:  true,
	}}
	args := s.buildArgs()

	if !slices.Contains(args, "--embeddings") {
		t.Errorf("--embeddings missing in embedding-mode args: %v", args)
	}
	if slices.Contains(args, "--reasoning-format") || slices.Contains(args, "--reasoning") {
		t.Errorf("--reasoning* flags leaked into embedding-mode args: %v", args)
	}
	if slices.Contains(args, "--model-draft") {
		t.Errorf("--model-draft leaked into embedding-mode args: %v", args)
	}
	if !slices.Contains(args, "--pooling") {
		t.Errorf("--pooling missing in embedding-mode args: %v", args)
	}
}

// TestBuildArgs_ChatMode confirms the inverse: chat mode emits
// --reasoning-format and does NOT emit --embeddings. The default format is
// "auto" (data-driven reasoning extraction), so the inline "--reasoning off"
// pair is absent.
func TestBuildArgs_ChatMode(t *testing.T) {
	s := &Server{cfg: Config{
		ModelPath:   "/models/qwen-7b.gguf",
		Host:        "127.0.0.1",
		Port:        8081,
		CtxSize:     4096,
		Parallel:    4,
		GPULayers:   99,
		KVCacheType: "q8_0",
	}}
	args := s.buildArgs()

	if slices.Contains(args, "--embeddings") {
		t.Errorf("--embeddings leaked into chat-mode args: %v", args)
	}
	idx := slices.Index(args, "--reasoning-format")
	if idx < 0 || idx+1 >= len(args) {
		t.Fatalf("--reasoning-format missing from chat-mode args: %v", args)
	}
	if args[idx+1] != "auto" {
		t.Errorf("default --reasoning-format = %q, want %q (args=%v)", args[idx+1], "auto", args)
	}
}

// TestBuildArgs_ReasoningDefault covers the model-agnostic ReasoningFormat
// resolution: empty defaults to "auto" (llama-server detects the format from
// the model's chat template — no hardcoded arch list), an explicit "none"
// forces inline output, and any other explicit value passes through.
func TestBuildArgs_ReasoningDefault(t *testing.T) {
	cases := []struct {
		name         string
		explicit     string // caller-set ReasoningFormat ("" = unset)
		wantFormat   string // expected --reasoning-format value
		wantInlineUp bool   // expect the "--reasoning off" inline pair
	}{
		{"default empty -> auto", "", "auto", false},
		{"explicit none -> inline", "none", "none", true},
		{"explicit auto", "auto", "auto", false},
		{"explicit passthrough", "deepseek", "deepseek", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := &Server{cfg: Config{
				ModelPath:       "/models/m.gguf",
				Host:            "127.0.0.1",
				Port:            8081,
				CtxSize:         4096,
				Parallel:        1,
				GPULayers:       99,
				KVCacheType:     "q8_0",
				ReasoningFormat: tc.explicit,
			}}
			args := s.buildArgs()

			idx := slices.Index(args, "--reasoning-format")
			if idx < 0 || idx+1 >= len(args) {
				t.Fatalf("--reasoning-format missing: %v", args)
			}
			if args[idx+1] != tc.wantFormat {
				t.Errorf("--reasoning-format = %q, want %q (args=%v)", args[idx+1], tc.wantFormat, args)
			}
			gotInline := slices.Contains(args, "--reasoning")
			if gotInline != tc.wantInlineUp {
				t.Errorf("--reasoning (inline off) present = %v, want %v (args=%v)", gotInline, tc.wantInlineUp, args)
			}
		})
	}
}

// TestBuildArgs_CacheReuse covers --cache-reuse for cross-slot prefix
// sharing: emitted in chat mode (stable 5-10K-token agent system
// prompts dominate prefill cost), skipped in embedding mode (one-shot,
// no reuse opportunity).
func TestBuildArgs_CacheReuse(t *testing.T) {
	chat := &Server{cfg: Config{
		ModelPath:   "/models/qwen-7b.gguf",
		Host:        "127.0.0.1",
		Port:        8081,
		CtxSize:     4096,
		Parallel:    4,
		GPULayers:   99,
		KVCacheType: "q8_0",
	}}
	args := chat.buildArgs()
	idx := slices.Index(args, "--cache-reuse")
	if idx < 0 {
		t.Fatalf("--cache-reuse missing from chat-mode args: %v", args)
	}
	if idx+1 >= len(args) || args[idx+1] != "256" {
		t.Errorf("--cache-reuse value = %q, want %q", args[idx+1], "256")
	}

	embed := &Server{cfg: Config{
		ModelPath:   "/models/bge-m3.gguf",
		Host:        "127.0.0.1",
		Port:        8082,
		CtxSize:     8192,
		Parallel:    1,
		GPULayers:   99,
		KVCacheType: "q8_0",
		Embeddings:  true,
	}}
	if slices.Contains(embed.buildArgs(), "--cache-reuse") {
		t.Errorf("--cache-reuse leaked into embedding-mode args: %v", embed.buildArgs())
	}
}

// TestNew_RejectsEmbeddingsWithDraft is the belt-and-suspenders
// check matching the doc comment on Config.Embeddings.
func TestNew_RejectsEmbeddingsWithDraft(t *testing.T) {
	_, err := New(Config{
		ModelPath:      "/does/not/exist.gguf",
		Embeddings:     true,
		DraftModelPath: "/also/missing.gguf",
	})
	if err == nil {
		t.Fatal("New(Embeddings+DraftModelPath) returned nil error")
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("error = %q, want it to mention mutual exclusion", err)
	}
}

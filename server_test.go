package llamafit

import (
	"slices"
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

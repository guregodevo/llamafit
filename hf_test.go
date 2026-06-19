package llamafit

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestParseHFRepo(t *testing.T) {
	cases := []struct {
		spec      string
		wantRepo  string
		wantQuant string
		wantErr   bool
	}{
		{"Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", "Qwen/Qwen2.5-7B-Instruct-GGUF", "Q4_K_M", false},
		{"Qwen/Qwen2.5-7B-Instruct-GGUF", "Qwen/Qwen2.5-7B-Instruct-GGUF", "", false},
		{"  user/repo:q8_0  ", "user/repo", "q8_0", false},
		{"", "", "", true},
		{"justrepo", "", "", true},
		{"too/many/slashes", "", "", true},
	}
	for _, tc := range cases {
		t.Run(tc.spec, func(t *testing.T) {
			repo, quant, err := parseHFRepo(tc.spec)
			if (err != nil) != tc.wantErr {
				t.Fatalf("parseHFRepo(%q) err = %v, wantErr %v", tc.spec, err, tc.wantErr)
			}
			if err != nil {
				return
			}
			if repo != tc.wantRepo || quant != tc.wantQuant {
				t.Errorf("parseHFRepo(%q) = (%q, %q), want (%q, %q)", tc.spec, repo, quant, tc.wantRepo, tc.wantQuant)
			}
		})
	}
}

func TestHFSelectFile(t *testing.T) {
	repo := []string{
		"model-q4_k_m.gguf",
		"model-Q8_0.gguf",
		"model-q4_k_m-00001-of-00002.gguf",
		"model-q4_k_m-00002-of-00002.gguf",
		"README.md",
	}
	cases := []struct {
		name     string
		ggufs    []string
		quant    string
		explicit string
		want     string
		wantErr  bool
	}{
		{"explicit wins", repo, "q8_0", "model-Q8_0.gguf", "model-Q8_0.gguf", false},
		{"quant case-insensitive", repo, "Q4_K_M", "", "model-q4_k_m.gguf", false},
		{"quant prefers non-sharded", []string{"a-q4_k_m-00001-of-00002.gguf", "a-q4_k_m.gguf"}, "q4_k_m", "", "a-q4_k_m.gguf", false},
		{"single gguf no quant", []string{"only.gguf"}, "", "", "only.gguf", false},
		{"ambiguous needs quant", []string{"a.gguf", "b.gguf"}, "", "", "", true},
		{"no quant match", repo, "iq2_xxs", "", "", true},
		{"empty repo", nil, "q4", "", "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := hfSelectFile(tc.ggufs, tc.quant, tc.explicit)
			if (err != nil) != tc.wantErr {
				t.Fatalf("hfSelectFile err = %v, wantErr %v", err, tc.wantErr)
			}
			if err == nil && got != tc.want {
				t.Errorf("hfSelectFile = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestHFCacheDir_RespectsEnv(t *testing.T) {
	t.Setenv("LLAMA_CACHE", "/tmp/my-cache")
	if got := hfCacheDir(); got != "/tmp/my-cache" {
		t.Errorf("hfCacheDir() = %q, want %q", got, "/tmp/my-cache")
	}
}

// TestEnsureHFModel_CachedHit verifies the cached path is returned without
// any network access: with an explicit file there's no repo listing, and a
// pre-existing cache file short-circuits the download.
func TestEnsureHFModel_CachedHit(t *testing.T) {
	cache := t.TempDir()
	t.Setenv("LLAMA_CACHE", cache)

	repo := "Qwen/Qwen2.5-7B-Instruct-GGUF"
	file := "qwen2.5-7b-instruct-q4_k_m.gguf"
	dest := filepath.Join(cache, "llamafit", "Qwen_Qwen2.5-7B-Instruct-GGUF", file)
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dest, []byte("fake gguf"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := EnsureHFModel(context.Background(), repo+":Q4_K_M", file, "")
	if err != nil {
		t.Fatalf("EnsureHFModel cached hit returned error: %v", err)
	}
	if got != dest {
		t.Errorf("EnsureHFModel = %q, want cached %q", got, dest)
	}
}

func TestEnsureHFModel_RejectsSharded(t *testing.T) {
	_, err := EnsureHFModel(context.Background(), "user/repo", "model-00001-of-00003.gguf", "")
	if err == nil {
		t.Fatal("EnsureHFModel accepted a sharded GGUF, want error")
	}
}

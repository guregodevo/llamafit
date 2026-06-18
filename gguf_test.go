package llamafit

import (
	"math"
	"os"
	"testing"
)

func testGGUFPath(t *testing.T) string {
	if p := os.Getenv("LLAMAFIT_TEST_GGUF"); p != "" {
		return p
	}
	t.Skip("set LLAMAFIT_TEST_GGUF to a .gguf file path to run this test")
	return ""
}

func TestReadGGUFMetadata(t *testing.T) {
	path := testGGUFPath(t)

	meta, err := ReadGGUFMetadata(path)
	if err != nil {
		t.Fatalf("ReadGGUFMetadata: %v", err)
	}

	if meta.Architecture == "" {
		t.Error("expected non-empty architecture")
	}
	if meta.Layers <= 0 {
		t.Errorf("expected positive layers, got %d", meta.Layers)
	}
	if meta.EmbeddingDim <= 0 {
		t.Errorf("expected positive embedding dim, got %d", meta.EmbeddingDim)
	}
	if meta.HeadCount <= 0 {
		t.Errorf("expected positive head count, got %d", meta.HeadCount)
	}
	if meta.KVHeadCount <= 0 {
		t.Errorf("expected positive KV head count, got %d", meta.KVHeadCount)
	}
	if meta.FileSizeBytes <= 0 {
		t.Errorf("expected positive file size, got %d", meta.FileSizeBytes)
	}

	t.Logf("arch=%s layers=%d embed=%d heads=%d kv_heads=%d ctx=%d size=%.1fGB",
		meta.Architecture, meta.Layers, meta.EmbeddingDim,
		meta.HeadCount, meta.KVHeadCount, meta.ContextSize,
		float64(meta.FileSizeBytes)/(1024*1024*1024))
	t.Logf("swa=%d shared_kv=%d key_len=%d key_len_swa=%d swa_layers=%d",
		meta.SlidingWindow, meta.SharedKVLayers, meta.KeyLength,
		meta.KeyLengthSWA, meta.SWALayerCount)
}

// TestReadGGUFMetadata_Gemma4Fields verifies the sliding window fields
// are correctly parsed from the sara-q4 (gemma4) GGUF file.
func TestReadGGUFMetadata_Gemma4Fields(t *testing.T) {
	path := testGGUFPath(t)

	meta, err := ReadGGUFMetadata(path)
	if err != nil {
		t.Fatalf("ReadGGUFMetadata: %v", err)
	}

	if meta.Architecture != "gemma4" {
		t.Skipf("not a gemma4 model (got %s), skipping gemma4-specific checks", meta.Architecture)
	}

	// Verify sliding window metadata matches llama-server output
	if meta.SlidingWindow != 512 {
		t.Errorf("SlidingWindow = %d, want 512", meta.SlidingWindow)
	}
	if meta.SharedKVLayers != 18 {
		t.Errorf("SharedKVLayers = %d, want 18", meta.SharedKVLayers)
	}
	if meta.KeyLength != 512 {
		t.Errorf("KeyLength = %d, want 512", meta.KeyLength)
	}
	if meta.KeyLengthSWA != 256 {
		t.Errorf("KeyLengthSWA = %d, want 256", meta.KeyLengthSWA)
	}

	// 42 layers - 18 shared = 24 unique; from llama-server: 4 non-SWA + 20 SWA
	uniqueLayers := meta.Layers - meta.SharedKVLayers
	if uniqueLayers != 24 {
		t.Errorf("unique layers = %d, want 24", uniqueLayers)
	}
	// Pattern has 35 true (SWA) and 7 false (non-SWA) entries
	if meta.SWALayerCount != 35 {
		t.Errorf("SWALayerCount = %d, want 35 (from pattern)", meta.SWALayerCount)
	}

	// The real test: compare our KV estimate to what llama-server actually allocated
	// llama-server output with 7 slots @ 16384 ctx, q8_0:
	//   Non-SWA: 952 MB (4 layers, q8_0)
	//   SWA:     280 MB (20 layers, f16)
	//   Total:  1232 MB / 7 slots = 176 MB per slot
	actualTotalKV := int64(1232 * 1024 * 1024)         // 1232 MB for 7 slots
	actualPerSlot := float64(actualTotalKV) / 7.0       // ~176 MB

	estimatedPerSlot := float64(meta.kvCachePerSlot(16384, "q8_0"))

	pctError := math.Abs(estimatedPerSlot-actualPerSlot) / actualPerSlot * 100
	t.Logf("estimated=%.0f MB, actual=%.0f MB, error=%.1f%%",
		estimatedPerSlot/(1024*1024), actualPerSlot/(1024*1024), pctError)

	if pctError > 10 {
		t.Errorf("KV estimate off by %.1f%% (estimated %.0f MB vs actual %.0f MB per slot)",
			pctError, estimatedPerSlot/(1024*1024), actualPerSlot/(1024*1024))
	}
}

func TestReadGGUFMetadata_InvalidFile(t *testing.T) {
	_, err := ReadGGUFMetadata("/nonexistent.gguf")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestKVCachePerSlot_NoSWA(t *testing.T) {
	tests := []struct {
		name      string
		meta      GGUFMetadata
		ctxSize   int
		cacheType string
		want      int64
	}{
		{
			name: "llama-q8_0",
			meta: GGUFMetadata{
				Layers:       32,
				EmbeddingDim: 4096,
				HeadCount:    32,
				KVHeadCount:  8,
			},
			ctxSize:   4096,
			cacheType: "q8_0",
			// 32 * 2 * 4096 * 8 * 128 * 1.0625 = 285,212,672
			// (q8_0 = 34 bytes per 32-elem block incl. the f16 scale)
			want: 285_212_672,
		},
		{
			name: "llama-f16",
			meta: GGUFMetadata{
				Layers:       32,
				EmbeddingDim: 4096,
				HeadCount:    32,
				KVHeadCount:  8,
			},
			ctxSize:   4096,
			cacheType: "f16",
			// 32 * 2 * 4096 * 8 * 128 * 2.0 = 536,870,912
			want: 536_870_912,
		},
		{
			name:      "zero-heads",
			meta:      GGUFMetadata{Layers: 32, EmbeddingDim: 4096},
			ctxSize:   4096,
			cacheType: "q8_0",
			want:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.meta.kvCachePerSlot(tt.ctxSize, tt.cacheType)
			if got != tt.want {
				t.Errorf("kvCachePerSlot = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestKVCachePerSlot_WithSWA(t *testing.T) {
	// Gemma4-like model: 42 layers, 18 shared, sliding_window=512
	// Pattern: 35 SWA + 7 non-SWA. Shared distributed proportionally:
	//   sharedSWA = round(18 * 35/42) = 15, sharedNonSWA = 3
	//   effective: 20 SWA layers + 4 non-SWA layers = 24 unique
	meta := GGUFMetadata{
		Layers:         42,
		EmbeddingDim:   2560,
		HeadCount:      8,
		KVHeadCount:    2,
		SlidingWindow:  512,
		SharedKVLayers: 18,
		KeyLength:      512,
		KeyLengthSWA:   256,
		SWALayerCount:  35, // raw from pattern (35 true entries)
	}

	// Non-SWA: 4 layers * 2(K+V) * 16384 ctx * 2 kv_heads * 512 dim * 1.0625(q8_0) = 142,606,336
	// SWA:    20 layers * 2(K+V) * 1024 swa_ctx * 2 kv_heads * 256 dim * 2.0(f16) = 41,943,040
	// Total: 184,549,376
	want := int64(142_606_336 + 41_943_040)

	got := meta.kvCachePerSlot(16384, "q8_0")
	if got != want {
		t.Errorf("kvCachePerSlot (SWA) = %d, want %d", got, want)
	}

	// Verify this matches actual llama-server output (~176 MB/slot)
	actualPerSlotMB := float64(1232) / 7.0 // 1232 MB total / 7 slots
	estimatedMB := float64(got) / (1024 * 1024)
	pctError := math.Abs(estimatedMB-actualPerSlotMB) / actualPerSlotMB * 100
	t.Logf("estimated=%.1f MB, actual=%.1f MB, error=%.1f%%", estimatedMB, actualPerSlotMB, pctError)

	if pctError > 10 {
		t.Errorf("SWA estimate off by %.1f%%", pctError)
	}
}

// TestKVCachePerSlot_PerLayer pins the exact per-layer accounting against
// real gemma-4-12B-it numbers verified against llama-server's own KV buffer
// report. The architecture is hybrid: a repeating 5:1 pattern of
// sliding-window layers (8 KV heads, 256 head-dim, windowed) and
// full-attention layers (1 KV head, 512 head-dim, full context). A single
// scalar KVHeadCount can't express this — the full layers have 1 KV head,
// not 8 — so the per-layer arrays drive the estimate.
func TestKVCachePerSlot_PerLayer(t *testing.T) {
	// 48 layers, pattern [SWA×5, full×1] repeating; head_count_kv mirrors it
	// (8 for SWA layers, 1 for the full-attention layers).
	headKV := make([]int, 48)
	pattern := make([]bool, 48)
	for i := 0; i < 48; i++ {
		full := (i+1)%6 == 0 // every 6th layer is full-attention
		if full {
			headKV[i] = 1
			pattern[i] = false
		} else {
			headKV[i] = 8
			pattern[i] = true
		}
	}

	meta := GGUFMetadata{
		Layers:              48,
		EmbeddingDim:        3840,
		HeadCount:           16,
		KVHeadCount:         8, // element[0]; only used by the scalar fallback
		SlidingWindow:       1024,
		KeyLength:           512,
		KeyLengthSWA:        256,
		SWALayerCount:       40,
		HeadCountKVPerLayer: headKV,
		SWAPattern:          pattern,
	}

	// Full (8 layers): 8 * 16384 cells * 2(K+V) * 1 head * 512 dim * 1.0625 = 142,606,336 (136 MiB)
	// SWA (40 layers): 40 * 1536 cells * 2(K+V) * 8 heads * 256 dim * 2.0(f16) = 503,316,480 (480 MiB)
	// Total: 645,922,816 bytes = 616 MiB — matches llama-server exactly.
	want := int64(142_606_336 + 503_316_480)
	got := meta.kvCachePerSlot(16384, "q8_0")
	if got != want {
		t.Errorf("per-layer kvCachePerSlot = %d (%.0f MiB), want %d (%.0f MiB)",
			got, float64(got)/(1024*1024), want, float64(want)/(1024*1024))
	}
}

// TestKVCachePerSlot_LinearHybrid pins the linear-attention hybrid path
// against real qwen3.5-9B numbers verified against llama-server. With 32
// layers and full_attention_interval=4, only 8 layers keep a KV cache; the
// other 24 use linear attention and contribute no context-scaling state.
func TestKVCachePerSlot_LinearHybrid(t *testing.T) {
	meta := GGUFMetadata{
		Layers:                32,
		EmbeddingDim:          4096,
		HeadCount:             16,
		KVHeadCount:           4,
		KeyLength:             256,
		FullAttentionInterval: 4, // every 4th layer is full attention → 8 layers
	}

	// 8 full-attention layers * 2(K+V) * 16384 ctx * 4 kv_heads * 256 dim * 1.0625(q8_0)
	//   = 285,212,672 bytes = 272 MiB — matches llama-server exactly.
	// (Without the hybrid fix, all 32 layers counted → 1088 MiB, 4x too high.)
	want := int64(285_212_672)
	got := meta.kvCachePerSlot(16384, "q8_0")
	if got != want {
		t.Errorf("linear-hybrid kvCachePerSlot = %d (%.0f MiB), want %d (%.0f MiB)",
			got, float64(got)/(1024*1024), want, float64(want)/(1024*1024))
	}
}

func TestKVCachePerSlot_SWAReducesMemory(t *testing.T) {
	noSWA := GGUFMetadata{
		Layers:       42,
		EmbeddingDim: 2560,
		HeadCount:    8,
		KVHeadCount:  2,
	}

	withSWA := GGUFMetadata{
		Layers:         42,
		EmbeddingDim:   2560,
		HeadCount:      8,
		KVHeadCount:    2,
		SlidingWindow:  512,
		SharedKVLayers: 18,
		KeyLength:      512,
		KeyLengthSWA:   256,
		SWALayerCount:  35,
	}

	kvNoSWA := noSWA.kvCachePerSlot(16384, "q8_0")
	kvWithSWA := withSWA.kvCachePerSlot(16384, "q8_0")

	t.Logf("no SWA: %d MB, with SWA: %d MB, ratio: %.1fx",
		kvNoSWA/(1024*1024), kvWithSWA/(1024*1024),
		float64(kvNoSWA)/float64(kvWithSWA))

	if kvWithSWA >= kvNoSWA {
		t.Errorf("SWA should reduce KV cache: %d >= %d", kvWithSWA, kvNoSWA)
	}

	// For gemma4, the reduction should be roughly 5x
	ratio := float64(kvNoSWA) / float64(kvWithSWA)
	if ratio < 3 {
		t.Errorf("expected significant reduction (>3x), got %.1fx", ratio)
	}
}

func TestOptimalSlots(t *testing.T) {
	// Standard model without SWA
	llama := &GGUFMetadata{
		Layers:        32,
		EmbeddingDim:  4096,
		HeadCount:     32,
		KVHeadCount:   8,
		FileSizeBytes: 4 * 1024 * 1024 * 1024,
	}

	// Gemma4 with SWA
	gemma4 := &GGUFMetadata{
		Layers:         42,
		EmbeddingDim:   2560,
		HeadCount:      8,
		KVHeadCount:    2,
		FileSizeBytes:  5 * 1024 * 1024 * 1024,
		SlidingWindow:  512,
		SharedKVLayers: 18,
		KeyLength:      512,
		KeyLengthSWA:   256,
		SWALayerCount:  35,
	}

	tests := []struct {
		name      string
		meta      *GGUFMetadata
		ctxSize   int
		available int64
		cacheType string
		want      int
	}{
		{
			name:      "llama-16GB-q8_0",
			meta:      llama,
			ctxSize:   16384,
			available: 12 * 1024 * 1024 * 1024,
			cacheType: "q8_0",
			want:      5,
		},
		{
			name:      "gemma4-16GB-q8_0",
			meta:      gemma4,
			ctxSize:   16384,
			available: 12 * 1024 * 1024 * 1024,
			cacheType: "q8_0",
			want:      30,
		},
		{
			name:      "gemma4-barely-fits",
			meta:      gemma4,
			ctxSize:   16384,
			available: 6 * 1024 * 1024 * 1024,
			cacheType: "q8_0",
			want:      1,
		},
		{
			name:      "gemma4-less-than-model",
			meta:      gemma4,
			ctxSize:   16384,
			available: 2 * 1024 * 1024 * 1024,
			cacheType: "q8_0",
			want:      1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.meta.OptimalSlots(tt.ctxSize, tt.available, tt.cacheType)
			if got != tt.want {
				kv := tt.meta.kvCachePerSlot(tt.ctxSize, tt.cacheType)
				t.Errorf("OptimalSlots = %d, want %d (kv/slot = %d MB)", got, tt.want, kv/(1024*1024))
			}
		})
	}
}

func TestOptimalSlots_SWAFitsMore(t *testing.T) {
	base := GGUFMetadata{
		Layers:        42,
		EmbeddingDim:  2560,
		HeadCount:     8,
		KVHeadCount:   2,
		FileSizeBytes: 5 * 1024 * 1024 * 1024,
	}

	withSWA := base
	withSWA.SlidingWindow = 512
	withSWA.SharedKVLayers = 18
	withSWA.KeyLength = 512
	withSWA.KeyLengthSWA = 256
	withSWA.SWALayerCount = 20

	available := int64(12 * 1024 * 1024 * 1024)
	slotsNoSWA := base.OptimalSlots(16384, available, "q8_0")
	slotsWithSWA := withSWA.OptimalSlots(16384, available, "q8_0")

	t.Logf("no SWA: %d slots, with SWA: %d slots", slotsNoSWA, slotsWithSWA)

	if slotsWithSWA <= slotsNoSWA {
		t.Errorf("SWA should allow more slots: %d <= %d", slotsWithSWA, slotsNoSWA)
	}
}

func TestModelMemory(t *testing.T) {
	meta := &GGUFMetadata{
		Layers:        32,
		EmbeddingDim:  4096,
		HeadCount:     32,
		KVHeadCount:   8,
		FileSizeBytes: 4 * 1024 * 1024 * 1024,
	}

	mem := meta.ModelMemory(16384)

	if mem.ModelBytes != 4*1024*1024*1024 {
		t.Errorf("ModelBytes = %d, want %d", mem.ModelBytes, 4*1024*1024*1024)
	}
	if mem.OverheadBytes <= 0 {
		t.Errorf("OverheadBytes should be positive, got %d", mem.OverheadBytes)
	}
	if mem.KVPerSlotBytes <= 0 {
		t.Errorf("KVPerSlotBytes should be positive, got %d", mem.KVPerSlotBytes)
	}

	total := mem.TotalBytes(4)
	expected := mem.ModelBytes + 4*mem.KVPerSlotBytes + mem.OverheadBytes
	if total != expected {
		t.Errorf("TotalBytes(4) = %d, want %d", total, expected)
	}
}

func TestModelMemory_DefaultCtx(t *testing.T) {
	meta := &GGUFMetadata{
		Layers:       32,
		EmbeddingDim: 4096,
		HeadCount:    32,
		KVHeadCount:  8,
	}

	mem := meta.ModelMemory(0)
	memExplicit := meta.ModelMemory(16384)

	if mem.KVPerSlotBytes != memExplicit.KVPerSlotBytes {
		t.Errorf("ModelMemory(0) should default to 16384 ctx, got KV=%d vs %d",
			mem.KVPerSlotBytes, memExplicit.KVPerSlotBytes)
	}
}

func TestKVBytesPerElement(t *testing.T) {
	tests := []struct {
		cacheType string
		want      float64
	}{
		{"q4_0", 0.5625}, // 18 bytes / 32-elem block (incl. f16 scale)
		{"q8_0", 1.0625}, // 34 bytes / 32-elem block (incl. f16 scale)
		{"f16", 2.0},
		{"f32", 4.0},
		{"", 2.0},
		{"xyz", 2.0},
	}

	for _, tt := range tests {
		t.Run(tt.cacheType, func(t *testing.T) {
			got := KVBytesPerElement(tt.cacheType)
			if got != tt.want {
				t.Errorf("KVBytesPerElement(%q) = %v, want %v", tt.cacheType, got, tt.want)
			}
		})
	}
}

func TestOptimalSlots_CacheTypeComparison(t *testing.T) {
	meta := &GGUFMetadata{
		Layers:        32,
		EmbeddingDim:  4096,
		HeadCount:     32,
		KVHeadCount:   8,
		FileSizeBytes: 4 * 1024 * 1024 * 1024,
	}
	available := int64(12 * 1024 * 1024 * 1024)

	q4 := meta.OptimalSlots(16384, available, "q4_0")
	q8 := meta.OptimalSlots(16384, available, "q8_0")
	f16 := meta.OptimalSlots(16384, available, "f16")

	if q4 <= q8 {
		t.Errorf("q4_0 (%d) should yield more slots than q8_0 (%d)", q4, q8)
	}
	if q8 <= f16 {
		t.Errorf("q8_0 (%d) should yield more slots than f16 (%d)", q8, f16)
	}
}

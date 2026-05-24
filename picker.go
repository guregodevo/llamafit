package llamafit

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DefaultDraftHeadroom is the recommended free-memory buffer for
// PickDraftModel when speculative decoding runs alongside a chat-
// embed workload. Calibrated against the pathology observed on a
// 16 GB Mac running qwen2.5-7B + qwen2.5-1.5B: residual free memory
// dropped to ~80 MB and llama-server began thrashing, inverting
// speculative decoding into a 4-8x SLOWDOWN. 8 GiB of headroom
// leaves room for the KV cache pool, the caller's host process,
// and any user apps sharing the machine.
//
// Dedicated inference hosts (no other workload, larger RAM tier)
// can pass a tighter value; pass 0 to PickDraftModel to disable
// the memory guard entirely.
const DefaultDraftHeadroom = int64(8) * 1024 * 1024 * 1024

// PickModel returns the largest .gguf file in dir, or an error
// when the directory can't be read or contains no GGUFs.
//
// "Largest" is the right heuristic for "the best quality model I
// have" in Q4-uniform pulls: bigger files mean more parameters at
// the same quantization, which means higher quality. Callers who
// pull mixed quantization tiers (Q4 + Q8 of the same model) should
// implement their own selection.
func PickModel(dir string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", fmt.Errorf("read models dir %q: %w", dir, err)
	}
	var (
		bestPath string
		bestSize int64
	)
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".gguf") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if info.Size() > bestSize {
			bestSize = info.Size()
			bestPath = filepath.Join(dir, e.Name())
		}
	}
	if bestPath == "" {
		return "", fmt.Errorf("no GGUFs under %q", dir)
	}
	return bestPath, nil
}

// PickDraftModel returns a draft GGUF from dir suitable for pairing
// with mainPath for speculative decoding, or "" when nothing in dir
// qualifies. The empty return is the normal "speculative decoding
// disabled, run single-model" signal — it is not an error.
//
// Selection rules:
//
//   - Skips mainPath itself.
//   - Considers only .gguf files at most 1/3 the size of mainPath.
//     The 1/3 threshold is a proxy for "same family, smaller tier"
//     — qwen2.5-1.5B paired with qwen2.5-7B (~1/4) or qwen2.5-0.5B
//     paired with qwen2.5-32B (~1/30) both pass.
//   - Picks the LARGEST qualifying file. Bigger drafts have higher
//     acceptance rates, so size near (but below) the threshold is
//     best.
//   - Returns "" when (main + draft + headroomBytes) would exceed
//     AvailableMemory(). See DefaultDraftHeadroom for the recommended
//     value and the rationale; pass 0 to skip the memory guard.
//
// Tokenizer compatibility is NOT checked — caller is responsible
// for ensuring the picked draft shares vocab with the main. If a
// non-matching GGUF is in dir, llama-server will refuse to start
// with a clear error and the caller can fall back to single-model.
func PickDraftModel(mainPath, dir string, headroomBytes int64) string {
	if mainPath == "" {
		return ""
	}
	mainInfo, err := os.Stat(mainPath)
	if err != nil {
		return ""
	}
	mainSize := mainInfo.Size()
	maxDraftSize := mainSize / 3

	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	var (
		bestPath string
		bestSize int64
	)
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".gguf") {
			continue
		}
		p := filepath.Join(dir, e.Name())
		if p == mainPath {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		if info.Size() > maxDraftSize {
			continue
		}
		if info.Size() > bestSize {
			bestSize = info.Size()
			bestPath = p
		}
	}
	if bestPath == "" {
		return ""
	}
	if headroomBytes > 0 {
		if avail := AvailableMemory(); avail > 0 {
			if mainSize+bestSize+headroomBytes > avail {
				return ""
			}
		}
	}
	return bestPath
}

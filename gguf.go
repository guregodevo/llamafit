package llamafit

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
)

// GGUFMetadata holds model parameters extracted from a GGUF file.
type GGUFMetadata struct {
	Architecture   string
	Layers         int
	EmbeddingDim   int
	HeadCount      int
	KVHeadCount    int
	ContextSize    int
	FileType       int
	FileSizeBytes  int64
	SlidingWindow  int // SWA window size (0 = no sliding window)
	SharedKVLayers int // layers sharing KV cache with others
	KeyLength      int // head dim for non-SWA layers (0 = derive from EmbeddingDim/HeadCount)
	KeyLengthSWA   int // head dim for SWA layers (0 = same as KeyLength)
	SWALayerCount  int // number of SWA layers (computed from sliding_window_pattern)
}

// headDimNonSWA returns the K/V head dimension for non-SWA layers.
func (m *GGUFMetadata) headDimNonSWA() int {
	if m.KeyLength > 0 {
		return m.KeyLength
	}
	if m.HeadCount == 0 {
		return 0
	}
	return m.EmbeddingDim / m.HeadCount
}

// headDimSWA returns the K/V head dimension for SWA layers.
func (m *GGUFMetadata) headDimSWA() int {
	if m.KeyLengthSWA > 0 {
		return m.KeyLengthSWA
	}
	return m.headDimNonSWA()
}

// KVBytesPerElement returns the bytes per element for a KV cache type.
func KVBytesPerElement(cacheType string) float64 {
	switch cacheType {
	case "q4_0":
		return 0.5
	case "q8_0":
		return 1.0
	case "f32":
		return 4.0
	default: // f16
		return 2.0
	}
}

// ModelMemory returns estimated memory usage for a given context size and KV cache type.
func (m *GGUFMetadata) ModelMemory(ctxSize int, kvCacheType ...string) MemoryEstimate {
	if ctxSize <= 0 {
		ctxSize = 16384
	}
	cacheType := "q8_0"
	if len(kvCacheType) > 0 && kvCacheType[0] != "" {
		cacheType = kvCacheType[0]
	}
	return MemoryEstimate{
		ModelBytes:     m.FileSizeBytes,
		KVPerSlotBytes: m.kvCachePerSlot(ctxSize, cacheType),
		OverheadBytes:  512 * 1024 * 1024, // ~512MB for compute buffers, scratch space
	}
}

// kvCachePerSlot calculates KV cache size per slot in bytes.
// For models with sliding window attention (e.g., Gemma4), only a subset of
// layers need full-context KV cache. SWA layers use a small fixed window,
// and shared KV layers need no cache at all.
func (m *GGUFMetadata) kvCachePerSlot(ctxSize int, cacheType string) int64 {
	if m.HeadCount == 0 || m.KVHeadCount == 0 {
		return 0
	}

	// No sliding window — all layers use full context
	if m.SlidingWindow == 0 {
		headDim := int64(m.headDimNonSWA())
		elements := int64(m.Layers) * 2 * int64(ctxSize) * int64(m.KVHeadCount) * headDim
		return int64(float64(elements) * KVBytesPerElement(cacheType))
	}

	// Sliding window architecture
	// Shared KV layers are distributed proportionally across SWA and non-SWA layers
	nonSWATotal := m.Layers - m.SWALayerCount
	sharedSWA := 0
	if m.Layers > 0 {
		sharedSWA = int(math.Round(float64(m.SharedKVLayers) * float64(m.SWALayerCount) / float64(m.Layers)))
	}
	sharedNonSWA := m.SharedKVLayers - sharedSWA
	swaLayers := m.SWALayerCount - sharedSWA
	nonSWALayers := nonSWATotal - sharedNonSWA

	// Non-SWA layers: full context, uses requested cache type
	nonSWABytes := int64(0)
	if nonSWALayers > 0 {
		headDim := int64(m.headDimNonSWA())
		elements := int64(nonSWALayers) * 2 * int64(ctxSize) * int64(m.KVHeadCount) * headDim
		nonSWABytes = int64(float64(elements) * KVBytesPerElement(cacheType))
	}

	// SWA layers: sliding window context (llama.cpp allocates 2x window), always F16
	swaBytes := int64(0)
	if swaLayers > 0 {
		swaCtx := int64(m.SlidingWindow) * 2
		headDim := int64(m.headDimSWA())
		elements := int64(swaLayers) * 2 * swaCtx * int64(m.KVHeadCount) * headDim
		swaBytes = int64(float64(elements) * KVBytesPerElement("f16"))
	}

	return nonSWABytes + swaBytes
}

// OptimalSlots calculates how many parallel slots fit in available memory.
func (m *GGUFMetadata) OptimalSlots(ctxSize int, availableBytes int64, kvCacheType ...string) int {
	cacheType := "q8_0"
	if len(kvCacheType) > 0 && kvCacheType[0] != "" {
		cacheType = kvCacheType[0]
	}
	mem := m.ModelMemory(ctxSize, cacheType)
	// Apply 10% safety margin for alignment overhead and runtime allocations
	safeAvailable := int64(float64(availableBytes) * 0.90)
	remaining := safeAvailable - mem.ModelBytes - mem.OverheadBytes
	if remaining <= 0 {
		return 1
	}
	kvPerSlot := m.kvCachePerSlot(ctxSize, cacheType)
	if kvPerSlot <= 0 {
		return 1
	}
	slots := int(remaining / kvPerSlot)
	if slots < 1 {
		return 1
	}
	if slots > 256 {
		return 256
	}
	return slots
}

// MemoryEstimate holds memory breakdown for a model configuration.
type MemoryEstimate struct {
	ModelBytes     int64
	KVPerSlotBytes int64
	OverheadBytes  int64
}

// TotalBytes returns total memory for a given number of slots.
func (e MemoryEstimate) TotalBytes(slots int) int64 {
	return e.ModelBytes + int64(slots)*e.KVPerSlotBytes + e.OverheadBytes
}

// AvailableMemory returns estimated available memory for model serving.
// Reserves ~4GB for OS on macOS, ~2GB on Linux.
func AvailableMemory() int64 {
	var total uint64

	// Use runtime.GOMAXPROCS as a proxy — actual memory detection below
	switch runtime.GOOS {
	case "darwin":
		// macOS: read hw.memsize
		total = detectTotalMemory()
	default:
		// Fallback: assume 16GB
		total = 16 * 1024 * 1024 * 1024
	}

	// Reserve memory for OS
	reserved := uint64(4 * 1024 * 1024 * 1024) // 4GB for macOS
	if runtime.GOOS == "linux" {
		reserved = 2 * 1024 * 1024 * 1024 // 2GB for Linux
	}

	if total <= reserved {
		return int64(total / 2)
	}
	return int64(total - reserved)
}

// ReadGGUFMetadata reads model parameters from a GGUF file header.
func ReadGGUFMetadata(path string) (*GGUFMetadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open GGUF: %w", err)
	}
	defer f.Close()

	stat, _ := f.Stat()
	meta := &GGUFMetadata{FileSizeBytes: stat.Size()}

	// Read GGUF magic and version
	var magic uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != 0x46554747 { // "GGUF"
		return nil, fmt.Errorf("not a GGUF file (magic: 0x%X)", magic)
	}

	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d", version)
	}

	// Read tensor count and metadata KV count
	var tensorCount, metadataKVCount uint64
	if err := binary.Read(f, binary.LittleEndian, &tensorCount); err != nil {
		return nil, fmt.Errorf("read tensor count: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &metadataKVCount); err != nil {
		return nil, fmt.Errorf("read metadata count: %w", err)
	}

	// Parse metadata key-value pairs
	for i := uint64(0); i < metadataKVCount; i++ {
		key, val, err := readGGUFKV(f, version)
		if err != nil {
			break // Best effort — stop on first error
		}

		switch key {
		case "general.architecture":
			if s, ok := val.(string); ok {
				meta.Architecture = s
			}
		case "general.file_type":
			if v, ok := val.(uint32); ok {
				meta.FileType = int(v)
			}
		}

		// Architecture-prefixed keys (e.g., "gemma4.block_count")
		arch := meta.Architecture
		if arch == "" {
			arch = "llama" // fallback
		}
		switch key {
		case arch + ".block_count":
			if v, ok := val.(uint32); ok {
				meta.Layers = int(v)
			}
		case arch + ".embedding_length":
			if v, ok := val.(uint32); ok {
				meta.EmbeddingDim = int(v)
			}
		case arch + ".attention.head_count":
			if v, ok := val.(uint32); ok {
				meta.HeadCount = int(v)
			}
		case arch + ".attention.head_count_kv":
			if v, ok := val.(uint32); ok {
				meta.KVHeadCount = int(v)
			}
		case arch + ".context_length":
			if v, ok := val.(uint32); ok {
				meta.ContextSize = int(v)
			}
		case arch + ".attention.sliding_window":
			if v, ok := val.(uint32); ok {
				meta.SlidingWindow = int(v)
			}
		case arch + ".attention.shared_kv_layers":
			if v, ok := val.(uint32); ok {
				meta.SharedKVLayers = int(v)
			}
		case arch + ".attention.key_length":
			if v, ok := val.(uint32); ok {
				meta.KeyLength = int(v)
			}
		case arch + ".attention.key_length_swa":
			if v, ok := val.(uint32); ok {
				meta.KeyLengthSWA = int(v)
			}
		case arch + ".attention.sliding_window_pattern":
			// Bool array: count SWA layers (true = SWA)
			if bools, ok := val.([]bool); ok {
				count := 0
				for _, isSWA := range bools {
					if isSWA {
						count++
					}
				}
				meta.SWALayerCount = count
			}
		}
	}

	return meta, nil
}

// GGUF value types
const (
	ggufTypeUint8   = 0
	ggufTypeInt8    = 1
	ggufTypeUint16  = 2
	ggufTypeInt16   = 3
	ggufTypeUint32  = 4
	ggufTypeInt32   = 5
	ggufTypeFloat32 = 6
	ggufTypeBool    = 7
	ggufTypeString  = 8
	ggufTypeArray   = 9
	ggufTypeUint64  = 10
	ggufTypeInt64   = 11
	ggufTypeFloat64 = 12
)

func readGGUFKV(r io.Reader, version uint32) (string, interface{}, error) {
	key, err := readGGUFString(r)
	if err != nil {
		return "", nil, err
	}

	var valueType uint32
	if err := binary.Read(r, binary.LittleEndian, &valueType); err != nil {
		return key, nil, err
	}

	val, err := readGGUFValue(r, valueType)
	return key, val, err
}

func readGGUFString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 1024*1024 { // sanity check
		return "", fmt.Errorf("string too long: %d", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readGGUFValue(r io.Reader, valueType uint32) (interface{}, error) {
	switch valueType {
	case ggufTypeUint8:
		var v uint8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeInt8:
		var v int8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeUint16:
		var v uint16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeInt16:
		var v int16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeUint32:
		var v uint32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeInt32:
		var v int32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeFloat32:
		var v float32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeBool:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	case ggufTypeString:
		return readGGUFString(r)
	case ggufTypeUint64:
		var v uint64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeInt64:
		var v int64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeFloat64:
		var v float64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case ggufTypeArray:
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, err
		}
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return nil, err
		}
		// Return bool arrays (needed for sliding_window_pattern)
		if elemType == ggufTypeBool && count <= 1024 {
			bools := make([]bool, count)
			for j := uint64(0); j < count; j++ {
				var v uint8
				if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
					return nil, err
				}
				bools[j] = v != 0
			}
			return bools, nil
		}
		// Skip other array contents
		for j := uint64(0); j < count; j++ {
			if _, err := readGGUFValue(r, elemType); err != nil {
				return nil, err
			}
		}
		return nil, nil
	default:
		return nil, fmt.Errorf("unknown GGUF type: %d", valueType)
	}
}

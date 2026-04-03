package spartacus

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"runtime"
)

// GGUFMetadata holds model parameters extracted from a GGUF file.
type GGUFMetadata struct {
	Architecture string
	Layers       int
	EmbeddingDim int
	HeadCount    int
	KVHeadCount  int
	ContextSize  int
	FileType     int
	FileSizeBytes int64
}

// ModelMemory returns estimated memory usage for a given context size.
func (m *GGUFMetadata) ModelMemory(ctxSize int) MemoryEstimate {
	if ctxSize <= 0 {
		ctxSize = 8192
	}
	return MemoryEstimate{
		ModelBytes:     m.FileSizeBytes,
		KVPerSlotBytes: m.kvCachePerSlot(ctxSize),
		OverheadBytes:  200 * 1024 * 1024, // ~200MB for compute buffers
	}
}

// kvCachePerSlot calculates KV cache size per slot in bytes.
// Formula: layers × 2(K+V) × ctx × kv_heads × head_dim × bytes_per_element
func (m *GGUFMetadata) kvCachePerSlot(ctxSize int) int64 {
	if m.HeadCount == 0 || m.KVHeadCount == 0 {
		return 0
	}
	headDim := int64(m.EmbeddingDim) / int64(m.HeadCount)
	// Q8_0 quantization ≈ 1 byte per element
	// All operands as int64 to avoid overflow
	return int64(m.Layers) * 2 * int64(ctxSize) * int64(m.KVHeadCount) * headDim
}

// OptimalSlots calculates how many parallel slots fit in available memory.
func (m *GGUFMetadata) OptimalSlots(ctxSize int, availableBytes int64) int {
	mem := m.ModelMemory(ctxSize)
	remaining := availableBytes - mem.ModelBytes - mem.OverheadBytes
	if remaining <= 0 {
		return 1
	}
	kvPerSlot := m.kvCachePerSlot(ctxSize)
	if kvPerSlot <= 0 {
		return 1
	}
	slots := int(remaining / kvPerSlot)
	if slots < 1 {
		return 1
	}
	if slots > 32 {
		return 32
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
		total = detectDarwinMemory()
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
		// Skip array contents (we don't need token arrays)
		for j := uint64(0); j < count; j++ {
			if _, err := readGGUFValue(r, elemType); err != nil {
				return nil, err
			}
		}
		return nil, nil // arrays skipped
	default:
		return nil, fmt.Errorf("unknown GGUF type: %d", valueType)
	}
}

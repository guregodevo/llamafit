package llamafit

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

// Calibration persists per-(model, host, config) Parallel choices so
// repeat boots don't recompute the auto-Parallel formula from scratch.
//
// The design is the Tangram "Profile" phase, adapted for a single-
// node llama.cpp wrapper: the FIRST time we boot with a given
// (modelID, hostRAM, ctxSize, kvCacheType, hasDraft) tuple we let
// the formula in server.New pick Parallel and persist the result
// as "known good" after llama-server reaches /health without OOM.
// Subsequent boots with the same tuple skip the formula entirely
// and reuse the persisted value.
//
// Why persist instead of always re-computing: the formula derives
// from GGUF metadata and host RAM, both of which are stable. The
// only thing that drifts is what we LEARN about the workload — e.g.
// "Parallel=8 has been known-good for 3 consecutive boots, the
// formula was conservative, next boot try 9 and see if it sticks."
// That growth path requires persistence across runs.

// Calibration is the on-disk cache of past auto-Parallel choices.
// Schema versioned so future format changes can be migrated rather
// than throwing the file away.
type Calibration struct {
	SchemaVersion int                `json:"schema_version"`
	Entries       []CalibrationEntry `json:"entries"`
}

// CalibrationEntry is one cached (key, choice, observation) record.
// Key fields (ModelID..HasDraft) must ALL match for the entry to be
// reused on a later boot. The recorded Parallel is what server.New
// chose during the entry's creation boot.
type CalibrationEntry struct {
	// Key — must match for reuse.
	ModelID      string `json:"model_id"`
	HostRAMBytes int64  `json:"host_ram_bytes"`
	OS           string `json:"os"`
	Arch         string `json:"arch"`
	CtxSize      int    `json:"ctx_size"`
	KVCacheType  string `json:"kv_cache_type"`
	HasDraft     bool   `json:"has_draft"`
	DraftSize    int64  `json:"draft_size_bytes,omitempty"`

	// Choice — what server.New ended up with.
	Parallel int `json:"parallel"`

	// Observation — bookkeeping that informs future scale-up probing.
	BootsKnownGood int       `json:"boots_known_good"`
	LastBootAt     time.Time `json:"last_boot_at"`
	CreatedAt      time.Time `json:"created_at"`
}

// modelIdentity returns a stable per-model key combining absolute
// path, file size, and a SHA-256 of the GGUF header (first 64 KB).
func modelIdentity(modelPath string) (string, error) {
	info, err := os.Stat(modelPath)
	if err != nil {
		return "", fmt.Errorf("stat model: %w", err)
	}
	f, err := os.Open(modelPath)
	if err != nil {
		return fmt.Sprintf("path:%s|size:%d", modelPath, info.Size()), nil
	}
	defer f.Close()
	hasher := sha256.New()
	if _, err := io.CopyN(hasher, f, 64*1024); err != nil && err != io.EOF {
		return fmt.Sprintf("path:%s|size:%d", modelPath, info.Size()), nil
	}
	headerHash := hex.EncodeToString(hasher.Sum(nil))
	return fmt.Sprintf("sha256:%s|size:%d", headerHash[:16], info.Size()), nil
}

// DefaultCalibrationPath returns the canonical on-disk location for
// the calibration file.
func DefaultCalibrationPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".aktapus", "llm", "calibration.json")
}

// LoadCalibration reads the calibration file. Missing file returns
// an empty Calibration without error — first-boot is the common case.
func LoadCalibration(path string) (*Calibration, error) {
	if path == "" {
		return &Calibration{SchemaVersion: 1}, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &Calibration{SchemaVersion: 1}, nil
		}
		return nil, fmt.Errorf("read calibration: %w", err)
	}
	var cal Calibration
	if err := json.Unmarshal(data, &cal); err != nil {
		// Corrupted file — start fresh rather than failing every boot.
		return &Calibration{SchemaVersion: 1}, nil
	}
	if cal.SchemaVersion == 0 {
		cal.SchemaVersion = 1
	}
	return &cal, nil
}

// SaveCalibration writes the calibration atomically.
func SaveCalibration(path string, cal *Calibration) error {
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("mkdir calibration dir: %w", err)
	}
	data, err := json.MarshalIndent(cal, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal calibration: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("write calibration tmp: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("rename calibration: %w", err)
	}
	return nil
}

// Find returns the entry matching the given key, or nil if no match.
func (c *Calibration) Find(key CalibrationEntry) *CalibrationEntry {
	for i := range c.Entries {
		e := &c.Entries[i]
		if e.ModelID == key.ModelID &&
			e.HostRAMBytes == key.HostRAMBytes &&
			e.OS == key.OS &&
			e.Arch == key.Arch &&
			e.CtxSize == key.CtxSize &&
			e.KVCacheType == key.KVCacheType &&
			e.HasDraft == key.HasDraft &&
			e.DraftSize == key.DraftSize {
			return e
		}
	}
	return nil
}

// Upsert replaces an existing entry with the same key, or appends.
func (c *Calibration) Upsert(entry CalibrationEntry) {
	for i := range c.Entries {
		e := &c.Entries[i]
		if e.ModelID == entry.ModelID &&
			e.HostRAMBytes == entry.HostRAMBytes &&
			e.OS == entry.OS &&
			e.Arch == entry.Arch &&
			e.CtxSize == entry.CtxSize &&
			e.KVCacheType == entry.KVCacheType &&
			e.HasDraft == entry.HasDraft &&
			e.DraftSize == entry.DraftSize {
			if e.Parallel == entry.Parallel {
				e.BootsKnownGood++
			} else {
				e.BootsKnownGood = 1
				e.Parallel = entry.Parallel
			}
			e.LastBootAt = entry.LastBootAt
			return
		}
	}
	entry.BootsKnownGood = 1
	if entry.CreatedAt.IsZero() {
		entry.CreatedAt = time.Now().UTC()
	}
	c.Entries = append(c.Entries, entry)
}

// buildCalibrationKey constructs the CalibrationEntry key fields.
func buildCalibrationKey(cfg Config) (CalibrationEntry, error) {
	modelID, err := modelIdentity(cfg.ModelPath)
	if err != nil {
		return CalibrationEntry{}, err
	}
	key := CalibrationEntry{
		ModelID:      modelID,
		HostRAMBytes: int64(detectTotalMemory()),
		OS:           runtime.GOOS,
		Arch:         runtime.GOARCH,
		CtxSize:      cfg.CtxSize,
		KVCacheType:  cfg.KVCacheType,
		HasDraft:     cfg.DraftModelPath != "",
	}
	if cfg.DraftModelPath != "" {
		if info, ierr := os.Stat(cfg.DraftModelPath); ierr == nil {
			key.DraftSize = info.Size()
		}
	}
	return key, nil
}

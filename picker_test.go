package llamafit

import (
	"os"
	"path/filepath"
	"testing"
)

// writeFakeGGUF creates a sparse file with .gguf suffix at the
// given path and size. Sparse means we get the right Size() in
// metadata without actually allocating size bytes — important
// because PickDraftModel's selection ratio assumes realistic
// model sizes (GB-scale), and tests would be flaky if forced to
// write multi-GB real files.
func writeFakeGGUF(t *testing.T, dir, name string, size int64) string {
	t.Helper()
	p := filepath.Join(dir, name)
	f, err := os.Create(p)
	if err != nil {
		t.Fatalf("create %s: %v", p, err)
	}
	defer f.Close()
	if err := f.Truncate(size); err != nil {
		t.Fatalf("truncate %s to %d: %v", p, size, err)
	}
	return p
}

func TestPickModel_PicksLargest(t *testing.T) {
	dir := t.TempDir()
	writeFakeGGUF(t, dir, "small.gguf", 1<<30)             // 1 GiB
	want := writeFakeGGUF(t, dir, "large.gguf", 20<<30)    // 20 GiB
	writeFakeGGUF(t, dir, "medium.gguf", 5<<30)            // 5 GiB
	// Non-GGUF files should be ignored.
	writeFakeGGUF(t, dir, "random.bin", 100<<30)

	got, err := PickModel(dir)
	if err != nil {
		t.Fatalf("PickModel: %v", err)
	}
	if got != want {
		t.Errorf("PickModel = %q, want %q", got, want)
	}
}

func TestPickModel_EmptyDir(t *testing.T) {
	dir := t.TempDir()
	_, err := PickModel(dir)
	if err == nil {
		t.Fatal("PickModel on empty dir returned nil error")
	}
}

func TestPickModel_NoSuchDir(t *testing.T) {
	_, err := PickModel("/does/not/exist")
	if err == nil {
		t.Fatal("PickModel on missing dir returned nil error")
	}
}

func TestPickDraftModel_PicksLargestQualifier(t *testing.T) {
	dir := t.TempDir()
	main := writeFakeGGUF(t, dir, "main-32b.gguf", 21<<30) // 21 GiB main
	// Qualifiers (≤ 1/3 of main = 7 GiB):
	tiny := writeFakeGGUF(t, dir, "tiny-0_5b.gguf", 500<<20)   // 500 MiB
	want := writeFakeGGUF(t, dir, "small-1_5b.gguf", 1<<30)    // 1 GiB ← largest qualifier
	// Disqualifier (too big — > 1/3 of main):
	writeFakeGGUF(t, dir, "big-14b.gguf", 9<<30) // 9 GiB

	// Headroom 0 disables the memory guard so this test is
	// independent of host RAM.
	got := PickDraftModel(main, dir, 0)
	if got != want {
		t.Errorf("PickDraftModel = %q, want %q (tiny=%q)", got, want, tiny)
	}
}

func TestPickDraftModel_SkipsMain(t *testing.T) {
	dir := t.TempDir()
	main := writeFakeGGUF(t, dir, "main.gguf", 21<<30)
	// Only file present is the main itself.
	got := PickDraftModel(main, dir, 0)
	if got != "" {
		t.Errorf("PickDraftModel picked the main itself: %q", got)
	}
}

func TestPickDraftModel_AllTooBig(t *testing.T) {
	dir := t.TempDir()
	main := writeFakeGGUF(t, dir, "main-7b.gguf", 5<<30) // 5 GiB
	// All siblings are > 1/3 of main (> ~1.67 GiB).
	writeFakeGGUF(t, dir, "sibling-a.gguf", 3<<30)
	writeFakeGGUF(t, dir, "sibling-b.gguf", 4<<30)

	got := PickDraftModel(main, dir, 0)
	if got != "" {
		t.Errorf("PickDraftModel returned %q when nothing should qualify", got)
	}
}

func TestPickDraftModel_MemoryGuardRejects(t *testing.T) {
	dir := t.TempDir()
	// AvailableMemory() reports total RAM minus a 4 GB OS reserve.
	// On any realistic test host that's at least 4 GiB. We make the
	// (main + draft + headroom) sum exceed that by setting absurd
	// sizes so the guard fires regardless of host RAM.
	main := writeFakeGGUF(t, dir, "main.gguf", 1<<40) // 1 TiB
	writeFakeGGUF(t, dir, "draft.gguf", 100<<30)      // 100 GiB
	// Headroom matters only for the guard math — pass the canonical
	// default so we exercise the same path Aktapus does.
	got := PickDraftModel(main, dir, DefaultDraftHeadroom)
	if got != "" {
		t.Errorf("PickDraftModel returned %q despite obvious memory pressure", got)
	}
}

func TestPickDraftModel_MemoryGuardDisabledWithZeroHeadroom(t *testing.T) {
	dir := t.TempDir()
	// Same impossible sizes as above, but headroom=0 disables the
	// guard so the picker should still return the draft.
	main := writeFakeGGUF(t, dir, "main.gguf", 21<<30)
	want := writeFakeGGUF(t, dir, "draft.gguf", 1<<30)
	got := PickDraftModel(main, dir, 0)
	if got != want {
		t.Errorf("PickDraftModel with headroom=0 returned %q, want %q", got, want)
	}
}

func TestPickDraftModel_NoMainPath(t *testing.T) {
	dir := t.TempDir()
	writeFakeGGUF(t, dir, "draft.gguf", 1<<30)
	got := PickDraftModel("", dir, 0)
	if got != "" {
		t.Errorf("PickDraftModel with empty mainPath returned %q, want \"\"", got)
	}
}

func TestPickModel_UpperCaseExtension(t *testing.T) {
	dir := t.TempDir()
	want := writeFakeGGUF(t, dir, "MODEL.GGUF", 2<<30)
	writeFakeGGUF(t, dir, "small.gguf", 1<<30)
	got, err := PickModel(dir)
	if err != nil || got != want {
		t.Fatalf("PickModel = %q, %v; want %q", got, err, want)
	}
}

// A cache may keep a second name (a symlink) for the very same file.
// That alias is the main model itself and must not become the draft.
func TestPickDraftModel_SkipsSymlinkAliasOfMain(t *testing.T) {
	dir := t.TempDir()
	main := writeFakeGGUF(t, dir, "main-7b.gguf", 21<<30)
	if err := os.Symlink(main, filepath.Join(dir, "main-7b.gguf.link.gguf")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if got := PickDraftModel(main, dir, 0); got != "" {
		t.Errorf("PickDraftModel returned the main model via its symlink alias: %q", got)
	}
}

// A draft reached through a symlink must be sized by its target, so a
// qualifying one still qualifies (lstat would report the link length).
func TestPickDraftModel_SymlinkedDraftIsMeasured(t *testing.T) {
	dir := t.TempDir()
	main := writeFakeGGUF(t, dir, "main-7b.gguf", 21<<30)
	sub := filepath.Join(dir, "cache")
	if err := os.Mkdir(sub, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", sub, err)
	}
	target := writeFakeGGUF(t, sub, "draft-1_5b.gguf", 1<<30)
	link := filepath.Join(dir, "draft-link.gguf")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if got := PickDraftModel(main, dir, 0); got != link {
		t.Errorf("PickDraftModel = %q, want the symlinked qualifying draft %q", got, link)
	}
}

// A broken symlink with a .gguf name must be skipped, not picked.
func TestPickDraftModel_SkipsBrokenSymlink(t *testing.T) {
	dir := t.TempDir()
	main := writeFakeGGUF(t, dir, "main-7b.gguf", 21<<30)
	if err := os.Symlink(filepath.Join(dir, "gone.gguf"), filepath.Join(dir, "broken.gguf")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if got := PickDraftModel(main, dir, 0); got != "" {
		t.Errorf("PickDraftModel picked a broken symlink: %q", got)
	}
}

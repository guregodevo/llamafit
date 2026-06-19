package llamafit

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Hugging Face direct loading. Llamafit's auto-tuning (slot count, KV-cache
// estimate, calibration) all read the GGUF *before* launch, so it needs the
// file locally. Rather than forward llama-server's -hf flag blindly (which
// would launch but skip all tuning), llamafit resolves an "user/repo[:quant]"
// spec to a cached local path first — download-then-tune — and then runs the
// normal local-file flow unchanged.

const hfEndpoint = "https://huggingface.co"

// hfCacheDir returns the base directory for downloaded GGUFs. It follows
// llama.cpp's own convention so a model pulled here shares a home with one
// pulled by `llama-server -hf`: $LLAMA_CACHE if set, else the platform cache
// dir (~/Library/Caches/llama.cpp on macOS, ~/.cache/llama.cpp elsewhere).
func hfCacheDir() string {
	if c := os.Getenv("LLAMA_CACHE"); c != "" {
		return c
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "llama.cpp")
	}
	if runtime.GOOS == "darwin" {
		return filepath.Join(home, "Library", "Caches", "llama.cpp")
	}
	return filepath.Join(home, ".cache", "llama.cpp")
}

// parseHFRepo splits "user/repo[:quant]" into its repo and (optional) quant.
func parseHFRepo(spec string) (repo, quant string, err error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return "", "", fmt.Errorf("empty Hugging Face repo spec")
	}
	repo = spec
	if i := strings.LastIndex(spec, ":"); i >= 0 {
		repo, quant = spec[:i], spec[i+1:]
	}
	if strings.Count(repo, "/") != 1 || strings.HasPrefix(repo, "/") || strings.HasSuffix(repo, "/") {
		return "", "", fmt.Errorf("Hugging Face repo %q must be in \"user/repo\" form", spec)
	}
	return repo, quant, nil
}

// hfToken resolves the access token: the explicit value, else $HF_TOKEN.
func hfToken(explicit string) string {
	if explicit != "" {
		return explicit
	}
	return os.Getenv("HF_TOKEN")
}

// isShardedGGUF reports whether a filename looks like one part of a
// multi-part GGUF (e.g. model-00001-of-00003.gguf).
func isShardedGGUF(name string) bool {
	return strings.Contains(strings.ToLower(name), "-of-")
}

// hfListGGUFs returns the .gguf filenames in a repo via the HF model API.
func hfListGGUFs(ctx context.Context, client *http.Client, repo, token string) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, hfEndpoint+"/api/models/"+repo, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("User-Agent", "llamafit")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HF API for %q returned %s", repo, resp.Status)
	}

	var info struct {
		Siblings []struct {
			RFilename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("decode HF API response: %w", err)
	}
	var ggufs []string
	for _, s := range info.Siblings {
		if strings.HasSuffix(strings.ToLower(s.RFilename), ".gguf") {
			ggufs = append(ggufs, s.RFilename)
		}
	}
	return ggufs, nil
}

// hfSelectFile picks one GGUF from a repo's listing. An explicit file wins;
// otherwise a quant is matched case-insensitively (preferring a single-file
// over a sharded match); a repo with exactly one GGUF needs neither.
func hfSelectFile(ggufs []string, quant, explicit string) (string, error) {
	if explicit != "" {
		return explicit, nil
	}
	if len(ggufs) == 0 {
		return "", fmt.Errorf("repo has no .gguf files")
	}
	if quant != "" {
		q := strings.ToLower(quant)
		var matches []string
		for _, f := range ggufs {
			if strings.Contains(strings.ToLower(f), q) {
				matches = append(matches, f)
			}
		}
		if len(matches) == 0 {
			return "", fmt.Errorf("no .gguf matching quant %q (available: %s)", quant, strings.Join(ggufs, ", "))
		}
		for _, f := range matches {
			if !isShardedGGUF(f) {
				return f, nil
			}
		}
		return matches[0], nil
	}
	if len(ggufs) == 1 {
		return ggufs[0], nil
	}
	return "", fmt.Errorf("repo has %d GGUFs; specify a quant or file (available: %s)", len(ggufs), strings.Join(ggufs, ", "))
}

// hfDownload streams repo/file to dest atomically (temp + rename), so a
// present dest is always a complete file.
func hfDownload(ctx context.Context, client *http.Client, repo, file, token, dest string, log *slog.Logger) error {
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return fmt.Errorf("mkdir HF cache dir: %w", err)
	}
	url := hfEndpoint + "/" + repo + "/resolve/main/" + file + "?download=true"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("User-Agent", "llamafit")

	log.Info("llamafit downloading HF model", slog.String("repo", repo), slog.String("file", file))
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: %s", url, resp.Status)
	}

	tmp := dest + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("create temp download: %w", err)
	}
	n, copyErr := io.Copy(f, resp.Body)
	closeErr := f.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("download %s: %w", file, copyErr)
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("flush download: %w", closeErr)
	}
	if err := os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("finalize download: %w", err)
	}
	log.Info("llamafit HF model ready", slog.String("path", dest), slog.String("size", formatBytes(n)))
	return nil
}

// EnsureHFModel resolves a Hugging Face GGUF repo spec ("user/repo[:quant]")
// to a local cached file path, downloading it on first use and reusing the
// cached copy thereafter. file optionally names the exact GGUF (overriding
// :quant); token (or $HF_TOKEN) authenticates to gated/private repos. The
// cache base follows llama.cpp (see hfCacheDir). Returns the local path.
//
// This is the standalone form of what Config.HFRepo does inside New — useful
// for pre-warming a cache or resolving a model without starting a server.
func EnsureHFModel(ctx context.Context, repoSpec, file, token string) (string, error) {
	repo, quant, err := parseHFRepo(repoSpec)
	if err != nil {
		return "", err
	}
	token = hfToken(token)
	client := &http.Client{} // no overall timeout: GGUFs are large; ctx cancels

	chosen := file
	if chosen == "" {
		ggufs, err := hfListGGUFs(ctx, client, repo, token)
		if err != nil {
			return "", fmt.Errorf("list HF repo %s: %w", repo, err)
		}
		chosen, err = hfSelectFile(ggufs, quant, "")
		if err != nil {
			return "", fmt.Errorf("select GGUF in %s: %w", repo, err)
		}
	}
	if isShardedGGUF(chosen) {
		return "", fmt.Errorf("HF file %q is a multi-part (sharded) GGUF, not yet supported; pick a single-file quant", chosen)
	}

	dest := filepath.Join(hfCacheDir(), "llamafit", strings.ReplaceAll(repo, "/", "_"), chosen)
	if fi, err := os.Stat(dest); err == nil && fi.Size() > 0 {
		return dest, nil // already cached
	}
	if err := hfDownload(ctx, client, repo, chosen, token, dest, slog.Default()); err != nil {
		return "", err
	}
	return dest, nil
}

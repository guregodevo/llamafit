package llamafit

import (
	"net/http"
	"os"
	"path/filepath"
)

// ServeModelFile returns a model file from the models directory, so a client
// can check what is installed.
func ServeModelFile(modelsDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		name := r.URL.Query().Get("name")
		data, err := os.ReadFile(filepath.Join(modelsDir, name))
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		w.Write(data)
	}
}

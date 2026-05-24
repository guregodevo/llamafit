MODEL ?=
PORT  ?= 8081

.PHONY: build serve inspect test clean

build:
	# -buildvcs=false avoids `bad CPU type in executable` when an x86_64
	# Homebrew git lands first on PATH on Apple Silicon. VCS stamping
	# isn't load-bearing (the module proxy carries the source hash) and
	# skipping it lets the build succeed cleanly on every Mac.
	go build -buildvcs=false -o spartacus ./cmd/spartacus

serve: build
ifndef MODEL
	$(error MODEL is required. Usage: make serve MODEL=path/to/model.gguf)
endif
	./spartacus --model $(MODEL) --port $(PORT)

inspect: build
ifndef MODEL
	$(error MODEL is required. Usage: make inspect MODEL=path/to/model.gguf)
endif
	./spartacus --model $(MODEL) --inspect

test:
	go test -v ./...

clean:
	rm -f spartacus

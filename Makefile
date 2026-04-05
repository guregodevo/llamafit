MODEL ?=
PORT  ?= 8081

.PHONY: build serve inspect test clean

build:
	go build -o spartacus ./cmd/spartacus

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

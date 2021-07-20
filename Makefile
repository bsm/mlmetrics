default: test

test:
	go test ./...

bench:
	go test ./... -run=NONE -bench=. -benchmem

lint:
	golangci-lint run

doc: README.md

README.md: README.md.tpl $(wildcard *.go)
	becca -package github.com/bsm/mlmetrics

default: vet test

vet:
	go vet ./...

test:
	go test ./...

bench:
	go test ./... -test.run=NONE -test.bench=. -benchmem -benchtime=1s

.PHONY: vet test bench

doc: README.md

README.md: README.md.tpl $(wildcard *.go)
	becca -package $(subst $(GOPATH)/src/,,$(PWD))

.PHONY: doc

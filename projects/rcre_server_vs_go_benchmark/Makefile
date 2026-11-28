# RCRE (C23) vs Go (1.27) High-Performance Web Server Benchmark Suite

CC        ?= clang
RCRE_ROOT := ../..
RCRE_INC  := $(RCRE_ROOT)/include
RCRE_LIB  := $(RCRE_ROOT)/lib/librcre.a

C_SRCS := server-rcre/src/main.c \
          server-rcre/src/http_parser.c \
          server-rcre/src/handlers.c

CFLAGS_COMMON := -std=c2x -Wall -Wextra -Werror -I$(RCRE_INC) -pthread
CFLAGS_REL    := $(CFLAGS_COMMON) -O3 -flto -mcpu=native -DNDEBUG
CFLAGS_ASAN   := $(CFLAGS_COMMON) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
LDFLAGS       := $(RCRE_LIB) -lm -lpthread

BIN_DIR     := bin
RCRE_BIN    := $(BIN_DIR)/server-rcre
RCRE_ASAN   := $(BIN_DIR)/server-rcre-asan
GO_BIN      := $(BIN_DIR)/server-go

.DEFAULT_GOAL := help
.PHONY: help all fixtures build build-rcre build-rcre-asan build-go verify bench clean test san

help:
	@echo "rcre-server-bench — High-Performance C23 Reactor vs Go Web Server Suite:"
	@echo "  make build         Build C23 Release, ASan/UBSan, and Go server binaries"
	@echo "  make verify        Run ASan safety gate + HTTP/1.1 conformance & parity checks"
	@echo "  make bench         Run full benchmark matrix (Phases A-D) & generate report"
	@echo "  make fixtures      Regenerate deterministic test fixtures (10k docs, GBDT, etc.)"
	@echo "  make test          Alias for make verify"
	@echo "  make san           Alias for make verify (runs ASan/UBSan server validation)"
	@echo "  make clean         Remove compiled binaries and benchmark results"

all: fixtures build

$(RCRE_LIB):
	@$(MAKE) -C $(RCRE_ROOT) lib/librcre.a

fixtures:
	python3 spec/generate_fixtures.py

build: build-rcre build-rcre-asan build-go

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

build-rcre: $(RCRE_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS_REL) $(C_SRCS) -o $(RCRE_BIN) $(LDFLAGS)

build-rcre-asan: $(RCRE_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS_ASAN) $(C_SRCS) -o $(RCRE_ASAN) $(LDFLAGS)

build-go: | $(BIN_DIR)
	cd server-go && GOWORK=off go build -o ../$(GO_BIN) ./cmd/server

verify: build
	python3 bench/run_verify.py

test: verify

san: verify

bench: build
	python3 bench/run_benchmark.py

clean:
	rm -rf $(BIN_DIR) results

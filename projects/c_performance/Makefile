CC ?= cc
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -fPIC
INCLUDES = -I../../include -I../../src/hal -Iinclude
LIBS = ../../lib/librcre.a -lm -lpthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIBS += -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework Foundation -framework Cocoa
endif

BIN = bin/rcre-perf
SRCS = src/main.c src/bench_cache.c src/bench_strides.c src/bench_simd.c

.PHONY: all test demo bench clean

all: $(BIN)

bin:
	@mkdir -p bin

$(BIN): bin $(SRCS) include/rcre_perf.h ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) $(SRCS) $(LIBS) -o $@
	@echo "✅ Built $@"

test: $(BIN)
	@echo "🧪 Running RCRE-PERF test verification (1024x1024 matrix)..."
	@./$(BIN) --all --matrix-size 1024 --iterations 1

demo: $(BIN)
	@echo "🚀 Running RCRE-PERF live demo (2048x2048 matrix)..."
	@./$(BIN) --all --matrix-size 2048 --iterations 2

bench: $(BIN)
	@echo "⚡ Running RCRE-PERF full benchmark (4096x4096 with Perfetto trace)..."
	@./$(BIN) --all --matrix-size 4096 --iterations 3 --perfetto bin/perf_trace.json --html bin/perf_dashboard.html

clean:
	@rm -rf bin *.o *.json *.html
	@echo "🧹 Cleaned RCRE-PERF build artifacts"

# ============================================================================
#  RCRE-SR — Document Image Super-Resolution (built on RCRE)
#  Zero external dependencies. C23. Portable (clang/gcc), M5-friendly.
# ============================================================================

CC      ?= cc
AR      ?= ar
CFLAGS  ?= -std=c2x -O3 -Wall -Wextra
LDFLAGS ?=
LDLIBS  := -lm -lpthread

# RCRE single header: prefer dist, else fallback to third_party
RCRE_DIR ?= ../../dist
ifeq ($(wildcard $(RCRE_DIR)/rcre.h),)
RCRE_DIR := third_party
endif

INCLUDES := -Iinclude -I$(RCRE_DIR)

SRCS := src/sr_glue.c \
        src/sr_io.c \
        src/sr_metrics.c \
        src/sr_assess.c \
        src/sr_resize.c \
        src/sr_enhance.c \
        src/sr_restore.c \
        src/sr_pipeline.c

OBJ  := $(SRCS:src/%.c=obj/%.o)
LIB  := lib/libsr.a
BIN  := bin
TESTS := bin/test_sr
BENCH := bin/bench_sr
CLI   := bin/sr_cli

.PHONY: all clean test demo bench cli samples web

all: $(LIB) $(CLI) $(BENCH) $(TESTS)
	@echo "✅ RCRE-SR built (RCRE header: $(RCRE_DIR)/rcre.h)"

$(LIB): $(OBJ)
	@mkdir -p lib
	$(AR) rcs $@ $(OBJ)

obj/%.o: src/%.c include/sr.h
	@mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(CLI): examples/sr_cli.c $(LIB)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

$(BENCH): benchmarks/bench_sr.c $(LIB)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

$(TESTS): tests/test_sr.c $(LIB)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

test: $(TESTS)
	@$(TESTS)

demo: $(CLI)
	@$(CLI) demo

bench: $(BENCH)
	@$(BENCH)

cli: $(CLI)

# Generate demo sample images (synthetic sharp document + degraded version)
samples: $(CLI)
	@$(CLI) generate samples/original.pgm samples/degraded.pgm 1200 1600
	@echo "✅ samples/original.pgm + samples/degraded.pgm written"

# Web demo: python3 stdlib only (no pip dependencies)
web: $(CLI)
	@cd web && python3 server.py

clean:
	rm -rf obj lib $(BIN)

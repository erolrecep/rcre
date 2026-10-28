# ============================================================================
#  RCRE-XGB — XGBoost-style gradient boosted trees (built on RCRE)
#  Zero external dependencies. C23. Portable (clang/gcc), M5-friendly.
# ============================================================================

CC      ?= cc
AR      ?= ar
CFLAGS  ?= -std=c2x -O3 -Wall -Wextra
LDFLAGS ?=
LDLIBS  := -lm -lpthread

# RCRE single header: prefer the parent repo's fresh dist, else vendored copy
RCRE_DIR ?= ../../dist
ifeq ($(wildcard $(RCRE_DIR)/rcre.h),)
RCRE_DIR := third_party
endif

INCLUDES := -Iinclude -Isrc -I$(RCRE_DIR)

SRCS := src/xgb_glue.c \
        src/xgb_core.c \
        src/xgb_boost.c \
        src/xgb_predict.c \
        src/xgb_io.c \
        src/xgb_metrics.c

OBJ  := $(SRCS:src/%.c=obj/%.o)
LIB  := lib/libxgb.a
BIN  := bin
TESTS := bin/test_xgb
BENCH := bin/bench_xgb
CLI   := bin/xgb_cli

.PHONY: all clean test bench cli demo

all: $(LIB) $(CLI) $(BENCH) $(TESTS)
	@echo "✅ RCRE-XGB built (RCRE header: $(RCRE_DIR)/rcre.h)"

$(LIB): $(OBJ)
	@mkdir -p lib
	$(AR) rcs $@ $(OBJ)

obj/%.o: src/%.c include/xgb.h src/xgb_internal.h
	@mkdir -p obj
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(CLI): examples/xgb_cli.c $(LIB)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

$(BENCH): benchmarks/bench_xgb.c $(LIB)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

$(TESTS): tests/test_xgb.c $(LIB)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(LIB) $(LDFLAGS) $(LDLIBS) -o $@

test: $(TESTS)
	@$(TESTS)

bench: $(BENCH)
	@$(BENCH)

cli: $(CLI)

demo: $(CLI)
	@$(CLI) demo

clean:
	rm -rf obj lib $(BIN)

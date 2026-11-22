# ===========================================================================
# rcre_e-means — e-means (Bessa et al. 2026) as an RCRE operator
#
# The operator itself lives in the CORE library (../../src/ml/e_means.c,
# API in ../../include/rcre.h) — this project is the development harness:
# the Python oracle (ref/), the stream fixtures (streams/), the
# oracle-dependent test suites (tests/), the benchmark (bench/) and the
# demo (examples/). Everything here builds against the core librcre.a.
#
# Control center. `make` (no target) shows help.
#
#   make help          show all targets
#   make all           build tests + bench + demo (vs the core library)
#   make test          build + run all C test suites
#   make san           build + run all suites under ASan + UBSan
#   make parity        build + run strict-parity tests (needs ref streams;
#                      generates them via `make refs` when missing)
#   make refs          (re)generate the Python oracle streams + recipes
#   make quality-par   statistical quality parity vs the Python oracle
#                      (needs the numpy venv; skips cleanly without it)
#   make bench         build + run the benchmark (vs RCRE Lloyd)
#   make clean         remove build artifacts
# ===========================================================================

.DEFAULT_GOAL := help

CC      ?= cc
PYTHON  ?= python3
VENV    := .venv
VENV_PY := $(VENV)/bin/python

# --- core RCRE ----------------------------------------------------------------
RCRE     := ../../
CORE_LIB := $(RCRE)/lib/librcre.a
# Re-built with ASan in the san target (coverage of the operator itself).
CORE_EM  := $(RCRE)/src/ml/e_means.c

INCLUDES := -I. -Istreams -I$(RCRE)/include -I$(RCRE)/src/ml
# -ffp-contract=off: the strict-parity contract assumes no FMA contraction
# (the core builds e_means.o the same way).
CFLAGS   ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -ffp-contract=off
CFLAGS   += $(INCLUDES)
LDFLAGS  := -lm -lpthread

TESTS     := test_recipe test_parity test_assign test_determinism \
             test_quality test_errors
TEST_BINS := $(addprefix build/,$(addsuffix .bin,$(TESTS)))
BENCH_BIN := build/bench_e_means.bin
DEMO_BIN  := build/emeans_demo.bin

STREAMS_DIR := streams

# Best-effort oracle generation: needs numpy (venv preferred). Writes
# streams/generated.h with EM_STREAMS_AVAILABLE 1 on success, 0 (degraded
# coverage, suites skip the oracle gates) when no numpy is reachable.
streams/generated.h: ref/ref_emeans.py
	@mkdir -p $(STREAMS_DIR)
	@if [ -x $(VENV_PY) ] && $(VENV_PY) -c 'import numpy' 2>/dev/null; then py=$(VENV_PY); \
	elif $(PYTHON) -c 'import numpy' 2>/dev/null; then py=$(PYTHON); \
	else py=; fi; \
	if [ -n "$$py" ] && [ ! -f $(STREAMS_DIR)/avail -o ref/ref_emeans.py -nt $(STREAMS_DIR)/avail ]; then \
		if $$py ref/ref_emeans.py --out $(STREAMS_DIR); then \
			touch $(STREAMS_DIR)/avail; \
			echo "#define EM_STREAMS_AVAILABLE 1" > $(STREAMS_DIR)/generated.h; \
			echo "oracle artifacts written to $(STREAMS_DIR)/"; \
		else \
			echo "#define EM_STREAMS_AVAILABLE 0" > $(STREAMS_DIR)/generated.h; \
			echo "WARN: oracle generation failed — degraded coverage"; \
		fi; \
	else \
		if [ -z "$$py" ]; then \
			echo "#define EM_STREAMS_AVAILABLE 0" > $(STREAMS_DIR)/generated.h; \
			echo "WARN: no numpy reachable ($(VENV_PY) / $(PYTHON)) — degraded coverage; make venv to fix"; \
		else \
			[ -f $(STREAMS_DIR)/generated.h ] || echo "#define EM_STREAMS_AVAILABLE 1" > $(STREAMS_DIR)/generated.h; \
		fi; \
	fi

.PHONY: help all clean test san parity refs quality-par bench demo venv

help:
	@echo "rcre_e-means — targets (operator: $(RCRE)src/ml/e_means.c):"
	@echo "  make all           build tests + bench + demo (vs the core library)"
	@echo "  make test          build + run all C test suites"
	@echo "  make demo          build + run the visual ASCII cluster demo"
	@echo "  make san           build + run all suites under ASan + UBSan"
	@echo "  make refs          generate Python oracle streams (needs $(VENV_PY))"
	@echo "  make parity        run strict-parity tests (auto-generates streams)"
	@echo "  make quality-par   statistical quality parity vs oracle (skip-clean)"
	@echo "  make bench         build + run benchmark vs RCRE Lloyd"
	@echo "  make venv          create the numpy venv at ./.venv"
	@echo "  make clean         remove build artifacts"

# Build the core library on demand.
$(CORE_LIB):
	@$(MAKE) -C $(RCRE) lib/librcre.a

all: streams/generated.h $(CORE_LIB) $(TEST_BINS) $(BENCH_BIN) $(DEMO_BIN)

build/%.bin: tests/%.c $(CORE_LIB)
	@mkdir -p build
	$(CC) $(CFLAGS) -O2 $< $(CORE_LIB) $(LDFLAGS) -o $@

$(BENCH_BIN): bench/bench_e_means.c $(CORE_LIB)
	@mkdir -p build
	$(CC) $(CFLAGS) $< $(CORE_LIB) $(LDFLAGS) -o $@

$(DEMO_BIN): examples/emeans_demo.c $(CORE_LIB)
	@mkdir -p build
	$(CC) $(CFLAGS) $< $(CORE_LIB) $(LDFLAGS) -o $@

demo: $(DEMO_BIN)
	@./$(DEMO_BIN)

# --- tests -----------------------------------------------------------------

test: all
	@echo "== rcre_e-means test suites =="
	@ok=0; fail=0; \
	for t in $(TESTS); do \
		if ./build/$$t.bin; then ok=$$((ok+1)); else fail=$$((fail+1)); fi; \
	done; \
	echo " test suites: $$ok passed, $$fail failed"; \
	test $$fail -eq 0

parity: all
	@if [ ! -f $(STREAMS_DIR)/strict_norev.json ]; then $(MAKE) refs; fi
	@./build/test_parity.bin || exit 1

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --quiet --upgrade pip
	$(VENV_PY) -m pip install --quiet "numpy<3"
	@echo "venv ready: $(VENV_PY)"

# Force regeneration of all oracle artifacts (hard-fails without numpy).
refs: venv
	$(VENV_PY) ref/ref_emeans.py --out $(STREAMS_DIR) || { echo "#define EM_STREAMS_AVAILABLE 0" > $(STREAMS_DIR)/generated.h; exit 1; }
	@touch $(STREAMS_DIR)/avail
	@echo "#define EM_STREAMS_AVAILABLE 1" > $(STREAMS_DIR)/generated.h
	@echo "oracle streams + recipes written to $(STREAMS_DIR)/"

bench: $(BENCH_BIN)
	@echo "== benchmark (e-means serial | pooled | lloyd-25) =="
	@./$(BENCH_BIN)

quality-par: all
	@if [ ! -f $(STREAMS_DIR)/fullrun.json ]; then $(MAKE) refs; fi
	@./build/test_quality.bin || exit 1
	@if [ -x $(VENV_PY) ] && $(VENV_PY) -c 'import numpy' 2>/dev/null; then py=$(VENV_PY); \
	elif $(PYTHON) -c 'import numpy' 2>/dev/null; then py=$(PYTHON); else py=; fi; \
	if [ -n "$$py" ]; then \
		$$py ref/compare_parity.py $(STREAMS_DIR) || exit 1; \
	else \
		echo "WARN: no numpy — skipping statistical parity comparison (degraded coverage)"; \
	fi

# --- sanitizers ------------------------------------------------------------
# The core e_means.c is recompiled here (with ASan) so the operator itself
# is instrumented; everything else links from the (non-instrumented) core
# library.

SANFLAGS := -fsanitize=address,undefined -fno-sanitize-recover=all -g -O1

san: all
	@echo "== sanitizer build (all suites) =="
	@for t in recipe parity assign determinism quality errors; do \
		$(CC) $(CFLAGS) $(SANFLAGS) $(CORE_EM) tests/test_$$t.c $(CORE_LIB) $(LDFLAGS) -o build/san_$$t.bin || exit 1; \
		if [ $$t = parity ] || [ $$t = quality ]; then \
			[ -f $(STREAMS_DIR)/strict_norev.json ] || $(MAKE) refs; \
		fi; \
		./build/san_$$t.bin || exit 1; \
	done
	@echo " SAN_EXIT=0"

clean:
	rm -rf build
	rm -f tests/*.o src/*.o

CC ?= clang
AR ?= ar
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -Iinclude -Isrc/hal -fPIC
SANITIZER_FLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O1 -g -fsanitize=address,undefined -Iinclude -Isrc/hal

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CAM_SRCS = src/extras/cam_core.c src/extras/cam_dummy.c src/extras/cam_avf.m src/extras/gui_mac.m
    CAM_OBJS = src/extras/cam_core.o src/extras/cam_dummy.o src/extras/cam_avf.o src/extras/gui_mac.o
    CAM_LDFLAGS = -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework Foundation -framework Cocoa
    OPEN_CMD = open
else ifeq ($(UNAME_S),Linux)
    CAM_SRCS = src/extras/cam_core.c src/extras/cam_dummy.c src/extras/cam_v4l2.c src/extras/gui_stub.c
    CAM_OBJS = src/extras/cam_core.o src/extras/cam_dummy.o src/extras/cam_v4l2.o src/extras/gui_stub.c
    CAM_LDFLAGS =
    OPEN_CMD = xdg-open
else
    CAM_SRCS = src/extras/cam_core.c src/extras/cam_dummy.c src/extras/gui_stub.c
    CAM_OBJS = src/extras/cam_core.o src/extras/cam_dummy.o src/extras/gui_stub.c
    CAM_LDFLAGS =
    OPEN_CMD = echo
endif

SRCS = src/core/arena.c \
       src/core/tensor.c \
       src/core/adapt.c \
       src/core/threadpool.c \
       src/cv/color.c \
       src/cv/filter.c \
       src/ml/linear.c \
       src/ml/cluster.c \
       src/ml/gbdt_infer.c \
       src/ml/hnsw.c \
       src/hal/cpu_probe.c \
       src/hal/hal_dispatch.c \
       src/hal/kernels_scalar.c \
       src/hal/kernels_neon.c \
       src/hal/kernels_avx2.c \
       src/hal/kernels_wasm.c

OBJS = $(SRCS:.c=.o)
ALL_OBJS = $(OBJS) $(CAM_OBJS)
LIB = lib/librcre.a
DYLIB = lib/librcre.dylib
AMALGAMATED_HEADER = dist/rcre.h

TESTS = test_arena test_tensor_strides test_adapt_spec test_threadpool test_cv_filters test_ml_linear test_hal test_cluster test_gbdt test_hnsw test_cam test_amalgamated
TEST_BINS = $(addprefix bin/,$(TESTS))
SAN_BINS = $(addprefix bin/,$(addsuffix _asan,$(TESTS)))

EXAMPLES = webcam_edge_filter tabular_gbdt_predict hnsw_vector_search
EXAMPLE_BINS = $(addprefix bin/,$(EXAMPLES))

.DEFAULT_GOAL := help

.PHONY: default help all clean test test-sanitizers test-parity test-all bench examples dist up_web_app webcam_test webcam_test_gui webcam_test_terminal webcam_test_dummy gbdt_test hnsw_test project_lens_test project_lens_demo project_facex_test project_facex_demo

default: help

all: $(LIB) $(DYLIB) $(AMALGAMATED_HEADER) $(EXAMPLE_BINS)

dist: $(AMALGAMATED_HEADER)

$(AMALGAMATED_HEADER): $(SRCS) include/rcre.h scripts/amalgamate.py
	@mkdir -p dist
	python3 scripts/amalgamate.py

$(LIB): $(ALL_OBJS)
	@mkdir -p lib
	$(AR) rcs $@ $^
	@echo "✅ Built $@"

$(DYLIB): $(ALL_OBJS)
	@mkdir -p lib
	$(CC) -shared $(CFLAGS) $^ $(CAM_LDFLAGS) -lm -lpthread -o $@
	@echo "✅ Built $@"

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

src/extras/%.o: src/extras/%.c
	$(CC) $(CFLAGS) -Isrc/extras -c $< -o $@

src/extras/%.o: src/extras/%.m
	$(CC) $(CFLAGS) -Isrc/extras -fobjc-arc -c $< -o $@

bin/test_amalgamated: tests/test_amalgamated.c $(AMALGAMATED_HEADER)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -lm -lpthread -o $@

bin/test_%: tests/test_%.c $(LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -Llib -lrcre $(CAM_LDFLAGS) -lm -lpthread -o $@

bin/%: examples/%.c $(LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -Llib -lrcre $(CAM_LDFLAGS) -lm -lpthread -o $@

examples: $(EXAMPLE_BINS)
	@echo "✅ Built all RCRE examples in bin/"

# =============================================================================
# Interactive Developer Operations
# =============================================================================

up_web_app:
	@echo "🌐 Launching RCRE Educational Suite & In-Browser Playground..."
	@echo "   Serving at: http://localhost:8080"
	@($(OPEN_CMD) http://localhost:8080 2>/dev/null &) || true
	@python3 -m http.server 8080 --directory web/

# Optional runtime parameters (calibrated to MacBook 16:10 aspect ratio)
FRAMES ?= 200
WIDTH ?= 768
HEIGHT ?= 480

# Support lowercase parameter overrides passed on command line (e.g. make width=640)
ifdef width
    WIDTH = $(width)
endif
ifdef height
    HEIGHT = $(height)
endif
ifdef frames
    FRAMES = $(frames)
endif

ARGS ?=

webcam_test: bin/webcam_edge_filter
	@bin/webcam_edge_filter --width $(WIDTH) --height $(HEIGHT) --frames $(FRAMES) $(ARGS)

webcam_test_gui: bin/webcam_edge_filter
	@echo "📷 Launching RCRE Live Camera Stream (16:10 MacBook Aspect)..."
	@echo "   👉 Controls: Spacebar to toggle views (Camera <-> Edge <-> Split), 'f' for fullscreen, 'q' to stop"
	@bin/webcam_edge_filter --gui --width $(WIDTH) --height $(HEIGHT) --frames $(FRAMES) $(ARGS)

webcam_test_terminal: bin/webcam_edge_filter
	@echo "📷 Launching RCRE Real-time Camera Stream in Terminal (ASCII/ANSI)..."
	@bin/webcam_edge_filter --terminal --width $(WIDTH) --height $(HEIGHT) --frames $(FRAMES) $(ARGS)

webcam_test_dummy: bin/webcam_edge_filter
	@echo "📷 Launching RCRE Synthetic Test Camera Stream (16:10 MacBook Aspect)..."
	@bin/webcam_edge_filter --dummy --terminal --width $(WIDTH) --height $(HEIGHT) --frames $(FRAMES) $(ARGS)

gbdt_test: bin/tabular_gbdt_predict
	@bin/tabular_gbdt_predict

hnsw_test: bin/hnsw_vector_search
	@bin/hnsw_vector_search

project_lens_test: $(LIB)
	@$(MAKE) -C projects/reverse_image_search test

project_lens_demo: $(LIB)
	@$(MAKE) -C projects/reverse_image_search demo

project_facex_test: $(LIB)
	@$(MAKE) -C projects/facial_expression_analysis test

project_facex_demo: $(LIB)
	@$(MAKE) -C projects/facial_expression_analysis demo

bin/bench_hal: benchmarks/bench_hal.c $(LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -Llib -lrcre -lm -lpthread -o $@

bench: bin/bench_hal
	@bin/bench_hal

# =============================================================================
# Verification & Testing Suites
# =============================================================================

test: $(LIB) $(AMALGAMATED_HEADER) $(TEST_BINS)
	@echo "🚀 Running RCRE Full Unit Test Suite (Core, Camera & Amalgamated)..."
	@bin/test_arena
	@bin/test_tensor_strides
	@bin/test_adapt_spec
	@bin/test_threadpool
	@bin/test_cv_filters
	@bin/test_ml_linear
	@bin/test_hal
	@bin/test_cluster
	@bin/test_gbdt
	@bin/test_hnsw
	@bin/test_cam
	@bin/test_amalgamated
	@echo "🎉 All RCRE Unit Tests Passed!"

test-sanitizers: $(AMALGAMATED_HEADER)
	@mkdir -p bin
	@echo "🛡️ Compiling and running tests with AddressSanitizer & UBSan..."
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_arena.c -lm -lpthread -o bin/test_arena_asan
	@bin/test_arena_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_tensor_strides.c -lm -lpthread -o bin/test_tensor_strides_asan
	@bin/test_tensor_strides_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_adapt_spec.c -lm -lpthread -o bin/test_adapt_spec_asan
	@bin/test_adapt_spec_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_threadpool.c -lm -lpthread -o bin/test_threadpool_asan
	@bin/test_threadpool_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_cv_filters.c -lm -lpthread -o bin/test_cv_filters_asan
	@bin/test_cv_filters_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_ml_linear.c -lm -lpthread -o bin/test_ml_linear_asan
	@bin/test_ml_linear_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_hal.c -lm -lpthread -o bin/test_hal_asan
	@bin/test_hal_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_cluster.c -lm -lpthread -o bin/test_cluster_asan
	@bin/test_cluster_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_gbdt.c -lm -lpthread -o bin/test_gbdt_asan
	@bin/test_gbdt_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_hnsw.c -lm -lpthread -o bin/test_hnsw_asan
	@bin/test_hnsw_asan
ifeq ($(UNAME_S),Darwin)
	$(CC) $(SANITIZER_FLAGS) $(SRCS) $(CAM_SRCS) tests/test_cam.c -Isrc/extras -fobjc-arc $(CAM_LDFLAGS) -lm -lpthread -o bin/test_cam_asan
else
	$(CC) $(SANITIZER_FLAGS) $(SRCS) $(CAM_SRCS) tests/test_cam.c -Isrc/extras $(CAM_LDFLAGS) -lm -lpthread -o bin/test_cam_asan
endif
	@bin/test_cam_asan
	$(CC) $(SANITIZER_FLAGS) tests/test_amalgamated.c -lm -lpthread -o bin/test_amalgamated_asan
	@bin/test_amalgamated_asan
	@echo "🎉 Zero Memory Leaks & Zero Undefined Behavior Detected!"

test-parity: $(DYLIB)
	@echo "🐍 Running Python Parity Tests against OpenCV & Scikit-Learn..."
	uv run --with numpy,opencv-python,scikit-learn,scipy python tests/test_against_scipy_opencv.py
	@echo "🐍 Running Python FFI Bindings Test..."
	uv run --with numpy python tests/test_python_bindings.py

test-all: test test-sanitizers test-parity
	@echo ""
	@echo "=================================================================="
	@echo "🎉 ALL VERIFICATION LAYERS (Units, Sanitizers, Parity) PASSED!  "
	@echo "=================================================================="

clean:
	rm -rf lib bin dist src/core/*.o src/cv/*.o src/ml/*.o src/hal/*.o src/extras/*.o
	@echo "🧹 Cleaned all build artifacts"

docs:
	@echo "📖 Generating C23 API documentation from headers..."
	python3 scripts/generate_docs.py
	@echo "✅ Documentation generated in web/docs/"

up_docs: docs
	@echo "🌐 Launching Documentation at http://localhost:8080/docs/ ..."
	python3 -m http.server 8080 --directory web & PID=$$!; \
	sleep 1; \
	open http://localhost:8080/docs/; \
	wait $$PID

sync_blog: docs
	@echo "📦 Syncing RCRE web suite & documentation to personal blog..."
	mkdir -p ../personal-blog/blog/receps_blog/static/rcre
	cp -r web/* ../personal-blog/blog/receps_blog/static/rcre/
	@echo "🔨 Building personal blog with Hugo..."
	cd ../personal-blog && make build
	@echo "✅ Synced and built into personal blog at erolrecep.github.io/rcre/!"

help:
	@echo "=================================================================="
	@echo "  RCRE — Recep's Code Run Everywhere: Command Center             "
	@echo "=================================================================="
	@echo "Interactive Operations:"
	@echo "  make up_web_app            Serve educational visualizers at http://localhost:8080"
	@echo "  make up_docs               Serve documentation at http://localhost:8080/docs/"
	@echo "  make webcam_test           Interactive camera test (choose Terminal or GUI window)"
	@echo "  make webcam_test_gui       Launch live webcam edge detection with Native GUI window"
	@echo "  make webcam_test_terminal  Launch live webcam edge detection in Terminal (ASCII)"
	@echo "  make webcam_test_dummy     Launch synthetic animated camera preview"
	@echo "  make gbdt_test             Run 100k-sample GBDT inference benchmark"
	@echo "  make hnsw_test             Run 64-D HNSW vector similarity search benchmark"
	@echo "  make project_lens_test     Run Reverse Image Search (rcre-lens) test suite"
	@echo "  make project_lens_demo     Run Reverse Image Search (rcre-lens) live demo"
	@echo "  make project_facex_test    Run Facial Expression (rcre-facex) verification suite"
	@echo "  make project_facex_demo    Run Facial Expression (rcre-facex) archetype demo"
	@echo ""
	@echo "Build & Documentation Targets:"
	@echo "  make all                Build static/shared libraries, dist header & examples"
	@echo "  make dist               Generate single-header amalgamated dist/rcre.h"
	@echo "  make docs               Extract API signatures and build web documentation"
	@echo "  make sync_blog          Sync web suite & docs to erolrecep.github.io via Hugo"
	@echo "  make examples           Compile all standalone examples into bin/"
	@echo "  make bench              Run hardware acceleration GEMM benchmark"
	@echo "  make clean              Remove all build artifacts and libraries"
	@echo ""
	@echo "Verification & Quality Gates:"
	@echo "  make test               Run full 12-test unit test suite"
	@echo "  make test-sanitizers    Run AddressSanitizer & UndefinedBehaviorSanitizer"
	@echo "  make test-parity        Run Python OpenCV/Scikit-Learn parity tests"
	@echo "  make test-all           Run all test suites & verification gates"
	@echo "=================================================================="


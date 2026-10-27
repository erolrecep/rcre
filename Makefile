CC ?= clang
AR ?= ar
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -Iinclude -Isrc/hal -Isrc/core -fPIC
SANITIZER_FLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O1 -g -fsanitize=address,undefined -Iinclude -Isrc/hal -Isrc/core

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CAM_SRCS = src/extras/cam_core.c src/extras/cam_dummy.c src/extras/cam_avf.m src/extras/gui_mac.m
    CAM_OBJS = src/extras/cam_core.o src/extras/cam_dummy.o src/extras/cam_avf.o src/extras/gui_mac.o
    CAM_LDFLAGS = -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework Foundation -framework Cocoa
    OPEN_CMD = open
else ifeq ($(UNAME_S),Linux)
    CAM_SRCS = src/extras/cam_core.c src/extras/cam_dummy.c src/extras/cam_v4l2.c src/extras/gui_stub.c
    CAM_OBJS = src/extras/cam_core.o src/extras/cam_dummy.o src/extras/cam_v4l2.o src/extras/gui_stub.o
    CAM_LDFLAGS =
    OPEN_CMD = xdg-open
    CFLAGS += -D_GNU_SOURCE
    SANITIZER_FLAGS += -D_GNU_SOURCE
else
    CAM_SRCS = src/extras/cam_core.c src/extras/cam_dummy.c src/extras/gui_stub.c
    CAM_OBJS = src/extras/cam_core.o src/extras/cam_dummy.o src/extras/gui_stub.o
    CAM_LDFLAGS =
    OPEN_CMD = echo
endif

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    AVX2_FLAGS ?= -mavx2 -mfma
else ifeq ($(UNAME_M),amd64)
    AVX2_FLAGS ?= -mavx2 -mfma
else
    AVX2_FLAGS ?=
endif

SRCS = src/core/arena.c \
       src/core/tensor.c \
       src/core/adapt.c \
       src/core/threadpool.c \
       src/core/prof.c \
       src/cv/color.c \
       src/cv/geom.c \
       src/cv/filter.c \
       src/cv/fft.c \
       src/cv/features.c \
       src/cv/homography.c \
       src/cv/optical_flow.c \
       src/cv/nn.c \
       src/ml/linear.c \
       src/ml/cluster.c \
       src/ml/gbdt_infer.c \
       src/ml/hnsw.c \
       src/autograd/scalar_autograd.c \
       src/autograd/tensor_autograd.c \
       src/nn/loss.c \
       src/nn/optim.c \
       src/nn/data.c \
       src/nn/vis.c \
       src/hal/cpu_probe.c \
       src/hal/hal_dispatch.c \
       src/hal/kernels_scalar.c \
       src/hal/kernels_neon.c \
       src/hal/kernels_avx2.c \
       src/hal/kernels_wasm.c \
       src/transformer/nn.c \
       src/transformer/model.c \
       src/transformer/kv_cache.c \
       src/transformer/tokenizer.c \
       src/transformer/dataset.c \
       src/transformer/optim.c \
       src/transformer/generate.c

OBJS = $(SRCS:.c=.o)
ALL_OBJS = $(OBJS) $(CAM_OBJS)
LIB = lib/librcre.a
DYLIB = lib/librcre.dylib
AMALGAMATED_HEADER = dist/rcre.h

TESTS = test_arena test_tensor_strides test_adapt_spec test_threadpool test_cv_filters test_fft test_nn test_ml_linear test_hal test_cluster test_gbdt test_hnsw test_cam test_amalgamated test_autograd_scalar test_autograd_tensor test_nn_train test_merger_report test_transformer_nn test_transformer_kv_cache test_transformer_bpe test_transformer_merge test_cv_geom test_cv_features test_cv_homography test_cv_optical_flow test_profiler
TEST_BINS = $(addprefix bin/,$(TESTS))
SAN_BINS = $(addprefix bin/,$(addsuffix _asan,$(TESTS)))

EXAMPLES = webcam_edge_filter tabular_gbdt_predict hnsw_vector_search mlp_autograd_train gpt_generate
EXAMPLE_BINS = $(addprefix bin/,$(EXAMPLES))

.DEFAULT_GOAL := help

.PHONY: default help all clean test test-sanitizers test-parity test-portable test-all bench profile examples dist up_web_app webcam_test webcam_test_gui webcam_test_terminal webcam_test_dummy gbdt_test hnsw_test mlp_train gpt_demo project_lens_test project_lens_demo project_facex_test project_facex_demo project_mobilenet_test project_mobilenet_demo project_perf_test project_perf_demo project_perf_bench

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

$(OBJS): include/rcre.h

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

src/hal/kernels_avx2.o: src/hal/kernels_avx2.c
	$(CC) $(CFLAGS) $(AVX2_FLAGS) -c $< -o $@

src/extras/%.o: src/extras/%.c
	$(CC) $(CFLAGS) -Isrc/extras -c $< -o $@

src/extras/%.o: src/extras/%.m
	$(CC) $(CFLAGS) -Isrc/extras -fobjc-arc -c $< -o $@

bin/test_amalgamated: tests/test_amalgamated.c $(AMALGAMATED_HEADER)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -lm -lpthread -o $@

bin/test_%: tests/test_%.c $(LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< $(LIB) $(CAM_LDFLAGS) -lm -lpthread -o $@

bin/%: examples/%.c $(LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< $(LIB) $(CAM_LDFLAGS) -lm -lpthread -o $@

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

mlp_train: bin/mlp_autograd_train
	@bin/mlp_autograd_train

gpt_demo: bin/gpt_generate
	@bin/gpt_generate

project_lens_test: $(LIB)
	@$(MAKE) -C projects/reverse_image_search test

project_lens_demo: $(LIB)
	@$(MAKE) -C projects/reverse_image_search demo

project_facex_test: $(LIB)
	@$(MAKE) -C projects/facial_expression_analysis test

project_facex_demo: $(LIB)
	@$(MAKE) -C projects/facial_expression_analysis demo

project_mobilenet_test: $(LIB)
	@$(MAKE) -C projects/cifar100_mobilenet test

project_mobilenet_demo: $(LIB)
	@$(MAKE) -C projects/cifar100_mobilenet demo

project_perf_test: $(LIB)
	@$(MAKE) -C projects/c_performance test

project_perf_demo: $(LIB)
	@$(MAKE) -C projects/c_performance demo

project_perf_bench: $(LIB)
	@$(MAKE) -C projects/c_performance bench

bin/bench_hal: benchmarks/bench_hal.c $(LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -Llib -lrcre -lm -lpthread -o $@

bench: bin/bench_hal
	@bin/bench_hal

# =============================================================================
# Profiling & Telemetry (rcre_prof) — CI threshold gate + trace export
# =============================================================================
PROF_DIR := prof
profile:
	@mkdir -p $(PROF_DIR) bin
	@echo "🔬 Building profiling workload with engine instrumentation (-DRCRE_ENABLE_PROFILING)..."
	$(CC) $(CFLAGS) -DRCRE_ENABLE_PROFILING $(SRCS) benchmarks/bench_prof.c $(CAM_LDFLAGS) -lm -lpthread -o bin/bench_prof
	@echo "🔬 Running RCRE profiling workload..."
	@bin/bench_prof $(PROF_DIR)/rcre_profile.json $(PROF_DIR)/rcre_profile.html

# P4: verify the amalgamated header compiles STANDALONE (no src/ tree).
# Catches relative includes that leak into dist/rcre.h (e.g. ../core/rcre_rng.h).
test-portable: $(AMALGAMATED_HEADER)
	@echo "🔒 Verifying dist/rcre.h compiles in a clean directory (no src/ tree)..."
	@tmpdir=$$(mktemp -d); \
	 cp $(AMALGAMATED_HEADER) $$tmpdir/rcre.h; \
	 printf '#define RCRE_IMPLEMENTATION\n#include "rcre.h"\nint main(void){RCRE_ProfContext*p=rcre_prof_init(64);rcre_hal_init();RCRE_Arena*a=rcre_arena_create(4096);RCRE_PROFILE_SCOPE(p,"boot",z);void*buf=rcre_arena_alloc(a,16,64);if(!buf)return 1;RCRE_PROFILE_END(z);rcre_prof_sample_arena(p,a);rcre_prof_free(p);rcre_arena_destroy(a);return 0;}\n' > $$tmpdir/probe.c; \
	 $(CC) $(CFLAGS) -I$$tmpdir $$tmpdir/probe.c -lm -lpthread -o $$tmpdir/probe && \
	 $$tmpdir/probe && echo "✅ dist/rcre.h is portable (compiles + runs standalone)"; \
	 rc=$$?; rm -rf $$tmpdir; exit $$rc

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
	@bin/test_fft
	@bin/test_nn
	@bin/test_ml_linear
	@bin/test_hal
	@bin/test_cluster
	@bin/test_gbdt
	@bin/test_hnsw
	@bin/test_cam
	@bin/test_amalgamated
	@bin/test_autograd_scalar
	@bin/test_autograd_tensor
	@bin/test_nn_train
	@bin/test_merger_report
	@bin/test_transformer_nn
	@bin/test_transformer_kv_cache
	@bin/test_transformer_bpe
	@bin/test_transformer_merge
	@bin/test_cv_geom
	@bin/test_cv_features
	@bin/test_cv_homography
	@bin/test_cv_optical_flow
	@bin/test_profiler
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
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_fft.c -lm -lpthread -o bin/test_fft_asan
	@bin/test_fft_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_nn.c -lm -lpthread -o bin/test_nn_asan
	@bin/test_nn_asan
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
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_autograd_scalar.c -lm -lpthread -o bin/test_autograd_scalar_asan
	@bin/test_autograd_scalar_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_autograd_tensor.c -lm -lpthread -o bin/test_autograd_tensor_asan
	@bin/test_autograd_tensor_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_nn_train.c -lm -lpthread -o bin/test_nn_train_asan
	@bin/test_nn_train_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_merger_report.c -lm -lpthread -o bin/test_merger_report_asan
	@bin/test_merger_report_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_transformer_nn.c -lm -lpthread -o bin/test_transformer_nn_asan
	@bin/test_transformer_nn_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_transformer_kv_cache.c -lm -lpthread -o bin/test_transformer_kv_cache_asan
	@bin/test_transformer_kv_cache_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_transformer_bpe.c -lm -lpthread -o bin/test_transformer_bpe_asan
	@bin/test_transformer_bpe_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_transformer_merge.c -lm -lpthread -o bin/test_transformer_merge_asan
	@bin/test_transformer_merge_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_cv_geom.c -lm -lpthread -o bin/test_cv_geom_asan
	@bin/test_cv_geom_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_cv_features.c -lm -lpthread -o bin/test_cv_features_asan
	@bin/test_cv_features_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_cv_homography.c -lm -lpthread -o bin/test_cv_homography_asan
	@bin/test_cv_homography_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_cv_optical_flow.c -lm -lpthread -o bin/test_cv_optical_flow_asan
	@bin/test_cv_optical_flow_asan
	$(CC) $(SANITIZER_FLAGS) $(SRCS) tests/test_profiler.c -lm -lpthread -o bin/test_profiler_asan
	@bin/test_profiler_asan
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
	rm -rf lib bin dist src/core/*.o src/cv/*.o src/ml/*.o src/hal/*.o src/extras/*.o src/autograd/*.o src/nn/*.o src/transformer/*.o
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
	@echo "  make project_perf_test     Run Hardware Benchmark (rcre-perf) verification suite"
	@echo "  make project_perf_demo     Run Hardware Benchmark (rcre-perf) live demo"
	@echo ""
	@echo "Build & Documentation Targets:"
	@echo "  make all                Build static/shared libraries, dist header & examples"
	@echo "  make dist               Generate single-header amalgamated dist/rcre.h"
	@echo "  make docs               Extract API signatures and build web documentation"
	@echo "  make sync_blog          Sync web suite & docs to erolrecep.github.io via Hugo"
	@echo "  make examples           Compile all standalone examples into bin/"
	@echo "  make bench              Run hardware acceleration GEMM benchmark"
	@echo "  make profile            Run telemetry profiler workload (CPU, RAM, storage, Perfetto trace)"
	@echo "  make clean              Remove all build artifacts and libraries"
	@echo ""
	@echo "Verification & Quality Gates:"
	@echo "  make test               Run full 27-test unit test suite"
	@echo "  make test-portable      Verify amalgamated dist/rcre.h standalone portability"
	@echo "  make test-sanitizers    Run AddressSanitizer & UndefinedBehaviorSanitizer"
	@echo "  make test-parity        Run Python OpenCV/Scikit-Learn parity tests"
	@echo "  make test-all           Run all test suites & verification gates"
	@echo "=================================================================="


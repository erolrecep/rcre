CC ?= clang
AR ?= ar
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -Iinclude -Isrc/hal -fPIC
SANITIZER_FLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O1 -g -fsanitize=address,undefined -Iinclude -Isrc/hal

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CAM_SRCS = src/extras/cam_core.c src/extras/cam_dummy.c src/extras/cam_avf.m
    CAM_OBJS = src/extras/cam_core.o src/extras/cam_dummy.o src/extras/cam_avf.o
    CAM_LDFLAGS = -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework Foundation
else ifeq ($(UNAME_S),Linux)
    CAM_SRCS = src/extras/cam_core.c src/extras/cam_dummy.c src/extras/cam_v4l2.c
    CAM_OBJS = src/extras/cam_core.o src/extras/cam_dummy.o src/extras/cam_v4l2.o
    CAM_LDFLAGS =
else
    CAM_SRCS = src/extras/cam_core.c src/extras/cam_dummy.c
    CAM_OBJS = src/extras/cam_core.o src/extras/cam_dummy.o
    CAM_LDFLAGS =
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
LIB = librcre.a
DYLIB = lib/librcre.dylib

TESTS = test_arena test_tensor_strides test_adapt_spec test_threadpool test_cv_filters test_ml_linear test_hal test_cluster test_gbdt test_hnsw test_cam
TEST_BINS = $(addprefix bin/,$(TESTS))
SAN_BINS = $(addprefix bin/,$(addsuffix _asan,$(TESTS)))

EXAMPLES = webcam_edge_filter tabular_gbdt_predict hnsw_vector_search
EXAMPLE_BINS = $(addprefix bin/,$(EXAMPLES))

.PHONY: all clean test test-sanitizers test-parity bench examples

all: $(LIB) $(DYLIB) $(EXAMPLE_BINS)

$(LIB): $(ALL_OBJS)
	@mkdir -p lib
	$(AR) rcs lib/$@ $^
	@echo "✅ Built lib/$@"

$(DYLIB): $(ALL_OBJS)
	@mkdir -p lib
	$(CC) -shared $(CFLAGS) $^ $(CAM_LDFLAGS) -lm -lpthread -o $@
	@echo "✅ Built $@"

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

src/extras/%.o: src/extras/%.c
	$(CC) $(CFLAGS) -Isrc/extras -c $< -o $@

src/extras/cam_avf.o: src/extras/cam_avf.m
	$(CC) $(CFLAGS) -Isrc/extras -fobjc-arc -c $< -o $@

bin/test_%: tests/test_%.c $(LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -Llib -lrcre $(CAM_LDFLAGS) -lm -lpthread -o $@

bin/%: examples/%.c $(LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -Llib -lrcre $(CAM_LDFLAGS) -lm -lpthread -o $@

examples: $(EXAMPLE_BINS)
	@echo "✅ Built all RCRE examples in bin/"

bin/bench_hal: benchmarks/bench_hal.c $(LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -Llib -lrcre -lm -lpthread -o $@

test: $(LIB) $(TEST_BINS)
	@echo "🚀 Running RCRE Core & Camera Unit Tests..."
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
	@echo "🎉 All RCRE Unit Tests Passed!"

test-sanitizers:
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
	@echo "🎉 Zero Memory Leaks & Zero Undefined Behavior Detected!"

test-parity: $(DYLIB)
	@echo "🐍 Running Python Parity Tests against OpenCV & Scikit-Learn..."
	uv run --with numpy,opencv-python,scikit-learn,scipy python tests/test_against_scipy_opencv.py

bench: bin/bench_hal
	@bin/bench_hal

clean:
	rm -rf lib bin src/core/*.o src/cv/*.o src/ml/*.o src/hal/*.o src/extras/*.o

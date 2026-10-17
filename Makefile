CC ?= clang
AR ?= ar
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -Iinclude -fPIC
SANITIZER_FLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O1 -g -fsanitize=address,undefined -Iinclude

SRCS = src/core/arena.c \
       src/core/tensor.c \
       src/core/adapt.c \
       src/core/threadpool.c \
       src/cv/color.c \
       src/cv/filter.c \
       src/ml/linear.c

OBJS = $(SRCS:.c=.o)
LIB = librcre.a
DYLIB = lib/librcre.dylib

TESTS = test_arena test_tensor_strides test_adapt_spec test_threadpool test_cv_filters test_ml_linear
TEST_BINS = $(addprefix bin/,$(TESTS))
SAN_BINS = $(addprefix bin/,$(addsuffix _asan,$(TESTS)))

.PHONY: all clean test test-sanitizers test-parity

all: $(LIB) $(DYLIB)

$(LIB): $(OBJS)
	@mkdir -p lib
	$(AR) rcs lib/$@ $^
	@echo "✅ Built lib/$@"

$(DYLIB): $(OBJS)
	@mkdir -p lib
	$(CC) -shared $(CFLAGS) $^ -lm -lpthread -o $@
	@echo "✅ Built $@"

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

bin/test_%: tests/test_%.c $(LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -Llib -lrcre -lm -lpthread -o $@

test: $(LIB) $(TEST_BINS)
	@echo "🚀 Running RCRE Core Unit Tests..."
	@bin/test_arena
	@bin/test_tensor_strides
	@bin/test_adapt_spec
	@bin/test_threadpool
	@bin/test_cv_filters
	@bin/test_ml_linear
	@echo "🎉 All RCRE Core Unit Tests Passed!"

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
	@echo "🎉 Zero Memory Leaks & Zero Undefined Behavior Detected!"

test-parity: $(DYLIB)
	@echo "🐍 Running Python Parity Tests against OpenCV & Scikit-Learn..."
	uv run --with numpy,opencv-python,scikit-learn,scipy python tests/test_against_scipy_opencv.py

clean:
	rm -rf lib bin src/core/*.o src/cv/*.o src/ml/*.o

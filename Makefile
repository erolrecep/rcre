CC ?= clang
AR ?= ar
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -Iinclude
SANITIZER_FLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O1 -g -fsanitize=address,undefined -Iinclude

SRCS = src/core/arena.c src/core/tensor.c src/core/adapt.c
OBJS = $(SRCS:.c=.o)
LIB = librcre.a

TEST_SRCS = tests/test_arena.c tests/test_tensor_strides.c tests/test_adapt_spec.c
TEST_BINS = bin/test_arena bin/test_tensor_strides bin/test_adapt_spec

.PHONY: all clean test test-sanitizers

all: $(LIB)

$(LIB): $(OBJS)
	@mkdir -p lib
	$(AR) rcs lib/$@ $^
	@echo "✅ Built lib/$@"

src/core/%.o: src/core/%.c
	$(CC) $(CFLAGS) -c $< -o $@

bin/test_%: tests/test_%.c $(LIB)
	@mkdir -p bin
	$(CC) $(CFLAGS) $< -Llib -lrcre -o $@

test: $(LIB) $(TEST_BINS)
	@echo "🚀 Running RCRE Core Unit Tests..."
	@bin/test_arena
	@bin/test_tensor_strides
	@bin/test_adapt_spec
	@echo "🎉 All RCRE Core Unit Tests Passed!"

test-sanitizers:
	@mkdir -p bin
	@echo "🛡️ Compiling and running tests with AddressSanitizer & UBSan..."
	$(CC) $(SANITIZER_FLAGS) src/core/arena.c tests/test_arena.c -o bin/test_arena_asan
	@bin/test_arena_asan
	$(CC) $(SANITIZER_FLAGS) src/core/arena.c src/core/tensor.c tests/test_tensor_strides.c -o bin/test_tensor_strides_asan
	@bin/test_tensor_strides_asan
	$(CC) $(SANITIZER_FLAGS) src/core/arena.c src/core/tensor.c src/core/adapt.c tests/test_adapt_spec.c -o bin/test_adapt_spec_asan
	@bin/test_adapt_spec_asan
	@echo "🎉 Zero Memory Leaks & Zero Undefined Behavior Detected!"

clean:
	rm -rf lib bin src/core/*.o

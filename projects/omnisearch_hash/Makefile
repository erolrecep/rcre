CC ?= clang
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -fPIC
INCLUDES = -I../../include -I../../src/hal -Iinclude -I.
LIBS = ../../lib/librcre.a -lm -lpthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIBS += -framework Foundation
endif

SRCS = src/hash_table.c src/distance_simd.c src/lsh_projector.c src/index_builder.c src/search_engine.c
OBJS = $(SRCS:.c=.o)

BINS = bin/rcre-omnisearch bin/test_hash_table bin/test_simd_distance bin/test_parity_lsh

.PHONY: all test demo bench asan clean

all: bin $(BINS)

bin:
	@mkdir -p bin

src/%.o: src/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

bin/rcre-omnisearch: main.c $(SRCS) ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) main.c $(SRCS) $(LIBS) -o $@
	@echo "✅ Built $@"

bin/test_hash_table: tests/test_hash_table.c src/hash_table.c ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) tests/test_hash_table.c src/hash_table.c $(LIBS) -o $@
	@echo "✅ Built $@"

bin/test_simd_distance: tests/test_simd_distance.c src/distance_simd.c src/lsh_projector.c ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) tests/test_simd_distance.c src/distance_simd.c src/lsh_projector.c $(LIBS) -o $@
	@echo "✅ Built $@"

bin/test_parity_lsh: tests/test_parity_lsh.c $(SRCS) ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) tests/test_parity_lsh.c $(SRCS) $(LIBS) -o $@
	@echo "✅ Built $@"

asan: bin
	$(CC) -std=c2x -Wall -Wextra -Wpedantic -O1 -g -fsanitize=address,undefined $(INCLUDES) tests/test_parity_lsh.c $(SRCS) ../../lib/librcre.a $(LIBS) -o bin/test_parity_asan
	@./bin/test_parity_asan
	@echo "🛡️ AddressSanitizer & UBSan clean!"

test: all
	@./bin/test_hash_table
	@./bin/test_simd_distance
	@./bin/test_parity_lsh

demo: bin/rcre-omnisearch
	@./bin/rcre-omnisearch demo

bench: bin/rcre-omnisearch
	@./bin/rcre-omnisearch bench

clean:
	@rm -rf bin src/*.o *.rcre_lsh
	@echo "🧹 Cleaned omnisearch_hash build artifacts"

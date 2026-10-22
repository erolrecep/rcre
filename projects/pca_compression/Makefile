CC ?= cc
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -fPIC
INCLUDES = -I../../include -I../../src/hal -I.
LIBS = ../../lib/librcre.a -lm -lpthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIBS += -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework Foundation -framework Cocoa
endif

BIN = bin/rcre-pca

.PHONY: all test demo bench clean

all: $(BIN)

bin:
	@mkdir -p bin

$(BIN): bin main.c pca_compressor.c ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) main.c pca_compressor.c $(LIBS) -o $@
	@echo "✅ Built $@"

test: $(BIN)
	@./$(BIN) --demo

demo: $(BIN)
	@./$(BIN) --demo

bench: $(BIN)
	@./$(BIN) --benchmark

clean:
	@rm -rf bin *.o *.bmp *.rpca
	@echo "🧹 Cleaned PCA compression build artifacts"

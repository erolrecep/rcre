CC ?= cc
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -fPIC
INCLUDES = -I../../include -I../../src/hal -I.
LIBS = -L../../lib -lrcre -lm -lpthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIBS += -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework Foundation -framework Cocoa
endif

SRC = feature_extractor.c test_extractor.c
BIN = bin/test_extractor

.PHONY: all test clean

all: $(BIN)

bin:
	@mkdir -p bin

$(BIN): bin $(SRC) ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) $(SRC) $(LIBS) -o $(BIN)
	@echo "✅ Built $(BIN)"

test: $(BIN)
	@./$(BIN)

clean:
	@rm -rf bin *.o
	@echo "🧹 Cleaned reverse_image_search build artifacts"

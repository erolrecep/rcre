# ==============================================================================
# RCRE Starter Project Makefile
# Targets:
#   make          - Builds the optimized release binary (bin/app)
#   make run      - Builds and executes bin/app, recording telemetry to logs/
#   make test     - Builds and executes test suite with AddressSanitizer & UBSan
#   make clean    - Removes all generated binaries and temporary files
#   make help     - Displays available targets and configuration options
# ==============================================================================

CC ?= clang
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3
SAN_FLAGS ?= -fsanitize=address,undefined -fno-omit-frame-pointer -g

INCLUDES = -Iinclude
LIBS = -lm -lpthread

# OS detection for optional platform flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS specific flags if needed
endif

APP_BIN = bin/app
TEST_BIN = bin/test_runner

.PHONY: all run test sanitize clean help

all: $(APP_BIN)

dirs:
	@mkdir -p bin logs

$(APP_BIN): dirs src/main.c include/rcre.h
	@echo "🔨 Compiling $(APP_BIN) (-O3)..."
	$(CC) $(CFLAGS) $(INCLUDES) src/main.c $(LIBS) -o $@
	@echo "✅ Built $(APP_BIN) successfully."

$(TEST_BIN): dirs tests/test_main.c include/rcre.h
	@echo "🔨 Compiling $(TEST_BIN) with AddressSanitizer & UBSan..."
	$(CC) -std=c2x -Wall -Wextra -Wpedantic $(SAN_FLAGS) $(INCLUDES) tests/test_main.c $(LIBS) -o $@
	@echo "✅ Built $(TEST_BIN) successfully."

run: $(APP_BIN)
	@echo "🚀 Executing $(APP_BIN)..."
	@./$(APP_BIN) | tee logs/sample_run.log

test: $(TEST_BIN)
	@echo "🧪 Running AddressSanitizer & UBSan test suite..."
	@./$(TEST_BIN)

sanitize: test

clean:
	@rm -rf bin/app bin/test_runner *.o *.dSYM
	@echo "🧹 Cleaned build artifacts."

help:
	@echo "RCRE Starter Project Build System"
	@echo ""
	@echo "Available Targets:"
	@echo "  make        - Build optimized binary ($(APP_BIN))"
	@echo "  make run    - Build and execute application (logs saved to logs/sample_run.log)"
	@echo "  make test   - Build and run verification test suite with ASan/UBSan"
	@echo "  make clean  - Remove compiled binaries and build artifacts"
	@echo "  make help   - Show this help message"


# ==============================================================================
# RCRE Galton Board (Quincunx) Project Makefile
# Targets:
#   make            - Builds optimized C23 release binary (bin/rcre-galton)
#   make run        - Executes demo CLI simulation
#   make animate    - Launches live ANSI interactive terminal visualizer
#   make benchmark  - Runs 20M-ball Monte-Carlo throughput benchmark
#   make test       - Compiles and runs test suite with AddressSanitizer & UBSan
#   make shared     - Builds lib/librcre_galton.dylib for cross-language C-API
#   make python-run - Launches Python 3.15 FastAPI streaming server (port 8000)
#   make python-test- Runs Python pytest suite with uv
#   make clean      - Removes generated binaries and build artifacts
# ==============================================================================

CC ?= clang
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -march=native
SAN_FLAGS ?= -fsanitize=address,undefined -fno-omit-frame-pointer -g
INCLUDES = -Iinclude
LIBS = -lm -lpthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    DYLIB_EXT = dylib
    SHARED_FLAGS = -dynamiclib
else
    DYLIB_EXT = so
    SHARED_FLAGS = -shared -fPIC
endif

BIN_DIR = bin
LIB_DIR = lib
LOG_DIR = logs

APP_BIN = $(BIN_DIR)/rcre-galton
TEST_BIN = $(BIN_DIR)/test_runner
SHARED_LIB = $(LIB_DIR)/librcre_galton.$(DYLIB_EXT)

SOURCES = src/galton_core.c src/galton_ansi.c
MAIN_SRC = src/main.c
TEST_SRC = tests/test_galton.c

PYTHON_DIR = python
UV ?= uv
PYTHON_BIN ?= /opt/homebrew/bin/python3.15

.PHONY: all run demo bench animate benchmark test sanitize shared python-run python-test clean help

all: $(APP_BIN) shared

demo: $(APP_BIN)
	@./$(APP_BIN) --rows 16 --balls 1000000

bench: benchmark


dirs:
	@mkdir -p $(BIN_DIR) $(LIB_DIR) $(LOG_DIR)

$(APP_BIN): dirs $(SOURCES) $(MAIN_SRC) include/rcre_galton.h
	@echo "🔨 Compiling $(APP_BIN) (ISO C23 -O3)..."
	$(CC) $(CFLAGS) $(INCLUDES) $(SOURCES) $(MAIN_SRC) $(LIBS) -o $@
	@echo "✅ Successfully built $(APP_BIN)"

$(SHARED_LIB): dirs $(SOURCES) include/rcre_galton.h
	@echo "🔨 Building shared C library $(SHARED_LIB)..."
	$(CC) $(CFLAGS) -fPIC $(SHARED_FLAGS) $(INCLUDES) $(SOURCES) $(LIBS) -o $@
	@echo "✅ Successfully built $(SHARED_LIB)"

$(TEST_BIN): dirs $(SOURCES) $(TEST_SRC) include/rcre_galton.h
	@echo "🧪 Compiling test runner with AddressSanitizer and UBSan..."
	$(CC) -std=c2x -Wall -Wextra -Wpedantic $(SAN_FLAGS) $(INCLUDES) $(SOURCES) $(TEST_SRC) $(LIBS) -o $@
	@echo "✅ Successfully built $(TEST_BIN)"

test: $(TEST_BIN)
	@echo "🚀 Executing C23 sanitizer test suite..."
	@./$(TEST_BIN)

run: $(APP_BIN)
	@./$(APP_BIN) --rows 16 --balls 1000000

animate: $(APP_BIN)
	@./$(APP_BIN) --animate --rows 12 --balls 200 --delay 20

benchmark: $(APP_BIN)
	@./$(APP_BIN) --benchmark --rows 16 --balls 20000000

python-run:
	@echo "🚀 Launching Python 3.15 FastAPI server on http://127.0.0.1:8000..."
	@cd $(PYTHON_DIR) && $(UV) run --python $(PYTHON_BIN) uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

python-test:
	@echo "🧪 Running Python 3.15 FastAPI test suite..."
	@cd $(PYTHON_DIR) && $(UV) run --extra dev --python $(PYTHON_BIN) pytest tests/ -v

clean:
	@rm -rf $(BIN_DIR) $(LIB_DIR) $(LOG_DIR) *.csv *.json /tmp/test_galton*
	@echo "🧹 Cleaned build artifacts"

# ==============================================================================
# Project 15: rcre_cam_stream — Zero-Copy Camera Stream & Live Studio
# Real-time Hardware Video Ingestion, Lock-Free SPSC Ring Buffer & OpenGL Studio
# ==============================================================================

CC        ?= clang
RCRE_ROOT := ../..
RCRE_INC  := $(RCRE_ROOT)/include
RCRE_LIB  := $(RCRE_ROOT)/lib/librcre.a

UNAME_S := $(shell uname -s)

C_SRCS := src/main.c src/render.c
BIN_DIR := bin
TARGET  := $(BIN_DIR)/rcre_cam_stream
SAN_BIN := $(BIN_DIR)/rcre_cam_stream_san

CFLAGS_COMMON := -std=c2x -Wall -Wextra -Wpedantic -I$(RCRE_INC) -I$(RCRE_ROOT)/src/extras -Isrc
CFLAGS_REL    := $(CFLAGS_COMMON) -O3 -flto -mcpu=native -DNDEBUG
CFLAGS_SAN    := $(CFLAGS_COMMON) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer

ifeq ($(UNAME_S),Darwin)
    SDL_CFLAGS := $(shell pkg-config --cflags sdl3 2>/dev/null || echo "-I/opt/homebrew/include")
    SDL_LIBS   := $(shell pkg-config --libs sdl3 2>/dev/null || echo "-L/opt/homebrew/lib -lSDL3")
    LDLIBS     := $(SDL_LIBS) $(RCRE_LIB) \
                  -framework AVFoundation -framework CoreMedia -framework CoreVideo \
                  -framework Foundation -framework OpenGL -framework Cocoa -framework Metal \
                  -L/opt/homebrew/lib -L/usr/local/lib -lzstd -lm -lpthread
else
    SDL_CFLAGS := $(shell pkg-config --cflags sdl3 2>/dev/null || echo "")
    SDL_LIBS   := $(shell pkg-config --libs sdl3 2>/dev/null || echo "-lSDL3")
    LDLIBS     := $(SDL_LIBS) $(RCRE_LIB) -lGL -lm -lpthread
endif

CFLAGS_REL += $(SDL_CFLAGS)
CFLAGS_SAN += $(SDL_CFLAGS)

.DEFAULT_GOAL := help
.PHONY: help all build demo run san test clean

help:
	@echo "rcre-cam-stream — Turnkey High-Performance Camera Stream Studio:"
	@echo "  make build         Build release binary (bin/rcre_cam_stream)"
	@echo "  make run           Launch live video camera stream (with GUI toolbar & HUD)"
	@echo "  make demo          Run automated synthetic camera demo (60 frames)"
	@echo "  make san           Build and run smoke test under AddressSanitizer + UBSan"
	@echo "  make test          Alias for make san"
	@echo "  make clean         Remove build artifacts and compiled binaries"

all: build

$(RCRE_LIB):
	@$(MAKE) -C $(RCRE_ROOT) lib/librcre.a

$(BIN_DIR):
	mkdir -p $(BIN_DIR)
	mkdir -p snapshots

build: $(RCRE_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS_REL) $(C_SRCS) -o $(TARGET) $(LDLIBS)

san: $(RCRE_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS_SAN) $(C_SRCS) -o $(SAN_BIN) $(LDLIBS)
	@echo "🧪 Running ASan/UBSan headless camera stream test..."
	@$(SAN_BIN) --dummy --headless --frames 60

test: san

demo: build
	@echo "🎥 Running synthetic camera streaming demo (60 frames)..."
	@$(TARGET) --dummy --headless --frames 60

run: build
	@./$(TARGET)

clean:
	rm -rf $(BIN_DIR)

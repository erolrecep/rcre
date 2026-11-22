# ===========================================================================
# rcre_manifold — Multi-Dimensional Manifold Learning & Untangling Engine
#
# Implements UMAP, Barnes-Hut t-SNE, PaCMAP, and PCA in pure ISO C23.
# Zero external runtime dependencies.
# ===========================================================================

.DEFAULT_GOAL := help

CC       ?= clang
RCRE     := ../..
CORE_LIB := $(RCRE)/lib/librcre.a

INCLUDES := -Iinclude -Isrc -I$(RCRE)/include -I$(RCRE)/src/core -I$(RCRE)/src/ml -I$(RCRE)/src/hal
CFLAGS   ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -fPIC $(INCLUDES)
SAN_FLAGS := -std=c2x -Wall -Wextra -Wpedantic -O1 -g -fsanitize=address,undefined $(INCLUDES)
LDFLAGS  := $(CORE_LIB) -lm -lpthread

SRCS     := src/manifold.c
OBJS     := build/manifold.o

BIN_DIR  := build
DEMO_BIN := $(BIN_DIR)/manifold_demo
ANIM_BIN := $(BIN_DIR)/manifold_animate
TEST_BIN := $(BIN_DIR)/test_manifold
SAN_BIN  := $(BIN_DIR)/test_manifold_san

.PHONY: help all clean test demo animate animate-swiss animate-trefoil animate-pacmap san bench

help:
	@echo "rcre_manifold — Available Targets:"
	@echo "  make all             Build demo, animator, and test binaries"
	@echo "  make test            Build & run comprehensive verification suite"
	@echo "  make demo            Build & run 10D ASCII cluster untangling demo"
	@echo "  make animate         Run interactive terminal timeline animation (10D Gaussians)"
	@echo "  make animate-swiss   Run 3D Swiss Roll unrolling animation in terminal"
	@echo "  make animate-trefoil Run 3D Trefoil Knot untangling animation in terminal"
	@echo "  make animate-pacmap  Run PaCMAP 3-stage annealing animation in terminal"
	@echo "  make san             Build & run test suite under AddressSanitizer + UBSan"
	@echo "  make clean           Remove all build artifacts"

$(CORE_LIB):
	@$(MAKE) -C $(RCRE) lib/librcre.a

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(OBJS): src/manifold.c include/rcre_manifold.h src/manifold_internal.h | $(BIN_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TEST_BIN): tests/test_manifold.c $(OBJS) $(CORE_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) tests/test_manifold.c $(OBJS) $(LDFLAGS) -o $@

$(DEMO_BIN): examples/manifold_demo.c $(OBJS) $(CORE_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) examples/manifold_demo.c $(OBJS) $(LDFLAGS) -o $@

$(ANIM_BIN): examples/manifold_animate.c $(OBJS) $(CORE_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) examples/manifold_animate.c $(OBJS) $(LDFLAGS) -o $@

$(SAN_BIN): tests/test_manifold.c src/manifold.c $(CORE_LIB) | $(BIN_DIR)
	$(CC) $(SAN_FLAGS) tests/test_manifold.c src/manifold.c $(LDFLAGS) -o $@

all: $(CORE_LIB) $(TEST_BIN) $(DEMO_BIN) $(ANIM_BIN)

test: $(TEST_BIN)
	@echo "🚀 Running RCRE Manifold Unit Test Suite..."
	@$(TEST_BIN)

demo: $(DEMO_BIN)
	@$(DEMO_BIN)

animate: $(ANIM_BIN)
	@$(ANIM_BIN)

animate-swiss: $(ANIM_BIN)
	@$(ANIM_BIN) --swiss

animate-trefoil: $(ANIM_BIN)
	@$(ANIM_BIN) --trefoil

animate-pacmap: $(ANIM_BIN)
	@$(ANIM_BIN) --pacmap

san: $(SAN_BIN)
	@echo "🛡️  Running RCRE Manifold under AddressSanitizer + UBSan..."
	@$(SAN_BIN)

clean:
	@rm -rf $(BIN_DIR)
	@echo "🧹 Cleaned rcre_manifold build artifacts"

CC ?= cc
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -fPIC
INCLUDES = -I../../include -I../../src/hal -I.
LIBS = ../../lib/librcre.a -lm -lpthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIBS += -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework Foundation -framework Cocoa
endif

BIN = bin/rcre-mobilenet

.PHONY: all test demo benchmark train clean

all: $(BIN)

bin:
	@mkdir -p bin

$(BIN): bin main.c cifar100_labels.h ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) main.c $(LIBS) -o $@
	@echo "✅ Built $@"

test: $(BIN)
	@./$(BIN) --input data/sample_0.bin

demo: $(BIN)
	@./$(BIN) --demo

benchmark: $(BIN)
	@./$(BIN) --benchmark 50

train:
	@echo "🔥 Running MobileNet Training & Weight Export for CIFAR-100..."
	uv run --with torch,torchvision python train/train_cifar100.py --epochs 5 --synthetic

clean:
	@rm -rf bin *.o
	@echo "🧹 Cleaned cifar100_mobilenet build artifacts"

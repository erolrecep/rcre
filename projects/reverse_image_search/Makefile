CC ?= cc
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -fPIC
INCLUDES = -I../../include -I../../src/hal -I.
LIBS = ../../lib/librcre.a -lm -lpthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIBS += -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework Foundation -framework Cocoa
endif

BINS = bin/test_extractor bin/test_serializer

.PHONY: all test clean

all: $(BINS)

bin:
	@mkdir -p bin

bin/test_extractor: bin feature_extractor.c test_extractor.c ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) feature_extractor.c test_extractor.c $(LIBS) -o $@
	@echo "✅ Built $@"

bin/test_serializer: bin index_serializer.c test_serializer.c ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) index_serializer.c test_serializer.c $(LIBS) -o $@
	@echo "✅ Built $@"

test: all
	@./bin/test_extractor
	@./bin/test_serializer

clean:
	@rm -rf bin *.o
	@echo "🧹 Cleaned reverse_image_search build artifacts"


CC ?= cc
CFLAGS ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -fPIC
INCLUDES = -I../../include -I../../src/hal -I.
LIBS = ../../lib/librcre.a -lm -lpthread

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIBS += -framework AVFoundation -framework CoreVideo -framework CoreMedia -framework Foundation -framework Cocoa
endif

BINS = bin/rcre-facex bin/test_face_detector bin/test_classifier

.PHONY: all test demo clean

all: $(BINS)

bin:
	@mkdir -p bin

bin/rcre-facex: bin main.c face_detector.c landmark_detector.c action_units.c expression_classifier.c ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) main.c face_detector.c landmark_detector.c action_units.c expression_classifier.c $(LIBS) -o $@
	@echo "✅ Built $@"

bin/test_face_detector: bin face_detector.c test_face_detector.c ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) face_detector.c test_face_detector.c $(LIBS) -o $@
	@echo "✅ Built $@"

bin/test_classifier: bin landmark_detector.c action_units.c expression_classifier.c test_classifier.c ../../lib/librcre.a
	$(CC) $(CFLAGS) $(INCLUDES) landmark_detector.c action_units.c expression_classifier.c test_classifier.c $(LIBS) -o $@
	@echo "✅ Built $@"

test: all
	@./bin/test_face_detector
	@./bin/test_classifier

demo: bin/rcre-facex
	@./bin/rcre-facex demo

clean:
	@rm -rf bin *.o *.bmp
	@echo "🧹 Cleaned facial_expression_analysis build artifacts"

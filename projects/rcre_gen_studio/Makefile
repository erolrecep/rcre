# ===========================================================================
# rcre_gen_studio — Turnkey Generative & Vision AI Studio (Pure ISO C23)
#
# Full ML Lifecycle across ALL 5 models (ae, vae, diffusion, vit, dit):
#   train -> compress (prune + RCRZ) -> quantize (F16/BF16/Q8_0/Q4_0) -> infer -> serve/deploy
# ===========================================================================

.DEFAULT_GOAL := help

CC       ?= clang
RCRE     := ../..
CORE_LIB := $(RCRE)/lib/librcre.a

INCLUDES  := -I$(RCRE)/include -I$(RCRE)/src/core -I$(RCRE)/src/hal
CFLAGS    ?= -std=c2x -Wall -Wextra -Wpedantic -O3 -ffp-contract=off $(INCLUDES)
SAN_FLAGS := -std=c2x -Wall -Wextra -Wpedantic -O1 -g -fsanitize=address,undefined -ffp-contract=off $(INCLUDES)
LDFLAGS   := $(CORE_LIB) -lm -lpthread

BIN_DIR    := build
STUDIO     := $(BIN_DIR)/rcre_gen_studio
STUDIO_SAN := $(BIN_DIR)/rcre_gen_studio_san

MODEL      ?= all
STEPS      ?= 50
SPARSITY   ?= 0.50
QTYPE      ?= q8_0
PORT       ?= 8090

.PHONY: help all clean demo demo-ae demo-vae demo-diffusion demo-vit demo-dit pipeline deploy serve test san

help:
	@echo "rcre_gen_studio — Full Lifecycle Targets (ae | vae | diffusion | vit | dit):"
	@echo "  make all                          Build rcre_gen_studio CLI binary"
	@echo "  make demo                         Run full lifecycle demo across all 5 models"
	@echo "  make demo-{ae,vae,diffusion,vit,dit}  Run lifecycle demo for one model"
	@echo "  make pipeline [MODEL=all] [STEPS=50]  Run scripts/pipeline.sh (train->compress->quantize->infer)"
	@echo "  make deploy   [MODEL=dit] [QTYPE=q8_0] Package standalone single-header edge bundle + Dockerfile"
	@echo "  make serve    [MODEL=vit] [PORT=8090]  Launch zero-dep HTTP JSON inference microservice"
	@echo "  make test                         Run RCRE core generative & lifecycle test suite (test_gen)"
	@echo "  make san                          Run studio smoke test under AddressSanitizer + UBSan"
	@echo "  make clean                        Remove build, pipeline, and deploy artifacts"

$(CORE_LIB):
	@$(MAKE) -C $(RCRE) lib/librcre.a

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(STUDIO): src/gen_studio.c $(CORE_LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) src/gen_studio.c $(LDFLAGS) -o $@

$(STUDIO_SAN): src/gen_studio.c $(CORE_LIB) | $(BIN_DIR)
	$(CC) $(SAN_FLAGS) src/gen_studio.c $(LDFLAGS) -o $@

all: $(STUDIO)

demo: $(STUDIO)
	@$(STUDIO) demo --model all --steps $(STEPS) --out-dir $(BIN_DIR)/outputs

demo-ae: $(STUDIO)
	@$(STUDIO) demo --model ae --steps $(STEPS) --out-dir $(BIN_DIR)/outputs

demo-vae: $(STUDIO)
	@$(STUDIO) demo --model vae --steps $(STEPS) --out-dir $(BIN_DIR)/outputs

demo-diffusion: $(STUDIO)
	@$(STUDIO) demo --model diffusion --steps $(STEPS) --out-dir $(BIN_DIR)/outputs

demo-vit: $(STUDIO)
	@$(STUDIO) demo --model vit --steps $(STEPS) --out-dir $(BIN_DIR)/outputs

demo-dit: $(STUDIO)
	@$(STUDIO) demo --model dit --steps $(STEPS) --out-dir $(BIN_DIR)/outputs

pipeline: $(STUDIO)
	@chmod +x scripts/pipeline.sh
	@SPARSITY=$(SPARSITY) ./scripts/pipeline.sh $(MODEL) $(STEPS) $(BIN_DIR)/pipeline_artifacts

deploy: $(STUDIO)
	@chmod +x scripts/pipeline.sh scripts/deploy_package.sh
	@m="$(if $(filter all,$(MODEL)),dit,$(MODEL))"; \
	  ./scripts/pipeline.sh "$$m" $(STEPS) $(BIN_DIR)/pipeline_artifacts && \
	  ./scripts/deploy_package.sh "$$m" "$(BIN_DIR)/pipeline_artifacts/$$m/$${m}_$(QTYPE).rgq1" "$(BIN_DIR)/deploy_$$m"

serve: $(STUDIO)
	@m="$(if $(filter all,$(MODEL)),vit,$(MODEL))"; \
	  w="$(BIN_DIR)/pipeline_artifacts/$$m/$${m}_$(QTYPE).rgq1"; \
	  if [ ! -f "$$w" ]; then ./scripts/pipeline.sh "$$m" $(STEPS) $(BIN_DIR)/pipeline_artifacts; fi; \
	  $(STUDIO) serve --model "$$m" --weights "$$w" --port $(PORT)

test:
	@$(MAKE) -C $(RCRE) bin/test_gen && $(RCRE)/bin/test_gen

san: $(STUDIO_SAN)
	@$(STUDIO_SAN) demo --model all --steps 6 --out-dir $(BIN_DIR)/outputs_san

clean:
	@rm -rf $(BIN_DIR)
	@echo "🧹 Cleaned rcre_gen_studio artifacts"

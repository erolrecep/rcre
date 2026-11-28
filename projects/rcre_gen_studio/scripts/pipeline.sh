#!/usr/bin/env bash
# =============================================================================
# RCRE Generative & Vision Full Lifecycle Pipeline
#   train -> compress (prune + RCRZ) -> quantize (F16 / Q8_0 / Q4_0) -> infer
#
# Usage:
#   ./scripts/pipeline.sh [ae|vae|diffusion|vit|dit|all] [steps] [out_dir]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STUDIO="${PROJECT_DIR}/build/rcre_gen_studio"

TARGET="${1:-all}"
STEPS="${2:-50}"
OUT_DIR="${3:-${PROJECT_DIR}/build/pipeline_artifacts}"
SPARSITY="${SPARSITY:-0.50}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${STUDIO}" ]]; then
    echo "Building rcre_gen_studio..."
    make -C "${PROJECT_DIR}" all
fi

run_one() {
    local m="$1"
    local mdir="${OUT_DIR}/${m}"
    mkdir -p "${mdir}"

    echo ""
    echo "======================================================================"
    echo "  PIPELINE: ${m}  (steps=${STEPS}, sparsity=${SPARSITY})"
    echo "======================================================================"

    echo "[1/4] Training F32 baseline -> ${mdir}/${m}_f32.safetensors"
    "${STUDIO}" train --model "${m}" --steps "${STEPS}" --out "${mdir}/${m}_f32.safetensors"

    echo "[2/4] Compressing (Magnitude Prune ${SPARSITY} + Byte-Shuffle RCRZ) -> ${mdir}/${m}_pruned.rgc1"
    "${STUDIO}" compress --model "${m}" --weights "${mdir}/${m}_f32.safetensors" \
        --sparsity "${SPARSITY}" --out "${mdir}/${m}_pruned.rgc1"

    echo "[3/4] Quantizing -> F16 / Q8_0 / Q4_0 (.rgq1)"
    "${STUDIO}" quantize --model "${m}" --weights "${mdir}/${m}_f32.safetensors" \
        --qtype f16  --out "${mdir}/${m}_f16.rgq1"
    "${STUDIO}" quantize --model "${m}" --weights "${mdir}/${m}_f32.safetensors" \
        --qtype q8_0 --out "${mdir}/${m}_q8_0.rgq1"
    "${STUDIO}" quantize --model "${m}" --weights "${mdir}/${m}_f32.safetensors" \
        --qtype q4_0 --out "${mdir}/${m}_q4_0.rgq1"

    echo "[4/4] Inference parity check across F32 / Pruned-RCRZ / F16 / Q8_0 / Q4_0"
    "${STUDIO}" infer --model "${m}" --weights "${mdir}/${m}_f32.safetensors"
    "${STUDIO}" infer --model "${m}" --weights "${mdir}/${m}_pruned.rgc1"
    "${STUDIO}" infer --model "${m}" --weights "${mdir}/${m}_f16.rgq1"
    "${STUDIO}" infer --model "${m}" --weights "${mdir}/${m}_q8_0.rgq1" --out-png "${mdir}/${m}_q8_preview.png"
    "${STUDIO}" infer --model "${m}" --weights "${mdir}/${m}_q4_0.rgq1"
}

if [[ "${TARGET}" == "all" ]]; then
    for m in ae vae diffusion vit dit; do
        run_one "${m}"
    done
else
    run_one "${TARGET}"
fi

echo ""
echo "Pipeline complete. Artifacts stored in: ${OUT_DIR}"

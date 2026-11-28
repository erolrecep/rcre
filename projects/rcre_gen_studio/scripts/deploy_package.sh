#!/usr/bin/env bash
# =============================================================================
# RCRE Self-Contained Edge Deployment Packager
#
# Creates a zero-dependency deployment bundle containing:
#   - rcre.h            (amalgamated single-header C23 engine)
#   - deploy_infer.c    (portable standalone inference entrypoint)
#   - weights.*         (chosen F32 .safetensors / .rgc1 / .rgq1 artifact)
#   - rcre_infer        (compiled native binary for current host)
#   - Dockerfile        (multi-stage minimal Alpine/Debian container recipe)
#   - run.sh            (one-line launcher)
#
# Usage:
#   ./scripts/deploy_package.sh <ae|vae|diffusion|vit|dit> <weights_file> [bundle_dir]
# =============================================================================
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <ae|vae|diffusion|vit|dit> <weights_file> [bundle_dir]"
    exit 1
fi

MODEL="$1"
WEIGHTS="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RCRE_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"
BUNDLE_DIR="${3:-${PROJECT_DIR}/build/deploy_${MODEL}}"
CC="${CC:-cc}"

if [[ ! -f "${WEIGHTS}" ]]; then
    echo "Error: weights file not found: ${WEIGHTS}" >&2
    exit 1
fi

if [[ ! -f "${RCRE_ROOT}/dist/rcre.h" ]]; then
    echo "Amalgamating dist/rcre.h..."
    make -C "${RCRE_ROOT}" dist
fi

mkdir -p "${BUNDLE_DIR}"
WEIGHT_BASENAME="$(basename "${WEIGHTS}")"

cp "${RCRE_ROOT}/dist/rcre.h"                   "${BUNDLE_DIR}/rcre.h"
cp "${PROJECT_DIR}/deploy/deploy_infer.c"       "${BUNDLE_DIR}/deploy_infer.c"
cp "${WEIGHTS}"                                 "${BUNDLE_DIR}/${WEIGHT_BASENAME}"

echo "Compiling standalone single-header edge binary -> ${BUNDLE_DIR}/rcre_infer"
"${CC}" -std=c2x -Wall -Wextra -Wpedantic -O3 -ffp-contract=off \
    -I"${BUNDLE_DIR}" "${BUNDLE_DIR}/deploy_infer.c" -lm -lpthread \
    -o "${BUNDLE_DIR}/rcre_infer"

cat > "${BUNDLE_DIR}/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
exec "\${DIR}/rcre_infer" --model "${MODEL}" --weights "\${DIR}/${WEIGHT_BASENAME}" "\$@"
EOF
chmod +x "${BUNDLE_DIR}/run.sh"

cat > "${BUNDLE_DIR}/Dockerfile" <<EOF
# Zero-dependency RCRE single-header deployment container (${MODEL})
FROM debian:bookworm-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends clang && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY rcre.h deploy_infer.c ./
RUN clang -std=c2x -O3 -ffp-contract=off deploy_infer.c -lm -lpthread -o rcre_infer

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/rcre_infer /app/rcre_infer
COPY ${WEIGHT_BASENAME} /app/${WEIGHT_BASENAME}
ENTRYPOINT ["/app/rcre_infer", "--model", "${MODEL}", "--weights", "/app/${WEIGHT_BASENAME}"]
EOF

echo ""
echo "Verifying deployed standalone binary:"
"${BUNDLE_DIR}/run.sh" --out-png "${BUNDLE_DIR}/smoke_output.png"
echo ""
echo "Deployment bundle ready: ${BUNDLE_DIR}"
ls -lh "${BUNDLE_DIR}"

#!/usr/bin/env bash
# ==============================================================================
# remote_gpu_train.sh: Remote CUDA Training Pipeline for MobileNet on CIFAR-100
# Targets: dlbox (NVIDIA GeForce RTX 5060 Ti, 16GB VRAM, CUDA 12.2)
# ==============================================================================

set -euo pipefail

REMOTE_HOST="dlbox"
REMOTE_DIR="/home/erolrecep/Developer/codebase/apps/rcre/projects/cifar100_mobilenet"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS="${1:-5}"
BATCH_SIZE="${2:-128}"

echo "=================================================================="
echo " 🚀 Remote CUDA Training Pipeline on $REMOTE_HOST (RTX 5060 Ti)"
echo "    Epochs: $EPOCHS | Batch Size: $BATCH_SIZE"
echo "=================================================================="

# 1. Sync local training code to dlbox
echo "📦 Step 1/4: Syncing training code to $REMOTE_HOST..."
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR"
rsync -avz --exclude '.git' --exclude 'bin' --exclude '*.o' "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

# 2. Run training on dlbox with CUDA AMP
echo ""
echo "🔥 Step 2/4: Commencing CUDA GPU training with AMP FP16 on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && /home/erolrecep/Developer/codebase/apps/scientific_python_workflow_hpc/.venv/bin/python train/train_cifar100.py --epochs $EPOCHS --batch-size $BATCH_SIZE --synthetic"

# 3. Pull trained weights and test samples back to local machine
echo ""
echo "📥 Step 3/4: Synchronizing trained binary weights to local project..."
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/mobilenet_cifar100.bin" "$LOCAL_DIR/mobilenet_cifar100.bin"
rsync -avz "$REMOTE_HOST:$REMOTE_DIR/data/" "$LOCAL_DIR/data/"

# 4. Verify native C23 neural engine inference with newly trained weights
echo ""
echo "🧪 Step 4/4: Verifying RCRE native C23 neural inference engine..."
make -C "$LOCAL_DIR" demo

echo ""
echo "=================================================================="
echo "🎉 REMOTE CUDA TRAINING & C23 VERIFICATION COMPLETED SUCCESSFULLY!"
echo "=================================================================="

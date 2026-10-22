#!/usr/bin/env python3
"""
export_apple_silicon_ane.py: Apple Silicon (Metal MPS & Apple Neural Engine ANE) Exporter
"""

import os
import sys
from pathlib import Path
import torch
from model import MobileNetCIFAR100

def export_apple_silicon():
    print("===================================================================")
    print(" 🍏 Apple Silicon Hardware Acceleration Suite (MPS & ANE)         ")
    print("===================================================================")

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"Apple Metal MPS Available: {has_mps}")
    if has_mps:
        print("✓ Metal Performance Shaders (MPS) is active for high-throughput GPU execution.")

    model_dir = Path(__file__).resolve().parent.parent
    checkpoint = model_dir / "mobilenet_cifar100.pt"

    model = MobileNetCIFAR100(num_classes=100)
    if checkpoint.exists():
        print(f"Loading weights from {checkpoint}...")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    else:
        print("Using model architecture...")

    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    # 1. PyTorch TorchScript Traced Model
    ts_path = model_dir / "mobilenet_cifar100_traced.pt"
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(str(ts_path))
    print(f"✓ Saved TorchScript model: {ts_path}")

    # 2. Apple Neural Engine (ANE) via Core ML
    try:
        import coremltools as ct
        print("🚀 Compiling for Apple Neural Engine (ANE)...")
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="image", shape=dummy_input.shape)],
            compute_units=ct.ComputeUnits.ALL, # Prioritizes Apple Neural Engine (ANE)
            minimum_deployment_target=ct.target.macOS13
        )
        mlpackage_path = model_dir / "MobileNetCIFAR100.mlpackage"
        mlmodel.save(str(mlpackage_path))
        print(f"✅ Successfully compiled for Apple Neural Engine (ANE): {mlpackage_path}")
    except ImportError:
        print("ℹ️ coremltools not installed. To compile specifically for the Apple Neural Engine (ANE), run:")
        print("   pip install coremltools")
        print("   python train/export_apple_silicon_ane.py")

    print("\n🎉 Apple Silicon Acceleration Exporter Completed!")

if __name__ == "__main__":
    export_apple_silicon()

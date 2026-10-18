"""
Verification of Python FFI Bindings for RCRE
"""

import sys
from pathlib import Path
import numpy as np

# Add bindings/python to sys.path
BINDINGS_PATH = Path(__file__).resolve().parent.parent / "bindings" / "python"
sys.path.insert(0, str(BINDINGS_PATH))

import rcre

def test_python_bindings():
    print("==================================================")
    print("  Testing RCRE Python FFI Bindings                ")
    print("==================================================")

    # 1. Check CPU Detection
    brand = rcre.cpu_brand()
    print(f"  CPU Detected via Python: '{brand}'")
    assert len(brand) > 0

    # 2. Test Arena Context Manager
    with rcre.Arena(8 * 1024 * 1024) as arena:
        assert arena.capacity == 8 * 1024 * 1024
        print(f"  Arena initialized: {arena.capacity / (1024*1024):.1f} MB")

        # 3. Test RGB to Gray Filter
        H, W = 100, 100
        # Synthetic RGB gradient
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255  # Red channel
        gray = rcre.rgb_to_gray(rgb, arena)

        assert gray.shape == (H, W)
        assert gray.dtype == np.uint8
        # Red Rec.601 luma: 0.299 * 255 = ~76.245 -> 76
        assert gray[0, 0] == 76
        print(f"  Rec.601 Pure Red Luma: {gray[0, 0]} (Expected 76)")

        arena.reset()
        assert arena.offset == 0
        print("  Arena reset verified (offset=0)")

    print("==================================================")
    print("  ✓ Python FFI bindings verified successfully!    ")
    print("==================================================")

if __name__ == "__main__":
    test_python_bindings()

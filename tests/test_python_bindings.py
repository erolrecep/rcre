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

        # 4. Test Data Science Metrics (ROC-AUC & R2)
        y_true = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        y_score = np.array([0.1, 0.4, 0.35, 0.8], dtype=np.float32)
        auc = rcre.roc_auc_score(y_true, y_score, arena)
        print(f"  ROC-AUC Score: {auc:.4f}")
        assert auc >= 0.75

        y_pred = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        r2 = rcre.r2_score(y_true, y_pred)
        print(f"  R2 Score (perfect fit): {r2:.4f}")
        assert abs(r2 - 1.0) < 1e-5

        # 5. Test Welford Statistics
        data = np.arange(1, 11, dtype=np.float32)
        stats = rcre.stat_describe(data, arena)
        print(f"  Welford Summary -> Mean: {stats['mean']}, Median: {stats['median']}, Min: {stats['min']}, Max: {stats['max']}")
        assert stats["count"] == 10
        assert abs(stats["mean"] - 5.5) < 1e-4
        assert abs(stats["min"] - 1.0) < 1e-4
        assert abs(stats["max"] - 10.0) < 1e-4

    print("==================================================")
    print("  ✓ Python FFI bindings verified successfully!    ")
    print("==================================================")

if __name__ == "__main__":
    test_python_bindings()

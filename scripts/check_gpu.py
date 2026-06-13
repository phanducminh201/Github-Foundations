"""
check_gpu.py — Kiểm tra toàn bộ môi trường trước khi train

Cách chạy:
    python scripts/check_gpu.py
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def check_environment():
    print("\n" + "="*55)
    print("  KIỂM TRA MÔI TRƯỜNG TRƯỚC KHI TRAIN")
    print("="*55)

    # ── Python
    import platform
    print(f"\n[1] Python   : {sys.version.split()[0]}")
    print(f"    Platform : {platform.system()} {platform.machine()}")

    # ── PyTorch + CUDA
    try:
        import torch
        print(f"\n[2] PyTorch  : {torch.__version__}")
        print(f"    CUDA     : {torch.version.cuda}")

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"    GPU      : {gpu}")
            print(f"    VRAM     : {vram:.1f} GB")
            print(f"    ✓ GPU sẵn sàng!")

            # Test tensor trên GPU
            x = torch.rand(1000, 1000).cuda()
            y = x @ x.T
            print(f"    Test GPU : PASSED (matrix mul 1000x1000)")
        else:
            print("    ✗ GPU KHÔNG khả dụng!")
            print("    → Kiểm tra lại cài đặt CUDA và driver NVIDIA")
    except ImportError:
        print("    ✗ PyTorch chưa được cài! Chạy:")
        print("      pip install torch torchvision --index-url ...")

    # ── Ultralytics (YOLOv8)
    try:
        import ultralytics
        print(f"\n[3] Ultralytics : {ultralytics.__version__} ✓")
    except ImportError:
        print("\n[3] Ultralytics : ✗ Chưa cài — pip install ultralytics")

    # ── torchvision
    try:
        import torchvision
        print(f"    torchvision : {torchvision.__version__} ✓")
    except ImportError:
        print("    torchvision : ✗ Chưa cài")

    # ── Các thư viện khác
    libs = {
        "roboflow"    : "roboflow",
        "opencv"      : "cv2",
        "PIL"         : "PIL",
        "numpy"       : "numpy",
        "matplotlib"  : "matplotlib",
        "sklearn"     : "sklearn",
        "pycocotools" : "pycocotools",
        "tqdm"        : "tqdm",
        "yaml"        : "yaml",
        "dotenv"      : "dotenv",
    }

    print("\n[4] Thư viện bổ sung:")
    all_ok = True
    for display_name, import_name in libs.items():
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "ok")
            print(f"    {display_name:<14}: {ver} ✓")
        except ImportError:
            print(f"    {display_name:<14}: ✗ Chưa cài")
            all_ok = False

    # ── Dataset
    print("\n[5] Dataset:")
    import os
    data_dir = "data/processed"
    if os.path.exists(data_dir):
        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(data_dir, split, "images")
            if os.path.exists(img_dir):
                n = len([f for f in os.listdir(img_dir)
                          if f.endswith((".jpg", ".png", ".jpeg"))])
                print(f"    {split:6s} : {n} ảnh ✓")
            else:
                print(f"    {split:6s} : ✗ Chưa có")
    else:
        print(f"    ✗ Chưa có dataset tại '{data_dir}'")
        print("    → Chạy: python scripts/download_dataset.py")

    # ── Tóm tắt
    print("\n" + "="*55)
    if torch.cuda.is_available() and all_ok:
        print("  ✅ Môi trường đầy đủ — sẵn sàng train!")
        print("  Chạy: python scripts/train_all.py")
    else:
        print("  ⚠ Có vấn đề cần khắc phục (xem các mục ✗ ở trên)")
    print("="*55 + "\n")


if __name__ == "__main__":
    check_environment()

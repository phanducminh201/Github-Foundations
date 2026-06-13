"""
download_dataset.py — Download dataset từ Roboflow về máy local

Cách chạy:
    python scripts/download_dataset.py
"""

import os
import sys
import yaml
import shutil
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import get_logger, plot_class_distribution

load_dotenv()
logger = get_logger("download_dataset")


def download():
    from roboflow import Roboflow

    api_key   = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE")
    project   = os.getenv("ROBOFLOW_PROJECT")
    version   = int(os.getenv("ROBOFLOW_VERSION", "1"))

    if not api_key or api_key == "your_api_key_here":
        logger.error("Chưa điền ROBOFLOW_API_KEY trong file .env !")
        sys.exit(1)

    logger.info(f"Đang download: {workspace}/{project} v{version}")

    rf      = Roboflow(api_key=api_key)
    proj    = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(
        "yolov8",
        location="data/processed",
        overwrite=True,
    )

    logger.info(f"✅ Download xong: {dataset.location}")

    # Kiểm tra số lượng ảnh
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset.location, split, "images")
        if os.path.exists(img_dir):
            n = len(list(Path(img_dir).glob("*")))
            logger.info(f"  {split:6s}: {n} ảnh")

    # Vẽ biểu đồ phân phối lớp
    _plot_distribution(dataset.location)


def _plot_distribution(dataset_path: str):
    """Đọc labels trong train và vẽ biểu đồ phân phối."""
    import glob
    from collections import Counter

    data_yaml = os.path.join(dataset_path, "data.yaml")
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    class_names = data["names"]

    label_files = glob.glob(f"{dataset_path}/train/labels/*.txt")
    counter = Counter()
    for lf in label_files:
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counter[int(parts[0])] += 1

    counts = [counter.get(i, 0) for i in range(len(class_names))]

    os.makedirs("runs/analysis", exist_ok=True)
    plot_class_distribution(
        class_names,
        counts,
        save_path="runs/analysis/class_distribution.png"
    )
    logger.info("Biểu đồ phân phối: runs/analysis/class_distribution.png")

    # In cảnh báo lớp thiếu dữ liệu
    for name, count in zip(class_names, counts):
        if count < 50:
            logger.warning(f"  ⚠ {name}: chỉ có {count} ảnh (< 50) — nên thu thập thêm")


if __name__ == "__main__":
    download()

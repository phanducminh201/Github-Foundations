"""
dataset.py — Dataset class và DataLoader cho Faster R-CNN & SSD
(YOLOv8 tự xử lý dataset qua ultralytics, không cần file này)
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from collections import Counter
from src.utils import get_logger

logger = get_logger("dataset")


# ─────────────────────────────────────────────────────────────
# Dataset class đọc định dạng YOLO (txt labels)
# ─────────────────────────────────────────────────────────────
class TrafficSignDataset(Dataset):
    """
    Dataset biển báo giao thông theo định dạng YOLO.

    Cấu trúc thư mục yêu cầu:
        split/
            images/   ← *.jpg, *.png
            labels/   ← *.txt (class xc yc w h, normalized)
    """

    def __init__(self,
                 img_dir: str,
                 lbl_dir: str,
                 class_names: list,
                 img_size: int = 512,
                 augment: bool = False):
        self.img_paths   = sorted(
            glob.glob(f"{img_dir}/*.jpg") +
            glob.glob(f"{img_dir}/*.jpeg") +
            glob.glob(f"{img_dir}/*.png")
        )
        self.lbl_dir     = lbl_dir
        self.class_names = class_names
        self.img_size    = img_size
        self.augment     = augment

        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"Không tìm thấy ảnh trong: {img_dir}")

        logger.info(f"Loaded {len(self.img_paths)} ảnh từ {img_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img      = Image.open(img_path).convert("RGB")
        W, H     = img.size

        # Resize ảnh
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        scale_x = self.img_size / W
        scale_y = self.img_size / H

        # Đọc labels
        lbl_path = os.path.join(
            self.lbl_dir,
            os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )

        boxes, labels = [], []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls, xc, yc, w, h = map(float, parts[:5])

                    # Chuyển YOLO format → xyxy (pixel)
                    x1 = (xc - w / 2) * W * scale_x
                    y1 = (yc - h / 2) * H * scale_y
                    x2 = (xc + w / 2) * W * scale_x
                    y2 = (yc + h / 2) * H * scale_y

                    # Clamp trong phạm vi ảnh
                    x1 = max(0, min(x1, self.img_size))
                    y1 = max(0, min(y1, self.img_size))
                    x2 = max(0, min(x2, self.img_size))
                    y2 = max(0, min(y2, self.img_size))

                    if x2 > x1 + 2 and y2 > y1 + 2:   # bbox hợp lệ
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls) + 1)     # 0 = background

        # Xử lý ảnh không có box (tránh lỗi khi train)
        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
        }

        img_tensor = T.ToTensor()(img)          # [0, 1] float32
        img_tensor = T.Normalize(               # ImageNet mean/std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img_tensor)

        return img_tensor, target

    def get_class_counts(self) -> dict:
        """Đếm số lượng instance mỗi lớp trong dataset."""
        counter = Counter()
        for img_path in self.img_paths:
            lbl_path = os.path.join(
                self.lbl_dir,
                os.path.splitext(os.path.basename(img_path))[0] + ".txt"
            )
            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            counter[int(parts[0])] += 1
        return {self.class_names[k]: v for k, v in sorted(counter.items())}


# ─────────────────────────────────────────────────────────────
# Collate function (bắt buộc với detection — batch có số box khác nhau)
# ─────────────────────────────────────────────────────────────
def collate_fn(batch):
    return tuple(zip(*batch))


# ─────────────────────────────────────────────────────────────
# Factory function tạo DataLoader cho cả 3 split
# ─────────────────────────────────────────────────────────────
def build_dataloaders(dataset_path: str,
                      class_names: list,
                      img_size: int = 512,
                      batch_size: int = 4,
                      num_workers: int = 2) -> dict:
    """
    Tạo DataLoader cho train / valid / test.

    Returns:
        dict với keys: 'train', 'valid', 'test'
    """
    loaders = {}
    splits  = {
        "train": True,   # augment = True khi train
        "valid": False,
        "test":  False,
    }

    for split, do_augment in splits.items():
        img_dir = os.path.join(dataset_path, split, "images")
        lbl_dir = os.path.join(dataset_path, split, "labels")

        if not os.path.exists(img_dir):
            logger.warning(f"Không tìm thấy thư mục: {img_dir}, bỏ qua.")
            continue

        ds = TrafficSignDataset(
            img_dir     = img_dir,
            lbl_dir     = lbl_dir,
            class_names = class_names,
            img_size    = img_size,
            augment     = do_augment,
        )

        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            collate_fn  = collate_fn,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )
        logger.info(f"DataLoader [{split}]: {len(ds)} ảnh, "
                    f"batch_size={batch_size}")

    return loaders

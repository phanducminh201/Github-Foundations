"""
utils.py — Hàm tiện ích dùng chung cho toàn bộ dự án
"""

import os
import time
import logging
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)  # màu terminal Windows


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
def get_logger(name: str, log_dir: str = None) -> logging.Logger:
    """Tạo logger ghi ra terminal và file cùng lúc."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler terminal
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Handler file
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}_{timestamp()}.log")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    """Đọc file YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────────────────────
# GPU / Device
# ─────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Tự động chọn GPU nếu có, ngược lại dùng CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{Fore.GREEN}✓ GPU: {gpu_name} ({vram:.1f} GB VRAM){Style.RESET_ALL}")
    else:
        device = torch.device("cpu")
        print(f"{Fore.YELLOW}⚠ GPU không khả dụng, dùng CPU (train sẽ rất chậm){Style.RESET_ALL}")
    return device


def print_gpu_memory():
    """In trạng thái VRAM hiện tại."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        total     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {allocated:.2f}GB dùng / {reserved:.2f}GB giữ / {total:.1f}GB tổng")


# ─────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────
def save_checkpoint(state: dict, save_dir: str, filename: str):
    """Lưu checkpoint vào thư mục chỉ định."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer=None, device=None):
    """Load checkpoint và restore model state."""
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_metric = ckpt.get("best_metric", 0.0)
    print(f"{Fore.CYAN}✓ Loaded checkpoint: epoch={ckpt.get('epoch')}, "
          f"best_metric={best_metric:.4f}{Style.RESET_ALL}")
    return model, optimizer, start_epoch, best_metric


# ─────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────
def plot_loss_curve(history: dict, save_path: str, title: str = "Training Loss"):
    """Vẽ loss curve từ history dict."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, values in history.items():
        ax.plot(range(1, len(values) + 1), values, linewidth=2, label=key)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Đã lưu biểu đồ: {save_path}")


def plot_class_distribution(class_names: list, counts: list, save_path: str):
    """Vẽ biểu đồ phân phối số lượng ảnh theo lớp."""
    colors = ["#E24B4A" if c < 50 else "#4A90D9" for c in counts]
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(class_names, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.6, label="Ngưỡng tối thiểu (50)")
    ax.set_title("Phân phối số lượng nhãn theo lớp", fontsize=13, fontweight="bold")
    ax.set_ylabel("Số lượng bounding box")
    ax.legend()
    plt.xticks(rotation=30, ha="right", fontsize=9)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────
# Timer
# ─────────────────────────────────────────────────────────────
class Timer:
    """Đo thời gian training."""
    def __init__(self):
        self._start = None

    def start(self):
        self._start = time.time()

    def elapsed(self) -> str:
        secs = int(time.time() - self._start)
        h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

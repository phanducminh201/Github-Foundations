"""
trainer.py — Training loop chung cho Faster R-CNN và SSD
(YOLOv8 dùng ultralytics API riêng trong scripts/train_all.py)
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style

from src.utils import (
    get_logger, save_checkpoint, load_checkpoint,
    plot_loss_curve, print_gpu_memory, timestamp
)

logger = get_logger("trainer")


class Trainer:
    """
    Training loop tổng quát cho Faster R-CNN và SSD.

    Tính năng:
    - Lưu best.pt khi val_loss giảm
    - Lưu last.pt sau mỗi epoch
    - Lưu checkpoint mỗi N epoch (save_period)
    - Early stopping
    - Vẽ và lưu loss curve tự động
    - Resume từ checkpoint bị ngắt giữa chừng
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer,
                 scheduler,
                 device: torch.device,
                 cfg: dict,
                 model_name: str):

        self.model      = model.to(device)
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.device     = device
        self.cfg        = cfg
        self.model_name = model_name

        # Thư mục lưu output
        project  = cfg["output"]["project"]
        run_name = cfg["output"]["name"]
        self.save_dir = os.path.join(project, run_name)
        os.makedirs(os.path.join(self.save_dir, "weights"), exist_ok=True)

        # Cấu hình training
        self.epochs       = cfg["training"]["epochs"]
        self.grad_clip    = cfg["training"].get("gradient_clip", 0.0)

        # Cấu hình checkpoint
        ckpt_cfg          = cfg.get("checkpoint", {})
        self.save_best    = ckpt_cfg.get("save_best",   True)
        self.save_last    = ckpt_cfg.get("save_last",   True)
        self.save_period  = ckpt_cfg.get("save_period", 5)

        # Tracking
        self.history      = {"train_loss": [], "val_loss": []}
        self.best_loss    = float("inf")
        self.start_epoch  = 0

        logger.info(f"Trainer khởi tạo cho [{model_name}] "
                    f"| Device: {device} "
                    f"| Epochs: {self.epochs} "
                    f"| Save dir: {self.save_dir}")

    # ─────────────────────────────────────────────────────────
    # Resume từ checkpoint
    # ─────────────────────────────────────────────────────────
    def resume(self, checkpoint_path: str):
        """Load lại checkpoint để tiếp tục train bị ngắt."""
        self.model, self.optimizer, self.start_epoch, self.best_loss = \
            load_checkpoint(checkpoint_path, self.model,
                            self.optimizer, self.device)
        logger.info(f"Resume từ epoch {self.start_epoch}")

    # ─────────────────────────────────────────────────────────
    # Internal: 1 epoch train
    # ─────────────────────────────────────────────────────────
    def _train_one_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc="  Train",
                    leave=False, ncols=90,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")

        for imgs, targets in pbar:
            imgs    = [img.to(self.device) for img in imgs]
            targets = [{k: v.to(self.device) for k, v in t.items()}
                       for t in targets]

            loss_dict  = self.model(imgs, targets)
            batch_loss = sum(loss_dict.values())

            self.optimizer.zero_grad()
            batch_loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += batch_loss.item()
            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

        return total_loss / len(loader)

    # ─────────────────────────────────────────────────────────
    # Internal: 1 epoch validation
    # ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def _validate_one_epoch(self, loader) -> float:
        """
        Tính val_loss bằng cách switch model về train mode tạm thời.
        (torchvision detection models chỉ tính loss khi train mode)
        """
        self.model.train()   # giữ train mode để lấy loss
        total_loss = 0.0

        pbar = tqdm(loader, desc="  Valid",
                    leave=False, ncols=90,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")

        for imgs, targets in pbar:
            imgs    = [img.to(self.device) for img in imgs]
            targets = [{k: v.to(self.device) for k, v in t.items()}
                       for t in targets]
            loss_dict = self.model(imgs, targets)
            total_loss += sum(loss_dict.values()).item()

        self.model.eval()
        return total_loss / len(loader)

    # ─────────────────────────────────────────────────────────
    # Lưu checkpoint
    # ─────────────────────────────────────────────────────────
    def _save(self, epoch: int, val_loss: float):
        state = {
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric":          self.best_loss,
            "val_loss":             val_loss,
            "config":               self.cfg,
        }
        weights_dir = os.path.join(self.save_dir, "weights")

        # 1. Luôn lưu last.pt
        if self.save_last:
            save_checkpoint(state, weights_dir, "last.pt")

        # 2. Lưu best.pt nếu cải thiện
        if self.save_best and val_loss < self.best_loss:
            self.best_loss = val_loss
            save_checkpoint(state, weights_dir, "best.pt")
            logger.info(f"{Fore.GREEN}  ✓ best.pt cập nhật "
                        f"(val_loss={val_loss:.4f}){Style.RESET_ALL}")

        # 3. Lưu checkpoint định kỳ mỗi save_period epoch
        if self.save_period > 0 and (epoch + 1) % self.save_period == 0:
            fname = f"epoch_{epoch+1:03d}.pt"
            save_checkpoint(state, weights_dir, fname)
            logger.info(f"  Saved periodic checkpoint: {fname}")

    # ─────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────
    def fit(self, train_loader, val_loader):
        logger.info(f"\n{'='*60}")
        logger.info(f"  Bắt đầu train [{self.model_name.upper()}]")
        logger.info(f"  Epochs: {self.start_epoch+1} → {self.epochs}")
        logger.info(f"  Save dir: {self.save_dir}")
        logger.info(f"{'='*60}")

        total_timer = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            print(f"\n{Fore.CYAN}Epoch [{epoch+1}/{self.epochs}]{Style.RESET_ALL}")

            # Train
            train_loss = self._train_one_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validation
            val_loss = self._validate_one_epoch(val_loader)
            self.history["val_loss"].append(val_loss)

            # Scheduler step
            if self.scheduler:
                if hasattr(self.scheduler, "step"):
                    self.scheduler.step()

            # Lưu checkpoint
            self._save(epoch, val_loss)

            # Log
            elapsed = time.time() - t0
            lr_now  = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch+1:3d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={lr_now:.6f} | "
                f"time={elapsed:.0f}s"
            )
            if torch.cuda.is_available():
                print_gpu_memory()

        # Kết thúc — vẽ loss curve
        total_time = time.time() - total_timer
        logger.info(f"\n✅ Train xong [{self.model_name}] "
                    f"trong {total_time/60:.1f} phút")
        logger.info(f"   Best val_loss: {self.best_loss:.4f}")
        logger.info(f"   Weights: {self.save_dir}/weights/")

        # Lưu loss curve
        plot_loss_curve(
            self.history,
            save_path = os.path.join(self.save_dir, "loss_curve.png"),
            title     = f"Training & Validation Loss — {self.model_name}"
        )

        return self.history

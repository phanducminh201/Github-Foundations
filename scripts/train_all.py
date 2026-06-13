"""
train_all.py — Huấn luyện cả 3 mô hình theo thứ tự:
    1. YOLOv8n        (ultralytics API)
    2. Faster R-CNN   (torchvision + Trainer)
    3. SSD300         (torchvision + Trainer)
 
Cách chạy (Baseline):
    python scripts/train_all.py
    python scripts/train_all.py --model yolov8
    python scripts/train_all.py --model fasterrcnn
    python scripts/train_all.py --model ssd
 
Cách chạy (Tuned — Giai đoạn 5):
    python scripts/train_all.py --tuned
    python scripts/train_all.py --model yolov8 --tuned
    python scripts/train_all.py --model fasterrcnn --tuned
    python scripts/train_all.py --model ssd --tuned
 
Resume nếu bị ngắt:
    python scripts/train_all.py --model fasterrcnn --resume runs/fasterrcnn/baseline/weights/last.pt
    python scripts/train_all.py --model fasterrcnn --resume runs/fasterrcnn/tuned/weights/last.pt --tuned
"""
 
import os
import sys
import argparse
import yaml
import torch
from dotenv import load_dotenv
from pathlib import Path
 
# Thêm thư mục gốc vào Python path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
 
from src.utils    import get_logger, get_device, load_config, timestamp
from src.dataset  import build_dataloaders
from src.models   import get_model
from src.trainer  import Trainer
 
load_dotenv()
logger = get_logger("train_all", log_dir="runs/logs")
 
 
# ─────────────────────────────────────────────────────────────
# Đọc data.yaml để lấy class names
# ─────────────────────────────────────────────────────────────
def get_class_names(dataset_path: str) -> list:
    data_yaml = os.path.join(dataset_path, "data.yaml")
    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["names"]
 
 
# ─────────────────────────────────────────────────────────────
# Mô hình 1: YOLOv8n
# ─────────────────────────────────────────────────────────────
def train_yolov8(dataset_path: str, cfg_path: str, tuned: bool = False):
    from ultralytics import YOLO
 
    cfg = load_config(cfg_path)
 
    # Nếu tuned=True → ghi đè các hyperparameter cần tuning
    if tuned:
        cfg["output"]["name"]              = "tuned"
        cfg["training"]["epochs"]          = 100
        cfg["training"]["lr0"]             = 0.0005
        cfg["training"]["lrf"]             = 0.005
        cfg["training"]["patience"]        = 20
        cfg["training"]["cls_loss_weight"] = 2.0
        cfg["training"]["warmup_epochs"]   = 5
 
    t = cfg["training"]
    c = cfg["checkpoint"]
    o = cfg["output"]
 
    mode = "TUNED" if tuned else "BASELINE"
    logger.info("\n" + "="*60)
    logger.info(f"  MÔ HÌNH 1: YOLOv8n  [{mode}]")
    logger.info("="*60)
 
    model = YOLO(f"{cfg['model']['name']}.pt")
 
    results = model.train(
        data          = os.path.join(dataset_path, "data.yaml"),
        epochs        = t["epochs"],
        imgsz         = t["image_size"],
        batch         = t["batch_size"],
        optimizer     = t["optimizer"],
        lr0           = t["lr0"],
        lrf           = t["lrf"],
        momentum      = t["momentum"],
        weight_decay  = t["weight_decay"],
        warmup_epochs = t["warmup_epochs"],
        cls           = t["cls_loss_weight"],
        patience      = t["patience"],
        save          = True,
        save_period   = c["save_period"],
        project       = o["project"],
        name          = o["name"],
        plots         = o["plots"],
        verbose       = o["verbose"],
        device        = 0 if torch.cuda.is_available() else "cpu",
        exist_ok      = True,
    )
 
    logger.info(f"✅ YOLOv8 xong! Weights: {results.save_dir}/weights/")
    return results
 
 
# ─────────────────────────────────────────────────────────────
# Mô hình 2: Faster R-CNN
# ─────────────────────────────────────────────────────────────
def train_fasterrcnn(dataset_path: str, cfg_path: str,
                     class_names: list, device: torch.device,
                     resume_path: str = None, tuned: bool = False):
 
    cfg = load_config(cfg_path)
    t   = cfg["training"]
 
    mode = "TUNED" if tuned else "BASELINE"
    logger.info("\n" + "="*60)
    logger.info(f"  MÔ HÌNH 2: Faster R-CNN (ResNet50-FPN)  [{mode}]")
    logger.info("="*60)
 
    # DataLoader
    loaders = build_dataloaders(
        dataset_path = dataset_path,
        class_names  = class_names,
        img_size     = t["image_size"],
        batch_size   = t["batch_size"],
        num_workers  = 0 if sys.platform == "win32" else 2,
    )
 
    # Model
    model = get_model("fasterrcnn", len(class_names), cfg)
 
    # Optimizer
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr           = t["lr"],
        momentum     = t["momentum"],
        weight_decay = t["weight_decay"],
    )
 
    # Scheduler — tuned dùng CosineAnnealingLR, baseline dùng StepLR
    scheduler_name = t.get("lr_scheduler", "StepLR")
    if scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t.get("t_max", t["epochs"])
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = t["step_size"],
            gamma     = t["gamma"],
        )
 
    # Trainer
    trainer = Trainer(model, optimizer, scheduler, device, cfg, "fasterrcnn")
 
    if resume_path:
        trainer.resume(resume_path)
 
    history = trainer.fit(loaders["train"], loaders["valid"])
    logger.info("✅ Faster R-CNN xong!")
    return history
 
 
# ─────────────────────────────────────────────────────────────
# Mô hình 3: SSD300
# ─────────────────────────────────────────────────────────────
def train_ssd(dataset_path: str, cfg_path: str,
              class_names: list, device: torch.device,
              resume_path: str = None, tuned: bool = False):
 
    cfg = load_config(cfg_path)
    t   = cfg["training"]
 
    mode = "TUNED" if tuned else "BASELINE"
    logger.info("\n" + "="*60)
    logger.info(f"  MÔ HÌNH 3: SSD300 (VGG16)  [{mode}]")
    logger.info("="*60)
 
    # DataLoader (SSD dùng 300x300)
    loaders = build_dataloaders(
        dataset_path = dataset_path,
        class_names  = class_names,
        img_size     = t["image_size"],
        batch_size   = t["batch_size"],
        num_workers  = 0 if sys.platform == "win32" else 2,
    )
 
    # Model
    model = get_model("ssd", len(class_names), cfg)
 
    # Optimizer
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr           = t["lr"],
        momentum     = t["momentum"],
        weight_decay = t["weight_decay"],
    )
 
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t["t_max"]
    )
 
    # Trainer
    trainer = Trainer(model, optimizer, scheduler, device, cfg, "ssd")
 
    if resume_path:
        trainer.resume(resume_path)
 
    history = trainer.fit(loaders["train"], loaders["valid"])
    logger.info("✅ SSD xong!")
    return history
 
 
# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train traffic sign detection models")
    parser.add_argument("--model",  type=str, default="all",
                        choices=["all", "yolov8", "fasterrcnn", "ssd"],
                        help="Chọn mô hình cần train")
    parser.add_argument("--data",   type=str, default="data/processed",
                        help="Đường dẫn thư mục dataset")
    parser.add_argument("--resume", type=str, default=None,
                        help="Đường dẫn checkpoint để resume (fasterrcnn/ssd)")
    parser.add_argument("--tuned",  action="store_true",
                        help="Dùng config tuned (Giai đoạn 5) thay vì baseline")
    args = parser.parse_args()
 
    # Kiểm tra dataset
    if not os.path.exists(args.data):
        logger.error(f"Dataset không tồn tại: {args.data}")
        logger.error("Chạy scripts/download_dataset.py trước!")
        sys.exit(1)
 
    device      = get_device()
    class_names = get_class_names(args.data)
    mode        = "TUNED" if args.tuned else "BASELINE"
 
    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(f"Chế độ: {mode}")
 
    run = args.model
 
    # ── YOLOv8 ───────────────────────────────────────────────
    if run in ("all", "yolov8"):
        train_yolov8(
            args.data,
            cfg_path = "configs/yolov8.yaml",   # 1 file duy nhất, tuned ghi đè qua code
            tuned    = args.tuned,
        )
 
    # ── Faster R-CNN ─────────────────────────────────────────
    if run in ("all", "fasterrcnn"):
        # Tuned dùng file config riêng để thay đổi scheduler & trainable_layers
        cfg_path = ("configs/fasterrcnn_tuned.yaml"
                    if args.tuned else "configs/fasterrcnn.yaml")
        train_fasterrcnn(
            args.data, cfg_path, class_names, device,
            resume_path = args.resume if run == "fasterrcnn" else None,
            tuned       = args.tuned,
        )
 
    # ── SSD ──────────────────────────────────────────────────
    if run in ("all", "ssd"):
        cfg_path = ("configs/ssd_tuned.yaml"
                    if args.tuned else "configs/ssd.yaml")
        train_ssd(
            args.data, cfg_path, class_names, device,
            resume_path = args.resume if run == "ssd" else None,
            tuned       = args.tuned,
        )
 
    logger.info(f"\n🎉 Hoàn thành train tất cả mô hình! [{mode}]")
    logger.info("Chạy tiếp: python scripts/evaluate.py")
 
 
if __name__ == "__main__":
    main()

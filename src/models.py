"""
models.py — Khởi tạo 3 mô hình Object Detection
    1. YOLOv8n        (ultralytics)
    2. Faster R-CNN   (torchvision)
    3. SSD300         (torchvision)
"""

import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    ssd300_vgg16,
    FasterRCNN_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDClassificationHead
from src.utils import get_logger

logger = get_logger("models")


# ─────────────────────────────────────────────────────────────
# Mô hình 2: Faster R-CNN ResNet50-FPN
# ─────────────────────────────────────────────────────────────
def build_fasterrcnn(num_classes: int,
                     pretrained: bool = True,
                     trainable_backbone_layers: int = 3) -> nn.Module:
    """
    Khởi tạo Faster R-CNN với backbone ResNet50-FPN pretrained.

    Args:
        num_classes: số lớp + 1 (background)
        pretrained: dùng pretrained COCO weights
        trainable_backbone_layers: số lớp backbone được unfreeze (0-5)

    Returns:
        model: nn.Module
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model   = fasterrcnn_resnet50_fpn(
        weights                    = weights,
        trainable_backbone_layers  = trainable_backbone_layers,
    )

    # Thay head classifier cho đúng số lớp của bài toán
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Đếm tham số
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Faster R-CNN — Tổng params: {total/1e6:.1f}M, "
                f"Trainable: {trainable/1e6:.1f}M")

    return model


# ─────────────────────────────────────────────────────────────
# Mô hình 3: SSD300 VGG16
# ─────────────────────────────────────────────────────────────
def build_ssd(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Khởi tạo SSD300 với backbone VGG16 pretrained.

    Args:
        num_classes: số lớp + 1 (background)
        pretrained: dùng pretrained COCO weights

    Returns:
        model: nn.Module
    """
    weights = SSD300_VGG16_Weights.DEFAULT if pretrained else None
    model   = ssd300_vgg16(weights=weights)

    # Thay classification head cho đúng số lớp
    num_anchors = model.anchor_generator.num_anchors_per_location()
    in_channels = [
        model.head.classification_head.module_list[i].in_channels
        for i in range(len(num_anchors))
    ]
    model.head.classification_head = SSDClassificationHead(
        in_channels, num_anchors, num_classes
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"SSD300 — Tổng params: {total/1e6:.1f}M, "
                f"Trainable: {trainable/1e6:.1f}M")

    return model


# ─────────────────────────────────────────────────────────────
# Factory — lấy model theo tên
# ─────────────────────────────────────────────────────────────
def get_model(model_name: str,
              num_classes: int,
              cfg: dict = None) -> nn.Module:
    """
    Trả về model theo tên.

    Args:
        model_name: 'fasterrcnn' | 'ssd'
        num_classes: số lớp (chưa cộng background, hàm tự +1)
        cfg: dict config từ file yaml

    Returns:
        model: nn.Module chưa move sang device
    """
    nc = num_classes + 1    # +1 cho background

    if model_name == "fasterrcnn":
        layers = cfg.get("model", {}).get("trainable_backbone_layers", 3)
        return build_fasterrcnn(nc, pretrained=True,
                                trainable_backbone_layers=layers)
    elif model_name == "ssd":
        return build_ssd(nc, pretrained=True)
    else:
        raise ValueError(f"Model không hỗ trợ: {model_name}. "
                         f"Chọn: 'fasterrcnn' | 'ssd'")

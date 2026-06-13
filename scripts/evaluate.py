"""
evaluate.py — Đánh giá toàn diện 3 mô hình sau khi train xong

Output:
    runs/evaluation/baseline/   ← kết quả trước Tuning
    runs/evaluation/tuned/      ← kết quả sau Tuning

Cách chạy:
    python scripts/evaluate.py                          ← baseline, cả 3 mô hình
    python scripts/evaluate.py --phase tuned            ← tuned, cả 3 mô hình
    python scripts/evaluate.py --phase tuned --model yolov8  ← tuned, 1 mô hình
"""

import os
import sys
import csv
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils   import get_logger, get_device, load_config
from src.dataset import TrafficSignDataset, collate_fn
from src.models  import get_model
from torch.utils.data import DataLoader

logger = get_logger("evaluate", log_dir="runs/logs")

# ── SỬA CHỖ 1 ────────────────────────────────────────────────
# EVAL_DIR không còn cố định — sẽ được gán trong main() theo --phase
# "runs/evaluation/baseline" hoặc "runs/evaluation/tuned"
EVAL_DIR = "runs/evaluation/baseline"   # giá trị mặc định, main() sẽ ghi đè
# ─────────────────────────────────────────────────────────────


def get_class_names(dataset_path: str) -> list:
    with open(os.path.join(dataset_path, "data.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["names"]


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    xi1 = max(box1[0], box2[0]); yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2]); yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_predictions(pred_boxes, pred_labels, pred_scores,
                      gt_boxes, gt_labels, iou_thresh=0.5):
    matched_gt = set()
    tp, fp = [], []
    if len(pred_scores) > 0:
        order = np.argsort(-np.array(pred_scores))
        pred_boxes  = [pred_boxes[i]  for i in order]
        pred_labels = [pred_labels[i] for i in order]
        pred_scores = [pred_scores[i] for i in order]

    for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
        best_iou, best_gt = 0.0, -1
        for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
            if gi in matched_gt or gl != pl:
                continue
            iou = compute_iou(np.array(pb), np.array(gb))
            if iou > best_iou:
                best_iou = iou; best_gt = gi
        if best_iou >= iou_thresh and best_gt >= 0:
            tp.append((pl, ps)); matched_gt.add(best_gt)
        else:
            fp.append((pl, ps))
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def compute_metrics_from_matches(all_tp, all_fp, all_fn_count, class_names):
    tp_per = defaultdict(int)
    fp_per = defaultdict(int)
    fn_per = defaultdict(int)
    for cls_id, _ in all_tp: tp_per[cls_id] += 1
    for cls_id, _ in all_fp: fp_per[cls_id] += 1
    for cls_id, cnt in all_fn_count.items(): fn_per[cls_id] += cnt

    results = {}
    for i, name in enumerate(class_names):
        tp = tp_per[i+1]; fp = fp_per[i+1]; fn = fn_per[i+1]
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        results[name] = {"precision": prec, "recall": rec, "f1": f1,
                         "tp": tp, "fp": fp, "fn": fn}
    if results:
        avg_p = np.mean([v["precision"] for v in results.values()])
        avg_r = np.mean([v["recall"]    for v in results.values()])
        avg_f = np.mean([v["f1"]        for v in results.values()])
        results["__overall__"] = {"precision": avg_p, "recall": avg_r, "f1": avg_f}
    return results


def build_confusion_matrix(pred_labels_all, gt_labels_all, nc):
    cm = np.zeros((nc, nc), dtype=int)
    for gt, pred in zip(gt_labels_all, pred_labels_all):
        if 0 <= gt < nc and 0 <= pred < nc:
            cm[gt][pred] += 1
    return cm


def plot_confusion_matrix(cm, class_names, save_path, title):
    fig, ax = plt.subplots(figsize=(max(10, len(class_names)),
                                    max(8,  len(class_names))))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0)
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax, linewidths=0.5, linecolor="white",
                annot_kws={"size": 9})
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j+0.5, i+0.72, f"({cm[i][j]})",
                    ha="center", va="center", fontsize=7, color="gray")
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# Đánh giá YOLOv8
# ── SỬA CHỖ 2 ────────────────────────────────────────────────
# Thêm tham số `phase` để tìm đúng weights baseline hoặc tuned
# ─────────────────────────────────────────────────────────────
def evaluate_yolov8(dataset_path: str, class_names: list,
                    phase: str = "baseline") -> dict:
    from ultralytics import YOLO

    # Tìm weights theo phase
    candidates = [
        f"runs/detect/runs/yolov8/{phase}/weights/best.pt",
        f"runs/yolov8/{phase}/weights/best.pt",
    ]
    weights = None
    for c in candidates:
        if os.path.exists(c):
            weights = c
            break
    if not weights:
        # Tìm tự động theo phase
        found = list(Path("runs").rglob(f"yolov8*/{phase}/weights/best.pt"))
        if not found:
            found = list(Path("runs").rglob("yolov8*/weights/best.pt"))
        if found:
            weights = str(found[0])
        else:
            logger.error(f"Không tìm thấy weights YOLOv8 [{phase}]!")
            return {}

    logger.info(f"  YOLOv8 [{phase}] weights: {weights}")
    model = YOLO(weights)

    metrics = model.val(
        data    = os.path.join(dataset_path, "data.yaml"),
        split   = "test",
        imgsz   = 512,
        conf    = 0.25,
        iou     = 0.5,
        device  = 0 if torch.cuda.is_available() else "cpu",
        verbose = False,
        plots   = False,
    )

    mp       = float(metrics.box.mp)
    mr       = float(metrics.box.mr)
    map50    = float(metrics.box.map50)
    map5095  = float(metrics.box.map)
    p_per    = metrics.box.p
    r_per    = metrics.box.r

    results = {}
    for i, name in enumerate(class_names):
        pi = float(p_per[i]) if i < len(p_per) else 0.0
        ri = float(r_per[i]) if i < len(r_per) else 0.0
        fi = 2*pi*ri/(pi+ri) if (pi+ri)>0 else 0.0
        results[name] = {"precision": pi, "recall": ri, "f1": fi}

    results["__overall__"] = {
        "precision": mp, "recall": mr,
        "f1": 2*mp*mr/(mp+mr) if (mp+mr)>0 else 0.0,
        "mAP50": map50, "mAP50_95": map5095,
    }

    # Confusion matrix — truyền EVAL_DIR động
    _build_yolo_confusion_matrix(model, dataset_path, class_names, phase)

    logger.info(f"  YOLOv8 [{phase}] — mAP50={map50:.4f} | "
                f"mAP50-95={map5095:.4f} | P={mp:.4f} | R={mr:.4f}")
    return results


def _build_yolo_confusion_matrix(model, dataset_path, class_names, phase):
    import glob as _glob
    nc = len(class_names)
    pred_flat, gt_flat = [], []
    test_imgs = (_glob.glob(f"{dataset_path}/test/images/*.jpg") +
                 _glob.glob(f"{dataset_path}/test/images/*.png"))
    for img_path in test_imgs:
        lbl_path = img_path.replace("images","labels").rsplit(".",1)[0]+".txt"
        if not os.path.exists(lbl_path): continue
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts: gt_flat.append(int(parts[0]))
        res = model.predict(img_path, conf=0.25, iou=0.5, verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            for box in res.boxes:
                pred_flat.append(int(box.cls.item()))
        else:
            with open(lbl_path) as f:
                for _ in f: pred_flat.append(nc)
    min_len = min(len(pred_flat), len(gt_flat))
    cm = build_confusion_matrix(pred_flat[:min_len], gt_flat[:min_len], nc)
    plot_confusion_matrix(
        cm, class_names,
        save_path = f"{EVAL_DIR}/confusion_matrix_yolov8.png",
        title     = f"Confusion Matrix — YOLOv8n [{phase}] (test set)"
    )


# ─────────────────────────────────────────────────────────────
# Đánh giá Faster R-CNN & SSD
# ── SỬA CHỖ 3 ────────────────────────────────────────────────
# Thêm tham số `phase` để tìm đúng weights và config
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_torchvision_model(model_name: str, dataset_path: str,
                                class_names: list, device: torch.device,
                                phase: str = "baseline") -> dict:

    # Tìm weights theo đúng phase
    weights_path = f"runs/{model_name}/{phase}/weights/best.pt"
    if not os.path.exists(weights_path):
        found = list(Path("runs").rglob(f"{model_name}*/{phase}/weights/best.pt"))
        if not found:
            found = list(Path("runs").rglob(f"{model_name}*/weights/best.pt"))
        if found:
            weights_path = str(found[0])
        else:
            logger.error(f"Không tìm thấy weights {model_name} [{phase}]!")
            return {}

    logger.info(f"  {model_name} [{phase}] weights: {weights_path}")

    # Đọc đúng file config theo phase
    cfg_path = (f"configs/{model_name}_tuned.yaml"
                if phase == "tuned" else f"configs/{model_name}.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = f"configs/{model_name}.yaml"   # fallback về baseline config

    cfg   = load_config(cfg_path)
    model = get_model(model_name, len(class_names), cfg)
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    img_size    = cfg["training"]["image_size"]
    test_ds     = TrafficSignDataset(
        img_dir     = f"{dataset_path}/test/images",
        lbl_dir     = f"{dataset_path}/test/labels",
        class_names = class_names,
        img_size    = img_size,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    all_tp, all_fp   = [], []
    all_fn_count     = defaultdict(int)
    pred_flat, gt_flat = [], []
    CONF_THRESH, IOU_THRESH = 0.3, 0.5

    for imgs, targets in test_loader:
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)
        for output, target in zip(outputs, targets):
            scores   = output["scores"].cpu().numpy()
            keep     = scores >= CONF_THRESH
            p_boxes  = output["boxes"].cpu().numpy()[keep].tolist()
            p_labels = output["labels"].cpu().numpy()[keep].tolist()
            p_scores = scores[keep].tolist()
            g_boxes  = target["boxes"].numpy().tolist()
            g_labels = target["labels"].numpy().tolist()

            tp, fp, fn = match_predictions(p_boxes, p_labels, p_scores,
                                           g_boxes, g_labels, IOU_THRESH)
            all_tp.extend(tp); all_fp.extend(fp)

            gt_matched = set()
            for pb, pl, ps in zip(p_boxes, p_labels, p_scores):
                for gi, (gb, gl) in enumerate(zip(g_boxes, g_labels)):
                    if gi in gt_matched or gl != pl: continue
                    if compute_iou(np.array(pb), np.array(gb)) >= IOU_THRESH:
                        gt_matched.add(gi); break
            for gi, gl in enumerate(g_labels):
                if gi not in gt_matched: all_fn_count[gl] += 1

            for gb, gl in zip(g_boxes, g_labels):
                best_iou, best_pl = 0.0, -1
                for pb, pl, ps in zip(p_boxes, p_labels, p_scores):
                    iou = compute_iou(np.array(pb), np.array(gb))
                    if iou > best_iou: best_iou = iou; best_pl = pl
                gt_flat.append(gl - 1)
                pred_flat.append(best_pl - 1 if best_pl > 0 else len(class_names))

    results = compute_metrics_from_matches(all_tp, all_fp, all_fn_count, class_names)
    overall = results.get("__overall__", {})
    mp = overall.get("precision", 0.0)
    mr = overall.get("recall",    0.0)
    logger.info(f"  {model_name} [{phase}] — P={mp:.4f} | R={mr:.4f} | "
                f"F1={overall.get('f1',0):.4f}")

    nc = len(class_names)
    cm = build_confusion_matrix(
        [p if 0<=p<nc else nc-1 for p in pred_flat],
        [g if 0<=g<nc else nc-1 for g in gt_flat], nc
    )
    plot_confusion_matrix(
        cm, class_names,
        save_path = f"{EVAL_DIR}/confusion_matrix_{model_name}.png",
        title     = f"Confusion Matrix — {model_name.title()} [{phase}] (test set)"
    )
    return results


# ─────────────────────────────────────────────────────────────
# Vẽ biểu đồ so sánh
# ── SỬA CHỖ 4 ────────────────────────────────────────────────
# Thêm tham số `phase` vào tiêu đề biểu đồ
# ─────────────────────────────────────────────────────────────
def plot_metrics_comparison(all_results: dict, class_names: list,
                             phase: str = "baseline"):
    model_names = list(all_results.keys())
    colors      = {"yolov8": "#4A90D9", "fasterrcnn": "#7B68EE", "ssd": "#FF7F50"}
    phase_label = "Baseline (trước Tuning)" if phase == "baseline" else "Tuned (sau Tuning)"

    # Biểu đồ 1: Overall P/R/F1
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"So sánh Metrics tổng thể — 3 Mô hình [{phase_label}]",
                 fontsize=14, fontweight="bold", y=1.02)
    for ax, metric in zip(axes, ["precision","recall","f1"]):
        label_map = {"precision":"Precision","recall":"Recall","f1":"F1-score"}
        vals = [all_results[m].get("__overall__",{}).get(metric,0) for m in model_names]
        bars = ax.bar(model_names, vals,
                      color=[colors.get(m,"#888") for m in model_names],
                      edgecolor="white", linewidth=0.5, width=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(label_map[metric], fontsize=12, fontweight="bold")
        ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.3)
        ax.set_xticklabels(model_names, rotation=10)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    plt.tight_layout()
    p1 = f"{EVAL_DIR}/overall_metrics_comparison.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  Saved: {p1}")

    # Biểu đồ 2: F1 per class
    nc = len(class_names); x = np.arange(nc); w = 0.25
    fig, ax = plt.subplots(figsize=(max(14, nc*1.5), 6))
    for i, mname in enumerate(model_names):
        f1_vals = [all_results[mname].get(cls,{}).get("f1",0) for cls in class_names]
        ax.bar(x+i*w, f1_vals, width=w, label=mname,
               color=colors.get(mname,"#888"), edgecolor="white", linewidth=0.5)
    ax.set_xticks(x+w); ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("F1-score", fontsize=11)
    ax.set_title(f"F1-score theo từng lớp — 3 Mô hình [{phase_label}]",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4)
    plt.tight_layout()
    p2 = f"{EVAL_DIR}/f1_per_class_comparison.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  Saved: {p2}")

    # Biểu đồ 3: Radar chart
    _plot_radar(all_results, model_names, colors, phase_label)


def _plot_radar(all_results, model_names, colors, phase_label):
    metrics = ["precision","recall","f1"]
    labels  = ["Precision","Recall","F1-score"]
    N = len(metrics)
    angles = [n/float(N)*2*np.pi for n in range(N)] ; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    for mname in model_names:
        vals = [all_results[mname].get("__overall__",{}).get(m,0) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=mname,
                color=colors.get(mname,"#888"))
        ax.fill(angles, vals, alpha=0.1, color=colors.get(mname,"#888"))
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0,1); ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8)
    ax.set_title(f"Radar Chart — [{phase_label}]",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1), fontsize=10)
    plt.tight_layout()
    p = f"{EVAL_DIR}/radar_chart_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    logger.info(f"  Saved: {p}")


# ─────────────────────────────────────────────────────────────
# Lưu CSV + báo cáo TXT
# ── SỬA CHỖ 5 ────────────────────────────────────────────────
# Thêm tham số `phase` vào header báo cáo và tên file CSV
# ─────────────────────────────────────────────────────────────
def save_summary_table(all_results: dict, class_names: list,
                        phase: str = "baseline"):
    csv_path = f"{EVAL_DIR}/summary_table_{phase}.csv"
    rows = [["Phase", "Model", "Class", "Precision", "Recall",
             "F1-score", "mAP50", "mAP50-95"]]
    for mname, res in all_results.items():
        for cls in class_names:
            r = res.get(cls, {})
            rows.append([phase, mname, cls,
                         f"{r.get('precision',0):.4f}",
                         f"{r.get('recall',   0):.4f}",
                         f"{r.get('f1',       0):.4f}",
                         "-", "-"])
        ov = res.get("__overall__", {})
        rows.append([phase, mname, "OVERALL",
                     f"{ov.get('precision',0):.4f}",
                     f"{ov.get('recall',   0):.4f}",
                     f"{ov.get('f1',       0):.4f}",
                     f"{ov.get('mAP50',    0):.4f}",
                     f"{ov.get('mAP50_95', 0):.4f}"])
        rows.append([])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    logger.info(f"  Saved: {csv_path}")


def save_text_report(all_results: dict, class_names: list,
                      phase: str = "baseline"):
    report_path = f"{EVAL_DIR}/evaluation_report_{phase}.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    phase_label = "Baseline (TRƯỚC Hyperparameter Tuning)" \
                  if phase == "baseline" else "Tuned (SAU Hyperparameter Tuning)"

    lines = []
    lines.append("=" * 70)
    lines.append("  BÁO CÁO ĐÁNH GIÁ — BIỂN BÁO GIAO THÔNG VIỆT NAM")
    lines.append(f"  Thời gian : {ts}")
    lines.append(f"  Giai đoạn : {phase_label}")
    lines.append("=" * 70)

    for mname, res in all_results.items():
        lines.append(f"\n{'─'*70}")
        lines.append(f"  MÔ HÌNH: {mname.upper()}  [{phase.upper()}]")
        lines.append(f"{'─'*70}")
        lines.append(f"  {'Lớp':<35} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        lines.append(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
        for cls in class_names:
            r  = res.get(cls, {})
            p  = r.get("precision", 0)
            rc = r.get("recall",    0)
            f1 = r.get("f1",        0)
            flag = " ⚠" if f1 < 0.5 else ""
            lines.append(f"  {cls:<35} {p:>10.4f} {rc:>10.4f} {f1:>10.4f}{flag}")
        ov = res.get("__overall__", {})
        lines.append(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10}")
        lines.append(f"  {'OVERALL (macro avg)':<35} "
                     f"{ov.get('precision',0):>10.4f} "
                     f"{ov.get('recall',   0):>10.4f} "
                     f"{ov.get('f1',       0):>10.4f}")
        if "mAP50" in ov:
            lines.append(f"\n  mAP@50      : {ov['mAP50']:.4f}")
            lines.append(f"  mAP@50-95   : {ov['mAP50_95']:.4f}")

    lines.append(f"\n{'='*70}")
    lines.append("  BẢNG SO SÁNH TỔNG HỢP")
    lines.append(f"{'='*70}")
    lines.append(f"  {'Mô hình':<15} {'Precision':>10} {'Recall':>10} "
                 f"{'F1':>10} {'mAP50':>10} {'mAP50-95':>10}")
    lines.append(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for mname, res in all_results.items():
        ov = res.get("__overall__", {})
        lines.append(f"  {mname:<15} "
                     f"{ov.get('precision',0):>10.4f} "
                     f"{ov.get('recall',   0):>10.4f} "
                     f"{ov.get('f1',       0):>10.4f} "
                     f"{ov.get('mAP50',    0):>10.4f} "
                     f"{ov.get('mAP50_95', 0):>10.4f}")

    lines.append(f"\n{'='*70}")
    lines.append("  OUTPUT FILES")
    lines.append(f"{'─'*70}")
    for fname in sorted(os.listdir(EVAL_DIR)):
        lines.append(f"  {EVAL_DIR}/{fname}")
    lines.append("=" * 70)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    logger.info(f"\n  Saved: {report_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    global EVAL_DIR   # cho phép ghi đè biến module-level

    parser = argparse.ArgumentParser(description="Evaluate traffic sign detection models")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all","yolov8","fasterrcnn","ssd"],
                        help="Mô hình cần đánh giá")
    parser.add_argument("--data",  type=str, default="data/processed",
                        help="Đường dẫn dataset")
    # ── SỬA CHỖ 1 (tiếp) ──────────────────────────────────────
    # Thêm --phase để chọn baseline hoặc tuned
    parser.add_argument("--phase", type=str, default="baseline",
                        choices=["baseline", "tuned"],
                        help="Đánh giá baseline (trước tuning) hay tuned (sau tuning)")
    # ──────────────────────────────────────────────────────────
    args = parser.parse_args()

    if not os.path.exists(args.data):
        logger.error(f"Dataset không tồn tại: {args.data}")
        sys.exit(1)

    # Gán EVAL_DIR động theo phase — kết quả 2 lần chạy không ghi đè nhau
    EVAL_DIR = f"runs/evaluation/{args.phase}"
    os.makedirs(EVAL_DIR, exist_ok=True)

    device      = get_device()
    class_names = get_class_names(args.data)
    phase_label = "Baseline" if args.phase == "baseline" else "Tuned"

    logger.info(f"Classes ({len(class_names)}): {class_names}")
    logger.info(f"Phase   : {phase_label}")
    logger.info(f"Lưu tại : {EVAL_DIR}/\n")

    all_results = {}
    run = args.model

    if run in ("all", "yolov8"):
        logger.info("\n" + "="*50)
        logger.info(f"  Đánh giá YOLOv8n [{phase_label}] ...")
        logger.info("="*50)
        res = evaluate_yolov8(args.data, class_names, phase=args.phase)
        if res: all_results["yolov8"] = res

    if run in ("all", "fasterrcnn"):
        logger.info("\n" + "="*50)
        logger.info(f"  Đánh giá Faster R-CNN [{phase_label}] ...")
        logger.info("="*50)
        res = evaluate_torchvision_model("fasterrcnn", args.data,
                                          class_names, device, phase=args.phase)
        if res: all_results["fasterrcnn"] = res

    if run in ("all", "ssd"):
        logger.info("\n" + "="*50)
        logger.info(f"  Đánh giá SSD300 [{phase_label}] ...")
        logger.info("="*50)
        res = evaluate_torchvision_model("ssd", args.data,
                                          class_names, device, phase=args.phase)
        if res: all_results["ssd"] = res

    if len(all_results) > 1:
        logger.info("\n" + "="*50)
        logger.info("  Vẽ biểu đồ so sánh ...")
        logger.info("="*50)
        plot_metrics_comparison(all_results, class_names, phase=args.phase)

    if all_results:
        save_summary_table(all_results, class_names, phase=args.phase)
        save_text_report(all_results,   class_names, phase=args.phase)

    logger.info(f"\n✅ Đánh giá [{phase_label}] hoàn tất! Xem kết quả: {EVAL_DIR}/")


if __name__ == "__main__":
    main()
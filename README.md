# Phát Hiện & Phân Loại Biển Báo Giao Thông Việt Nam
**Deep Learning | Object Detection | YOLOv8 · Faster R-CNN · SSD300**

---

## Cấu trúc dự án
```
bien_bao_detection/
├── data/processed/          ← Dataset (download từ Roboflow)
├── src/                     ← Source code module
│   ├── dataset.py           ← Dataset class & DataLoader
│   ├── models.py            ← Khởi tạo 3 mô hình
│   ├── trainer.py           ← Training loop
│   └── utils.py             ← Hàm tiện ích
├── configs/                 ← Cấu hình từng mô hình
├── scripts/
│   ├── check_gpu.py         ← Kiểm tra môi trường
│   ├── download_dataset.py  ← Download dataset
│   ├── train_all.py         ← Train 3 mô hình
│   └── evaluate.py          ← Đánh giá kết quả
├── runs/                    ← Kết quả training (tự sinh)
├── .env                     ← API keys (không commit)
└── requirements.txt
```

---

## Cài đặt

### Bước 1 — Tạo môi trường ảo
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Bước 2 — Cài PyTorch với CUDA
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Bước 3 — Cài các thư viện còn lại
```bash
pip install -r requirements.txt
```

### Bước 4 — Cấu hình API key
```bash
# Sao chép file .env mẫu và điền thông tin Roboflow
# Mở file .env và điền:
ROBOFLOW_API_KEY=your_key
ROBOFLOW_WORKSPACE=your_workspace
ROBOFLOW_PROJECT=bien-bao-giao-thong-vn
ROBOFLOW_VERSION=1
```

---

## Chạy dự án
**Run the following command:

```bash
# 1. Kiểm tra GPU và môi trường
python scripts/check_gpu.py

# 2. Download dataset từ Roboflow
python scripts/download_dataset.py

# 3. Train tất cả 3 mô hình
python scripts/train_all.py

# Hoặc train từng mô hình
python scripts/train_all.py --model yolov8
python scripts/train_all.py --model fasterrcnn
python scripts/train_all.py --model ssd

# Resume nếu bị ngắt giữa chừng
python scripts/train_all.py --model fasterrcnn --resume runs/fasterrcnn/baseline/weights/last.pt

# 4. Đánh giá kết quả
python scripts/evaluate.py
```

---

## Các lớp biển báo (10 lớp)
**The project uses a Vietnamese Traffic Sign Dataset for object detection.

| ID | Tên lớp | Mã biển |
|---|---|---|
| 0 | cam_di_nguoc_chieu | 102 |
| 1 | cam_do_xe | 131a |
| 2 | cam_do_xe_ngay_chan | 131c |
| 3 | cam_do_xe_ngay_le | 131b |
| 4 | cam_dung_va_do | 130 |
| 5 | cam_oto | 103a |
| 6 | cam_oto_re_phai | 103b |
| 7 | duong_mot_chieu | 407a |
| 8 | giao_nhau_duong_uu_tien | 208 |
| 9 | nguoi_di_bo_sang_ngang | 423b |

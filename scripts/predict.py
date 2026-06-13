"""
predict.py — Demo inference với giọng đọc tiếng Việt
    - Chạy webcam realtime
    - Chạy trên ảnh hoặc video từ file
    - Phát giọng đọc tên biển báo khi detect được

Cách chạy:
    # Webcam realtime (mặc định)
    python scripts/predict.py

    # Chạy trên ảnh
    python scripts/predict.py --source path/to/image.jpg

    # Chạy trên video
    python scripts/predict.py --source path/to/video.mp4

    # Tắt giọng đọc
    python scripts/predict.py --no-voice

    # Chọn mô hình (mặc định yolov8)
    python scripts/predict.py --model fasterrcnn
    python scripts/predict.py --model ssd

Phím tắt khi đang chạy:
    Q hoặc ESC  : Thoát
    S           : Chụp ảnh màn hình lưu vào runs/demo/
    V           : Bật/tắt giọng đọc
    P           : Pause / Resume
"""

import os
import sys
import cv2
import time
import queue
import threading
import argparse
import warnings
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils  import get_logger, get_device, load_config

logger = get_logger("predict")

# ─────────────────────────────────────────────────────────────
# Tên tiếng Việt đầy đủ cho từng lớp biển báo
# ─────────────────────────────────────────────────────────────
VIET_NAMES = {
    "cam_di_nguoc_chieu_102"     : "Cấm đi ngược chiều",
    "cam_do_xe_131a"             : "Cấm đỗ xe",
    "cam_do_xe_ngay_chan_131c"   : "Cấm đỗ xe ngày chẵn",
    "cam_do_xe_ngay_le_131b"     : "Cấm đỗ xe ngày lẻ",
    "cam_dung_va_do_130"         : "Cấm dừng và đỗ xe",
    "cam_oto_103a"               : "Cấm ô tô",
    "cam_oto_re_phai_103b"       : "Cấm ô tô rẽ phải",
    "duong_mot_chieu_407a"       : "Đường một chiều",
    "giao_nhau_duong_uu_tien_208": "Giao nhau với đường ưu tiên",
    "nguoi_di_bo_sang_ngang_423b": "Người đi bộ sang ngang",
}

# Màu bounding box cho từng lớp (BGR)
CLASS_COLORS = [
    (0,   200, 200),   # teal
    (200, 100,   0),   # blue
    (0,   150, 255),   # orange
    (150,   0, 255),   # purple
    (0,   255, 150),   # green
    (255, 100,   0),   # cyan
    (100,   0, 255),   # red-purple
    (0,   255, 255),   # yellow
    (255, 150,   0),   # light blue
    (180, 255,   0),   # lime
]


# ─────────────────────────────────────────────────────────────
# Voice Engine — chạy trên thread riêng để không block video
# ─────────────────────────────────────────────────────────────
class VoiceEngine:
    """
    Phát giọng đọc tiếng Việt trên thread riêng.
    Dùng Windows SAPI (win32com) — ổn định nhất trên Windows.
    Fallback sang pyttsx3 nếu không có win32com.
    """

    def __init__(self, cooldown: float = 3.0):
        self.cooldown     = cooldown
        self.enabled      = True
        self._queue       = queue.Queue()
        self._last_spoken = {}
        self._ready       = False

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _init_engine(self):
        """Thử Windows SAPI trước, fallback pyttsx3."""
        # Cách 1: Windows SAPI qua win32com — ổn định nhất
        try:
            from gtts import gTTS
            import pygame
            pygame.mixer.init()
            # Test thử 1 câu ngắn
            import tempfile, os
            tts = gTTS(text="xin chào", lang="vi", slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name
            tts.save(tmp)
            os.unlink(tmp)

            self._mode  = "gtts"
            self._ready = True
            logger.info("✓ Voice: gTTS tiếng Việt (Google)")
            return
        except Exception as e:
            logger.debug(f"gTTS không khả dụng: {e}")

        # Cách 2: Windows SAPI — fallback
        try:
            import win32com.client
            self._sapi = win32com.client.Dispatch("SAPI.SpVoice")
            voices = self._sapi.GetVoices()
            for i in range(voices.Count):
                v    = voices.Item(i)
                desc = v.GetDescription()
                if any(kw in desc for kw in ["Vietnamese","Viet","vi-VN"]):
                    self._sapi.Voice = v
                    logger.info(f"  Giọng Việt SAPI: {desc}")
                    break
            self._mode  = "sapi"
            self._ready = True
            logger.info("✓ Voice: Windows SAPI")
            return
        except Exception as e:
            logger.debug(f"SAPI không khả dụng: {e}")

        logger.warning("✗ Không tìm thấy TTS engine")
        self._ready = False


    def _speak_now(self, text: str):
        """Phát âm trực tiếp — gọi từ worker thread."""
        try:
            if self._mode == "gtts":
                import tempfile, os
                from gtts import gTTS
                import pygame

                if not pygame.mixer.get_init():
                    pygame.mixer.init()

                tts = gTTS(text=text, lang="vi", slow=False)
                with tempfile.NamedTemporaryFile(
                        suffix=".mp3", delete=False) as f:
                    tmp_path = f.name
                tts.save(tmp_path)

                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                # Chờ đọc xong rồi mới xóa file
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                pygame.mixer.music.unload()
                os.unlink(tmp_path)

            elif self._mode == "sapi":
                self._sapi.Speak(text)

        except Exception as e:
            logger.debug(f"TTS error: {e}")

    def _worker(self):
        """Thread worker — khởi tạo engine rồi đọc queue."""
        self._init_engine()
        while True:
            text = self._queue.get()
            if text is None:
                break
            if self._ready and self.enabled:
                self._speak_now(text)
            self._queue.task_done()

    def speak(self, label: str):
        """Đọc tên biển báo — có cooldown."""
        if not self.enabled or not self._ready:
            return
        now  = time.time()
        last = self._last_spoken.get(label, 0)
        if now - last >= self.cooldown:
            self._last_spoken[label] = now
            viet_name = VIET_NAMES.get(label, label.replace("_", " "))
            # Xóa queue cũ — ưu tiên biển mới nhất
            while not self._queue.empty():
                try: self._queue.get_nowait()
                except: pass
            self._queue.put(viet_name)
            logger.info(f"  🔊 Đọc: {viet_name}")

    def toggle(self):
        self.enabled = not self.enabled
        logger.info(f"Giọng đọc: {'BẬT' if self.enabled else 'TẮT'}")
        return self.enabled

    def stop(self):
        self._queue.put(None)


# ─────────────────────────────────────────────────────────────
# Vẽ kết quả lên frame
# ─────────────────────────────────────────────────────────────
def get_font(size: int = 18):
    """Lấy font hỗ trợ tiếng Việt."""
    # Thử các font có sẵn trên Windows theo thứ tự ưu tiên
    font_candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except:
                continue
    # Fallback — font mặc định của Pillow
    return ImageFont.load_default()


def put_text_viet(frame: np.ndarray, text: str,
                  pos: tuple, font, color_bg: tuple,
                  color_text: tuple = (255, 255, 255)) -> np.ndarray:
    """Vẽ chữ tiếng Việt lên frame bằng Pillow."""
    # Chuyển frame OpenCV sang Pillow
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)

    x, y = pos
    # Đo kích thước text
    bbox = draw.textbbox((x, y), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]

    # Vẽ nền màu
    padding = 4
    draw.rectangle(
        [x - padding, y - padding,
         x + tw + padding, y + th + padding],
        fill=color_bg
    )
    # Vẽ chữ trắng
    draw.text((x, y), text, font=font, fill=color_text)

    # Chuyển lại sang OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_detections(frame: np.ndarray, detections: list,
                    class_names: list,
                    conf_thresh: float = 0.35) -> np.ndarray:
    """Vẽ bounding box và nhãn tiếng Việt lên frame."""
    font = get_font(size=17)

    for det in detections:
        conf = det["conf"]
        if conf < conf_thresh:
            continue

        x1, y1, x2, y2 = map(int, det["box"])
        cls_id    = det.get("label_id", 0) % len(CLASS_COLORS)
        color_bgr = CLASS_COLORS[cls_id]
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR→RGB cho Pillow

        viet_name = VIET_NAMES.get(det["label_name"],
                                   det["label_name"].replace("_", " "))
        text = f"{viet_name}  {conf:.0%}"

        # Vẽ bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0, 0, 0), 1)

        # Vẽ nhãn tiếng Việt bằng Pillow
        label_y = max(y1 - 26, 4)
        frame = put_text_viet(frame, text, (x1, label_y),
                              font, color_rgb)

    return frame


def draw_overlay(frame: np.ndarray, fps: float, n_det: int,
                 voice_on: bool, paused: bool,
                 model_name: str) -> np.ndarray:
    """Vẽ thanh thông tin góc trên trái bằng Pillow."""
    h, w   = frame.shape[:2]
    font_s = get_font(size=15)

    # Nền mờ
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (340, 85), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    voice_str = "BẬT" if voice_on  else "TẮT"
    pause_str = "  [TẠM DỪNG]" if paused else ""
    fps_color = (0, 200, 0) if fps >= 15 else (0, 165, 255)

    # Dùng Pillow để vẽ chữ tiếng Việt
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)

    lines = [
        (f"Mô hình : {model_name.upper()}",    (200, 200, 200)),
        (f"FPS     : {fps:.1f}{pause_str}",    (0, 200, 0) if fps >= 15 else (255, 165, 0)),
        (f"Detect  : {n_det}  |  Giọng: {voice_str}", (200, 200, 200)),
    ]
    for i, (line, color) in enumerate(lines):
        draw.text((8, 8 + i * 24), line, font=font_s, fill=color)

    # Hướng dẫn phím góc dưới
    guide = "[Q] Thoát  [S] Chụp  [V] Giọng  [P] Dừng"
    draw.text((8, h - 24), guide, font=font_s, fill=(180, 180, 180))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────
# Load mô hình
# ─────────────────────────────────────────────────────────────
def load_model(model_name: str, phase: str, device: torch.device,
               class_names: list):
    """Load model theo tên và phase (baseline/tuned)."""

    if model_name == "yolov8":
        from ultralytics import YOLO
        # Tìm weights
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
            found = list(Path("runs").rglob(f"yolov8*/{phase}/weights/best.pt"))
            if not found:
                found = list(Path("runs").rglob("yolov8*/weights/best.pt"))
            weights = str(found[0]) if found else None

        if not weights:
            raise FileNotFoundError(f"Không tìm thấy weights YOLOv8 [{phase}]")

        logger.info(f"Loaded YOLOv8 [{phase}]: {weights}")
        return YOLO(weights), "yolov8"

    else:
        from src.models import get_model
        cfg_path = (f"configs/{model_name}_tuned.yaml"
                    if phase == "tuned" else f"configs/{model_name}.yaml")
        if not os.path.exists(cfg_path):
            cfg_path = f"configs/{model_name}.yaml"

        cfg    = load_config(cfg_path)
        model  = get_model(model_name, len(class_names), cfg)
        w_path = f"runs/{model_name}/{phase}/weights/best.pt"
        if not os.path.exists(w_path):
            found = list(Path("runs").rglob(f"{model_name}*/weights/best.pt"))
            w_path = str(found[0]) if found else None

        if not w_path:
            raise FileNotFoundError(f"Không tìm thấy weights {model_name} [{phase}]")

        ckpt = torch.load(w_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        logger.info(f"Loaded {model_name} [{phase}]: {w_path}")
        return model, model_name


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────
def infer_yolov8(model, frame: np.ndarray, class_names: list,
                 conf: float = 0.35, imgsz: int = 512) -> list:
    """Chạy inference YOLOv8 và trả về list detections."""
    results = model.predict(frame, conf=conf, imgsz=imgsz,
                            verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls.item())
            conf_  = float(box.conf.item())
            detections.append({
                "box"       : [x1, y1, x2, y2],
                "label_id"  : cls_id,
                "label_name": class_names[cls_id] if cls_id < len(class_names) else "unknown",
                "conf"      : conf_,
            })
    return detections


@torch.no_grad()
def infer_torchvision(model, frame: np.ndarray, class_names: list,
                       device: torch.device, conf: float = 0.35,
                       img_size: int = 512) -> list:
    """Chạy inference Faster R-CNN / SSD."""
    from torchvision import transforms as T
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    tensor = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])(img).unsqueeze(0).to(device)

    h_orig, w_orig = frame.shape[:2]
    output = model(tensor)[0]

    scores  = output["scores"].cpu().numpy()
    boxes   = output["boxes"].cpu().numpy()
    labels  = output["labels"].cpu().numpy()

    detections = []
    for score, box, lbl in zip(scores, boxes, labels):
        if score < conf:
            continue
        # Scale bbox về kích thước gốc
        x1 = box[0] / img_size * w_orig
        y1 = box[1] / img_size * h_orig
        x2 = box[2] / img_size * w_orig
        y2 = box[3] / img_size * h_orig
        cls_id = int(lbl) - 1   # remove background offset
        if cls_id < 0 or cls_id >= len(class_names):
            continue
        detections.append({
            "box"       : [x1, y1, x2, y2],
            "label_id"  : cls_id,
            "label_name": class_names[cls_id],
            "conf"      : float(score),
        })
    return detections


# ─────────────────────────────────────────────────────────────
# Demo Loop chính
# ─────────────────────────────────────────────────────────────
def run_demo(args):
    # ── Setup ──────────────────────────────────────────────
    device      = get_device()
    data_yaml   = os.path.join(args.data, "data.yaml")
    with open(data_yaml, encoding="utf-8") as f:
        class_names = yaml.safe_load(f)["names"]

    logger.info(f"Classes: {class_names}")
    logger.info(f"Model  : {args.model} [{args.phase}]")

    # Load model
    model, model_type = load_model(args.model, args.phase,
                                   device, class_names)

    # Voice engine
    voice = VoiceEngine(cooldown=args.voice_cooldown)
    voice.enabled = not args.no_voice

    # Output dir cho ảnh chụp
    demo_dir = "runs/demo"
    os.makedirs(demo_dir, exist_ok=True)

    # ── Mở source ──────────────────────────────────────────
    if args.source == "0" or args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        is_webcam = True
        logger.info("Mở webcam...")
    else:
        if not os.path.exists(args.source):
            logger.error(f"Không tìm thấy file: {args.source}")
            return
        cap       = cv2.VideoCapture(args.source)
        is_webcam = False
        logger.info(f"Mở file: {args.source}")

    if not cap.isOpened():
        logger.error("Không mở được source! Kiểm tra webcam hoặc đường dẫn file.")
        return

    # ── FPS tracker ────────────────────────────────────────
    fps_deque  = deque(maxlen=30)
    paused     = False
    prev_time  = time.time()

    logger.info("\n" + "="*50)
    logger.info("  DEMO BẮT ĐẦU")
    logger.info("  [Q/ESC] Thoát  [S] Chụp  [V] Voice  [P] Pause")
    logger.info("="*50 + "\n")

    # Thông báo bắt đầu bằng giọng đọc
    if voice.enabled:
        voice._queue.put("Bắt đầu nhận diện biển báo giao thông")
        time.sleep(0.5)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if not is_webcam:
                    logger.info("Video kết thúc.")
                    break
                continue

            # ── Inference ──────────────────────────────────
            t0 = time.time()
            if model_type == "yolov8":
                detections = infer_yolov8(model, frame, class_names,
                                          conf=args.conf)
            else:
                img_size = 300 if args.model == "ssd" else 512
                detections = infer_torchvision(model, frame, class_names,
                                               device, conf=args.conf,
                                               img_size=img_size)

            # ── FPS ────────────────────────────────────────
            now       = time.time()
            fps_deque.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            fps       = np.mean(fps_deque)

            # ── Giọng đọc ──────────────────────────────────
            # Đọc biển báo có confidence cao nhất trong frame
            if detections and voice.enabled:
                best = max(detections, key=lambda d: d["conf"])
                if best["conf"] >= args.conf:
                    voice.speak(best["label_name"])

            # ── Vẽ kết quả ─────────────────────────────────
            frame = draw_detections(frame, detections, class_names, args.conf)
            frame = draw_overlay(frame, fps, len(detections),
                                 voice.enabled, paused, args.model)

        # ── Hiển thị ───────────────────────────────────────
        cv2.imshow("Demo — Nhan Dien Bien Bao Giao Thong  [Q de thoat]", frame)

        # ── Phím tắt ───────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):   # Q hoặc ESC
            logger.info("Thoát demo.")
            break

        elif key in (ord("s"), ord("S")):      # Chụp ảnh
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            savepath = os.path.join(demo_dir, f"capture_{ts}.png")
            cv2.imwrite(savepath, frame)
            logger.info(f"  Đã lưu: {savepath}")

        elif key in (ord("v"), ord("V")):      # Bật/tắt giọng
            state = voice.toggle()

        elif key in (ord("p"), ord("P")):      # Pause
            paused = not paused
            logger.info(f"  {'PAUSED' if paused else 'RESUMED'}")

    # ── Cleanup ────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    voice.stop()
    logger.info("Demo kết thúc.")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Demo nhận diện biển báo giao thông với giọng đọc")

    parser.add_argument("--source", type=str, default="0",
                        help="Nguồn ảnh/video: '0' = webcam, "
                             "hoặc đường dẫn file ảnh/video")
    parser.add_argument("--model",  type=str, default="yolov8",
                        choices=["yolov8", "fasterrcnn", "ssd"],
                        help="Mô hình dùng để detect")
    parser.add_argument("--phase",  type=str, default="baseline",
                        choices=["baseline", "tuned"],
                        help="Dùng weights baseline hay tuned")
    parser.add_argument("--data",   type=str, default="data/processed",
                        help="Đường dẫn dataset (lấy class names)")
    parser.add_argument("--conf",   type=float, default=0.35,
                        help="Ngưỡng confidence (mặc định 0.35)")
    parser.add_argument("--no-voice", action="store_true",
                        help="Tắt giọng đọc")
    parser.add_argument("--voice-cooldown", type=float, default=3.0,
                        help="Giây chờ giữa 2 lần đọc cùng nhãn (mặc định 3.0)")

    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()

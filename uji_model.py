import os
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO

# 1. Konfigurasi Model Lokal (best_v2.pt)
MODEL_CANDIDATES = [
    Path("runs/detect/runs/mango/yolo11n_mango/weights/best_v2.pt"),
    Path("runs/mango/yolo11n_mango/weights/best_v2.pt"),
    Path("runs/mango/weights/best_v2.pt"),
    Path("best_v2.pt"),
    Path("runs/detect/runs/mango/yolo11n_mango/weights/best.pt"),
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if p.exists()), None)
IMAGE_DIR = Path("images")
NUM_IMAGES_TO_INFER = 5
OUTPUT_DIR = Path("results")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

if MODEL_PATH is None or not MODEL_PATH.exists():
    raise FileNotFoundError(
        "File model lokal (best_v2.pt) tidak ditemukan!\n"
        "Pastikan file best_v2.pt sudah diletakkan di runs/ atau direktori proyek."
    )

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(f"Folder gambar tidak ditemukan: {IMAGE_DIR}")

if NUM_IMAGES_TO_INFER < 1:
    raise ValueError("NUM_IMAGES_TO_INFER harus lebih dari 0.")

all_image_paths = sorted(
    path
    for path in IMAGE_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
)

if not all_image_paths:
    raise FileNotFoundError(f"Tidak ada gambar yang didukung di folder: {IMAGE_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = []
for image_path in all_image_paths:
    output_path = OUTPUT_DIR / f"{image_path.stem}_hasil{image_path.suffix}"
    if output_path.exists():
        print(f"Lewati {image_path.name}: hasil sudah ada di {output_path}")
        continue

    image_paths.append(image_path)
    if len(image_paths) == NUM_IMAGES_TO_INFER:
        break

if not image_paths:
    print("Semua gambar sudah memiliki hasil inference. Menjalankan ulang untuk seluruh gambar sampel...")
    image_paths = all_image_paths[:NUM_IMAGES_TO_INFER]

# 2. Load model lokal
print(f"Memuat model lokal dari: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print(f"[OK] Model berhasil dimuat! Kelas: {model.names}")
print(
    f"\nMemulai inference untuk {len(image_paths)} gambar "
    f"(maksimal {NUM_IMAGES_TO_INFER} gambar pada proses ini)..."
)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 3. Inferensi dan simpan hasil untuk tiap gambar
for index, image_path in enumerate(image_paths, start=1):
    print(f"\n[{index}/{len(image_paths)}] Memproses: {image_path.name}")

    image = cv2.imread(str(image_path))
    if image is None:
        print("Gambar gagal dibaca OpenCV, dilewati.")
        continue

    # Prediksi menggunakan Ultralytics YOLO lokal dengan NMS optimal
    results = model.predict(
        source=image,
        conf=0.20,
        iou=0.45,
        agnostic_nms=True,
        imgsz=640,
        verbose=False,
    )[0]
    detections = sv.Detections.from_ultralytics(results)

    labels = [
        f"{model.names[cls_id]} {conf:.2f}"
        for cls_id, conf in zip(detections.class_id, detections.confidence)
    ]

    print(f"Terdeteksi {len(detections)} objek:")
    for cls_id, conf in zip(detections.class_id, detections.confidence):
        print(f" - {model.names[cls_id]}: {conf * 100:.1f}%")

    annotated_image = box_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels
    )

    output_path = OUTPUT_DIR / f"{image_path.stem}_hasil{image_path.suffix}"
    if not cv2.imwrite(str(output_path), annotated_image):
        print(f"Gagal menyimpan hasil anotasi ke: {output_path}")
    else:
        print(f"Hasil disimpan di: {output_path}")

print("\nProses inferensi selesai.")

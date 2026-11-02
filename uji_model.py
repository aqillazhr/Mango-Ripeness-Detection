import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import cv2
import supervision as sv
from inference import get_model

# 1. Konfigurasi
API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "carabao-ripeness-level-0x9me/19"
IMAGE_DIR = Path("images")
NUM_IMAGES_TO_INFER = 5
OUTPUT_DIR = Path("results")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

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
    print("Semua gambar sudah memiliki hasil inference. Tidak ada yang diproses.")
    raise SystemExit(0)

if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY belum terbaca!")

# 2. Load model CARABAO
print("Memuat model CARABAO...")

model = get_model(
    model_id=MODEL_ID,
    api_key=API_KEY
)

print("Model berhasil dimuat!")
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

    results = model.infer(str(image_path))
    predictions = results[0]
    detections = sv.Detections.from_inference(predictions)
    labels = [
        f"{prediction.class_name} {prediction.confidence:.2f}"
        for prediction in predictions.predictions
    ]

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
        print(f"Gagal menyimpan hasil ke: {output_path}")
        continue

    if predictions.predictions:
        for prediction in predictions.predictions:
            print(
                f"Kelas: {prediction.class_name} | "
                f"Confidence: {prediction.confidence:.2%}"
            )
    else:
        print("Tidak ada objek mangga yang terdeteksi.")

    print(f"Hasil disimpan di: {output_path}")

print("\nInference selesai.")

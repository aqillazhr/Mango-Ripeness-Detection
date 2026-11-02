import os
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# 1. Konfigurasi
API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "mango-aiyie/1"
IMAGE_DIR = Path("images")
NUM_IMAGES_TO_INFER = 5
OUTPUT_DIR = Path("results")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

if not API_KEY:
    raise ValueError("ROBOFLOW_API_KEY belum terbaca di file .env!")

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

# 2. Inisialisasi Klien Model Roboflow
print("Memuat klien inferensi Roboflow...")
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY,
)
client.configure(InferenceConfiguration(api_key_transport="header"))
print(f"Model ID: {MODEL_ID} siap digunakan.")
print(
    f"\nMemulai inference untuk {len(image_paths)} gambar "
    f"(maksimal {NUM_IMAGES_TO_INFER} gambar pada proses ini)..."
)


def annotate_classification_image(
    image: np.ndarray, top_class: str, confidence: float
) -> np.ndarray:
    """Menambahkan banner visual status klasifikasi di bagian atas citra."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Skala proporsional ukuran banner & teks terhadap resolusi gambar
    banner_height = max(40, int(h * 0.08))
    font_scale = max(0.6, banner_height / 50.0)
    thickness = max(1, int(font_scale * 2))

    # Tentukan warna banner (BGR): Hijau untuk Fresh, Merah/Marun untuk Rotten
    is_fresh = "fresh" in top_class.lower()
    banner_color = (34, 180, 70) if is_fresh else (40, 40, 220)

    # Buat banner dengan semi-transparansi
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_color, -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    # Teks label
    text = f"{top_class} ({confidence * 100:.1f}%)"
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    text_x = 15
    text_y = int((banner_height + text_h) / 2)

    cv2.putText(
        annotated,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return annotated


# 3. Inferensi dan simpan hasil untuk tiap gambar
for index, image_path in enumerate(image_paths, start=1):
    print(f"\n[{index}/{len(image_paths)}] Memproses: {image_path.name}")

    image = cv2.imread(str(image_path))
    if image is None:
        print("Gambar gagal dibaca OpenCV, dilewati.")
        continue

    try:
        results = client.infer(str(image_path), model_id=MODEL_ID)
        predictions = results.get("predictions", {})
        predicted_classes = results.get("predicted_classes", [])

        if predicted_classes:
            top_class = predicted_classes[0]
            top_conf = predictions.get(top_class, {}).get("confidence", 0.0)
        else:
            # Fallback jika predicted_classes kosong, ambil conf tertinggi
            sorted_preds = sorted(
                predictions.items(),
                key=lambda item: item[1].get("confidence", 0),
                reverse=True,
            )
            top_class = sorted_preds[0][0] if sorted_preds else "Unknown"
            top_conf = sorted_preds[0][1].get("confidence", 0.0) if sorted_preds else 0.0

        print(f"Hasil Klasifikasi: {top_class} ({top_conf * 100:.2f}%)")
        for cls_name, cls_info in predictions.items():
            conf = cls_info.get("confidence", 0.0)
            print(f" - {cls_name}: {conf * 100:.2f}%")

        annotated_image = annotate_classification_image(image, top_class, top_conf)

        output_path = OUTPUT_DIR / f"{image_path.stem}_hasil{image_path.suffix}"
        if not cv2.imwrite(str(output_path), annotated_image):
            print(f"Gagal menyimpan hasil ke: {output_path}")
        else:
            print(f"Hasil disimpan di: {output_path}")

    except Exception as e:
        print(f"Gagal melakukan inferensi pada {image_path.name}: {e}")

print("\nProses selesai.")
